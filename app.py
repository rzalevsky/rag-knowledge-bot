"""Stage 3 of the pipeline: hybrid retrieval -> grounded generation (see
README).

One endpoint, `POST /ask`. Flow, per the README's own pipeline table
("metadata filter narrows the set, then vector search ranks"):

1. Build a Qdrant `Filter` from whichever *hard* constraints the caller
   actually provided (`max_price`, `condition`, `location_country`) —
   omitted fields aren't turned into filter conditions at all.
2. Embed the free-text `query` with the same model `embed.py` used for
   passages (`embeddings.py`, prefixed with `"query: "` per E5's convention)
   so query and passage vectors land in the same space.
3. `client.query_points(...)` with that filter + vector.
4. Zero hits -> return a fixed, deterministic refusal immediately. No LLM
   call. This is the honest-refusal the README's intro is built around:
   "nothing matches" is a real answer, and it shouldn't depend on an LLM
   being reachable or behaving to say so.
5. One or more hits -> hand the retrieved listings to the LLM with a
   grounding/refusal system prompt (never invent a listing/price/detail not
   present, say plainly if none of the *soft* asks — condition quality,
   "quiet", "sturdy" — are actually satisfied rather than recommending a
   poor match, cite URLs when recommending) and return its answer alongside
   the retrieved items.

LLM: LM Studio's local OpenAI-compatible API (`config.LLM_API_BASE`,
`config.LLM_MODEL`) — see README's Generation row for why. Called with
plain `httpx` (already a dependency, same minimal-deps preference as
`ebay_client.py`) rather than pulling in the `openai` package.

Qdrant client, embed function, and LLM function are all injectable via
FastAPI dependency overrides — same DI intent as `EbayClient`'s
`http_client` param and `embed.py`'s `default_embed_fn` — so
`tests/test_app.py` runs without a live Qdrant, a downloaded model, or a
reachable LM Studio server.

Usage:
    docker compose up -d                      # Qdrant on :6333
    python embed.py                           # local/offers.json -> Qdrant
    # LM Studio running locally, a model loaded, serving on LLM_API_BASE
    uvicorn app:app --reload
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Literal

import httpx
from fastapi import Depends, FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

import config
from embeddings import QUERY_PREFIX, EmbedFn, default_embed_fn

REFUSAL_MESSAGE = (
    "No listings match your constraints — nothing to show rather than a forced guess."
)

# Takes OpenAI-shaped chat messages, returns the assistant's reply text.
LlmFn = Callable[[list[dict]], str]


class AskRequest(BaseModel):
    query: str
    max_price: float | None = None
    condition: Literal["new", "refurbished", "used", "for_parts"] | None = None
    location_country: str | None = None
    top_k: int = config.RETRIEVAL_TOP_K


class RetrievedItem(BaseModel):
    id: str
    title: str
    price_amount: float | None
    price_currency: str | None
    condition_bucket: str
    location_country: str | None
    url: str
    score: float


class AskResponse(BaseModel):
    answer: str
    matched: bool  # True iff the hard-constraint filter + vector search returned >=1 item
    items: list[RetrievedItem]


def build_filter(request: AskRequest) -> qmodels.Filter | None:
    """Hard-constraint filter from whichever fields the caller provided —
    omitted fields aren't turned into filter conditions."""
    conditions: list[qmodels.FieldCondition] = []
    if request.max_price is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="price_amount", range=qmodels.Range(lte=request.max_price)
            )
        )
    if request.condition is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="condition_bucket", match=qmodels.MatchValue(value=request.condition)
            )
        )
    if request.location_country is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="location_country",
                match=qmodels.MatchValue(value=request.location_country),
            )
        )
    if not conditions:
        return None
    return qmodels.Filter(must=conditions)


def to_retrieved_item(point: qmodels.ScoredPoint) -> RetrievedItem:
    payload = point.payload or {}
    return RetrievedItem(
        id=str(payload.get("id", point.id)),
        title=payload.get("title", ""),
        price_amount=payload.get("price_amount"),
        price_currency=payload.get("price_currency"),
        condition_bucket=payload.get("condition_bucket", "unknown"),
        location_country=payload.get("location_country"),
        url=payload.get("url", ""),
        score=point.score,
    )


_SYSTEM_PROMPT = (
    "You are a shopping assistant helping a buyer choose a laptop listing. "
    "Answer only using the listings given below — never invent a listing, price, "
    "or detail that isn't present in them. These listings already satisfy the "
    "buyer's hard constraints (price, condition, location, if any were given); "
    "your job is to judge whether any of them actually satisfy the buyer's "
    "free-text request (quality, noise, sturdiness, and similar things a plain "
    "filter can't check). If none of them genuinely do, say so plainly instead of "
    "recommending a poor match anyway. When you do recommend a listing, cite its URL."
)


def build_messages(query: str, items: list[RetrievedItem]) -> list[dict]:
    lines = ["Retrieved listings:"]
    for i, item in enumerate(items, start=1):
        price = (
            f"{item.price_amount} {item.price_currency}"
            if item.price_amount is not None
            else "price unknown"
        )
        lines.append(
            f"{i}. {item.title} — {price} — condition: {item.condition_bucket} — {item.url}"
        )
    user_content = f"Buyer's request: {query}\n\n" + "\n".join(lines)
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def default_llm_fn() -> LlmFn:
    """Calls LM Studio's local OpenAI-compatible chat completions endpoint.
    Imported/constructed lazily via Depends so tests inject a stub instead
    of hitting a real server."""
    http_client = httpx.Client(timeout=60.0)

    def call(messages: list[dict]) -> str:
        response = http_client.post(
            f"{config.LLM_API_BASE}/chat/completions",
            json={"model": config.LLM_MODEL, "messages": messages},
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    return call


# lru_cache so the real model/client are only built once per process (not
# once per request) while still being fully overridable per-test via
# app.dependency_overrides.
@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL)


@lru_cache(maxsize=1)
def get_embed_fn() -> EmbedFn:
    return default_embed_fn()


@lru_cache(maxsize=1)
def get_llm_fn() -> LlmFn:
    return default_llm_fn()


app = FastAPI(title="rag-knowledge-bot")


@app.post("/ask", response_model=AskResponse)
def ask(
    request: AskRequest,
    qdrant: QdrantClient = Depends(get_qdrant_client),
    embed_fn: EmbedFn = Depends(get_embed_fn),
    llm_fn: LlmFn = Depends(get_llm_fn),
) -> AskResponse:
    query_filter = build_filter(request)
    vector = embed_fn([QUERY_PREFIX + request.query])[0]

    # `search()` was removed from qdrant-client in favor of `query_points()`
    # (confirmed live: newer qdrant-client raises AttributeError on `.search`).
    hits = qdrant.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=vector,
        query_filter=query_filter,
        limit=request.top_k,
    ).points

    if not hits:
        return AskResponse(answer=REFUSAL_MESSAGE, matched=False, items=[])

    items = [to_retrieved_item(hit) for hit in hits]
    answer = llm_fn(build_messages(request.query, items))
    return AskResponse(answer=answer, matched=True, items=items)
