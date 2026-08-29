"""Stage 2 of the pipeline: chunks (one item = one chunk, from ingest.py) ->
embeddings -> Qdrant upsert (see README).

Reads `local/offers.json` (produced by `ingest.py`), embeds each item's
title, and upserts one Qdrant point per item, ready for the hybrid
retrieval stage (`app.py`, not built yet): a metadata filter on the payload
fields below narrows the candidate set, then vector search ranks by the
free-text ask.

Embedding model: intfloat/multilingual-e5-small (sentence-transformers,
384-dim, runs locally — no API key, no per-call cost). Chosen because a
live ingest run showed listing titles mixing German/Italian/Polish/English
("Laptop Dell Latitude 3420", "Notebook usato", etc.) and E5 is
contrastively trained specifically for retrieval (query/passage pairs),
not just general sentence similarity — this is a retrieval system, not a
paraphrase-similarity demo. E5's own convention requires prefixing text
before encoding: "passage: " for indexed documents (applied here),
"query: " for search queries (deferred to app.py, which does the querying).

Condition normalization uses `conditionId` from the raw eBay payload, not
the localized `condition` string — `condition` comes back in the seller's
marketplace language ("Gebraucht"/"Used"/"Używany"/"Usato" all mean "used"),
while `conditionId` is numeric and locale-independent (confirmed live:
conditionId "1000" paired with condition "Neu", i.e. German for "New").
Bucketed coarsely — a buyer's filter doesn't need eBay's full tier list:

    1000-1750 -> new
    2000-2750 -> refurbished
    3000-6000 -> used
    7000      -> for_parts
    otherwise -> unknown

The exact numeric ranges above come from a secondary source (eBay's own
condition-id-values reference page timed out when fetched directly while
writing this) corroborated by one live data point (1000 == "Neu"), not
independently confirmed end to end the way ebay_client.py's own findings
are — flagged here rather than presented as equally certain. Both the
bucket (`condition_bucket`, for filtering) and the original raw string
(`condition_raw`, for display) are kept in the payload; normalization is
lossy by design but the source string is never discarded.

Qdrant point IDs: eBay's `itemId` (e.g. "v1|198586449169|0") isn't a valid
Qdrant point ID (Qdrant requires an unsigned int or a UUID). `point_id`
derives a stable UUID via `uuid.uuid5(uuid.NAMESPACE_URL, item_id)`, so
re-running this after a fresh ingest updates existing points (upsert)
instead of duplicating them.

Both the embedder and the Qdrant client are injectable (same pattern as
`EbayClient`'s `http_client` param) so tests can run without downloading
the real model or needing a live Qdrant instance.

The model name/prefixes/vector size and `default_embed_fn` loader live in
`embeddings.py`, shared with `app.py` — passage and query vectors are only
comparable if they come from the same model, so that config isn't
duplicated here.

Usage:
    docker compose up -d                   # starts Qdrant on :6333
    python embed.py                        # local/offers.json -> config.QDRANT_COLLECTION
    python embed.py --input other.json
"""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

import config
from embeddings import PASSAGE_PREFIX, VECTOR_SIZE, EmbedFn, default_embed_fn

INPUT_PATH = Path("local/offers.json")

# See module docstring for the confidence caveat on these ranges.
_CONDITION_BUCKETS: list[tuple[int, int, str]] = [
    (1000, 1750, "new"),
    (2000, 2750, "refurbished"),
    (3000, 6000, "used"),
    (7000, 7000, "for_parts"),
]


def condition_bucket(condition_id: Any) -> str:
    """Coarse, locale-independent condition bucket from eBay's conditionId."""
    try:
        cid = int(condition_id)
    except (TypeError, ValueError):
        return "unknown"
    for lo, hi, bucket in _CONDITION_BUCKETS:
        if lo <= cid <= hi:
            return bucket
    return "unknown"


def point_id(item_id: str) -> str:
    """Deterministic Qdrant-compatible point id for an eBay itemId."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, item_id))


def build_payload(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("raw") or {}
    return {
        "id": item.get("id"),
        "title": item.get("title"),
        "price_amount": item.get("price_amount"),
        "price_currency": item.get("price_currency"),
        "condition_raw": item.get("condition"),
        "condition_bucket": condition_bucket(raw.get("conditionId")),
        "location_country": item.get("location_country"),
        "url": item.get("url"),
    }


def embed_items(items: list[dict[str, Any]], embed_fn: EmbedFn) -> list[qmodels.PointStruct]:
    texts = [PASSAGE_PREFIX + item.get("title", "") for item in items]
    vectors = embed_fn(texts)
    return [
        qmodels.PointStruct(
            id=point_id(item["id"]),
            vector=list(vector),
            payload=build_payload(item),
        )
        for item, vector in zip(items, vectors)
    ]


def ensure_collection(client: QdrantClient, name: str, vector_size: int = VECTOR_SIZE) -> None:
    """Creates the collection if it doesn't exist yet; no-op otherwise."""
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return
    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    items = json.loads(args.input.read_text())

    embed_fn = default_embed_fn()
    points = embed_items(items, embed_fn)
    for item in items:
        print(f"  {item['id']}  {item['title'][:70]}")

    client = QdrantClient(url=config.QDRANT_URL)
    ensure_collection(client, config.QDRANT_COLLECTION)
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)

    print(
        f"\n{len(points)} items embedded -> upserted into "
        f"'{config.QDRANT_COLLECTION}' @ {config.QDRANT_URL}"
    )


if __name__ == "__main__":
    main()
