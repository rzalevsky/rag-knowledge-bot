from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from qdrant_client.http import models as qmodels

from app import (
    REFUSAL_MESSAGE,
    AskRequest,
    app,
    build_filter,
    get_embed_fn,
    get_llm_fn,
    get_qdrant_client,
)


def stub_embed_fn(texts: list[str]) -> list[list[float]]:
    """Deterministic fixed-size vector per text — no real model involved."""
    return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]


def make_scored_point(**overrides) -> qmodels.ScoredPoint:
    payload = {
        "id": "v1|198603108529|0",
        "title": "Laptop Dell Latitude 3420 i5-1135G7 16/512GB SSD NVMe FHD",
        "price_amount": 617.67,
        "price_currency": "PLN",
        "condition_raw": "Używany",
        "condition_bucket": "used",
        "location_country": "PL",
        "url": "https://www.ebay.pl/itm/198603108529",
    }
    payload.update(overrides.pop("payload", {}))
    return qmodels.ScoredPoint(
        id=overrides.pop("id", "00000000-0000-0000-0000-000000000001"),
        version=0,
        score=overrides.pop("score", 0.9),
        payload=payload,
    )


def make_client(qdrant_client=None, embed_fn=None, llm_fn=None) -> TestClient:
    app.dependency_overrides[get_qdrant_client] = lambda: qdrant_client or MagicMock()
    app.dependency_overrides[get_embed_fn] = lambda: embed_fn or stub_embed_fn
    app.dependency_overrides[get_llm_fn] = lambda: llm_fn or (lambda messages: "stub answer")
    return TestClient(app)


def teardown_function() -> None:
    app.dependency_overrides.clear()


# --- filter building -------------------------------------------------------


def test_build_filter_returns_none_when_nothing_provided():
    request = AskRequest(query="quiet laptop")

    assert build_filter(request) is None


def test_build_filter_includes_only_provided_hard_constraints():
    request = AskRequest(query="quiet laptop", max_price=500.0)

    result = build_filter(request)

    assert result is not None
    assert len(result.must) == 1
    condition = result.must[0]
    assert condition.key == "price_amount"
    assert condition.range == qmodels.Range(lte=500.0)


def test_build_filter_combines_all_provided_constraints():
    request = AskRequest(
        query="quiet laptop",
        max_price=500.0,
        condition="used",
        location_country="PL",
    )

    result = build_filter(request)

    assert result is not None
    keys = {c.key for c in result.must}
    assert keys == {"price_amount", "condition_bucket", "location_country"}

    by_key = {c.key: c for c in result.must}
    assert by_key["price_amount"].range == qmodels.Range(lte=500.0)
    assert by_key["condition_bucket"].match == qmodels.MatchValue(value="used")
    assert by_key["location_country"].match == qmodels.MatchValue(value="PL")


# --- /ask endpoint -----------------------------------------------------------


def test_ask_zero_hits_returns_deterministic_refusal_without_calling_llm():
    qdrant = MagicMock()
    qdrant.query_points.return_value = MagicMock(points=[])
    llm_calls: list[list[dict]] = []

    def recording_llm_fn(messages: list[dict]) -> str:
        llm_calls.append(messages)
        return "should not be reached"

    client = make_client(qdrant_client=qdrant, llm_fn=recording_llm_fn)

    response = client.post("/ask", json={"query": "quiet laptop under $500"})

    assert response.status_code == 200
    body = response.json()
    assert body["matched"] is False
    assert body["items"] == []
    assert body["answer"] == REFUSAL_MESSAGE
    assert llm_calls == []  # LLM must not be called on the zero-hit path


def test_ask_with_hits_returns_matched_items_and_calls_llm_with_grounded_prompt():
    qdrant = MagicMock()
    qdrant.query_points.return_value = MagicMock(
        points=[
            make_scored_point(),
            make_scored_point(
                id="00000000-0000-0000-0000-000000000002",
                score=0.5,
                payload={
                    "id": "v1|999|0",
                    "title": "Lenovo ThinkPad T14 quiet fan",
                    "url": "https://www.ebay.pl/itm/999",
                },
            ),
        ]
    )
    llm_calls: list[list[dict]] = []

    def recording_llm_fn(messages: list[dict]) -> str:
        llm_calls.append(messages)
        return "The ThinkPad T14 looks like a good fit."

    client = make_client(qdrant_client=qdrant, llm_fn=recording_llm_fn)

    response = client.post("/ask", json={"query": "quiet laptop under $500"})

    assert response.status_code == 200
    body = response.json()
    assert body["matched"] is True
    assert body["answer"] == "The ThinkPad T14 looks like a good fit."
    assert len(body["items"]) == 2
    assert body["items"][0]["id"] == "v1|198603108529|0"
    assert body["items"][0]["condition_bucket"] == "used"
    assert body["items"][1]["title"] == "Lenovo ThinkPad T14 quiet fan"

    assert len(llm_calls) == 1
    prompt_text = " ".join(m["content"] for m in llm_calls[0])
    assert "Laptop Dell Latitude 3420 i5-1135G7 16/512GB SSD NVMe FHD" in prompt_text
    assert "https://www.ebay.pl/itm/198603108529" in prompt_text
    assert "Lenovo ThinkPad T14 quiet fan" in prompt_text
    assert "https://www.ebay.pl/itm/999" in prompt_text


def test_ask_embeds_query_with_query_prefix_and_uses_it_for_search():
    seen_texts: list[str] = []

    def recording_embed_fn(texts: list[str]) -> list[list[float]]:
        seen_texts.extend(texts)
        return stub_embed_fn(texts)

    qdrant = MagicMock()
    qdrant.query_points.return_value = MagicMock(points=[])

    client = make_client(qdrant_client=qdrant, embed_fn=recording_embed_fn)

    client.post("/ask", json={"query": "quiet laptop"})

    assert seen_texts == ["query: quiet laptop"]


def test_ask_passes_filter_and_top_k_to_qdrant_search():
    qdrant = MagicMock()
    qdrant.query_points.return_value = MagicMock(points=[])

    client = make_client(qdrant_client=qdrant)

    client.post(
        "/ask",
        json={"query": "quiet laptop", "max_price": 500.0, "top_k": 3},
    )

    qdrant.query_points.assert_called_once()
    _, kwargs = qdrant.query_points.call_args
    assert kwargs["limit"] == 3
    assert kwargs["query_filter"] is not None
    assert kwargs["query_filter"].must[0].key == "price_amount"
