import uuid
from unittest.mock import MagicMock

from qdrant_client.http import models as qmodels

from embed import (
    build_payload,
    condition_bucket,
    embed_items,
    ensure_collection,
    point_id,
)


def stub_embed_fn(texts: list[str]) -> list[list[float]]:
    """Deterministic fixed-size vector per text — no real model involved."""
    return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]


def make_item(**overrides) -> dict:
    item = {
        "id": "v1|198603108529|0",
        "title": "Laptop Dell Latitude 3420 i5-1135G7 16/512GB SSD NVMe FHD",
        "url": "https://www.ebay.pl/itm/198603108529",
        "price_amount": 617.67,
        "price_currency": "PLN",
        "condition": "Używany",
        "location_country": "PL",
        "raw": {"conditionId": "3000"},
    }
    item.update(overrides)
    return item


def test_condition_bucket_boundaries():
    assert condition_bucket("1000") == "new"
    assert condition_bucket("1750") == "new"
    assert condition_bucket("2000") == "refurbished"
    assert condition_bucket("2750") == "refurbished"
    assert condition_bucket("3000") == "used"
    assert condition_bucket("6000") == "used"
    assert condition_bucket("7000") == "for_parts"


def test_condition_bucket_unknown_for_missing_or_gaps():
    assert condition_bucket(None) == "unknown"
    assert condition_bucket("") == "unknown"
    assert condition_bucket("not-a-number") == "unknown"
    assert condition_bucket("1800") == "unknown"  # gap between new and refurbished


def test_point_id_is_deterministic_and_matches_uuid5():
    item_id = "v1|198603108529|0"
    expected = str(uuid.uuid5(uuid.NAMESPACE_URL, item_id))

    assert point_id(item_id) == expected
    assert point_id(item_id) == point_id(item_id)  # stable across calls


def test_point_id_differs_for_different_items():
    assert point_id("v1|111|0") != point_id("v1|222|0")


def test_build_payload_shape_and_condition_bucket():
    item = make_item()

    payload = build_payload(item)

    assert payload == {
        "id": "v1|198603108529|0",
        "title": item["title"],
        "price_amount": 617.67,
        "price_currency": "PLN",
        "condition_raw": "Używany",
        "condition_bucket": "used",
        "location_country": "PL",
        "url": item["url"],
    }


def test_build_payload_keeps_condition_raw_even_when_bucket_is_unknown():
    item = make_item(raw={})

    payload = build_payload(item)

    assert payload["condition_raw"] == "Używany"
    assert payload["condition_bucket"] == "unknown"


def test_embed_items_applies_passage_prefix_before_embedding():
    seen_texts: list[str] = []

    def recording_embed_fn(texts: list[str]) -> list[list[float]]:
        seen_texts.extend(texts)
        return stub_embed_fn(texts)

    items = [make_item(id="a", title="Foo"), make_item(id="b", title="Bar")]

    embed_items(items, recording_embed_fn)

    assert seen_texts == ["passage: Foo", "passage: Bar"]


def test_embed_items_builds_one_point_per_item_with_expected_id_and_payload():
    items = [make_item()]

    points = embed_items(items, stub_embed_fn)

    assert len(points) == 1
    point = points[0]
    assert isinstance(point, qmodels.PointStruct)
    assert point.id == point_id(items[0]["id"])
    assert point.payload == build_payload(items[0])
    assert point.vector == stub_embed_fn(["passage: " + items[0]["title"]])[0]


def test_ensure_collection_creates_when_missing():
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])

    ensure_collection(client, "laptop_offers", vector_size=4)

    client.create_collection.assert_called_once()
    _, kwargs = client.create_collection.call_args
    assert kwargs["collection_name"] == "laptop_offers"
    assert kwargs["vectors_config"].size == 4
    assert kwargs["vectors_config"].distance == qmodels.Distance.COSINE


def test_ensure_collection_skips_when_already_present():
    client = MagicMock()
    existing = MagicMock()
    existing.name = "laptop_offers"
    client.get_collections.return_value = MagicMock(collections=[existing])

    ensure_collection(client, "laptop_offers", vector_size=4)

    client.create_collection.assert_not_called()


def test_upsert_called_with_expected_points():
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    items = [make_item()]

    points = embed_items(items, stub_embed_fn)
    ensure_collection(client, "laptop_offers", vector_size=4)
    client.upsert(collection_name="laptop_offers", points=points)

    client.upsert.assert_called_once()
    _, kwargs = client.upsert.call_args
    assert kwargs["collection_name"] == "laptop_offers"
    assert kwargs["points"] == points
    assert kwargs["points"][0].payload["id"] == items[0]["id"]
