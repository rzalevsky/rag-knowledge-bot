"""Stage 1 of the pipeline: documents -> chunking (see README).

Right now this only fetches and normalizes eBay item summaries into local
JSON — embedding and the Qdrant upsert are stage 2, added once this stage is
proven against real data. Splitting it this way means a broken eBay
credential or a schema surprise shows up here, not buried inside an embedding
run.

Usage:
    python ingest.py                       # SEARCH_PHRASE / MAX_OFFERS from .env
    python ingest.py --phrase "laptop i7" --max-offers 50
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import config
from ebay_client import EbayClient

OUTPUT_PATH = Path("local/offers.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phrase", default=config.SEARCH_PHRASE)
    parser.add_argument("--max-offers", type=int, default=config.MAX_OFFERS)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = EbayClient(
        client_id=config.EBAY_CLIENT_ID,
        client_secret=config.EBAY_CLIENT_SECRET,
        auth_url=config.EBAY_AUTH_URL,
        api_url=config.EBAY_API_URL,
        marketplace_id=config.EBAY_MARKETPLACE_ID,
        category_ids=config.EBAY_CATEGORY_IDS,
        delivery_country=config.EBAY_DELIVERY_COUNTRY,
    )

    items = []
    try:
        for item in client.iter_items(args.phrase, args.max_offers):
            items.append(asdict(item))
            print(f"  {item.id}  {item.price_amount} {item.price_currency}  {item.title[:70]}")
    finally:
        client.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(items, ensure_ascii=False, indent=2))
    print(f"\n{len(items)} items -> {args.output}")


if __name__ == "__main__":
    main()
