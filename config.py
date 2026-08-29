"""Environment-backed settings, loaded once at import time."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if required and not value:
        raise RuntimeError(f"{name} is not set — copy .env.example to .env and fill it in")
    return value or ""


EBAY_CLIENT_ID = _env("EBAY_CLIENT_ID")
EBAY_CLIENT_SECRET = _env("EBAY_CLIENT_SECRET")
EBAY_AUTH_URL = _env("EBAY_AUTH_URL", "https://api.ebay.com/identity/v1/oauth2/token")
EBAY_API_URL = _env("EBAY_API_URL", "https://api.ebay.com")
# Poland-specific site. Optional per eBay's own docs (X-EBAY-C-MARKETPLACE-ID is not
# marked required) — unconfirmed against a live response yet, adjust here if a real
# call rejects it or a different site ID returns more relevant Polish listings.
EBAY_MARKETPLACE_ID = _env("EBAY_MARKETPLACE_ID", "EBAY_PL")

SEARCH_PHRASE = _env("SEARCH_PHRASE", "laptop")
MAX_OFFERS = int(_env("MAX_OFFERS", "200"))

# "PC Laptops & Netbooks" — confirmed live via the Taxonomy API's
# get_category_suggestions for q="laptop" on EBAY_PL, not guessed. Without
# this, a plain-text "laptop" search also returns batteries, chargers, cases,
# and replacement screens. Browse API rejects more than one category_ids value
# per request (live 400, errorId 12030, allowedMaxCategories=1) — "Apple
# Laptops" (111422) is out of scope for now rather than run as a second,
# separately-paginated search for a portfolio-sized corpus.
EBAY_CATEGORY_IDS = _env("EBAY_CATEGORY_IDS", "177")
# X-EBAY-C-MARKETPLACE-ID sets currency/locale only, not item location or
# shipping — this is the actual "ships to Poland" filter.
EBAY_DELIVERY_COUNTRY = _env("EBAY_DELIVERY_COUNTRY", "PL")

QDRANT_URL = _env("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = _env("QDRANT_COLLECTION", "laptop_offers")

# How many candidates the vector search stage returns after the metadata
# filter has narrowed the set (app.py).
RETRIEVAL_TOP_K = int(_env("RETRIEVAL_TOP_K", "5"))

# LM Studio's local OpenAI-compatible API — see README's Generation row for
# why: no external API key or per-query cost, keeps the whole pipeline
# runnable offline.
LLM_API_BASE = _env("LLM_API_BASE", "http://localhost:1234/v1")
LLM_MODEL = _env("LLM_MODEL", "qwen2.5-7b-instruct")
