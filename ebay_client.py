"""Thin client for the eBay Browse API — public item search only.

Uses the OAuth2 client_credentials grant for an Application access token, per
eBay's own Browse API docs (developer.ebay.com/api-docs/buy/browse/overview.html):
"All methods in the Browse API require an Application access token, which is
obtained using the client credentials grant flow." Confirmed there before
writing this client, and separately checked live: both endpoints below return
401/403 for missing auth rather than 404, confirming the paths are current.

This replaced an Allegro-based version of this client. Allegro's REST API has
no marketplace-wide public search for third-party apps — its only offers-search
endpoint, GET /sale/offers, is scoped to the authenticated seller's own
listings (confirmed against Allegro's official OpenAPI spec,
developer.allegro.pl/swagger.yaml, after two live 403s pointed at something
being fundamentally unsupported rather than misconfigured). eBay's Browse API
is built for exactly this use case — buyer-facing search across all sellers —
and says so directly in its own docs, which is the whole reason it was chosen.

Field names inside a 200 /item_summary/search response are taken from eBay's
public API reference rather than a live authenticated call — recheck `raw` on
the first real response and adjust `_normalize_item` if a field has moved.
Item summaries are lighter than full item detail: structured specs like RAM or
CPU (eBay calls these "aspects") are not reliably present on the summary and
may need a separate GET /item/{item_id} call per item — left for the ingest
stage that actually needs them rather than guessed at here.
"""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx

SEARCH_SCOPE = "https://api.ebay.com/oauth/api_scope"
PAGE_SIZE = 50  # eBay's documented default/typical page size for item_summary/search


@dataclass
class Item:
    """One normalized search result. `raw` keeps the untouched API payload —
    normalization is lossy by design (we only pull what retrieval needs), and
    debugging a bad chunk later is much easier with the source record on hand.
    """

    id: str
    title: str
    url: str
    price_amount: float | None
    price_currency: str | None
    condition: str | None
    location_country: str | None
    raw: dict[str, Any] = field(repr=False)


class EbayAuthError(RuntimeError):
    pass


class EbayClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        auth_url: str,
        api_url: str,
        marketplace_id: str,
        category_ids: str | None = None,
        delivery_country: str | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        if not client_id or not client_secret:
            raise EbayAuthError(
                "EBAY_CLIENT_ID / EBAY_CLIENT_SECRET are not set — "
                "register an app at developer.ebay.com and fill in .env"
            )
        self._client_id = client_id
        self._client_secret = client_secret
        self._auth_url = auth_url
        self._api_url = api_url.rstrip("/")
        self._marketplace_id = marketplace_id
        # X-EBAY-C-MARKETPLACE-ID only sets currency/locale context (confirmed
        # live: PLN prices come back with convertedFromCurrency once this is
        # set) — it does NOT restrict results to items located in or shipping
        # to that country. category_ids / delivery_country below do the actual
        # filtering: category_ids keeps results to "PC Laptops & Netbooks" (177)
        # and "Apple Laptops" (111422) — ids confirmed live via the Taxonomy
        # API's get_category_suggestions for q="laptop" on marketplace EBAY_PL,
        # not guessed — since a plain-text "laptop" query otherwise returns
        # batteries, chargers, cases, and replacement screens alongside actual
        # laptops.
        self._category_ids = category_ids
        self._delivery_country = delivery_country
        self._http = http_client or httpx.Client(timeout=15.0)
        self._access_token: str | None = None
        self._access_token_expires_at: float = 0.0

    def _basic_auth_header(self) -> str:
        raw = f"{self._client_id}:{self._client_secret}".encode()
        return "Basic " + base64.b64encode(raw).decode()

    def _get_access_token(self) -> str:
        # 30s safety margin so a token doesn't expire mid-request on a slow page fetch.
        if self._access_token and time.monotonic() < self._access_token_expires_at - 30:
            return self._access_token

        response = self._http.post(
            self._auth_url,
            data={"grant_type": "client_credentials", "scope": SEARCH_SCOPE},
            headers={
                "Authorization": self._basic_auth_header(),
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
        if response.status_code != 200:
            raise EbayAuthError(
                f"token request failed: {response.status_code} {response.text}"
            )
        data = response.json()
        self._access_token = data["access_token"]
        self._access_token_expires_at = time.monotonic() + float(data.get("expires_in", 7200))
        return self._access_token

    def search_items_page(
        self, phrase: str, limit: int = PAGE_SIZE, offset: int = 0
    ) -> list[dict[str, Any]]:
        """One page of raw item dicts from itemSummaries."""
        params: dict[str, Any] = {"q": phrase, "limit": limit, "offset": offset}
        if self._category_ids:
            params["category_ids"] = self._category_ids
        if self._delivery_country:
            params["filter"] = f"deliveryCountry:{self._delivery_country}"
        response = self._http.get(
            f"{self._api_url}/buy/browse/v1/item_summary/search",
            params=params,
            headers={
                "Authorization": f"Bearer {self._get_access_token()}",
                "X-EBAY-C-MARKETPLACE-ID": self._marketplace_id,
            },
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"item search request failed: {response.status_code} {response.text}"
            )
        return response.json().get("itemSummaries", [])

    def iter_items(self, phrase: str, max_items: int) -> Iterator[Item]:
        """Paginates until max_items is reached or the API runs out of results."""
        seen_ids: set[str] = set()
        offset = 0
        yielded = 0
        while yielded < max_items:
            page = self.search_items_page(phrase, limit=PAGE_SIZE, offset=offset)
            if not page:
                return
            for raw in page:
                item_id = raw.get("itemId")
                if not item_id or item_id in seen_ids:
                    continue
                seen_ids.add(item_id)
                yield _normalize_item(raw)
                yielded += 1
                if yielded >= max_items:
                    return
            # A short page means this was the last one — skip the extra request
            # that would otherwise be needed to confirm it via an empty page.
            if len(page) < PAGE_SIZE:
                return
            offset += PAGE_SIZE

    def close(self) -> None:
        self._http.close()


def _normalize_item(raw: dict[str, Any]) -> Item:
    price = raw.get("price", {})
    return Item(
        id=raw.get("itemId", ""),
        title=raw.get("title", ""),
        url=raw.get("itemWebUrl", ""),
        price_amount=_to_float(price.get("value")),
        price_currency=price.get("currency"),
        condition=raw.get("condition"),
        location_country=raw.get("itemLocation", {}).get("country"),
        raw=raw,
    )


def _to_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
