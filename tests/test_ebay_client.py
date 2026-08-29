import httpx
import pytest
import respx

from ebay_client import EbayAuthError, EbayClient

AUTH_URL = "https://ebay.test/identity/v1/oauth2/token"
API_URL = "https://api.ebay.test"
MARKETPLACE_ID = "EBAY_PL"


def make_client() -> EbayClient:
    return EbayClient(
        client_id="id123",
        client_secret="secret456",
        auth_url=AUTH_URL,
        api_url=API_URL,
        marketplace_id=MARKETPLACE_ID,
    )


def test_missing_credentials_raise_before_any_request():
    with pytest.raises(EbayAuthError):
        EbayClient(
            client_id="",
            client_secret="",
            auth_url=AUTH_URL,
            api_url=API_URL,
            marketplace_id=MARKETPLACE_ID,
        )


@respx.mock
def test_get_access_token_sends_client_credentials_and_caches():
    token_route = respx.post(AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "tok-1", "expires_in": 7200})
    )
    client = make_client()

    first = client._get_access_token()
    second = client._get_access_token()  # should hit the cache, not fire a second request

    assert first == second == "tok-1"
    assert token_route.call_count == 1

    request = token_route.calls[0].request
    assert request.headers["authorization"].startswith("Basic ")
    body = request.content.decode()
    assert "grant_type=client_credentials" in body
    assert "scope=" in body


@respx.mock
def test_get_access_token_raises_on_bad_credentials():
    respx.post(AUTH_URL).mock(
        return_value=httpx.Response(401, json={"error": "invalid_client"})
    )
    client = make_client()

    with pytest.raises(EbayAuthError):
        client._get_access_token()


@respx.mock
def test_search_items_page_sends_bearer_token_and_marketplace_header():
    respx.post(AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "tok-1", "expires_in": 7200})
    )
    search_route = respx.get(f"{API_URL}/buy/browse/v1/item_summary/search").mock(
        return_value=httpx.Response(200, json={"itemSummaries": []})
    )
    client = make_client()

    client.search_items_page("laptop", limit=10, offset=0)

    request = search_route.calls[0].request
    assert request.headers["authorization"] == "Bearer tok-1"
    assert request.headers["x-ebay-c-marketplace-id"] == MARKETPLACE_ID
    assert request.url.params["q"] == "laptop"


@respx.mock
def test_search_items_page_sends_category_ids_and_delivery_country_filter():
    respx.post(AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "tok-1", "expires_in": 7200})
    )
    search_route = respx.get(f"{API_URL}/buy/browse/v1/item_summary/search").mock(
        return_value=httpx.Response(200, json={"itemSummaries": []})
    )
    client = EbayClient(
        client_id="id123",
        client_secret="secret456",
        auth_url=AUTH_URL,
        api_url=API_URL,
        marketplace_id=MARKETPLACE_ID,
        category_ids="177",
        delivery_country="PL",
    )

    client.search_items_page("laptop", limit=10, offset=0)

    request = search_route.calls[0].request
    assert request.url.params["category_ids"] == "177"
    assert request.url.params["filter"] == "deliveryCountry:PL"


@respx.mock
def test_iter_items_normalizes_paginates_and_dedupes():
    respx.post(AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "tok-1", "expires_in": 7200})
    )

    def item(item_id: str, price: str = "199.00") -> dict:
        return {
            "itemId": item_id,
            "title": f"Laptop {item_id}",
            "itemWebUrl": f"https://www.ebay.com/itm/{item_id}",
            "price": {"value": price, "currency": "USD"},
            "condition": "Used",
            "itemLocation": {"country": "PL"},
        }

    # A full first page (PAGE_SIZE items) so iter_items requests a second page
    # rather than treating this as the last one — item "0" is repeated there
    # on purpose to check dedup by itemId across pages.
    page_1 = httpx.Response(200, json={"itemSummaries": [item(str(i)) for i in range(50)]})
    page_2 = httpx.Response(200, json={"itemSummaries": [item("0"), item("50", "219.00")]})
    respx.get(f"{API_URL}/buy/browse/v1/item_summary/search").mock(side_effect=[page_1, page_2])

    client = make_client()
    items = list(client.iter_items("laptop", max_items=100))

    assert [i.id for i in items] == [str(i) for i in range(50)] + ["50"]
    assert items[0].price_amount == 199.00
    assert items[0].price_currency == "USD"
    assert items[0].condition == "Used"
    assert items[0].location_country == "PL"
    assert items[-1].price_amount == 219.00


@respx.mock
def test_iter_items_stops_at_max_items_without_requesting_next_page():
    respx.post(AUTH_URL).mock(
        return_value=httpx.Response(200, json={"access_token": "tok-1", "expires_in": 7200})
    )
    many_items = [
        {
            "itemId": str(i),
            "title": f"Laptop {i}",
            "price": {"value": "100.00", "currency": "USD"},
        }
        for i in range(50)
    ]
    search_route = respx.get(f"{API_URL}/buy/browse/v1/item_summary/search").mock(
        return_value=httpx.Response(200, json={"itemSummaries": many_items})
    )

    client = make_client()
    items = list(client.iter_items("laptop", max_items=5))

    assert len(items) == 5
    assert search_route.call_count == 1
