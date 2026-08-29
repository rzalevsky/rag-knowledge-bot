# rag-knowledge-bot

Retrieval-augmented answers over a real marketplace search: "find me a used laptop
that actually fits these constraints" against live eBay listings, with the
retrieval decisions written down rather than left to defaults.

**Status: in progress.** Stage 1 (eBay ingest) and Stage 2 (embeddings +
Qdrant upsert, both below) are done and tested. The answering API is not
built yet — see Roadmap.

## The problem this solves

A plain keyword search on a marketplace answers "does this text contain 'laptop'
and 'i7'?" It can't answer "quiet fan, no visible wear, good for long compiles,
under $500, ships to Poland" — that's a mix of hard constraints (price, RAM,
location) and soft, free-text ones (condition, noise, "feels sturdy") that plain
filters and plain vector search each handle badly on their own.

This project is not a synthetic RAG demo. I'm actually shopping for a laptop —
the corpus is real listings, the questions are the ones I'm actually asking, and
an honest "nothing matches" answer is exactly as useful to me as a match. That
last part matters more than it sounds: a bot that quietly relaxes your budget to
find *something* is worse than no bot at all.

## Why eBay, not Allegro

The first version of this pointed at Allegro's REST API. That was a dead end,
and worth stating plainly rather than quietly rewriting history:

Allegro's REST API has **no marketplace-wide public search for third-party
apps.** `/offers/listing`, which looked like the right endpoint (it authenticates,
returns 401/403 rather than 404), isn't in Allegro's own official OpenAPI spec
(`developer.allegro.pl/swagger.yaml`) at all. The only offers-search endpoint
that *is* in the spec, `GET /sale/offers`, is scoped to the authenticated
seller's own listings — useless for "search everyone's laptops." Two live 403s
against a validly-authenticated token (first via `client_credentials`, then via
a real user-authorized Device Flow token) were the signal to stop guessing and
check the spec directly, which is what should have happened before writing any
client code in the first place.

eBay's Browse API says outright, in its own docs, that it's built for this:
"All methods in the Browse API require an Application access token, which is
obtained using the client credentials grant flow" — i.e. buyer-facing search
across all sellers, no user login, confirmed against the docs before writing
`ebay_client.py` this time.

## Pipeline (target)

```
eBay Browse API → normalize → chunking → embeddings → Qdrant → retrieval → reranking → answer with citations
                                              ↑ you are here (embed.py)
```

| Stage | Decision | Reasoning |
|---|---|---|
| Source | eBay Browse API (`item_summary/search`, OAuth2 client_credentials) | Official, public, read-only, explicitly designed for third-party buyer search — no scraping, no ToS risk |
| Chunking | one item = one chunk | Listings are short (title + a handful of fields); splitting further would lose context, not add it |
| Metadata filter | price, condition, location as structured fields, applied *before* vector search | These are exact constraints ("under $500" isn't a similarity question) — forcing them through embeddings would be slower and less precise than a plain filter. `condition` comes back in the seller's marketplace language (`Gebraucht`/`Used`/`Używany`/`Usato` all mean "used") — needs normalizing to a fixed vocabulary before it's filterable, done in `embed.py` via eBay's locale-independent `conditionId` |
| Embeddings | `intfloat/multilingual-e5-small` (sentence-transformers, 384-dim, local inference) | A live ingest run showed titles mixing German/Italian/Polish/English; E5 is contrastively trained for retrieval (query/passage pairs), not just general sentence similarity — this is a retrieval system, not a paraphrase-similarity demo |
| Vector store | Qdrant | Payload filtering alongside vector search in one query, runs locally in Docker |
| Retrieval | metadata filter narrows the set, then vector search ranks by the free-text ask | Hybrid, not pure dense retrieval — see `docs/adr` (not written yet) for why |
| Generation | ⟨model TBD⟩, answer constrained to retrieved items, cites listing URLs | Refuses rather than guessing when no item meets the hard constraints |

## Stage 1 — eBay ingest (done)

`ebay_client.py` wraps the two endpoints this needs: the OAuth2 token endpoint
(`client_credentials` grant, scope `https://api.ebay.com/oauth/api_scope`) and
`GET /item_summary/search`. Both were checked against the live API and eBay's
own docs while writing this. Item *summaries* are lighter than full item detail —
structured specs like RAM or CPU ("aspects", in eBay's terms) aren't reliably on
the summary and may need a separate `GET /item/{item_id}` call per item; that's
deferred to whichever stage actually needs it rather than guessed at here.

`ingest.py` paginates through search results, normalizes each item (title,
price, condition, location, URL), dedupes across pages, and writes the result
to `local/offers.json`. No embeddings or Qdrant yet — this stage exists so a
bad credential or an API schema surprise shows up here, not buried inside a
later, harder-to-debug step.

Confirmed against a live production run (200 items, `EBAY_PL` marketplace):

- A plain `q=laptop` search returns batteries, chargers, cases, and
  replacement screens alongside actual laptops. Fixed by scoping the search to
  `category_ids=177` ("PC Laptops & Netbooks"), an id confirmed via the
  Taxonomy API's `get_category_suggestions`, not guessed. The Browse API
  rejects more than one category id per request (live 400, `errorId 12030`),
  so "Apple Laptops" (`111422`) is out of scope for now rather than run as a
  second, separately-paginated search.
- `X-EBAY-C-MARKETPLACE-ID` only sets currency/locale context (prices come
  back converted to PLN) — it does **not** restrict results to items located
  in or shipping to that country. The actual "ships to Poland" constraint is
  `filter=deliveryCountry:PL`, added once this was confirmed live.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest                            # 18 tests (eBay client + embed.py), all mocked/stubbed — no credentials, no live Qdrant, no model download needed

cp .env.example .env              # then fill in EBAY_CLIENT_ID / EBAY_CLIENT_SECRET
python ingest.py                  # -> local/offers.json
```

Getting eBay credentials is a manual, one-time step outside this repo: register
a free app at [developer.ebay.com](https://developer.ebay.com/) with your own
eBay account — "Application access" (client_credentials) is enough, this
project never logs in as a buyer or seller.

`EBAY_MARKETPLACE_ID` defaults to `EBAY_PL` in `.env.example` — confirmed live
(prices come back converted to PLN). It only sets currency/locale, though; see
the confirmed findings above for the filters that actually scope results to
laptops shipping to Poland.

## Stage 2 — embeddings (done)

`embed.py` reads `local/offers.json`, embeds each item's title with
`intfloat/multilingual-e5-small` (sentence-transformers, 384-dim, runs
locally — no API key, no per-call cost — see the Embeddings row above for
why this model), and upserts one Qdrant point per item. E5's own convention
requires prefixing text before encoding: `"passage: "` for indexed
documents (applied here), `"query: "` for search queries (deferred to
`app.py`, which does the querying).

Condition normalization uses `conditionId` from the raw eBay payload
instead of the localized `condition` string, since `conditionId` is
numeric and locale-independent (confirmed live: `conditionId "1000"`
paired with `condition "Neu"`, German for "New"). It's bucketed coarsely —
a buyer's filter doesn't need eBay's full tier list:

```
1000-1750 -> new
2000-2750 -> refurbished
3000-6000 -> used
7000      -> for_parts
otherwise -> unknown
```

The exact numeric ranges come from a secondary source (eBay's own
condition-id-values reference page timed out when fetched directly while
writing this) corroborated by one live data point, not independently
confirmed end to end the way Stage 1's findings are — flagged here rather
than presented as equally certain. Both the bucket (`condition_bucket`,
for filtering) and the original raw string (`condition_raw`, for display)
land in the Qdrant payload; normalization is lossy by design but the
source string is never discarded.

eBay's `itemId` (e.g. `"v1|198586449169|0"`) isn't a valid Qdrant point ID
(Qdrant requires an unsigned int or a UUID), so each point's id is derived
via `uuid.uuid5(uuid.NAMESPACE_URL, item.id)` — stable across runs, so
re-running `embed.py` after a fresh `ingest.py` updates existing points
(upsert) instead of duplicating them.

Same dependency-injection pattern as `EbayClient`'s `http_client` param:
both the embedder and the Qdrant client are injectable, so
`tests/test_embed.py` runs without downloading the real model or needing a
live Qdrant instance.

```bash
docker compose up -d              # starts Qdrant on :6333
python embed.py                   # local/offers.json -> config.QDRANT_COLLECTION
```

## Roadmap

- [x] eBay client + ingest, tested against a mocked API
- [x] Embedding + Qdrant upsert (`embed.py`)
- [ ] FastAPI service: hybrid retrieval + grounded generation (`app.py`)
- [ ] Worked examples against real listings — including at least one query with
      zero matches, showing the honest refusal rather than a forced answer
- [ ] Write up what I'd improve — most likely hybrid dense+BM25 search, since pure
      embeddings tend to lose exact model numbers and rare terms

**Allegro and OLX** are both out of scope: Allegro for the reason above (no
third-party marketplace search in its public API), OLX because it has no public
API for third-party read access and this project only sources data through
channels that don't require working around a site's own protections.

## Stack

Python · httpx · sentence-transformers · Qdrant · FastAPI (planned) · Docker

Qdrant's dashboard, once `docker compose up -d` is running: http://localhost:6333/dashboard
