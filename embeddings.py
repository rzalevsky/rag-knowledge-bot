"""Shared embedding config/loader — the one model both `embed.py` (passages)
and `app.py` (queries) must use, kept in one place so they can't drift apart.
Passage and query vectors are only comparable in the same vector space if
they came from the same model; duplicating the model name/prefixes in two
files risked exactly that.

E5's own convention requires prefixing text before encoding: "passage: " for
indexed documents (used in `embed.py`), "query: " for search queries (used
in `app.py`).
"""
from __future__ import annotations

from typing import Callable

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
VECTOR_SIZE = 384
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

# Takes a batch of already-prefixed texts, returns one vector per text —
# batched rather than one-at-a-time so the real sentence-transformers model
# can use its own internal batching instead of one forward pass per text.
EmbedFn = Callable[[list[str]], list[list[float]]]


def default_embed_fn(model_name: str = EMBEDDING_MODEL_NAME) -> EmbedFn:
    """Loads the real sentence-transformers model. Imported lazily so tests
    that inject a stub embedder never need the model downloaded."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, show_progress_bar=False).tolist()

    return embed
