"""Gemini text-embedding-004 wrapper for ChromaDB.

Provides a ChromaDB-compatible embedding function that uses Google's
text-embedding-004 model (768 dims) instead of the local all-MiniLM-L6-v2
(384 dims). Supports dual API key failover for rate-limit resilience.

Usage:
    from gemini_embeddings import get_embedding_fn
    embed_fn = get_embedding_fn()
    # Works as a drop-in for any ChromaDB embedding_function parameter
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)

_MODEL = "models/gemini-embedding-2"
_BATCH_SIZE = 100  # Gemini supports up to 100 texts per request
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0


class GeminiEmbeddingFunction(EmbeddingFunction[Documents]):
    """ChromaDB-compatible embedding function backed by Gemini text-embedding-004."""

    def __init__(
        self,
        api_key: str | None = None,
        api_key_fallback: str | None = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ):
        self._keys = []
        k1 = api_key or os.getenv("GEMINI_API_KEY", "")
        k2 = api_key_fallback or os.getenv("GEMINI_API_KEY_2", "")
        if k1:
            self._keys.append(k1)
        if k2:
            self._keys.append(k2)
        if not self._keys:
            raise ValueError(
                "No Gemini API key found. Set GEMINI_API_KEY in .env or pass api_key=."
            )
        self._task_type = task_type
        self._current_key_idx = 0
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        import google.generativeai as genai
        genai.configure(api_key=self._keys[self._current_key_idx])
        self._client = genai

    def _rotate_key(self) -> bool:
        """Switch to the next API key. Returns False if no more keys."""
        next_idx = self._current_key_idx + 1
        if next_idx >= len(self._keys):
            self._current_key_idx = 0
            self._init_client()
            return False
        self._current_key_idx = next_idx
        self._init_client()
        logger.info("Rotated to Gemini API key %d", self._current_key_idx + 1)
        return True

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch with retry + key rotation."""
        last_err = None
        for attempt in range(_MAX_RETRIES):
            try:
                result = self._client.embed_content(
                    model=_MODEL,
                    content=texts,
                    task_type=self._task_type,
                )
                return result["embedding"]
            except Exception as e:
                last_err = e
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "quota" in err_str:
                    if self._rotate_key():
                        continue
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Gemini rate limit (attempt %d/%d), waiting %.1fs",
                        attempt + 1, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                elif "400" in err_str or "invalid" in err_str:
                    raise
                else:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "Gemini embed error (attempt %d/%d): %s — retrying in %.1fs",
                        attempt + 1, _MAX_RETRIES, e, delay,
                    )
                    time.sleep(delay)

        raise RuntimeError(f"Gemini embedding failed after {_MAX_RETRIES} retries: {last_err}")

    def __call__(self, input: Documents) -> Embeddings:
        """Embed a list of documents. ChromaDB calls this."""
        if not input:
            return []

        all_embeddings: list[list[float]] = []
        for start in range(0, len(input), _BATCH_SIZE):
            batch = input[start : start + _BATCH_SIZE]
            # Truncate very long texts to avoid API limits (Gemini has a ~10K token limit)
            batch = [t[:8000] if len(t) > 8000 else t for t in batch]
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)

        return all_embeddings


# Query variant uses RETRIEVAL_QUERY task type for better search quality
class GeminiQueryEmbeddingFunction(GeminiEmbeddingFunction):
    """Same as GeminiEmbeddingFunction but uses RETRIEVAL_QUERY task type.

    Gemini recommends using RETRIEVAL_DOCUMENT when indexing and
    RETRIEVAL_QUERY when querying for optimal retrieval performance.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault("task_type", "RETRIEVAL_QUERY")
        super().__init__(**kwargs)


_embed_fn_cache: GeminiEmbeddingFunction | None = None
_query_fn_cache: GeminiQueryEmbeddingFunction | None = None


def get_embedding_fn() -> GeminiEmbeddingFunction:
    """Get the shared document embedding function (for indexing)."""
    global _embed_fn_cache
    if _embed_fn_cache is None:
        _embed_fn_cache = GeminiEmbeddingFunction()
    return _embed_fn_cache


def get_query_embedding_fn() -> GeminiQueryEmbeddingFunction:
    """Get the shared query embedding function (for retrieval)."""
    global _query_fn_cache
    if _query_fn_cache is None:
        _query_fn_cache = GeminiQueryEmbeddingFunction()
    return _query_fn_cache
