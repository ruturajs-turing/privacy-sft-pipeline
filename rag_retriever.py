"""RAG retriever — semantic search API for the privacy trajectory pipeline.

Provides three retrieval functions:
  1. get_similar_conversations() — find real conversation phases similar to a query
  2. get_tool_examples() — find realistic tool call+result pairs for a given tool
  3. get_privacy_patterns() — find real privacy exchanges (refusals, consent gates, etc.)

All functions return formatted text blocks ready to inject into LLM prompts.

The ChromaDB must be built first using rag_index.py.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CHROMA_DIR = _ROOT / "privacy-sft-pipeline" / "chroma_db"

_client: chromadb.ClientAPI | None = None
_embed_fn = None


def _get_client(chroma_dir: Path | None = None) -> chromadb.ClientAPI:
    global _client
    if _client is not None:
        return _client

    path = str(chroma_dir or _DEFAULT_CHROMA_DIR)
    if not Path(path).exists():
        logger.warning("ChromaDB not found at %s — RAG retrieval will return empty results", path)
        _client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        return _client

    _client = chromadb.PersistentClient(
        path=path,
        settings=Settings(anonymized_telemetry=False),
    )
    return _client


def _get_embed_fn():
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn
    from gemini_embeddings import get_query_embedding_fn
    _embed_fn = get_query_embedding_fn()
    return _embed_fn


def _safe_query(collection_name: str, query_text: str, n: int, where: dict | None = None) -> list[dict]:
    """Query a collection safely, returning empty list on errors."""
    try:
        client = _get_client()
        embed_fn = _get_embed_fn()
        col = client.get_collection(name=collection_name, embedding_function=embed_fn)

        kwargs: dict = {
            "query_texts": [query_text],
            "n_results": min(n, col.count()),
        }
        if where:
            kwargs["where"] = where

        if kwargs["n_results"] <= 0:
            return []

        results = col.query(**kwargs)

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(docs, metas, distances)
        ]
    except Exception as e:
        logger.debug("RAG query failed for %s: %s", collection_name, e)
        return []


def get_similar_conversations(
    query: str,
    n: int = 5,
    tools_filter: str | None = None,
) -> str:
    """Find real conversation phases similar to the query.

    Args:
        query: description of the conversation context (task goal, current topic, etc.)
        n: number of results to return
        tools_filter: comma-separated tool names to filter by (optional)

    Returns:
        Formatted text block with similar conversation excerpts.
    """
    where = None
    if tools_filter:
        where = {"tools": {"$contains": tools_filter.split(",")[0]}}

    results = _safe_query("conversation_chunks", query, n, where)
    if not results:
        return ""

    blocks = []
    for i, r in enumerate(results):
        meta = r.get("metadata", {})
        tools = meta.get("tools", "")
        blocks.append(
            f"--- Example {i + 1} (tools: {tools or 'none'}) ---\n"
            f"{r['text']}"
        )

    return (
        "REAL CONVERSATION EXAMPLES (from actual human-agent sessions):\n\n"
        + "\n\n".join(blocks)
    )


def get_tool_examples(
    tool_name: str,
    context_query: str = "",
    n: int = 3,
) -> list[dict]:
    """Find realistic tool call + result pairs for a given tool.

    Args:
        tool_name: name of the tool to find examples for
        context_query: optional context to find more relevant examples
        n: number of results

    Returns:
        List of dicts with 'arguments' and 'result' keys parsed from the examples.
    """
    query = f"Tool: {tool_name}"
    if context_query:
        query += f" {context_query}"

    results = _safe_query(
        "tool_examples",
        query,
        n,
        where={"tool_name": tool_name},
    )

    examples = []
    for r in results:
        text = r.get("text", "")
        args_str = ""
        result_str = ""
        for line in text.split("\n"):
            if line.startswith("Arguments:"):
                args_str = line[len("Arguments:"):].strip()
            elif line.startswith("Result:"):
                result_str = line[len("Result:"):].strip()

        try:
            args = json.loads(args_str) if args_str else {}
        except (json.JSONDecodeError, ValueError):
            args = {}

        examples.append({
            "tool_name": tool_name,
            "arguments": args,
            "result": result_str,
            "distance": r.get("distance", 0),
        })

    return examples


def get_tool_examples_formatted(
    tool_name: str,
    context_query: str = "",
    n: int = 3,
) -> str:
    """Get tool examples as a formatted text block for LLM prompts."""
    examples = get_tool_examples(tool_name, context_query, n)
    if not examples:
        return ""

    blocks = []
    for i, ex in enumerate(examples):
        args_str = json.dumps(ex["arguments"])[:300]
        blocks.append(
            f"  Example {i + 1}:\n"
            f"    Args: {args_str}\n"
            f"    Result: {ex['result'][:300]}"
        )

    return (
        f"REAL {tool_name.upper()} EXAMPLES (from production trajectories):\n"
        + "\n".join(blocks)
    )


def get_privacy_patterns(
    query: str,
    pattern_type: str | None = None,
    n: int = 4,
) -> str:
    """Find real privacy-relevant exchanges similar to the query.

    Args:
        query: description of the privacy situation
        pattern_type: one of "refusal", "consent_gate", "persistence_attempt", or None for all
        n: number of results

    Returns:
        Formatted text block with real privacy exchange examples.
    """
    where = None
    if pattern_type:
        where = {"pattern_type": pattern_type}

    results = _safe_query("privacy_patterns", query, n, where)
    if not results:
        return ""

    blocks = []
    for i, r in enumerate(results):
        meta = r.get("metadata", {})
        ptype = meta.get("pattern_type", "general")
        blocks.append(
            f"--- Privacy Example {i + 1} (type: {ptype}) ---\n"
            f"{r['text']}"
        )

    return (
        "REAL PRIVACY EXCHANGES (from actual human-agent sessions):\n\n"
        + "\n\n".join(blocks)
    )


def get_adversarial_examples(
    query: str = "user pressures agent to save sensitive data permanently",
    n: int = 3,
) -> str:
    """Shortcut: find real adversarial/refusal exchanges."""
    return get_privacy_patterns(
        query=query,
        pattern_type="refusal",
        n=n,
    )


def get_consent_examples(
    query: str = "agent asks user for consent before writing sensitive data",
    n: int = 3,
) -> str:
    """Shortcut: find real consent gate exchanges."""
    return get_privacy_patterns(
        query=query,
        pattern_type="consent_gate",
        n=n,
    )


def is_index_ready(chroma_dir: Path | None = None) -> bool:
    """Check if the RAG index has been built and has data."""
    path = chroma_dir or _DEFAULT_CHROMA_DIR
    if not path.exists():
        return False
    try:
        client = _get_client(path)
        collections = client.list_collections()
        if not collections:
            return False
        for col_obj in collections:
            col = client.get_collection(name=col_obj.name, embedding_function=_get_embed_fn())
            if col.count() > 0:
                return True
        return False
    except Exception:
        return False
