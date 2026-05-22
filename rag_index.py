"""RAG indexer — build ChromaDB collections from 4000 trajectories + 131K tool calls.

Creates three collections:
  1. conversation_chunks — conversation phases from real trajectories
     (user-assistant exchange pairs, ~3-5 turns each, with task/tool metadata)
  2. tool_examples — tool call + result pairs from tool_calls.db
     (keyed by tool_name, with full arguments + response)
  3. privacy_patterns — privacy-relevant exchanges (consent gates, refusals,
     adversarial attempts) extracted from real trajectories

Usage:
    python rag_index.py [--trajectories-dir PATH] [--db-path PATH] [--chroma-dir PATH]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_TRAJ_DIR = _ROOT / "extracted-trajectories"
_DEFAULT_DB_PATH = _ROOT / "tool_calls.db"
_DEFAULT_CHROMA_DIR = _ROOT / "privacy-sft-pipeline" / "chroma_db"

_PRIVACY_KEYWORDS = {
    "consent", "permission", "authorize", "approve", "sensitive", "privacy",
    "hatch trust", "cannot retain", "refusal", "deny", "block", "redact",
    "memory_write", "vault", "encrypt", "pii", "medical", "ssn", "password",
    "health", "insurance", "diagnosis",
}


def _get_embedding_fn():
    """Return a ChromaDB-compatible embedding function using Gemini text-embedding-004."""
    from gemini_embeddings import get_embedding_fn
    return get_embedding_fn()


def _parse_trajectory_file(path: Path) -> list[dict]:
    """Parse a JSONL trajectory into a list of events."""
    events = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except (json.JSONDecodeError, ValueError):
                continue
    return events


def _extract_conversation_chunks(events: list[dict], file_id: str) -> list[dict]:
    """Extract conversation phase chunks from a trajectory.

    Groups events into phases of 3-6 turns (user+assistant exchanges).
    Each chunk includes the conversation text, tools used, and metadata.
    """
    messages = []
    for evt in events:
        if evt.get("type") != "message":
            continue
        msg = evt.get("message", {})
        role = msg.get("role", "")
        if role not in ("user", "assistant", "toolResult"):
            continue

        text_parts = []
        tool_names = []
        for c in msg.get("content", []):
            if isinstance(c, dict):
                if c.get("type") == "text":
                    t = c.get("text", "")
                    if "HEARTBEAT" in t or "Bootstrap" in t:
                        continue
                    if t.startswith("Sender (untrusted"):
                        idx = t.find("]")
                        if idx > 0:
                            t = t[idx + 1:].strip()
                    if t:
                        text_parts.append(t[:500])
                elif c.get("type") == "toolCall":
                    tool_names.append(c.get("name", ""))

        if not text_parts and not tool_names:
            continue

        messages.append({
            "role": role,
            "text": " ".join(text_parts),
            "tools": tool_names,
        })

    chunks = []
    phase_size = 6
    for i in range(0, len(messages), phase_size):
        phase = messages[i:i + phase_size]
        if len(phase) < 2:
            continue

        text_lines = []
        tools_in_phase = set()
        has_privacy = False

        for m in phase:
            prefix = m["role"].upper()
            text_lines.append(f"[{prefix}] {m['text'][:300]}")
            tools_in_phase.update(m["tools"])
            lower = m["text"].lower()
            if any(kw in lower for kw in _PRIVACY_KEYWORDS):
                has_privacy = True

        chunk_text = "\n".join(text_lines)
        if len(chunk_text) < 50:
            continue

        chunks.append({
            "id": f"{file_id}_phase_{i // phase_size}",
            "text": chunk_text[:2000],
            "metadata": {
                "source_file": file_id,
                "phase_index": i // phase_size,
                "turn_count": len(phase),
                "tools": ",".join(sorted(tools_in_phase)) if tools_in_phase else "",
                "has_privacy": has_privacy,
            },
        })

    return chunks


def _extract_privacy_patterns(events: list[dict], file_id: str) -> list[dict]:
    """Extract privacy-relevant exchanges from a trajectory.

    Looks for consent gates, refusals, memory writes near sensitive data,
    and adversarial-style exchanges.
    """
    patterns = []
    messages = []
    for evt in events:
        if evt.get("type") != "message":
            continue
        msg = evt.get("message", {})
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue

        text = ""
        for c in msg.get("content", []):
            if isinstance(c, dict) and c.get("type") == "text":
                text += c.get("text", "")

        messages.append({"role": role, "text": text[:1000]})

    for i, msg in enumerate(messages):
        lower = msg["text"].lower()
        is_privacy = any(kw in lower for kw in _PRIVACY_KEYWORDS)
        if not is_privacy:
            continue

        window_start = max(0, i - 1)
        window_end = min(len(messages), i + 2)
        window = messages[window_start:window_end]

        text_lines = [f"[{m['role'].upper()}] {m['text'][:400]}" for m in window]
        pattern_text = "\n".join(text_lines)

        pattern_type = "general_privacy"
        if "cannot retain" in lower or "hatch trust" in lower:
            pattern_type = "refusal"
        elif "consent" in lower or "permission" in lower or "approve" in lower:
            pattern_type = "consent_gate"
        elif "memory_write" in lower or "save" in lower or "store" in lower:
            pattern_type = "persistence_attempt"

        patterns.append({
            "id": f"{file_id}_privacy_{i}",
            "text": pattern_text[:2000],
            "metadata": {
                "source_file": file_id,
                "pattern_type": pattern_type,
                "turn_index": i,
            },
        })

    return patterns


def index_trajectories(
    traj_dir: Path,
    chroma_dir: Path,
    max_files: int = 0,
) -> tuple[int, int]:
    """Index trajectory files into ChromaDB conversation_chunks and privacy_patterns.

    Returns (chunks_indexed, patterns_indexed).
    """
    embed_fn = _get_embedding_fn()
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    conv_col = client.get_or_create_collection(
        name="conversation_chunks",
        embedding_function=embed_fn,
        metadata={"description": "Conversation phase chunks from real trajectories"},
    )
    priv_col = client.get_or_create_collection(
        name="privacy_patterns",
        embedding_function=embed_fn,
        metadata={"description": "Privacy-relevant exchanges from real trajectories"},
    )

    files = sorted(f for f in os.listdir(traj_dir) if f.endswith(".jsonl"))
    if max_files > 0:
        files = files[:max_files]

    total_chunks = 0
    total_patterns = 0
    batch_docs = []
    batch_ids = []
    batch_metas = []
    priv_docs = []
    priv_ids = []
    priv_metas = []

    BATCH_SIZE = 200

    for idx, fname in enumerate(files):
        if idx % 500 == 0:
            logger.info("Processing trajectory %d/%d (%s)...", idx, len(files), fname[:20])

        file_id = fname.replace(".jsonl", "")
        events = _parse_trajectory_file(traj_dir / fname)
        if not events:
            continue

        chunks = _extract_conversation_chunks(events, file_id)
        for chunk in chunks:
            batch_docs.append(chunk["text"])
            batch_ids.append(chunk["id"])
            batch_metas.append(chunk["metadata"])

        patterns = _extract_privacy_patterns(events, file_id)
        for pat in patterns:
            priv_docs.append(pat["text"])
            priv_ids.append(pat["id"])
            priv_metas.append(pat["metadata"])

        if len(batch_docs) >= BATCH_SIZE:
            conv_col.upsert(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
            total_chunks += len(batch_docs)
            batch_docs, batch_ids, batch_metas = [], [], []

        if len(priv_docs) >= BATCH_SIZE:
            priv_col.upsert(documents=priv_docs, ids=priv_ids, metadatas=priv_metas)
            total_patterns += len(priv_docs)
            priv_docs, priv_ids, priv_metas = [], [], []

    if batch_docs:
        conv_col.upsert(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
        total_chunks += len(batch_docs)
    if priv_docs:
        priv_col.upsert(documents=priv_docs, ids=priv_ids, metadatas=priv_metas)
        total_patterns += len(priv_docs)

    logger.info(
        "Indexed %d conversation chunks and %d privacy patterns from %d files",
        total_chunks, total_patterns, len(files),
    )
    return total_chunks, total_patterns


def index_tool_calls(
    db_path: Path,
    chroma_dir: Path,
    max_rows: int = 0,
) -> int:
    """Index tool call + result pairs from tool_calls.db into ChromaDB.

    Returns number of tool examples indexed.
    """
    if not db_path.exists():
        logger.error("tool_calls.db not found at %s", db_path)
        return 0

    embed_fn = _get_embedding_fn()
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    tool_col = client.get_or_create_collection(
        name="tool_examples",
        embedding_function=embed_fn,
        metadata={"description": "Tool call + result pairs from real trajectories"},
    )

    conn = sqlite3.connect(str(db_path))

    limit_clause = f"LIMIT {max_rows}" if max_rows > 0 else ""
    query = f"""
        SELECT tc.call_id, tc.tool_name, tc.arguments,
               tr.content, tr.is_error
        FROM tool_calls tc
        JOIN tool_results tr ON tc.call_id = tr.call_id
        WHERE tr.is_error = 0
          AND length(tr.content) > 30
          AND length(tr.content) < 5000
        {limit_clause}
    """

    rows = conn.execute(query).fetchall()
    conn.close()

    logger.info("Fetched %d tool call+result pairs from DB", len(rows))

    batch_docs = []
    batch_ids = []
    batch_metas = []
    BATCH_SIZE = 500
    total = 0

    for row_idx, (call_id, tool_name, args_str, result_content, is_error) in enumerate(rows):
        try:
            args = json.loads(args_str) if args_str else {}
        except (json.JSONDecodeError, TypeError):
            args = {}

        args_preview = json.dumps(args)[:300] if args else "{}"
        result_preview = result_content[:500] if result_content else ""

        doc_text = (
            f"Tool: {tool_name}\n"
            f"Arguments: {args_preview}\n"
            f"Result: {result_preview}"
        )

        batch_docs.append(doc_text)
        batch_ids.append(f"tc_{row_idx}_{tool_name}")
        batch_metas.append({
            "tool_name": tool_name,
            "has_path": bool(args.get("path") or args.get("file_path")),
            "result_length": len(result_content) if result_content else 0,
        })

        if len(batch_docs) >= BATCH_SIZE:
            tool_col.upsert(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
            total += len(batch_docs)
            batch_docs, batch_ids, batch_metas = [], [], []
            if total % 10000 == 0:
                logger.info("Indexed %d tool examples...", total)

    if batch_docs:
        tool_col.upsert(documents=batch_docs, ids=batch_ids, metadatas=batch_metas)
        total += len(batch_docs)

    logger.info("Indexed %d tool examples into ChromaDB", total)
    return total


def main():
    parser = argparse.ArgumentParser(description="Build RAG index from trajectories + tool calls")
    parser.add_argument("--trajectories-dir", default=str(_DEFAULT_TRAJ_DIR))
    parser.add_argument("--db-path", default=str(_DEFAULT_DB_PATH))
    parser.add_argument("--chroma-dir", default=str(_DEFAULT_CHROMA_DIR))
    parser.add_argument("--max-trajectories", type=int, default=0, help="Limit trajectory files (0=all)")
    parser.add_argument("--max-tool-rows", type=int, default=0, help="Limit tool call rows (0=all)")
    parser.add_argument("--skip-trajectories", action="store_true", help="Skip trajectory indexing")
    parser.add_argument("--skip-tools", action="store_true", help="Skip tool call indexing")
    args = parser.parse_args()

    chroma_dir = Path(args.chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_trajectories:
        traj_dir = Path(args.trajectories_dir)
        if traj_dir.exists():
            chunks, patterns = index_trajectories(traj_dir, chroma_dir, args.max_trajectories)
            logger.info("Trajectory indexing complete: %d chunks, %d privacy patterns", chunks, patterns)
        else:
            logger.warning("Trajectories directory not found: %s", traj_dir)

    if not args.skip_tools:
        db_path = Path(args.db_path)
        if db_path.exists():
            tools = index_tool_calls(db_path, chroma_dir, args.max_tool_rows)
            logger.info("Tool call indexing complete: %d examples", tools)
        else:
            logger.warning("tool_calls.db not found: %s", db_path)

    logger.info("RAG index build complete! ChromaDB at: %s", chroma_dir)


if __name__ == "__main__":
    main()
