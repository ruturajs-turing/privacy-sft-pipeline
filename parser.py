"""Stage 1: Download export zips from GCS, extract session JSONL, parse into ParsedTrajectory."""
from __future__ import annotations

import io
import json
import logging
import re
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import unquote

from google.cloud import storage
from google.oauth2 import service_account

from config import GCS_BUCKET, GCS_SERVICE_ACCOUNT_PATH, PERSONAS_PATH, TASKS_PATH, WORKSPACE_PREFIX
from models import AssistantTurn, ParsedTrajectory, ToolCall, ToolResult
from trajectory_structure import normalize_event_stream

logger = logging.getLogger(__name__)

_personas_cache: dict | None = None
_tasks_cache: dict | None = None


def _load_personas() -> dict[str, dict]:
    global _personas_cache
    if _personas_cache is None:
        data = json.loads(Path(PERSONAS_PATH).read_text())
        personas = data.get("personas", data) if isinstance(data, dict) else data
        _personas_cache = {p["persona_id"]: p for p in personas}
    return _personas_cache


def _load_tasks() -> dict[str, dict]:
    global _tasks_cache
    if _tasks_cache is None:
        tasks = json.loads(Path(TASKS_PATH).read_text())
        _tasks_cache = {t["task_id"]: t for t in tasks}
    return _tasks_cache


def _get_gcs_client() -> storage.Client:
    creds = service_account.Credentials.from_service_account_file(GCS_SERVICE_ACCOUNT_PATH)
    return storage.Client(credentials=creds, project=creds.project_id)


def _download_zip(export_url: str) -> bytes:
    """Download a zip from GCS given its full storage URL, or read from local path."""
    # Support local file paths for testing
    if export_url.startswith("/") or export_url.startswith("file://"):
        local_path = export_url.removeprefix("file://")
        return Path(local_path).read_bytes()

    # Parse: .../o/exports%2F...%2F....zip?alt=media
    if "/o/" in export_url:
        obj_path = export_url.split("/o/")[1].split("?")[0]
        obj_path = unquote(obj_path)
    else:
        obj_path = export_url

    client = _get_gcs_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(obj_path)
    return blob.download_as_bytes()


def _normalize_ts(val) -> int:
    """Convert any timestamp format to comparable int (epoch ms)."""
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str) and val:
        from datetime import datetime, timezone
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return int(dt.timestamp() * 1000)
        except (ValueError, TypeError):
            return 0
    return 0


def _looks_like_valid_jsonl(zf: zipfile.ZipFile, name: str) -> bool:
    """Return True when a zip member has at least one parseable JSONL event."""
    try:
        if zf.getinfo(name).file_size <= 0:
            return False
        text = zf.read(name).decode("utf-8", errors="replace")
    except Exception:
        return False

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            json.loads(line)
            return True
        except json.JSONDecodeError:
            return False
    return False


def _find_jsonl_for_session_id(zf: zipfile.ZipFile, session_id: str) -> str | None:
    """Find the best standard session JSONL for a session id."""
    candidates: list[tuple[int, int, str]] = []
    for name in zf.namelist():
        if session_id not in name:
            continue
        if not name.endswith(".jsonl") or "trajectory-path" in name:
            continue
        if name.startswith("__MACOSX"):
            continue
        if not _looks_like_valid_jsonl(zf, name):
            continue

        is_final = "final-snapshot" in name
        is_session = "sessions/" in name
        is_trajectory = name.endswith(".trajectory.jsonl")
        if is_final and is_session and not is_trajectory:
            priority = 0
        elif is_session and not is_trajectory:
            priority = 1
        elif is_final and is_trajectory:
            priority = 2
        else:
            priority = 3
        candidates.append((priority, -zf.getinfo(name).file_size, name))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _task_ref_timestamp(name: str, task_id: str = "") -> int | None:
    basename = name.split("/")[-1]
    pattern = r"^sess-(\d+)-(T-\d{3}-\d{2})\.jsonl$"
    match = re.match(pattern, basename)
    if not match:
        return None
    if task_id and match.group(2) != task_id:
        return None
    try:
        raw = int(match.group(1))
    except ValueError:
        return None
    return raw // 1000 if raw > 10_000_000_000_000 else raw


def _load_final_sessions(zf: zipfile.ZipFile) -> dict:
    sessions_candidates = [
        n for n in zf.namelist()
        if n.endswith("sessions.json") and "sessions/" in n and not n.startswith("__MACOSX")
    ]
    sessions_candidates.sort(key=lambda n: (0 if "final-snapshot" in n else 1, n))
    if not sessions_candidates:
        return {}
    try:
        return json.loads(zf.read(sessions_candidates[0]))
    except Exception:
        return {}


def _find_session_near_task_ref(zf: zipfile.ZipFile, task_id: str) -> str | None:
    ref_times: list[int] = []
    for name in zf.namelist():
        if (
            "final-snapshot" in name
            and "workspace/trajectories/" in name
            and name.endswith(f"-{task_id}.jsonl")
        ):
            ts = _task_ref_timestamp(name, task_id)
            if ts is not None:
                ref_times.append(ts)
    if not ref_times:
        return None

    target_ts = max(ref_times)
    sessions_json = _load_final_sessions(zf)
    scored: list[tuple[int, int, str]] = []
    for key, info in sessions_json.items():
        if not isinstance(info, dict):
            continue
        session_id = info.get("sessionId", key)
        if not session_id:
            continue
        started = _normalize_ts(info.get("sessionStartedAt", info.get("createdAt", 0)))
        updated = _normalize_ts(info.get("updatedAt", started))
        score = min(abs(started - target_ts), abs(updated - target_ts))
        scored.append((score, updated, session_id))

    for _score, _updated, session_id in sorted(scored):
        chosen = _find_jsonl_for_session_id(zf, session_id)
        if chosen:
            logger.info(
                "Selected session %s nearest task ref %s for %s",
                session_id, target_ts, task_id,
            )
            return chosen
    return None


def _find_session_jsonl(zf: zipfile.ZipFile, task_id: str = "") -> str | None:
    """Find the correct session JSONL for the given task_id.

    Strategy:
      1. Look for a session reference file matching the task_id (e.g. sess-*-T-033-02.jsonl)
         in final-snapshot workspace/trajectories/ to find the session timestamp, then locate
         the corresponding full session .jsonl in final-snapshot sessions/.
      2. Use final-snapshot sessions.json to find the most recently updated session.
      3. Fallback: largest .jsonl in final-snapshot sessions/.
      4. Last resort: largest .jsonl anywhere.
    """

    # Strategy 1: Match via task_id reference file in workspace/trajectories/
    if task_id:
        chosen = _find_session_near_task_ref(zf, task_id)
        if chosen:
            return chosen

    # Strategy 2: Use final-snapshot sessions.json (prefer over initial-snapshot)
    sessions_json = _load_final_sessions(zf)
    if sessions_json:
        sessions_by_recency: list[tuple[int, str]] = []

        for key, info in sessions_json.items():
            if not isinstance(info, dict):
                continue
            raw_ts = info.get("updatedAt", info.get("createdAt", 0))
            session_id = info.get("sessionId", key)
            if session_id:
                sessions_by_recency.append((_normalize_ts(raw_ts), session_id))

        # Meeting requirement: choose the latest session id that actually has a
        # usable JSONL file, not merely the latest id listed in sessions.json.
        for _ts, session_id in sorted(sessions_by_recency, reverse=True):
            chosen = _find_jsonl_for_session_id(zf, session_id)
            if chosen:
                return chosen

    # Strategy 3: Largest .jsonl in final-snapshot sessions/ (standard format preferred)
    final_session_jsonls = [
        n for n in zf.namelist()
        if "final-snapshot" in n and "sessions/" in n and n.endswith(".jsonl")
        and not n.endswith(".trajectory.jsonl") and "trajectory-path" not in n
        and not n.startswith("__MACOSX")
    ]
    if final_session_jsonls:
        sizes = [
            (n, zf.getinfo(n).file_size)
            for n in final_session_jsonls
            if _looks_like_valid_jsonl(zf, n)
        ]
        sizes.sort(key=lambda x: x[1], reverse=True)
        if sizes:
            return sizes[0][0]

    # Strategy 4: Largest .jsonl anywhere in sessions/
    session_jsonls = [
        n for n in zf.namelist()
        if "sessions/" in n and n.endswith(".jsonl")
        and not n.endswith(".trajectory.jsonl") and "trajectory-path" not in n
        and not n.startswith("__MACOSX")
    ]
    if session_jsonls:
        sizes = [
            (n, zf.getinfo(n).file_size)
            for n in session_jsonls
            if _looks_like_valid_jsonl(zf, n)
        ]
        sizes.sort(key=lambda x: x[1], reverse=True)
        if sizes:
            return sizes[0][0]

    # Last resort: any .jsonl
    all_jsonl = [n for n in zf.namelist() if n.endswith(".jsonl") and not n.startswith("__MACOSX")]
    if all_jsonl:
        sizes = [
            (n, zf.getinfo(n).file_size)
            for n in all_jsonl
            if _looks_like_valid_jsonl(zf, n)
        ]
        sizes.sort(key=lambda x: x[1], reverse=True)
        if sizes:
            return sizes[0][0]

    return None


def _extract_workspace(zf: zipfile.ZipFile) -> dict[str, str]:
    """Extract workspace/ files from the zip (handles final-snapshot/workspace/ or initial-snapshot/workspace/)."""
    files = {}
    # Prefer final-snapshot workspace, fall back to initial-snapshot
    workspace_entries = [
        n for n in zf.namelist()
        if "/workspace/" in n and not n.endswith("/") and not n.startswith("__MACOSX")
    ]

    # Prefer final-snapshot entries
    final_entries = [n for n in workspace_entries if "final-snapshot" in n]
    entries_to_use = final_entries if final_entries else workspace_entries

    for name in entries_to_use:
        basename = name.split("/workspace/")[-1]
        if basename.startswith(".git/") or basename.startswith(".openclaw/"):
            continue
        # Skip large binary files
        info = zf.getinfo(name)
        if info.file_size > 500_000:  # skip files > 500KB
            continue
        try:
            content = zf.read(name).decode("utf-8", errors="replace")
            files[basename] = content
        except Exception:
            pass
    return files


def _extract_text(content) -> str:
    """Extract plain text from message content (string or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            c.get("text", "") for c in content
            if isinstance(c, dict) and c.get("type") == "text"
        )
    return str(content) if content else ""


def _parse_jsonl_events(jsonl_text: str) -> list[dict]:
    """Parse JSONL into list of event dicts."""
    events = []
    for line in jsonl_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _parse_standard_session(events: list[dict]) -> tuple[list[str], list[AssistantTurn], dict[str, ToolResult], list[tuple], str]:
    """Parse standard session log format (type='message' events with role-based messages)."""
    user_messages: list[str] = []
    assistant_turns: list[AssistantTurn] = []
    tool_results: dict[str, ToolResult] = {}
    thread_order: list[tuple] = []
    session_uuid = ""

    # Extract session UUID
    for event in events:
        if event.get("type") == "session":
            session_uuid = event.get("id", "")
            break

    turn_idx = 0
    for event in events:
        if event.get("type") != "message":
            continue

        msg = event.get("message", {})
        role = msg.get("role", "")
        content = msg.get("content", [])

        if role == "user":
            text = _extract_text(content)
            if text.strip() and "heartbeat" not in text.lower():
                user_messages.append(text)
                thread_order.append(("user", len(user_messages) - 1))

        elif role == "assistant":
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]

            thinking = []
            texts = []
            calls = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "thinking":
                    thinking.append(block.get("thinking", block.get("text", "")))
                elif btype == "text":
                    texts.append(block.get("text", ""))
                elif btype in ("toolCall", "tool_use"):
                    tc = ToolCall(
                        call_id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("arguments", block.get("input", {})),
                    )
                    calls.append(tc)

            turn = AssistantTurn(
                event_id=event.get("id", ""),
                turn_index=turn_idx,
                thinking_blocks=thinking,
                text_blocks=texts,
                tool_calls=calls,
                timestamp=msg.get("timestamp"),
            )
            assistant_turns.append(turn)
            thread_order.append(("assistant", turn_idx))
            turn_idx += 1

        elif role == "toolResult":
            call_id = msg.get("toolCallId", "")
            result_content = msg.get("content", [])
            content_text = _extract_text(result_content) if isinstance(result_content, list) else str(result_content)
            is_error = msg.get("isError", False)

            tr = ToolResult(
                call_id=call_id,
                tool_name=msg.get("toolName", ""),
                content=content_text,
                is_error=is_error,
                is_empty=len(content_text.strip()) < 10,
            )
            tool_results[call_id] = tr
            thread_order.append(("tool", call_id))

    return user_messages, assistant_turns, tool_results, thread_order, session_uuid


def _parse_trajectory(events: list[dict]) -> tuple[list[str], list[AssistantTurn], dict[str, ToolResult], list[tuple], str]:
    """Parse OpenClaw trajectory events into structured components.

    Supports two formats:
    - Standard session log: events with type="message", message={role, content}
    - Native trajectory: events with type="model.completed" containing messagesSnapshot
    """
    user_messages: list[str] = []
    assistant_turns: list[AssistantTurn] = []
    tool_results: dict[str, ToolResult] = {}
    thread_order: list[tuple] = []
    session_uuid = ""

    # Detect format: standard session log has type="message" events with message.role
    has_standard_messages = any(
        e.get("type") == "message" and isinstance(e.get("message"), dict) and e["message"].get("role") in ("user", "assistant", "toolResult")
        for e in events
    )

    if has_standard_messages:
        return _parse_standard_session(events)

    # --- Native trajectory format (model.completed / prompt.submitted) ---

    # Extract session UUID
    for event in events:
        if event.get("type") == "session.started":
            session_uuid = event.get("sessionId", event.get("traceId", ""))
            break

    # Find the final model.completed event with the fullest messagesSnapshot
    final_snapshot = None
    for event in reversed(events):
        if event.get("type") == "model.completed":
            data = event.get("data", {})
            prompt_text = data.get("finalPromptText", "")
            if "heartbeat" in prompt_text.lower():
                continue
            snapshot = data.get("messagesSnapshot", [])
            if snapshot:
                final_snapshot = snapshot
                break

    if not final_snapshot:
        for event in events:
            if event.get("type") == "model.completed":
                data = event.get("data", {})
                snapshot = data.get("messagesSnapshot", [])
                if snapshot and (not final_snapshot or len(snapshot) > len(final_snapshot)):
                    final_snapshot = snapshot

    if not final_snapshot:
        for event in events:
            if event.get("type") == "prompt.submitted":
                data = event.get("data", {})
                prompt = data.get("prompt", data.get("finalPromptText", ""))
                if prompt and "heartbeat" not in prompt.lower():
                    user_messages.append(prompt)
        return user_messages, assistant_turns, tool_results, thread_order, session_uuid

    # Parse the messagesSnapshot (standard role-based format)
    turn_idx = 0
    for msg in final_snapshot:
        role = msg.get("role", "")

        if role == "user":
            text = _extract_text(msg.get("content", ""))
            if text.strip():
                user_messages.append(text)
                thread_order.append(("user", len(user_messages) - 1))

        elif role == "assistant":
            content = msg.get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]

            thinking = []
            texts = []
            calls = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")
                if btype == "thinking":
                    thinking.append(block.get("thinking", block.get("text", "")))
                elif btype == "text":
                    t = block.get("text", "")
                    if t.strip():
                        texts.append(t)
                elif btype in ("toolCall", "tool_use"):
                    tc = ToolCall(
                        call_id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("arguments", block.get("input", {})),
                    )
                    calls.append(tc)

            turn = AssistantTurn(
                event_id=f"turn-{turn_idx}",
                turn_index=turn_idx,
                thinking_blocks=thinking,
                text_blocks=texts,
                tool_calls=calls,
                timestamp=msg.get("timestamp"),
            )
            assistant_turns.append(turn)
            thread_order.append(("assistant", turn_idx))
            turn_idx += 1

        elif role in ("toolResult", "tool"):
            call_id = msg.get("toolCallId", msg.get("tool_call_id", ""))
            tool_name = msg.get("toolName", msg.get("name", ""))
            content_text = _extract_text(msg.get("content", ""))
            is_error = msg.get("isError", msg.get("is_error", False))

            tr = ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                content=content_text,
                is_error=bool(is_error),
                is_empty=len(content_text.strip()) < 10,
            )
            tool_results[call_id] = tr
            thread_order.append(("tool", call_id))

    return user_messages, assistant_turns, tool_results, thread_order, session_uuid


def _detect_task_id(zf: zipfile.ZipFile) -> str | None:
    """Auto-detect the task_id (T-NNN-NN) from session file names inside a zip."""
    final_refs: list[tuple[int, str]] = []
    for name in zf.namelist():
        if (
            "final-snapshot" in name
            and "workspace/trajectories/" in name
            and not name.startswith("__MACOSX")
        ):
            match = re.match(r"^sess-(\d+)-(T-\d{3}-\d{2})\.jsonl$", name.split("/")[-1])
            if match:
                ts = _task_ref_timestamp(name)
                if ts is not None:
                    final_refs.append((ts, match.group(2)))
    if final_refs:
        ordered = sorted(final_refs, reverse=True)
        candidates: list[str] = []
        for _ts, tid in ordered:
            if tid not in candidates:
                candidates.append(tid)
        logger.info("Auto-detected task_id(s) in final refs by recency: %s", candidates)
        return ordered[0][1]

    pattern = re.compile(r"(T-\d{3}-\d{2})")
    candidates: list[str] = []
    for name in zf.namelist():
        if name.startswith("__MACOSX"):
            continue
        m = pattern.search(name.split("/")[-1])
        if m:
            tid = m.group(1)
            if tid != "OCW-TEST" and tid not in candidates:
                candidates.append(tid)
    if candidates:
        logger.info("Auto-detected task_id(s) in zip: %s", candidates)
        return candidates[0]
    return None


def parse_export(
    task_id: str,
    submission_id: str,
    worker_id: str,
    export_url: str,
) -> ParsedTrajectory | None:
    """Download and parse a single export into a ParsedTrajectory."""
    try:
        zip_bytes = _download_zip(export_url)
    except Exception as e:
        logger.error("Failed to download %s: %s", submission_id, e)
        return None

    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        logger.error("Bad zip file for %s", submission_id)
        return None

    if task_id == "AUTO" or not task_id:
        detected = _detect_task_id(zf)
        if detected:
            task_id = detected
            logger.info("Using auto-detected task_id: %s", task_id)
        else:
            logger.warning("Could not auto-detect task_id for %s, using submission_id", submission_id)
            task_id = submission_id

    jsonl_path = _find_session_jsonl(zf, task_id=task_id)
    if not jsonl_path:
        logger.error("No session JSONL found in %s", submission_id)
        return None

    logger.info("Selected session file: %s (for task %s)", jsonl_path, task_id)

    jsonl_text = zf.read(jsonl_path).decode("utf-8", errors="replace")
    events = _parse_jsonl_events(jsonl_text)
    events = normalize_event_stream(events)

    if not events:
        logger.error("Empty trajectory for %s", submission_id)
        return None

    user_msgs, assistant_turns, tool_results, thread_order, session_uuid = _parse_trajectory(events)
    workspace_files = _extract_workspace(zf)

    persona_id = None
    m = re.match(r"^T-(\d+)-\d+$", task_id)
    if m:
        persona_id = f"P-{m.group(1)}"

    persona = {}
    if persona_id:
        personas = _load_personas()
        persona = personas.get(persona_id, {})

    task_spec = {}
    tasks = _load_tasks()
    internal_task_id = task_id.replace("T-", "P-", 1) if task_id.startswith("T-") else task_id
    task_spec = tasks.get(internal_task_id, {})

    return ParsedTrajectory(
        task_id=task_id,
        submission_id=submission_id,
        worker_id=worker_id,
        session_uuid=session_uuid,
        jsonl_path=jsonl_path,
        workspace_files=workspace_files,
        user_messages=user_msgs,
        assistant_turns=assistant_turns,
        tool_results_by_call_id=tool_results,
        thread_order=thread_order,
        ordered_events=events,
        persona=persona,
        task_spec=task_spec,
    )
