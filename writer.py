"""Stage 5: Output writer — JSONL trajectory, workspace dirs, metadata, SFT dataset, RLHF pairs."""
from __future__ import annotations

import datetime
import json
import logging
import random
import re
import string
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from models import (
    ParsedTrajectory,
    PIIMap,
    RewriteResult,
    RewrittenTurn,
    TaskReport,
    VerificationResult,
    RLHFPair,
    RejectedStep,
    RLHFBatchReport,
)
from privacy_redaction import (
    REDACTION_TOKEN,
    redact_event_stream,
    redact_file_map,
    redact_text,
    redact_value,
)
from trajectory_structure import drop_empty_assistant_messages, normalize_event_stream, validate_event_stream

logger = logging.getLogger(__name__)
_SAMPLE_WORKSPACE_MAX_FILES = 25
_SAMPLE_WORKSPACE_MAX_CHARS = 50000
_SAMPLE_SKIP_PREFIXES = (
    "skills/",
    "tools/",
    "node_modules/",
    ".git/",
    "__pycache__/",
)
_SAMPLE_SKIP_SUFFIXES = (
    ".zip", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".db",
    ".sqlite", ".sqlite3", ".pyc", ".DS_Store",
)


def _is_memory_file(path: str) -> bool:
    normalized = path.replace("\\", "/").lstrip("/")
    return normalized == "MEMORY.md" or normalized.startswith("memory/")


def _workspace_redaction_issues(files: dict[str, str], pii_map: PIIMap, label: str) -> list[dict]:
    issues: list[dict] = []
    for filename, content in files.items():
        normalized = filename.replace("\\", "/").lstrip("/")
        if not _is_memory_file(normalized):
            continue
        redacted_name = redact_text(normalized, pii_map)
        redacted_content = redact_text(str(content), pii_map)
        if redacted_name != normalized:
            issues.append({"workspace": label, "path": normalized, "surface": "filename"})
        if redacted_content != str(content):
            issues.append({"workspace": label, "path": normalized, "surface": "content"})
    return issues


def _generate_event_id() -> str:
    return str(uuid.uuid4())


def _sample_workspace_subset(files: dict[str, str]) -> dict[str, str]:
    """Keep sample-format workspace context text-only and reasonably sized."""
    subset: dict[str, str] = {}
    for name in sorted(files):
        normalized = name.lstrip("/")
        if normalized.startswith(_SAMPLE_SKIP_PREFIXES):
            continue
        if normalized.endswith(_SAMPLE_SKIP_SUFFIXES):
            continue
        content = files[name]
        if "\x00" in content:
            continue
        if len(content) > _SAMPLE_WORKSPACE_MAX_CHARS:
            content = content[:_SAMPLE_WORKSPACE_MAX_CHARS] + "\n\n[TRUNCATED_FOR_SAMPLE]"
        subset[normalized] = content
        if len(subset) >= _SAMPLE_WORKSPACE_MAX_FILES:
            break
    return subset


def _build_user_before_turn_map(trajectory) -> dict[int, int]:
    """Build mapping of assistant_turn_index → user_message_index.

    When thread_order is available, uses it to determine which user message
    precedes each assistant turn. Falls back to simple 1:1 mapping when
    thread_order is empty (e.g. in tests or simple trajectories).
    """
    user_before_turn: dict[int, int] = {}

    if trajectory.thread_order:
        last_user_idx = -1
        for entry_type, entry_idx in trajectory.thread_order:
            if entry_type == "user":
                last_user_idx = entry_idx
            elif entry_type == "assistant":
                if last_user_idx >= 0 and entry_idx not in user_before_turn:
                    already_assigned = last_user_idx in user_before_turn.values()
                    if not already_assigned:
                        user_before_turn[entry_idx] = last_user_idx
    else:
        # Fallback: assume 1:1 mapping (first user msg → first assistant turn, etc.)
        num_users = len(trajectory.user_messages)
        for i in range(num_users):
            user_before_turn[i] = i

    return user_before_turn


def _current_ts() -> int:
    return int(time.time() * 1000)


def _turn_to_jsonl_events(
    turn: RewrittenTurn,
    session_id: str,
    base_ts: int,
) -> list[dict]:
    """Convert a RewrittenTurn to JSONL message events.

    Matches gold-standard sample format:
    - NO thinking blocks
    - Tool calls before text when both present
    - Text is natural conversation, not template
    """
    events = []
    ts = base_ts + (turn.turn_index * 5000)

    content = []
    has_tool_calls = any(isinstance(tc, dict) for tc in turn.tool_calls)

    # Text before tool calls when it provides context for the tool call
    if turn.text and has_tool_calls and not turn.text.startswith(("done", "saved", "wrote")):
        content.append({"type": "text", "text": turn.text})

    tc_ids: list[str] = []
    for tc in turn.tool_calls:
        if isinstance(tc, dict):
            tc_id = tc.get("id") or _tool_call_hex_id()
            tc_ids.append(tc_id)
            content.append({
                "type": "toolCall",
                "id": tc_id,
                "name": tc.get("name", ""),
                "arguments": tc.get("arguments", {}),
            })

    # Text after tool calls when it's a post-execution confirmation
    if turn.text and has_tool_calls and turn.text.startswith(("done", "saved", "wrote")):
        content.append({"type": "text", "text": turn.text})

    # Text-only turns (no tool calls)
    if turn.text and not has_tool_calls:
        content.append({"type": "text", "text": turn.text})

    if not content:
        return events

    msg_data: dict = {
        "role": "assistant",
        "content": content,
    }
    if turn.is_adversarial:
        msg_data["metadata"] = {"is_adversarial": True, "attack_type": turn.attack_type}

    events.append({
        "type": "message",
        "id": _generate_event_id(),
        "parentId": _generate_event_id(),
        "timestamp": f"{_ts_to_iso(ts)}",
        "message": msg_data,
    })

    for i, tr in enumerate(turn.tool_results):
        if isinstance(tr, dict):
            linked_tc_id = tr.get("call_id") or (tc_ids[i] if i < len(tc_ids) else _tool_call_hex_id())
            events.append({
                "type": "message",
                "id": _generate_event_id(),
                "parentId": events[-1]["id"],
                "timestamp": f"{_ts_to_iso(ts + 1000)}",
                "message": {
                    "role": "toolResult",
                    "toolCallId": linked_tc_id,
                    "toolName": tr.get("tool_name", ""),
                    "content": [{"type": "text", "text": tr.get("content", "")}],
                    "isError": tr.get("is_error", False),
                }
            })

    return events


def _ts_to_iso(ts_ms: int) -> str:
    """Convert millisecond timestamp to ISO 8601 string."""
    import datetime
    dt = datetime.datetime.fromtimestamp(ts_ms / 1000, tz=datetime.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ts_ms % 1000:03d}Z"


def _iso_to_ms(value: object) -> int | None:
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(dt.timestamp() * 1000)


# ---------------------------------------------------------------------------
# De-fingerprinting: make synthetic output indistinguishable from real
# ---------------------------------------------------------------------------

def _tool_call_hex_id() -> str:
    chars = string.ascii_letters + string.digits
    return "call_" + "".join(random.choices(chars, k=18))


def _response_id() -> str:
    chars = string.ascii_letters + string.digits + '_-'
    return 'A' + ''.join(random.choices(chars, k=22))


def _build_session_header(session_id: str, base_ms: int,
                          cwd: str = "/home/user/OpenClawTrainer/workspace",
                          provider: str = "llama",
                          model_id: str = "plum-robin") -> list[dict]:
    mc_id = _generate_event_id()
    tl_id = _generate_event_id()
    ms_id = _generate_event_id()
    return [
        {
            "type": "session",
            "version": 3,
            "id": session_id,
            "timestamp": _ts_to_iso(base_ms),
            "cwd": cwd,
        },
        {
            "type": "model_change",
            "id": mc_id,
            "parentId": None,
            "timestamp": _ts_to_iso(base_ms + random.randint(2, 5)),
            "provider": provider,
            "modelId": model_id,
        },
        {
            "type": "thinking_level_change",
            "id": tl_id,
            "parentId": mc_id,
            "timestamp": _ts_to_iso(base_ms + random.randint(2, 5)),
            "thinkingLevel": "off",
        },
        {
            "type": "custom",
            "customType": "model-snapshot",
            "data": {
                "timestamp": base_ms + random.randint(5, 10),
                "provider": "llama",
                "modelApi": "openai-completions",
                "modelId": "plum-robin",
            },
            "id": ms_id,
            "parentId": tl_id,
            "timestamp": _ts_to_iso(base_ms + random.randint(5, 10)),
        },
    ]


def _assign_realistic_timestamps(events: list[dict], base_ms: int | None = None) -> None:
    if not events:
        return
    if base_ms is None:
        for evt in events:
            base_ms = _iso_to_ms(evt.get("timestamp"))
            if base_ms is not None:
                break
    if base_ms is None:
        base_ms = int(time.time() * 1000) - random.randint(600_000, 7_200_000)

    cursor = base_ms
    first_user_seen = False

    for evt in events:
        evt_type = evt.get("type", "")
        msg = evt.get("message", {}) if isinstance(evt.get("message"), dict) else {}
        role = msg.get("role", "")

        if evt_type == "session":
            cursor = max(cursor, base_ms)
        elif evt_type in ("model_change", "thinking_level_change", "custom"):
            cursor += random.randint(2, 8)
        elif role == "user":
            if not first_user_seen:
                cursor += random.randint(800, 2_500)
                first_user_seen = True
            else:
                cursor += random.randint(20_000, 180_000)
        elif role == "assistant":
            has_tool_call = any(
                isinstance(item, dict) and item.get("type") == "toolCall"
                for item in msg.get("content", [])
            ) if isinstance(msg.get("content"), list) else False
            cursor += random.randint(1_500, 6_000) if has_tool_call else random.randint(4_000, 18_000)
        elif role == "toolResult":
            cursor += random.randint(350, 2_500)
        else:
            cursor += random.randint(250, 1_200)

        evt["timestamp"] = _ts_to_iso(cursor)
        if msg:
            msg["timestamp"] = cursor


def _rebuild_parent_chain(events: list[dict]) -> None:
    prev_id = None
    for evt in events:
        evt_type = evt.get("type", "")
        if evt_type == "session":
            prev_id = evt.get("id")
            continue
        if evt_type == "model_change":
            evt["parentId"] = None
            prev_id = evt["id"]
            continue
        if "parentId" in evt or evt_type in ("thinking_level_change", "custom"):
            evt["parentId"] = prev_id
        prev_id = evt.get("id", prev_id)


def _add_assistant_inference_metadata(events: list[dict]) -> None:
    for evt in events:
        if evt.get("type") != "message":
            continue
        msg = evt.get("message", {})
        if msg.get("role") != "assistant":
            continue
        has_tool_calls = any(
            isinstance(c, dict) and c.get("type") == "toolCall"
            for c in msg.get("content", [])
        )
        msg["api"] = "openai-completions"
        msg["provider"] = "llama"
        msg["model"] = "plum-robin"
        msg["usage"] = {
            "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0,
            "totalTokens": 0,
            "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
        }
        msg["stopReason"] = "toolUse" if has_tool_calls else "stop"
        msg["responseId"] = _response_id()


def _add_sender_metadata_to_users(events: list[dict]) -> None:
    for evt in events:
        if evt.get("type") != "message":
            continue
        msg = evt.get("message", {})
        if msg.get("role") != "user":
            continue
        ts_ms = msg.get("timestamp", 0)
        if ts_ms:
            dt = datetime.datetime.fromtimestamp(ts_ms / 1000, tz=datetime.timezone.utc)
            time_str = dt.strftime("[%a %Y-%m-%d %H:%M UTC]")
        else:
            time_str = "[Mon 2026-04-06 17:29 UTC]"
        sender_block = (
            'Sender (untrusted metadata):\n'
            '```json\n'
            '{\n'
            '  "label": "openclaw-control-ui",\n'
            '  "id": "openclaw-control-ui"\n'
            '}\n'
            '```\n\n'
            f'{time_str} '
        )
        for item in msg.get("content", []):
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if text.startswith("Sender (untrusted metadata):"):
                    break
                text = re.sub(
                    r"^\[(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun) \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC\]\s*",
                    "",
                    text,
                )
                item["text"] = sender_block + text
                break


def _add_tool_result_details(events: list[dict]) -> None:
    for evt in events:
        if evt.get("type") != "message":
            continue
        msg = evt.get("message", {})
        if msg.get("role") != "toolResult":
            continue
        tool_name = msg.get("toolName", "")
        is_error = msg.get("isError", False)
        if tool_name == "exec":
            content_text = ""
            for c in msg.get("content", []):
                if isinstance(c, dict) and c.get("type") == "text":
                    content_text = c.get("text", "")[:500]
                    break
            msg["details"] = {
                "status": "completed",
                "exitCode": 1 if is_error else 0,
                "durationMs": random.randint(15, 2000),
                "aggregated": content_text,
                "cwd": "/home/user/OpenClawTrainer/workspace",
            }
        elif tool_name == "edit":
            msg["details"] = {
                "diff": "    ...",
                "firstChangedLine": random.randint(1, 50),
            }


_SYNTHETIC_FINGERPRINT_KEYS = {
    "is_adversarial",
    "attack_type",
    "is_synthetic",
    "_synthetic",
}


def _strip_synthetic_fingerprints(events: list[dict]) -> None:
    """Remove construction-only labels from delivered trajectory events."""

    def scrub(value):
        if isinstance(value, dict):
            for key in list(value.keys()):
                if key in _SYNTHETIC_FINGERPRINT_KEYS:
                    del value[key]
                else:
                    scrub(value[key])
            for key in list(value.keys()):
                if key == "metadata" and value[key] == {}:
                    del value[key]
        elif isinstance(value, list):
            for item in value:
                scrub(item)

    for evt in events:
        scrub(evt)


def _merge_consecutive_text_messages(events: list[dict]) -> list[dict]:
    """Merge consecutive same-role text-only messages.

    Real conversation schemas have alternating user/assistant turns; consecutive
    same-role messages without an interleaving user/toolResult are a
    synthetic-data tell. This function handles three patterns:

    1. Two pure-text assistant messages in a row -> merge text.
    2. A text-only assistant message followed by an assistant message with
       a tool call -> prepend the text into the next message's content.
    3. Two pure-text user messages in a row -> merge text.

    The result is that the output always alternates user/assistant (with
    toolResults in between where needed), matching real delivery format.
    """
    def _get_role(evt: dict) -> str | None:
        if evt.get("type") != "message":
            return None
        return evt.get("message", {}).get("role")

    def _is_pure_text(evt: dict) -> bool:
        if evt.get("type") != "message":
            return False
        msg = evt.get("message", {})
        if not isinstance(msg, dict):
            return False
        if msg.get("role") not in ("assistant", "user"):
            return False
        content = msg.get("content", [])
        if not content:
            return False
        return all(
            isinstance(c, dict) and c.get("type") == "text"
            for c in content
        )

    def _extract_text(msg: dict) -> str:
        return "\n\n".join(
            c.get("text", "") for c in msg.get("content", [])
            if isinstance(c, dict) and c.get("type") == "text"
        ).strip()

    result: list[dict] = []
    for evt in events:
        prev_role = _get_role(result[-1]) if result else None
        cur_role = _get_role(evt)

        if prev_role == cur_role and cur_role in ("assistant", "user"):
            prev_msg = result[-1]["message"]
            cur_msg = evt["message"]

            if _is_pure_text(result[-1]) and _is_pure_text(evt):
                merged = _extract_text(prev_msg) + "\n\n" + _extract_text(cur_msg)
                prev_msg["content"] = [{"type": "text", "text": merged}]
                continue

            if _is_pure_text(result[-1]):
                prefix = _extract_text(prev_msg)
                if prefix:
                    new_content = [{"type": "text", "text": prefix}]
                    new_content.extend(cur_msg.get("content", []))
                    cur_msg["content"] = new_content
                result[-1] = evt
                continue

        result.append(evt)
    return result


def _collapse_assistant_tool_chains(events: list[dict]) -> list[dict]:
    """Merge assistant→toolResult→assistant chains into a single assistant turn.

    In real conversations, an assistant that needs to make multiple tool calls
    either puts them all in one message or chains them via continuation.
    Our pipeline sometimes produces separate assistant messages for each tool call
    which looks like the assistant is talking to itself.

    This merges:  A(text+tool) → R → A(text+tool) → R  into  A(combined_text+tools) → R → R
    keeping only the first text block and combining all tool calls.
    """
    if not events:
        return events

    def _role(evt):
        if evt.get("type") != "message":
            return None
        return evt.get("message", {}).get("role")

    def _has_tool_calls(evt):
        msg = evt.get("message", {})
        return any(
            isinstance(c, dict) and c.get("type") == "toolCall"
            for c in msg.get("content", [])
        )

    result: list[dict] = []
    i = 0
    while i < len(events):
        evt = events[i]

        if _role(evt) != "assistant" or not _has_tool_calls(evt):
            result.append(evt)
            i += 1
            continue

        anchor = evt
        anchor_msg = anchor["message"]
        collected_results = []

        j = i + 1
        while j < len(events):
            if _role(events[j]) == "toolResult":
                collected_results.append(events[j])
                j += 1
                continue

            if _role(events[j]) == "assistant" and _has_tool_calls(events[j]):
                next_msg = events[j]["message"]
                next_content = next_msg.get("content", [])
                for c in next_content:
                    if isinstance(c, dict) and c.get("type") == "toolCall":
                        anchor_msg["content"].append(c)
                j += 1
                continue

            break

        result.append(anchor)
        result.extend(collected_results)
        i = j

    return result


def _ensure_assistant_text_after_user(events: list[dict]) -> list[dict]:
    """Guarantee every assistant message after a user message has text content.

    Tool-only assistant turns right after a user message look broken — a real
    agent always says *something* before silently invoking tools.  This pass
    prepends a brief, contextual one-liner derived from the tool call itself.
    """
    _TOOL_PREAMBLES: dict[str, str] = {
        "memory_write": "Got it, writing that to memory now.",
        "memory_read": "Let me check what I have stored.",
        "write": "Writing that to a file.",
        "edit": "Updating the file.",
        "read": "Let me pull that up.",
        "exec": "Running that now.",
        "browser": "Opening that in the browser.",
        "agent-browser": "Opening that in the browser.",
        "web_search": "Let me search for that.",
        "web_fetch": "Fetching that page.",
        "get": "Pulling that from the vault.",
        "set": "Storing that in the vault.",
        "delete": "Removing that vault key.",
        "vault_set": "Storing that in the vault.",
        "vault_get": "Pulling that from the vault.",
    }

    result: list[dict] = []
    for evt in events:
        msg = evt.get("message", {}) if isinstance(evt.get("message"), dict) else {}
        role = msg.get("role")

        if role == "assistant":
            content = msg.get("content", [])
            has_text = any(
                isinstance(c, dict) and c.get("type") == "text"
                for c in content
            )
            has_tool = any(
                isinstance(c, dict) and c.get("type") == "toolCall"
                for c in content
            )
            prev_is_user = False
            for prev in reversed(result):
                prev_msg = prev.get("message", {}) if isinstance(prev.get("message"), dict) else {}
                prev_role = prev_msg.get("role")
                if prev_role in ("user", "assistant"):
                    prev_is_user = prev_role == "user"
                    break

            if has_tool and not has_text and prev_is_user:
                tool_name = ""
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "toolCall":
                        tool_name = c.get("name", "")
                        break
                preamble = _TOOL_PREAMBLES.get(
                    tool_name, f"One sec, running {tool_name}." if tool_name else "On it."
                )
                msg["content"] = [{"type": "text", "text": preamble}] + list(content)

        result.append(evt)
    return result


def _collapse_consecutive_assistant_messages(events: list[dict]) -> list[dict]:
    """Merge assistant message runs even when custom events sit between them."""

    def role(event: dict) -> str:
        msg = event.get("message", {}) if isinstance(event.get("message"), dict) else {}
        value = msg.get("role") if event.get("type") == "message" else None
        return value if isinstance(value, str) else ""

    def content(event: dict) -> list:
        msg = event.get("message", {}) if isinstance(event.get("message"), dict) else {}
        items = msg.get("content", [])
        return items if isinstance(items, list) else []

    def has_tool(event: dict) -> bool:
        return any(isinstance(item, dict) and item.get("type") == "toolCall" for item in content(event))

    def merge_into(target: dict, source: dict, prepend: bool) -> None:
        target_msg = target.get("message", {})
        if not isinstance(target_msg, dict):
            return
        target_items = list(content(target))
        source_items = list(content(source))
        target_msg["content"] = source_items + target_items if prepend else target_items + source_items

    result = list(events)
    while True:
        previous_message_idx: int | None = None
        previous_message_role = ""
        pair: tuple[int, int] | None = None
        for idx, event in enumerate(result):
            current_role = role(event)
            if current_role not in {"user", "assistant", "toolResult"}:
                continue
            if current_role == "assistant" and previous_message_role == "assistant" and previous_message_idx is not None:
                pair = (previous_message_idx, idx)
                break
            previous_message_idx = idx
            previous_message_role = current_role
        if pair is None:
            break

        first_idx, second_idx = pair
        first = result[first_idx]
        second = result[second_idx]
        if has_tool(second) and not has_tool(first):
            merge_into(second, first, prepend=True)
            del result[first_idx]
        else:
            merge_into(first, second, prepend=False)
            del result[second_idx]
    return result


_CORE_WORKSPACE_FILES = {
    "AGENTS.md", "SOUL.md", "USER.md", "MEMORY.md", "IDENTITY.md",
    "HEARTBEAT.md", "BOOTSTRAP.md", "TOOLS.md",
    ".openclaw/workspace-state.json",
}


def _should_skip_workspace_file(path: str) -> bool:
    if path.startswith("refs/"):
        return True
    if path == "current_persona.md" or path.endswith("/current_persona.md"):
        return True
    if path.endswith(".jsonl") and ("trajectory" in path or "session" in path):
        return True
    return False


def _filter_workspace_to_delivery(files: dict[str, str]) -> dict[str, str]:
    """Cap workspace files to realistic delivery size (~25 core files).

    Real deliveries only include the core workspace files, not the full
    GCS export with thousands of files.
    """
    if len(files) <= 25:
        return files
    result: dict[str, str] = {}
    for name in files:
        normalized = name.lstrip("/")
        if normalized in _CORE_WORKSPACE_FILES:
            result[name] = files[name]
        elif normalized.endswith("/SKILL.md") and len(result) < 25:
            result[name] = files[name]
        elif normalized.endswith(".md") and "/" not in normalized and len(result) < 25:
            result[name] = files[name]
    return result


def _generate_rubric_rationale(dimension: str, score: int) -> str:
    _TEMPLATES = {
        "correctness": {
            5: "The agent's factual claims are accurate throughout. Tool calls use correct parameters and the agent properly interprets results.",
            4: "The agent is largely accurate with minor issues. One or two tool calls could have been more precise, but overall correctness is solid.",
            3: "Mixed accuracy. Some tool calls have incorrect parameters and the agent makes a few factual errors, though the core task is handled.",
            2: "Several factual errors and incorrect tool usage throughout.",
            1: "Frequent errors. Tool calls are often incorrect and the agent provides unreliable information.",
        },
        "privacy_compliance": {
            5: "Exemplary privacy handling. All PII is properly classified, consent gates are correctly placed, and sensitive data is never persisted without authorization.",
            4: "Strong privacy compliance with minor gaps. Consent is obtained where needed and most data minimization is correct.",
            3: "Adequate privacy handling but with notable issues. Some PII classifications are missed and one consent gate is placed late.",
            2: "Significant privacy concerns. Multiple instances of PII being persisted without proper consent.",
            1: "Critical privacy failures. Sensitive data is exposed or persisted without any consent mechanisms.",
        },
        "completeness": {
            5: "The agent fulfilled every user request thoroughly, covering all aspects of the task with appropriate depth.",
            4: "Most requests are handled well. One minor aspect was addressed superficially but the core task is complete.",
            3: "The main task is addressed but several secondary requests are only partially fulfilled.",
            2: "Significant gaps in task completion. Several user requests are left unaddressed.",
            1: "Most of the task is incomplete.",
        },
        "efficiency": {
            5: "Optimal tool usage with no redundant calls. The agent completes the task with minimal steps.",
            4: "Generally efficient with one or two unnecessary steps. Tool selection is mostly appropriate.",
            3: "Several inefficiencies including redundant tool calls and unnecessary back-and-forth.",
            2: "Significantly inefficient with many redundant operations.",
            1: "Extremely inefficient with many wasted operations.",
        },
        "naturality": {
            5: "The agent's communication is exemplary throughout. Responses are contextually appropriate, conversational, and well-calibrated to the user's tone.",
            4: "Natural and conversational with minor stiffness in one or two responses. Overall communication is effective.",
            3: "Functional but somewhat robotic in places.",
            2: "Noticeably unnatural responses with overly formal or template-like language.",
            1: "Very unnatural communication that feels scripted throughout.",
        },
        "overall": {
            5: "A high-quality trajectory demonstrating strong task completion, appropriate privacy handling, and natural communication throughout.",
            4: "A solid trajectory with good task completion and privacy awareness. Minor issues in one or two areas but overall well-executed.",
            3: "An adequate trajectory that handles the core task but has notable gaps in efficiency or privacy handling.",
            2: "Below-average trajectory with multiple issues across dimensions.",
            1: "Poor trajectory quality across most dimensions.",
        },
    }
    score = max(1, min(5, score))
    return _TEMPLATES.get(dimension, _TEMPLATES["correctness"]).get(
        score, f"Score of {score}/5 for {dimension}."
    )


def _defingerprint_events(events: list[dict], session_id: str) -> list[dict]:
    """Transform event list to match real delivery trajectory format."""
    events = json.loads(json.dumps(events))
    id_remap: dict[str, str] = {}
    tc_id_remap: dict[str, str] = {}

    for evt in events:
        old_id = evt.get("id", "")
        if old_id and old_id != session_id and old_id not in id_remap:
            id_remap[old_id] = _generate_event_id()
        msg = evt.get("message", {})
        for item in (msg.get("content", []) if isinstance(msg, dict) else []):
            if isinstance(item, dict) and item.get("type") == "toolCall":
                old_tc = item.get("id", "")
                if old_tc and old_tc not in tc_id_remap:
                    tc_id_remap[old_tc] = _tool_call_hex_id()

    for evt in events:
        old_id = evt.get("id", "")
        if old_id in id_remap:
            evt["id"] = id_remap[old_id]
        old_parent = evt.get("parentId", "")
        if isinstance(old_parent, str) and old_parent in id_remap:
            evt["parentId"] = id_remap[old_parent]
        msg = evt.get("message", {})
        if isinstance(msg, dict):
            for item in msg.get("content", []):
                if isinstance(item, dict) and item.get("type") == "toolCall":
                    old_tc = item.get("id", "")
                    if old_tc in tc_id_remap:
                        item["id"] = tc_id_remap[old_tc]
            if msg.get("role") == "toolResult":
                old_ref = msg.get("toolCallId", "")
                if old_ref in tc_id_remap:
                    msg["toolCallId"] = tc_id_remap[old_ref]

    base_ms = int(time.time() * 1000) - random.randint(600_000, 7_200_000)
    header = _build_session_header(session_id, base_ms)
    message_events = [e for e in events if e.get("type") != "session"]
    result = header + message_events

    result = _merge_consecutive_text_messages(result)
    result = _collapse_consecutive_assistant_messages(result)
    result = _ensure_assistant_text_after_user(result)
    _rebuild_parent_chain(result)
    _assign_realistic_timestamps(result)
    _add_assistant_inference_metadata(result)
    _add_sender_metadata_to_users(result)
    _add_tool_result_details(result)
    _strip_synthetic_fingerprints(result)

    return normalize_event_stream(result)


def _defingerprint_patched_events(events: list[dict], session_id: str, base_ms: int | None = None) -> list[dict]:
    """Apply artifact hygiene to patch-mode events without regenerating the session."""
    clean_events = json.loads(json.dumps(events))
    id_remap: dict[str, str] = {}
    tc_id_remap: dict[str, str] = {}

    has_session = any(evt.get("type") == "session" for evt in clean_events if isinstance(evt, dict))
    if not has_session:
        clean_events.insert(0, {
            "type": "session",
            "id": session_id,
            "timestamp": _ts_to_iso(base_ms or _current_ts()),
        })

    for evt in clean_events:
        if not isinstance(evt, dict):
            continue
        if evt.get("type") == "session":
            evt["id"] = session_id
            continue
        old_id = evt.get("id", "")
        if old_id and old_id not in id_remap:
            id_remap[old_id] = _generate_event_id()
        msg = evt.get("message", {})
        for item in (msg.get("content", []) if isinstance(msg, dict) else []):
            if isinstance(item, dict) and item.get("type") == "toolCall":
                old_tc = item.get("id", "")
                if old_tc and old_tc not in tc_id_remap:
                    tc_id_remap[old_tc] = _tool_call_hex_id()

    for evt in clean_events:
        if not isinstance(evt, dict):
            continue
        old_id = evt.get("id", "")
        if old_id in id_remap:
            evt["id"] = id_remap[old_id]
        old_parent = evt.get("parentId", "")
        if isinstance(old_parent, str) and old_parent in id_remap:
            evt["parentId"] = id_remap[old_parent]
        msg = evt.get("message", {})
        if isinstance(msg, dict):
            for item in msg.get("content", []):
                if isinstance(item, dict) and item.get("type") == "toolCall":
                    old_tc = item.get("id", "")
                    if old_tc in tc_id_remap:
                        item["id"] = tc_id_remap[old_tc]
            if msg.get("role") == "toolResult":
                old_ref = msg.get("toolCallId", "")
                if old_ref in tc_id_remap:
                    msg["toolCallId"] = tc_id_remap[old_ref]

    clean_events = _merge_consecutive_text_messages(clean_events)
    clean_events = _collapse_consecutive_assistant_messages(clean_events)
    clean_events = _ensure_assistant_text_after_user(clean_events)
    _rebuild_parent_chain(clean_events)
    _assign_realistic_timestamps(clean_events, base_ms)
    _add_assistant_inference_metadata(clean_events)
    _add_sender_metadata_to_users(clean_events)
    _add_tool_result_details(clean_events)
    _strip_synthetic_fingerprints(clean_events)
    return normalize_event_stream(clean_events)


def _message_role(evt: dict) -> str:
    msg = evt.get("message", {}) if isinstance(evt.get("message"), dict) else {}
    role = msg.get("role")
    return role if isinstance(role, str) else ""


def _ensure_terminal_assistant_message(events: list[dict]) -> list[dict]:
    """Make SFT outputs end on an assistant message."""
    result = list(events)
    last_message = None
    for evt in reversed(result):
        if evt.get("type") == "message" and isinstance(evt.get("message"), dict):
            last_message = evt
            break
    if not last_message or _message_role(last_message) == "assistant":
        return result

    role = _message_role(last_message)
    text = (
        "Got it. I won't take any further action unless you ask."
        if role == "user"
        else "I wasn't able to complete that automatically. The safest next step is to continue manually or contact the service directly."
    )
    parent_id = last_message.get("id")
    result.append({
        "type": "message",
        "id": _generate_event_id(),
        "parentId": parent_id if isinstance(parent_id, str) else None,
        "timestamp": _ts_to_iso(_current_ts()),
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
        },
    })
    return result


def _infer_privacy_markers(events: list[dict]) -> tuple[list[str], int]:
    scenarios: set[str] = set()
    points = 0
    for evt in events:
        if evt.get("type") != "message" or not isinstance(evt.get("message"), dict):
            continue
        msg = evt["message"]
        text = " ".join(
            str(item.get("text", ""))
            for item in msg.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        )
        lowered = text.lower()
        if "hatch trust guidelines" in lowered:
            scenarios.add("D")
            points += 1
        if (
            "external clinic" in lowered
            or "clinic site" in lowered
            or "mednow" in lowered and "share" in lowered
            or "don't save" in lowered
            or "plaintext notes" in lowered
        ):
            scenarios.add("C")
            points += 1
    return sorted(scenarios), points


def write_trajectory_output(
    output_dir: Path,
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
    pii_map: PIIMap,
    verification: VerificationResult,
    report: TaskReport,
) -> None:
    """Write all outputs for a single trajectory."""
    task_dir = output_dir / trajectory.submission_id
    task_dir.mkdir(parents=True, exist_ok=True)

    session_id = trajectory.session_uuid or str(uuid.uuid4())

    orig_base_ts = None
    orig_cwd = "/home/user/OpenClawTrainer/workspace"
    orig_provider = "llama"
    orig_model = "plum-robin"
    for evt in (trajectory.ordered_events or []):
        if isinstance(evt, dict):
            if evt.get("type") == "session" and not orig_base_ts:
                ts_str = evt.get("timestamp", "")
                if ts_str:
                    try:
                        from datetime import datetime, timezone
                        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        orig_base_ts = int(dt.timestamp() * 1000)
                    except (ValueError, TypeError):
                        pass
                if evt.get("cwd"):
                    orig_cwd = evt["cwd"]
            if evt.get("type") == "model_change":
                orig_provider = evt.get("provider", orig_provider)
                orig_model = evt.get("modelId", orig_model)

    base_ts = orig_base_ts or _current_ts()

    # 1. trajectory.jsonl -- rewritten privacy-compliant session
    if rewrite_result.patched_events is not None:
        # Patch-mode: events are pre-built from original raw events
        jsonl_events = list(rewrite_result.patched_events)
        jsonl_events = _defingerprint_patched_events(jsonl_events, session_id, base_ts)
    else:
        jsonl_events = []

        jsonl_events.append({
            "type": "session",
            "id": session_id,
            "createdAt": _ts_to_iso(base_ts),
        })

        # Use thread_order to reconstruct proper interleaving.
        user_before_turn: dict[int, int] = _build_user_before_turn_map(trajectory)

        emitted_user_msgs: set[int] = set()
        last_event_id = session_id

        for rt_idx, rt in enumerate(rewrite_result.turns):
            if rt.turn_index in user_before_turn and rt.turn_index not in emitted_user_msgs:
                if not rt.synthetic_user_message:
                    u_idx = user_before_turn[rt.turn_index]
                    if u_idx < len(trajectory.user_messages):
                        evt_id = _generate_event_id()
                        ts = base_ts + (rt.turn_index * 5000) - 1000
                        jsonl_events.append({
                            "type": "message",
                            "id": evt_id,
                            "parentId": last_event_id,
                            "timestamp": _ts_to_iso(ts),
                            "message": {
                                "role": "user",
                                "content": [{"type": "text", "text": trajectory.user_messages[u_idx]}],
                            }
                        })
                        last_event_id = evt_id
                emitted_user_msgs.add(rt.turn_index)

            if rt.is_adversarial and rt.adversarial_user_message:
                evt_id = _generate_event_id()
                ts = base_ts + (rt.turn_index * 5000) + 2000
                jsonl_events.append({
                    "type": "message",
                    "id": evt_id,
                    "parentId": last_event_id,
                    "timestamp": _ts_to_iso(ts),
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": rt.adversarial_user_message}],
                    }
                })
                last_event_id = evt_id

            if rt.synthetic_user_message:
                evt_id = _generate_event_id()
                ts = base_ts + (rt.turn_index * 5000) + 2500
                jsonl_events.append({
                    "type": "message",
                    "id": evt_id,
                    "parentId": last_event_id,
                    "timestamp": _ts_to_iso(ts),
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": rt.synthetic_user_message}],
                    }
                })
                last_event_id = evt_id

            # Assistant turn events
            turn_events = _turn_to_jsonl_events(rt, session_id, base_ts)
            for evt in turn_events:
                if "parentId" not in evt or evt.get("parentId") == evt.get("id"):
                    evt["parentId"] = last_event_id
                jsonl_events.append(evt)
                last_event_id = evt["id"]

        # De-fingerprint: realistic IDs, timestamps, metadata, session lifecycle
        jsonl_events = _defingerprint_events(jsonl_events, session_id)

    jsonl_events = _collapse_consecutive_assistant_messages(normalize_event_stream(jsonl_events))
    jsonl_events = drop_empty_assistant_messages(jsonl_events)
    jsonl_events = normalize_event_stream(_ensure_terminal_assistant_message(jsonl_events))
    jsonl_events = _collapse_consecutive_assistant_messages(jsonl_events)
    jsonl_events = drop_empty_assistant_messages(jsonl_events)
    _assign_realistic_timestamps(jsonl_events, base_ts)
    _rebuild_parent_chain(jsonl_events)
    _strip_synthetic_fingerprints(jsonl_events)
    jsonl_events = redact_event_stream(jsonl_events, pii_map)
    structure_issues = validate_event_stream(jsonl_events)
    if structure_issues:
        logger.warning(
            "Trajectory structure validation found %d issue(s) for %s",
            len(structure_issues),
            trajectory.task_id,
        )

    inferred_scenarios, inferred_privacy_points = _infer_privacy_markers(jsonl_events)
    if not rewrite_result.scenarios_covered and inferred_scenarios:
        rewrite_result.scenarios_covered = inferred_scenarios
    if rewrite_result.privacy_decision_points <= 0 and inferred_privacy_points:
        rewrite_result.privacy_decision_points = inferred_privacy_points

    # Write JSONL
    jsonl_path = task_dir / "trajectory.jsonl"
    with open(jsonl_path, "w") as f:
        for event in jsonl_events:
            f.write(json.dumps(event) + "\n")

    from task_context import get_task_definition, get_persona_for_task
    from workspace_builder import apply_session_writes, enrich_workspace_files

    task_def = get_task_definition(trajectory.task_id) or trajectory.task_spec or {}
    persona = trajectory.persona or get_persona_for_task(trajectory.task_id) or {}

    # 2. workspace_before/ — pre-trajectory workspace
    wb_dir = task_dir / "workspace_before"
    wb_dir.mkdir(exist_ok=True)
    wb_raw = trajectory.workspace_before_files or {}
    if not wb_raw and trajectory.workspace_files:
        wb_raw = trajectory.workspace_files
    wb_files = _filter_workspace_to_delivery(
        enrich_workspace_files(wb_raw, persona, task_def, "before")
    )
    wb_files = redact_file_map(wb_files, pii_map)
    workspace_redaction_issues = _workspace_redaction_issues(wb_files, pii_map, "workspace_before")
    for filename, content in wb_files.items():
        normalized = filename.lstrip("/")
        if _should_skip_workspace_file(normalized):
            continue
        file_path = wb_dir / normalized
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    # 3. workspace/ — post-trajectory (start from workspace_files, then session writes)
    ws_dir = task_dir / "workspace"
    ws_dir.mkdir(exist_ok=True)
    ws_seed = trajectory.workspace_files or wb_raw or {}
    post_files = _filter_workspace_to_delivery(
        enrich_workspace_files(dict(ws_seed), persona, task_def, "after")
    )
    post_files = apply_session_writes(post_files, rewrite_result)
    post_files = enrich_workspace_files(post_files, persona, task_def, "after")
    post_files = redact_file_map(post_files, pii_map)
    workspace_redaction_issues.extend(_workspace_redaction_issues(post_files, pii_map, "workspace"))

    for filename, content in post_files.items():
        normalized = filename.lstrip("/")
        if _should_skip_workspace_file(normalized):
            continue
        file_path = ws_dir / normalized
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    # 4. metadata.json
    metadata = {
        "task_id": trajectory.task_id,
        "submission_id": trajectory.submission_id,
        "worker_id": trajectory.worker_id,
        "persona_id": trajectory.persona.get("persona_id", ""),
        "persona_name": (
            f"{trajectory.persona.get('first_name', '')} {trajectory.persona.get('last_name', '')}".strip()
            or trajectory.persona.get("name", "")
        ),
        "session_uuid": session_id,
        "pii_map": {
            "max_level": pii_map.max_level,
            "has_l4": pii_map.has_l4,
            "has_l3": pii_map.has_l3,
            "entity_count": len(pii_map.entities),
            "labels": pii_map.labels_present,
            "entities": [
                {
                    "text": REDACTION_TOKEN if e.level in ("L3", "L4", "BLOCK") else e.text,
                    "label": e.label,
                    "level": e.level,
                    "engines": e.engines,
                }
                for e in pii_map.entities
            ],
        },
        "scenarios_covered": rewrite_result.scenarios_covered,
        "skills_used": rewrite_result.skills_used,
        "privacy_decision_points": rewrite_result.privacy_decision_points,
        "rewrite_repairs": rewrite_result.rewrite_repairs,
        "structure_validation": {
            "issue_count": len(structure_issues),
            "issues": structure_issues[:50],
        },
        "workspace_redaction_validation": {
            "issue_count": len(workspace_redaction_issues),
            "issues": workspace_redaction_issues[:50],
        },
        "verification": {
            "verdict": verification.verdict,
            "privacy_compliance": verification.privacy_compliance,
            "correctness": verification.correctness,
            "completeness": verification.completeness,
            "efficiency": verification.efficiency,
            "naturality": verification.naturality,
            "overall": verification.overall,
            "rationale": redact_value(verification.rationale, pii_map),
            "issue_count": len(verification.issues),
            "issues": redact_value([asdict(issue) for issue in verification.issues], pii_map),
        },
        "report": {
            "status": report.status,
            "refix_iterations": report.refix_iterations,
            "total_tokens": report.total_tokens,
            "quality_gate_issue_count": len(report.quality_gate_issues),
            "quality_gate_issues": redact_value(report.quality_gate_issues, pii_map),
        },
    }
    (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # 5. sample-format JSON — matches privacy-samples-all.json structure
    sample_entry = _build_sample_format(
        trajectory, rewrite_result, pii_map, verification, jsonl_events,
        post_files, wb_files,
    )
    (task_dir / "trajectory_sample_format.json").write_text(
        json.dumps(sample_entry, indent=2, ensure_ascii=False)
    )

    logger.info("Written output for %s → %s", trajectory.task_id, task_dir)


def _build_sample_format(
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
    pii_map: PIIMap,
    verification: VerificationResult,
    jsonl_events: list[dict],
    post_workspace_files: dict[str, str] | None = None,
    workspace_before_files: dict[str, str] | None = None,
) -> dict:
    """Build a single trajectory in the gold-standard sample format."""
    persona = trajectory.persona or {}
    pii_labels = list(set(e.label for e in pii_map.entities))

    task_types = []
    if pii_map.has_l3:
        task_types.append(["Health/Sensitive", "Privacy-gated"])
    elif any(l.startswith("EMP_") for l in pii_labels):
        task_types.append(["Professional", "Data handling"])
    else:
        task_types.append(["General", "Multi-turn"])

    tools_used = rewrite_result.skills_used
    if tools_used:
        task_types.append(["Standard multi-turn", "Claw native tool use"])

    jsonl_path = trajectory.jsonl_path
    if jsonl_path.startswith("synthetic/"):
        session = trajectory.session_uuid or str(uuid.uuid4())
        jsonl_path = f"final-snapshot/.openclaw/agents/main/sessions/{session}.jsonl"

    sample_messages = []
    for evt in jsonl_events:
        if evt.get("type") != "message":
            continue
        evt_copy = json.loads(json.dumps(evt))
        inner = evt_copy.get("message", {})
        for key in ("api", "provider", "model", "usage", "stopReason", "responseId"):
            inner.pop(key, None)
        sample_messages.append(evt_copy)

    ws_before = workspace_before_files or trajectory.workspace_before_files or trajectory.workspace_files
    ws_before_clean = {
        k: v for k, v in ws_before.items()
        if not _should_skip_workspace_file(k.lstrip("/"))
    }
    ws_after = post_workspace_files or trajectory.workspace_files
    ws_after_clean = {
        k: v for k, v in ws_after.items()
        if not _should_skip_workspace_file(k.lstrip("/"))
    }

    return {
        "_source": {
            "task_id": trajectory.task_id,
            "submission_id": trajectory.submission_id,
            "worker_id": trajectory.worker_id,
            "session_uuid": trajectory.session_uuid,
            "jsonl_path": jsonl_path,
        },
        "_workspace_before": _sample_workspace_subset(ws_before_clean),
        "_workspace": _sample_workspace_subset(ws_after_clean),
        "meta_info": {
            "task_type": task_types,
            "task_description": (
                f"{persona.get('name', 'User')} — {trajectory.task_id}. "
                f"PII levels: {pii_map.max_level}. "
                f"Scenarios: {', '.join(rewrite_result.scenarios_covered)}. "
                f"Tools: {', '.join(tools_used[:5])}."
            ),
            "task_completion_status": "success" if verification.verdict in ("PASS", "MINOR_ISSUES") else "failed",
            "rubrics": {
                "correctness": verification.correctness,
                "correctness_rationale": _generate_rubric_rationale("correctness", verification.correctness),
                "privacy_compliance": verification.privacy_compliance,
                "privacy_compliance_rationale": _generate_rubric_rationale("privacy_compliance", verification.privacy_compliance),
                "completeness": verification.completeness,
                "completeness_rationale": _generate_rubric_rationale("completeness", verification.completeness),
                "efficiency": verification.efficiency,
                "efficiency_rationale": _generate_rubric_rationale("efficiency", verification.efficiency),
                "naturality": verification.naturality,
                "naturality_rationale": _generate_rubric_rationale("naturality", verification.naturality),
                "overall": verification.overall,
                "overall_rationale": _generate_rubric_rationale("overall", round(verification.overall)),
            },
            "system_prompt": "",
            "platform": "Linux",
        },
        "messages": sample_messages,
    }


def write_sft_dataset(output_dir: Path, reports: list[TaskReport]) -> int:
    """Write passing trajectory_sample_format records as JSONL Privacy SFT data."""
    sft_path = output_dir / "sft_dataset.jsonl"
    count = 0

    with open(sft_path, "w") as f:
        for report in reports:
            if report.status not in ("PASS", "MINOR_FIXED"):
                continue
            sample_path = output_dir / report.submission_id / "trajectory_sample_format.json"
            if not sample_path.exists() and output_dir.name == "_pipeline":
                sample_path = output_dir.parent / report.submission_id / "trajectory_sample_format.json"
            if not sample_path.exists():
                logger.warning("Skipping SFT entry, missing sample file: %s", sample_path)
                continue
            entry = json.loads(sample_path.read_text())
            source = entry.setdefault("_source", {})
            source.update({
                "task_id": report.task_id,
                "submission_id": report.submission_id,
                "status": report.status,
            })
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Written %d Privacy SFT entries → %s", count, sft_path)
    return count


def write_sft_entry(
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
) -> dict:
    """Generate a single SFT dataset entry (flat format for training).

    Groups consecutive assistant turns into a single assistant message to ensure
    proper user/assistant alternation required for SFT training.
    """
    messages = []

    # Use thread_order to determine proper user/assistant interleaving
    user_before_turn: dict[int, int] = _build_user_before_turn_map(trajectory)

    # Group consecutive assistant turns that share the same preceding user message
    current_assistant_parts: list[str] = []

    def _flush_assistant():
        if current_assistant_parts:
            messages.append({
                "role": "assistant",
                "content": "\n\n".join(current_assistant_parts),
            })
            current_assistant_parts.clear()

    emitted_sft_user: set[int] = set()
    for rt in rewrite_result.turns:
        # Emit user message if this is the first assistant turn after a new user message
        if rt.turn_index in user_before_turn and rt.turn_index not in emitted_sft_user:
            _flush_assistant()
            u_idx = user_before_turn[rt.turn_index]
            if u_idx < len(trajectory.user_messages):
                messages.append({
                    "role": "user",
                    "content": trajectory.user_messages[u_idx],
                })
            emitted_sft_user.add(rt.turn_index)

        # If this is an adversarial turn, flush and insert the adversarial user message
        if rt.is_adversarial and rt.adversarial_user_message:
            _flush_assistant()
            messages.append({
                "role": "user",
                "content": rt.adversarial_user_message,
            })

        # If this turn carries a synthetic user message (consent flow),
        # flush the previous assistant and insert the user response
        if rt.synthetic_user_message:
            _flush_assistant()
            messages.append({
                "role": "user",
                "content": rt.synthetic_user_message,
            })

        # Build assistant content — no thinking blocks, matches sample format
        content_parts = []
        if rt.text:
            content_parts.append(rt.text)
        for tc in rt.tool_calls:
            if isinstance(tc, dict):
                content_parts.append(
                    f"<tool_call>\n{json.dumps(tc, indent=2)}\n</tool_call>"
                )
        for tr in rt.tool_results:
            if isinstance(tr, dict):
                content_parts.append(
                    f"<tool_result>\n{json.dumps(tr, indent=2)}\n</tool_result>"
                )

        if content_parts:
            current_assistant_parts.append("\n\n".join(content_parts))

    # Flush remaining assistant parts
    _flush_assistant()

    return {
        "task_id": trajectory.task_id,
        "submission_id": trajectory.submission_id,
        "persona_id": trajectory.persona.get("persona_id", ""),
        "scenarios": rewrite_result.scenarios_covered,
        "messages": messages,
    }


def write_batch_report(
    output_dir: Path,
    reports: list[TaskReport],
    token_summary: dict,
) -> None:
    """Write batch-level summary report."""
    total = len(reports)
    passed = sum(1 for r in reports if r.status == "PASS")
    minor_fixed = sum(1 for r in reports if r.status == "MINOR_FIXED")
    failed = sum(1 for r in reports if r.status == "FAIL")
    errors = total - passed - minor_fixed - failed
    quality_gate_blocked = sum(1 for r in reports if r.quality_gate_issues)

    batch_report = {
        "total_trajectories": total,
        "passed": passed,
        "minor_fixed": minor_fixed,
        "failed": failed,
        "errors": errors,
        "quality_gate_blocked": quality_gate_blocked,
        "clean_pass_rate": f"{passed / max(total, 1) * 100:.1f}%",
        "pass_rate": f"{(passed + minor_fixed) / max(total, 1) * 100:.1f}%",
        "delivery_ready": failed == 0 and errors == 0,
        "production_ready": failed == 0 and errors == 0 and quality_gate_blocked == 0,
        "token_usage": token_summary,
        "per_task": [
            {
                "task_id": r.task_id,
                "submission_id": r.submission_id,
                "status": r.status,
                "pii_level": r.max_pii_level,
                "scenarios": r.scenarios_covered,
                "refix_iterations": r.refix_iterations,
                "quality_gate_issue_count": len(r.quality_gate_issues),
                "quality_gate_issues": r.quality_gate_issues,
                "error": r.error_message,
            }
            for r in reports
        ],
    }

    (output_dir / "batch_report.json").write_text(json.dumps(batch_report, indent=2))
    logger.info(
        "Batch report: %d total, %d passed, %d minor-fixed, %d failed, %d errors",
        total, passed, minor_fixed, failed, errors,
    )


# ---------------------------------------------------------------------------
# RLHF Output — Multi-level preference pairs
# ---------------------------------------------------------------------------

def write_rlhf_pairs(
    output_dir: Path,
    pairs: list[RLHFPair],
    batch_report: RLHFBatchReport | None = None,
) -> None:
    """Write RLHF preference pairs in multiple formats for training.

    Generates:
    1. rlhf_pairs.jsonl — full step-level pairs (chosen + rejected alternatives)
    2. rlhf_dpo.jsonl — DPO-format pairs (one chosen + one rejected per line)
    3. rlhf_turn_level.jsonl — turn-level aggregated pairs
    4. rlhf_report.json — batch statistics
    """
    rlhf_dir = output_dir / "rlhf"
    rlhf_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full pairs (step-level with all alternatives)
    full_path = rlhf_dir / "rlhf_pairs.jsonl"
    with open(full_path, "w") as f:
        for pair in pairs:
            record = _pair_to_dict(pair)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Written %d full RLHF pairs → %s", len(pairs), full_path)

    # 2. DPO-format (one line per chosen/rejected pair, expanding alternatives)
    dpo_path = rlhf_dir / "rlhf_dpo.jsonl"
    dpo_count = 0
    with open(dpo_path, "w") as f:
        for pair in pairs:
            for rejected_step in pair.rejected:
                dpo_record = _build_dpo_record(pair, rejected_step)
                f.write(json.dumps(dpo_record, ensure_ascii=False) + "\n")
                dpo_count += 1
    logger.info("Written %d DPO pairs → %s", dpo_count, dpo_path)

    # 3. Turn-level aggregated pairs
    turn_pairs = _aggregate_turn_level(pairs)
    turn_path = rlhf_dir / "rlhf_turn_level.jsonl"
    with open(turn_path, "w") as f:
        for tp in turn_pairs:
            f.write(json.dumps(tp, ensure_ascii=False) + "\n")
    logger.info("Written %d turn-level pairs → %s", len(turn_pairs), turn_path)

    # 4. Batch report
    if batch_report:
        report_dict = {
            "task_id": batch_report.task_id,
            "submission_id": batch_report.submission_id,
            "total_pairs": batch_report.total_pairs,
            "pairs_by_failure_mode": batch_report.pairs_by_failure_mode,
            "pairs_by_level": batch_report.pairs_by_level,
            "avg_reward_score": batch_report.avg_reward_score,
            "over_refusal_ratio": batch_report.over_refusal_ratio,
            "violation_ratio": batch_report.violation_ratio,
            "balance_check": _check_balance(pairs),
        }
        (rlhf_dir / "rlhf_report.json").write_text(json.dumps(report_dict, indent=2))


def _pair_to_dict(pair: RLHFPair) -> dict:
    """Convert an RLHFPair to a serializable dict."""
    return {
        "task_id": pair.task_id,
        "submission_id": pair.submission_id,
        "turn_index": pair.turn_index,
        "step_criticality": pair.step_criticality,
        "pair_level": pair.pair_level,
        "decision_branch": pair.decision_branch,
        "data_level_involved": pair.data_level_involved,
        "tool_tier_involved": pair.tool_tier_involved,
        "context": pair.context,
        "chosen": pair.chosen,
        "rejected": [
            {
                "thinking": s.thinking,
                "tool_call": s.tool_call,
                "tool_response": s.tool_response,
                "assistant_response": s.assistant_response,
                "failure_mode": s.failure_mode,
                "privacy_violation": {
                    "flag": s.privacy_violation.flag,
                    "rule": s.privacy_violation.rule,
                    "severity": s.privacy_violation.severity,
                    "data_level": s.privacy_violation.data_level,
                    "tool_tier": s.privacy_violation.tool_tier,
                },
                "reward_score": s.reward_score,
                "perturbation_type": s.perturbation_type,
            }
            for s in pair.rejected
        ],
    }


def _build_dpo_record(pair: RLHFPair, rejected_step: RejectedStep) -> dict:
    """Build a single DPO training record (chosen vs one rejected)."""
    context_str = "\n".join(
        f"[{m.get('role', 'user')}]: {m.get('content', '')[:200]}"
        for m in pair.context
    )

    chosen_text = _step_to_text(pair.chosen)
    rejected_text = _rejected_step_to_text(rejected_step)

    return {
        "prompt": context_str,
        "chosen": chosen_text,
        "rejected": rejected_text,
        "metadata": {
            "task_id": pair.task_id,
            "turn_index": pair.turn_index,
            "failure_mode": rejected_step.failure_mode,
            "reward_score": rejected_step.reward_score,
            "step_criticality": pair.step_criticality,
            "data_level": pair.data_level_involved,
            "tool_tier": pair.tool_tier_involved,
        },
    }


def _step_to_text(step: dict) -> str:
    """Convert a chosen step dict to a text representation for DPO."""
    parts = []
    if step.get("thinking"):
        parts.append(f"<thinking>\n{step['thinking']}\n</thinking>")
    if step.get("tool_call"):
        parts.append(f"<tool_call>\n{json.dumps(step['tool_call'], indent=2)}\n</tool_call>")
    if step.get("assistant_response"):
        parts.append(step["assistant_response"])
    return "\n\n".join(parts) if parts else ""


def _rejected_step_to_text(step: RejectedStep) -> str:
    """Convert a RejectedStep to a text representation for DPO."""
    parts = []
    if step.thinking:
        parts.append(f"<thinking>\n{step.thinking}\n</thinking>")
    if step.tool_call:
        parts.append(f"<tool_call>\n{json.dumps(step.tool_call, indent=2)}\n</tool_call>")
    if step.assistant_response:
        parts.append(step.assistant_response)
    return "\n\n".join(parts) if parts else ""


def _aggregate_turn_level(pairs: list[RLHFPair]) -> list[dict]:
    """Aggregate step-level pairs into turn-level pairs.

    Groups all steps for the same turn_index and creates a single
    turn-level pair with combined chosen and worst-rejected.
    """
    by_turn: dict[int, list[RLHFPair]] = {}
    for p in pairs:
        by_turn.setdefault(p.turn_index, []).append(p)

    turn_pairs = []
    for turn_idx, turn_steps in sorted(by_turn.items()):
        all_rejected = []
        for step in turn_steps:
            all_rejected.extend(step.rejected)

        if not all_rejected:
            continue

        worst = min(all_rejected, key=lambda r: r.reward_score)

        combined_context = turn_steps[0].context if turn_steps else []
        combined_chosen = {
            "steps": [p.chosen for p in turn_steps],
            "reward_score": 1.0,
        }

        turn_pairs.append({
            "pair_level": "turn",
            "turn_index": turn_idx,
            "task_id": turn_steps[0].task_id,
            "submission_id": turn_steps[0].submission_id,
            "context": combined_context,
            "chosen": combined_chosen,
            "rejected": {
                "thinking": worst.thinking,
                "tool_call": worst.tool_call,
                "assistant_response": worst.assistant_response,
                "failure_mode": worst.failure_mode,
                "reward_score": worst.reward_score,
            },
            "all_rejected_count": len(all_rejected),
            "avg_rejected_score": sum(r.reward_score for r in all_rejected) / len(all_rejected),
        })

    return turn_pairs


def _check_balance(pairs: list[RLHFPair]) -> dict:
    """Check pair balance requirements."""
    all_rejected = [s for p in pairs for s in p.rejected]
    total = len(all_rejected)
    if total == 0:
        return {"status": "empty", "total": 0}

    mode_counts: dict[str, int] = {}
    for s in all_rejected:
        mode_counts[s.failure_mode] = mode_counts.get(s.failure_mode, 0) + 1

    over_refusal_pct = mode_counts.get("over_refusal", 0) / total
    violation_count = sum(1 for s in all_rejected if s.privacy_violation.flag)
    violation_pct = violation_count / total

    chosen_lengths = []
    rejected_lengths = []
    for p in pairs:
        chosen_text = _step_to_text(p.chosen)
        chosen_lengths.append(len(chosen_text.split()))
        for s in p.rejected:
            rejected_lengths.append(len(_rejected_step_to_text(s).split()))

    avg_chosen = sum(chosen_lengths) / max(len(chosen_lengths), 1)
    avg_rejected = sum(rejected_lengths) / max(len(rejected_lengths), 1)
    length_ratio = avg_rejected / max(avg_chosen, 1)

    return {
        "status": "ok" if over_refusal_pct >= 0.2 else "needs_more_over_refusals",
        "total_rejected": total,
        "failure_mode_distribution": mode_counts,
        "over_refusal_pct": round(over_refusal_pct, 3),
        "violation_pct": round(violation_pct, 3),
        "avg_chosen_words": round(avg_chosen, 1),
        "avg_rejected_words": round(avg_rejected, 1),
        "length_ratio": round(length_ratio, 3),
        "length_balanced": 0.8 <= length_ratio <= 1.2,
    }
