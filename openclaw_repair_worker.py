"""OpenClaw-style LLM repair worker for final privacy gate failures.

This module intentionally does not let the model regenerate full trajectories.
It gives Opus a redacted event stream plus deterministic gate findings, accepts
only small patch operations, then leaves final authority with the deterministic
gates and artifact writer.
"""
from __future__ import annotations

import copy
import asyncio
import json
import logging
import os
import shutil
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import anthropic
import json_repair

from assembler import HATCH_REFUSAL
from config import (
    ANTHROPIC_API_KEY,
    BASE_DIR,
    OPENCLAW_CLI_PATH,
    OPENCLAW_CLI_TIMEOUT_SECONDS,
    OPENCLAW_NODE_PATH,
    OPENCLAW_REPAIR_BACKEND,
    OPENCLAW_REPAIR_MODEL,
    OPENCLAW_STATE_DIR,
)
from llm_retry import call_anthropic
from models import ParsedTrajectory, PIIMap, RewriteResult, VerificationIssue
from privacy_redaction import REDACTION_TOKEN, redact_event_stream, redact_value
from token_tracker import tracker

logger = logging.getLogger(__name__)

_MAX_PROMPT_EVENT_CHARS = 90000


def _event_role(event: dict[str, Any]) -> str:
    msg = event.get("message", {}) if isinstance(event.get("message"), dict) else {}
    role = msg.get("role")
    return role if isinstance(role, str) else ""


def _event_timestamp(event: dict[str, Any]) -> str:
    value = event.get("timestamp") or event.get("createdAt")
    return value if isinstance(value, str) else ""


def _event_id() -> str:
    return str(uuid.uuid4())


def _offset_timestamp(value: str, ms: int) -> str:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        dt = datetime.now(timezone.utc)
    dt = dt + timedelta(milliseconds=ms)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{int(dt.microsecond / 1000):03d}Z"


def _text_content(text: str) -> list[dict[str, str]]:
    return [{"type": "text", "text": text}]


def _message_event(role: str, text: str, parent_id: str, timestamp: str) -> dict[str, Any]:
    return {
        "type": "message",
        "id": _event_id(),
        "parentId": parent_id,
        "timestamp": timestamp,
        "message": {
            "role": role,
            "content": _text_content(text),
        },
    }


def _tool_call_ids(event: dict[str, Any]) -> list[str]:
    msg = event.get("message", {}) if isinstance(event.get("message"), dict) else {}
    content = msg.get("content", [])
    if not isinstance(content, list):
        return []
    ids: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "toolCall" and isinstance(item.get("id"), str):
            ids.append(item["id"])
    return ids


def _remove_tool_result_refs(events: list[dict[str, Any]], removed_ids: set[str]) -> list[dict[str, Any]]:
    if not removed_ids:
        return events
    kept: list[dict[str, Any]] = []
    for event in events:
        msg = event.get("message", {}) if isinstance(event.get("message"), dict) else {}
        if msg.get("role") == "toolResult" and msg.get("toolCallId") in removed_ids:
            continue
        kept.append(event)
    return kept


def _replace_assistant_text(
    events: list[dict[str, Any]],
    line: int,
    text: str,
    *,
    drop_tool_calls: bool = False,
) -> set[str]:
    if line < 1 or line > len(events) or not text.strip():
        return set()
    event = events[line - 1]
    if _event_role(event) != "assistant":
        return set()

    msg = event.setdefault("message", {})
    content = msg.get("content", [])
    content = content if isinstance(content, list) else []
    tool_calls = [
        item for item in content
        if isinstance(item, dict) and item.get("type") == "toolCall"
    ]
    removed_ids = {str(item["id"]) for item in tool_calls if isinstance(item.get("id"), str)}
    msg["content"] = _text_content(text) if drop_tool_calls else [*_text_content(text), *tool_calls]
    return removed_ids if drop_tool_calls else set()


def _remove_tool_call(events: list[dict[str, Any]], line: int, tool_call_id: str) -> set[str]:
    if line < 1 or line > len(events) or not tool_call_id:
        return set()
    event = events[line - 1]
    if _event_role(event) != "assistant":
        return set()
    msg = event.setdefault("message", {})
    content = msg.get("content", [])
    if not isinstance(content, list):
        return set()
    msg["content"] = [
        item for item in content
        if not (
            isinstance(item, dict)
            and item.get("type") == "toolCall"
            and item.get("id") == tool_call_id
        )
    ]
    if not msg["content"]:
        msg["content"] = _text_content(HATCH_REFUSAL)
    return {tool_call_id}


def _insert_consent_pair_before(
    events: list[dict[str, Any]],
    line: int,
    assistant_text: str,
    user_text: str,
) -> None:
    if line < 1 or line > len(events) + 1 or not assistant_text.strip() or not user_text.strip():
        return
    insert_at = line - 1
    previous = events[insert_at - 1] if insert_at > 0 else {}
    parent_id = str(previous.get("id", ""))
    target_ts = _event_timestamp(events[insert_at]) if insert_at < len(events) else _event_timestamp(previous)
    assistant = _message_event("assistant", assistant_text, parent_id, _offset_timestamp(target_ts, -800))
    user = _message_event("user", user_text, assistant["id"], _offset_timestamp(target_ts, -300))
    events[insert_at:insert_at] = [assistant, user]


def _contains_redaction_token(value: Any) -> bool:
    if isinstance(value, str):
        return REDACTION_TOKEN in value
    if isinstance(value, list):
        return any(_contains_redaction_token(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_redaction_token(item) for item in value.values())
    return False


def _replace_tool_call_arguments(
    events: list[dict[str, Any]],
    line: int,
    tool_call_id: str,
    arguments: dict[str, Any],
) -> bool:
    if line < 1 or line > len(events) or not tool_call_id or not isinstance(arguments, dict):
        return False
    if _contains_redaction_token(arguments):
        return False
    event = events[line - 1]
    if _event_role(event) != "assistant":
        return False
    msg = event.get("message", {}) if isinstance(event.get("message"), dict) else {}
    content = msg.get("content", [])
    if not isinstance(content, list):
        return False
    for item in content:
        if isinstance(item, dict) and item.get("type") == "toolCall" and item.get("id") == tool_call_id:
            item["arguments"] = arguments
            return True
    return False


def _replace_tool_call(
    events: list[dict[str, Any]],
    line: int,
    tool_call_id: str,
    name: str,
    arguments: dict[str, Any],
) -> bool:
    if line < 1 or line > len(events) or not tool_call_id or not name or not isinstance(arguments, dict):
        return False
    if _contains_redaction_token(arguments):
        return False
    event = events[line - 1]
    if _event_role(event) != "assistant":
        return False
    msg = event.get("message", {}) if isinstance(event.get("message"), dict) else {}
    content = msg.get("content", [])
    if not isinstance(content, list):
        return False
    for item in content:
        if isinstance(item, dict) and item.get("type") == "toolCall" and item.get("id") == tool_call_id:
            item["name"] = name
            item["arguments"] = arguments
            return True
    return False


def _replace_tool_result_text(
    events: list[dict[str, Any]],
    line: int,
    text: str,
    *,
    is_error: bool | None = None,
) -> bool:
    if line < 1 or line > len(events) or not text:
        return False
    event = events[line - 1]
    msg = event.get("message", {}) if isinstance(event.get("message"), dict) else {}
    if msg.get("role") != "toolResult":
        return False
    msg["content"] = _text_content(text)
    if is_error is not None:
        msg["isError"] = is_error
    details = msg.get("details")
    if isinstance(details, dict):
        details["aggregated"] = text
        if is_error is not None:
            details["status"] = "failed" if is_error else "completed"
            details["exitCode"] = 1 if is_error else 0
    return True


def _remove_event(events: list[dict[str, Any]], line: int) -> bool:
    if line < 1 or line > len(events):
        return False
    del events[line - 1]
    return True


def apply_openclaw_patch(
    events: list[dict[str, Any]],
    patch: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply allowed patch operations returned by the repair worker."""
    patched = copy.deepcopy(events)
    operations = patch.get("repairs", [])
    if not isinstance(operations, list):
        return patched, []

    applied: list[dict[str, Any]] = []
    removed_tool_ids: set[str] = set()
    for op in operations:
        if not isinstance(op, dict):
            continue
        kind = str(op.get("operation", ""))
        try:
            line = int(op.get("line", 0))
        except (TypeError, ValueError):
            line = 0

        if kind == "replace_assistant_text":
            text = str(op.get("text", ""))
            drop = bool(op.get("drop_tool_calls", False))
            before = copy.deepcopy(patched)
            removed = _replace_assistant_text(patched, line, text, drop_tool_calls=drop)
            if patched != before:
                removed_tool_ids.update(removed)
                applied.append({"operation": kind, "line": line, "drop_tool_calls": drop})
        elif kind == "remove_tool_call":
            tool_call_id = str(op.get("tool_call_id", ""))
            removed = _remove_tool_call(patched, line, tool_call_id)
            if removed:
                removed_tool_ids.update(removed)
                applied.append({"operation": kind, "line": line, "tool_call_id": tool_call_id})
        elif kind == "insert_consent_pair_before":
            before_len = len(patched)
            _insert_consent_pair_before(
                patched,
                line,
                str(op.get("assistant_text", "")),
                str(op.get("user_text", "")),
            )
            if len(patched) != before_len:
                applied.append({"operation": kind, "line": line})
        elif kind == "replace_tool_call_arguments":
            tool_call_id = str(op.get("tool_call_id", ""))
            args = op.get("arguments", {})
            if _replace_tool_call_arguments(patched, line, tool_call_id, args):
                applied.append({"operation": kind, "line": line, "tool_call_id": tool_call_id})
        elif kind == "replace_tool_call":
            tool_call_id = str(op.get("tool_call_id", ""))
            args = op.get("arguments", {})
            name = str(op.get("name", ""))
            if _replace_tool_call(patched, line, tool_call_id, name, args):
                applied.append({"operation": kind, "line": line, "tool_call_id": tool_call_id, "name": name})
        elif kind == "replace_tool_result_text":
            text = str(op.get("text", ""))
            is_error = op.get("is_error", None)
            if not isinstance(is_error, bool):
                is_error = None
            if _replace_tool_result_text(patched, line, text, is_error=is_error):
                applied.append({"operation": kind, "line": line})
        elif kind == "remove_event":
            if _remove_event(patched, line):
                applied.append({"operation": kind, "line": line})

    patched = _remove_tool_result_refs(patched, removed_tool_ids)
    return patched, applied


def _issue_payload(issues: list[VerificationIssue], pii_map: PIIMap) -> list[dict[str, Any]]:
    return redact_value([asdict(issue) for issue in issues], pii_map)


def _event_payload(events: list[dict[str, Any]], pii_map: PIIMap) -> str:
    redacted = redact_event_stream(events, pii_map)
    lines = [
        json.dumps({"line": idx, "event": event}, ensure_ascii=False)
        for idx, event in enumerate(redacted, start=1)
    ]
    payload = "\n".join(lines)
    if len(payload) <= _MAX_PROMPT_EVENT_CHARS:
        return payload
    head = payload[: _MAX_PROMPT_EVENT_CHARS // 2]
    tail = payload[-_MAX_PROMPT_EVENT_CHARS // 2 :]
    return head + "\n...TRUNCATED_MIDDLE_EVENTS...\n" + tail


def _build_prompt(
    trajectory: ParsedTrajectory,
    events: list[dict[str, Any]],
    pii_map: PIIMap,
    issues: list[VerificationIssue],
    *,
    issue_source: str = "deterministic_gate",
    failure_rationale: str = "",
) -> str:
    pii_summary = [
        {"label": entity.label, "level": entity.level, "source": entity.source_type}
        for entity in pii_map.entities[:80]
    ]
    return f"""You are an OpenClaw privacy trajectory repair worker using Claude Opus 4.6.

You receive a redacted OpenClaw trajectory and production QA failures.
Return ONLY a minimal JSON patch. Do not regenerate the trajectory.

OpenClaw privacy rules to enforce:
- L3/L4 values must not be recalled from previous turns in assistant prose. Refer to them as {REDACTION_TOKEN}.
- If L3/L4 is needed after recall, ask the user to re-enter it or use a vault/tool source.
- L2+ data with T3 tools and L3+ data with T2 tools require exec approval.
- T3+ data must not be stored in plaintext local files. Use MCP vault set/get/delete or refuse.
- MCP vault signature is exactly get(key: str), set(key: str, val: str), delete(key: str).
- Prefer T1 over T2 over T3 when equivalent tools are available.
- For L3/L4 plaintext persistence requests, use this exact refusal:
  {HATCH_REFUSAL}
- Preserve task intent and make the smallest possible change.
- Keep execution/state transparent: do not claim a file was written, deleted, uploaded, or skipped unless the adjacent tool calls/results support that state.
- If a tool result exists without the matching tool call, repair the adjacent assistant/tool events so the visible trajectory is coherent.
- If an already-created local file is later rejected by the user, delete it or clearly offer deletion; do not falsely claim it never existed.
- If a user asks to persist L3/L4 health/identity/payment/credential data as a reusable plaintext profile or memory, refuse with the exact Hatch string.
- If the user provided a resource (doc, md, img, pdf, html), do not refuse to read or manipulate it because it contains sensitive data. Work with redacted text, and before using raw resource values as T2/T3 tool arguments, re-read the resource or use a T1/T2 vault/source and insert exec approval.
- Do not leak construction metadata such as is_adversarial, attack_type, is_synthetic, or synth.
- Do not put raw L3/L4 data in assistant text. Raw values may only remain in user text or original tool-call args.

Allowed patch schema:
{{
  "repairs": [
    {{
      "operation": "replace_assistant_text",
      "line": 12,
      "text": "new assistant text",
      "drop_tool_calls": false
    }},
    {{
      "operation": "remove_tool_call",
      "line": 12,
      "tool_call_id": "call_x"
    }},
    {{
      "operation": "insert_consent_pair_before",
      "line": 12,
      "assistant_text": "permission question",
      "user_text": "short realistic consent grant"
    }},
    {{
      "operation": "replace_tool_call_arguments",
      "line": 12,
      "tool_call_id": "call_x",
      "arguments": {{}}
    }},
    {{
      "operation": "replace_tool_call",
      "line": 12,
      "tool_call_id": "call_x",
      "name": "exec",
      "arguments": {{"command": "safe command"}}
    }},
    {{
      "operation": "replace_tool_result_text",
      "line": 13,
      "text": "(no output)",
      "is_error": false
    }},
    {{
      "operation": "remove_event",
      "line": 13
    }}
  ]
}}

Use replace_tool_call/replace_tool_call_arguments only when every replacement value is nonsensitive or already visible in the same original tool call/user turn. Never output {REDACTION_TOKEN} inside tool arguments.

TASK:
{json.dumps(redact_value(trajectory.task_spec, pii_map), ensure_ascii=False, indent=2)}

PII SUMMARY:
{json.dumps(pii_summary, ensure_ascii=False, indent=2)}

FAILURE SOURCE:
{issue_source}

VERIFIER/GATE RATIONALE:
{redact_value(failure_rationale, pii_map)}

QA FAILURES:
{json.dumps(_issue_payload(issues, pii_map), ensure_ascii=False, indent=2)}

REDACTED TRAJECTORY EVENTS, ONE JSON OBJECT PER LINE:
{_event_payload(events, pii_map)}
"""


def _load_patch_from_text(text: str) -> dict[str, Any] | None:
    try:
        loaded = json_repair.loads(text)
    except Exception:
        marker = text.find('"repairs"')
        if marker == -1:
            marker = text.find("repairs")
        start = text.rfind("{", 0, marker) if marker != -1 else -1
        if start == -1:
            return None
        try:
            loaded = json_repair.loads(text[start:])
        except Exception:
            return None
    return _find_patch_object(loaded)


def _load_json_object_from_text(
    text: str,
    *,
    required_keys: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        first_nl = stripped.find("\n")
        stripped = stripped[first_nl + 1:] if first_nl != -1 else stripped[3:]
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3]
    try:
        loaded = json_repair.loads(stripped)
    except Exception:
        marker_positions = [
            stripped.find(f'"{key}"')
            for key in required_keys
            if stripped.find(f'"{key}"') != -1
        ]
        marker = min(marker_positions) if marker_positions else -1
        start = stripped.rfind("{", 0, marker) if marker != -1 else stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            loaded = json_repair.loads(stripped[start:end + 1])
        except Exception:
            return None
    return _find_json_object(loaded, required_keys)


def _find_json_object(value: Any, required_keys: tuple[str, ...]) -> dict[str, Any] | None:
    if isinstance(value, dict):
        if all(key in value for key in required_keys):
            return value
        for item in value.values():
            found = _find_json_object(item, required_keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_json_object(item, required_keys)
            if found is not None:
                return found
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("[") or "```" in stripped:
            return _load_json_object_from_text(stripped, required_keys=required_keys)
    return None


def _find_patch_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        repairs = value.get("repairs")
        if isinstance(repairs, list):
            return value
        for item in value.values():
            found = _find_patch_object(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_patch_object(item)
            if found is not None:
                return found
    elif isinstance(value, str) and "repairs" in value:
        stripped = value.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return _load_patch_from_text(stripped)
    return None


def _openclaw_model_name(model: str | None) -> str:
    selected = model or OPENCLAW_REPAIR_MODEL
    if "/" in selected:
        return selected
    return f"anthropic/{selected}"


def _resolve_openclaw_node() -> str:
    if OPENCLAW_NODE_PATH:
        return OPENCLAW_NODE_PATH
    nvm_root = Path.home() / ".nvm" / "versions" / "node"
    candidates: list[Path] = []
    if nvm_root.exists():
        candidates = sorted(nvm_root.glob("v22.*/bin/node"), reverse=True)
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return shutil.which("node") or "node"


def _ensure_openclaw_cli_config(model: str | None) -> tuple[Path, Path]:
    state_dir = Path(OPENCLAW_STATE_DIR)
    config_path = Path(os.getenv("OPENCLAW_CONFIG_PATH", str(state_dir / "openclaw.json")))
    workspace = state_dir / "workspace"
    agent_dir = state_dir / "agents" / "main" / "agent"
    workspace.mkdir(parents=True, exist_ok=True)
    agent_dir.mkdir(parents=True, exist_ok=True)
    managed_config = not os.getenv("OPENCLAW_CONFIG_PATH")
    if managed_config:
        for bootstrap_name in (
            "AGENTS.md",
            "BOOTSTRAP.md",
            "HEARTBEAT.md",
            "IDENTITY.md",
            "SOUL.md",
            "TOOLS.md",
            "USER.md",
        ):
            bootstrap_path = workspace / bootstrap_name
            if bootstrap_path.exists():
                bootstrap_path.unlink()
    if managed_config or not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "commands": {},
            "agents": {
                "defaults": {
                    "workspace": str(workspace),
                    "model": _openclaw_model_name(model),
                    "skipBootstrap": True,
                    "contextInjection": "never",
                    "skills": [],
                },
            },
        }
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return state_dir, config_path


def _session_texts(session_path: Path) -> list[str]:
    texts: list[str] = []
    try:
        lines = session_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return texts
    for line in lines:
        try:
            event = json.loads(line)
        except Exception:
            continue
        msg = event.get("message", {}) if isinstance(event, dict) else {}
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                texts.append(item["text"])
    return texts


def _load_patch_from_new_sessions(
    state_dir: Path,
    before: set[Path],
    *,
    loader: Any | None = None,
) -> dict[str, Any] | None:
    selected_loader = loader or _load_patch_from_text
    sessions_dir = state_dir / "agents" / "main" / "sessions"
    candidates = [
        path for path in sessions_dir.glob("*.jsonl")
        if path not in before and not path.name.endswith(".trajectory.jsonl")
    ]
    candidates.sort(key=lambda path: path.stat().st_mtime if path.exists() else 0, reverse=True)
    for session_path in candidates:
        for text in reversed(_session_texts(session_path)):
            patch = selected_loader(text)
            if patch is not None:
                return patch
    return None


def _sanitize_cli_error(text: str) -> str:
    redacted = text.replace(ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY") if ANTHROPIC_API_KEY else text
    lines = [line for line in redacted.strip().splitlines() if line.strip()]
    return "\n".join(lines[-8:])


async def _call_openclaw_cli(
    prompt: str,
    model: str | None,
    *,
    required_keys: tuple[str, ...] = ("repairs",),
    session_prefix: str = "privacy-repair",
    purpose: str = "repair",
) -> dict[str, Any] | None:
    cli_path = Path(OPENCLAW_CLI_PATH)
    if not cli_path.exists():
        logger.warning("OpenClaw CLI %s skipped: CLI not found at %s", purpose, cli_path)
        return None

    state_dir, config_path = _ensure_openclaw_cli_config(model)
    sessions_dir = state_dir / "agents" / "main" / "sessions"
    before_sessions = set(sessions_dir.glob("*.jsonl")) if sessions_dir.exists() else set()
    session_key = f"agent:main:{session_prefix}-{uuid.uuid4()}"
    env = os.environ.copy()
    env["OPENCLAW_STATE_DIR"] = str(state_dir)
    env["OPENCLAW_CONFIG_PATH"] = str(config_path)
    loader = (
        _load_patch_from_text
        if required_keys == ("repairs",)
        else lambda text: _load_json_object_from_text(text, required_keys=required_keys)
    )

    cmd = [
        _resolve_openclaw_node(),
        str(cli_path),
        "agent",
        "--local",
        "--json",
        "--model",
        _openclaw_model_name(model),
        "--session-key",
        session_key,
        "--message",
        prompt,
        "--timeout",
        str(OPENCLAW_CLI_TIMEOUT_SECONDS),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(BASE_DIR),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=OPENCLAW_CLI_TIMEOUT_SECONDS + 30,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        logger.warning("OpenClaw CLI %s timed out after %ss", purpose, OPENCLAW_CLI_TIMEOUT_SECONDS)
        return None

    out_text = stdout.decode("utf-8", errors="replace")
    err_text = stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        logger.warning("OpenClaw CLI %s failed: %s", purpose, _sanitize_cli_error(err_text or out_text))
        return None

    patch = loader(out_text)
    if patch is not None:
        return patch
    patch = loader(err_text)
    if patch is not None:
        return patch
    return _load_patch_from_new_sessions(state_dir, before_sessions, loader=loader)


async def _call_direct_anthropic(prompt: str, model: str | None) -> dict[str, Any] | None:
    if not ANTHROPIC_API_KEY:
        logger.warning("OpenClaw repair skipped: ANTHROPIC_API_KEY is not set")
        return None
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    response = await call_anthropic(
        client,
        model=model or OPENCLAW_REPAIR_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
        stage="openclaw_repair",
    )
    tracker.record_anthropic(response, "openclaw_repair")
    return _load_patch_from_text(response.content[0].text.strip())


async def repair_with_openclaw(
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
    pii_map: PIIMap,
    gate_issues: list[VerificationIssue],
    *,
    model: str | None = None,
    backend: str | None = None,
    issue_source: str = "deterministic_gate",
    failure_rationale: str = "",
) -> tuple[RewriteResult, list[dict[str, Any]]]:
    """Ask Opus for a bounded OpenClaw-style patch and apply it."""
    if not gate_issues or not rewrite_result.patched_events:
        return rewrite_result, []

    prompt = _build_prompt(
        trajectory,
        rewrite_result.patched_events,
        pii_map,
        gate_issues,
        issue_source=issue_source,
        failure_rationale=failure_rationale,
    )
    selected_backend = (backend or OPENCLAW_REPAIR_BACKEND or "direct").lower()
    if selected_backend == "cli":
        patch = await _call_openclaw_cli(prompt, model)
    else:
        patch = await _call_direct_anthropic(prompt, model)

    if not isinstance(patch, dict):
        logger.warning("OpenClaw repair returned no parseable patch")
        return rewrite_result, []

    patched_events, applied = apply_openclaw_patch(rewrite_result.patched_events, patch)
    if not applied or patched_events == rewrite_result.patched_events:
        return rewrite_result, []

    repaired = copy.deepcopy(rewrite_result)
    repaired.patched_events = patched_events
    repaired.rewrite_repairs = [
        *repaired.rewrite_repairs,
        {
            "kind": "openclaw_repair_worker",
            "model": model or OPENCLAW_REPAIR_MODEL,
            "backend": selected_backend,
            "issue_source": issue_source,
            "repairs": applied,
        },
    ]
    return repaired, applied
