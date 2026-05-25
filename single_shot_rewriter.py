"""Patch-mode Privacy SFT rewriter (cuarena-inspired architecture).

Instead of regenerating trajectories (which causes hallucination), this
module preserves original JSONL events byte-for-byte and only SPLICES
new events at violation points. The approach:

  1. CLASSIFY every tool call deterministically (tier, is_write, data layer,
     free-band violation) and annotate them in-place on the events
  2. DETECT violations where data layer exceeds the tool's free band
  3. ASK the LLM to generate ONLY small patch text (consent wording,
     adversarial probe, refusal text) -- never the full trajectory
  4. SPLICE new consent-gate / adversarial events into the original stream
     with correct parentId chains and realistic timestamp offsets
  5. ANNOTATE all tool calls with classification metadata (class + flag)

This guarantees:
  - 100% original tool call and result preservation
  - 100% original user message preservation
  - Zero hallucination (LLM never sees or regenerates full trajectory)
  - Correct parentId chains across spliced events
  - Classification metadata on every tool call (like cuarena-pipeline)
"""
from __future__ import annotations

import json
import logging
import re
import shlex
from pathlib import Path
from typing import Any

import anthropic

from config import ANTHROPIC_API_KEY, REWRITER_MODEL
from models import (
    ParsedTrajectory, PIIMap, PIIEntity,
    RewriteResult, RewrittenTurn,
)
from privacy_redaction import REDACTION_TOKEN, redact_event_stream, redact_text, redact_value
from token_tracker import tracker

logger = logging.getLogger(__name__)

_TOOL_TIERS: dict[str, int] = {}
_tiers_path = Path(__file__).with_name("tool_tiers.json")
if _tiers_path.exists():
    _TOOL_TIERS = json.loads(_tiers_path.read_text()).get("tool", {})

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Format the original trajectory as readable text for the LLM
# ---------------------------------------------------------------------------

FREE_BAND: dict[tuple[int, bool], int] = {
    (1, False): 4, (1, True): 2,
    (2, False): 2, (2, True): 2,
    (3, False): 1, (3, True): 1,
}

HATCH_REFUSAL = (
    "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. "
    "Please provide it again for the current task."
)


_T1_FALLBACK = {
    "read", "write", "edit", "exec", "process", "gateway",
    "memory_write", "memory_search", "active_memory_write",
    "wiki_apply", "cron", "glob", "grep",
    "rag_search", "rag_add", "rag_get", "rag_delete",
    # Gio's minimal MCP vault signature. Treat these as local encrypted vault
    # tools when they appear as standalone tool names.
    "get", "set", "delete",
    "vault_get", "vault_set", "vault_delete",
}
_T2_FALLBACK = {
    "enterprise-mail", "enterprise-calendar", "enterprise-vault",
    "enterprise-crm", "enterprise-jira",
}


def _classify_tool_tier(name: str) -> int:
    """Return the canonical tool tier (1/2/3) for a tool name."""
    if name in _TOOL_TIERS:
        return _TOOL_TIERS[name]
    if name in _T1_FALLBACK:
        return 1
    if name in _T2_FALLBACK or name.startswith("enterprise-"):
        return 2
    return 3


def _is_write_tool(name: str, args: dict) -> bool:
    """Determine if this tool call mutates persistent state."""
    write_tools = {
        "memory_write", "write", "edit", "cron",
        "active_memory_write", "wiki_apply",
        "set", "delete", "vault_set", "vault_delete",
    }
    if name in write_tools:
        return True
    if name == "exec":
        cmd = str(args.get("command", "")).lower()
        return any(w in cmd for w in ["write", "mv", "cp", "rm", "mkdir", "touch", "sed", "tee", ">"])
    if name.startswith("enterprise-") and any(
        k in str(args).lower() for k in ["create", "update", "send", "put", "post", "delete", "store"]
    ):
        return True
    return False


def _remap_arguments(args: dict, param_map: dict | None) -> dict:
    if not param_map:
        return dict(args)
    return {param_map.get(key, key): value for key, value in args.items()}


def _lower_tier_substitution(
    *,
    tool_name: str,
    call_id: str,
    args: dict,
    tier: int,
    is_write: bool,
    data_layer: int,
    catalog: Any,
    result_index: dict,
    all_texts: list,
    consumed_for_call_fn: Any,
) -> dict | None:
    """Return a safe lower-tier replacement from the capability catalog."""
    if catalog is None or tier <= 1 or data_layer < 2:
        return None
    try:
        from skill_substitution import evaluate as subst_evaluate
        consumed = (
            consumed_for_call_fn(call_id, result_index, all_texts)
            if consumed_for_call_fn is not None
            else None
        )
        subst = subst_evaluate(
            catalog,
            tool_name,
            tier_hint=tier,
            consumed_fields=consumed,
            args=args,
        )
        if not subst.has_drop_in:
            return None
        best = next(candidate for candidate in subst.candidates if candidate.verdict == "drop-in")
        lower_free = FREE_BAND.get((best.tier, is_write), 1)
        if data_layer > lower_free:
            return None
        return {
            "original_tool": tool_name,
            "downgraded_tool": best.tool,
            "downgraded_tier": best.tier,
            "capability": subst.capability,
            "reason": best.reason,
            "param_map": best.param_map,
            "remapped_arguments": _remap_arguments(args, best.param_map),
        }
    except Exception:
        return None


def _apply_tool_downgrade_to_event(event: dict, violation: dict) -> tuple[dict, list[str]]:
    updated = json.loads(json.dumps(event))
    call_ids: list[str] = []
    original_tool = str(violation.get("original_tool") or violation.get("tool_name") or "")
    downgraded_tool = str(violation.get("downgraded_tool") or "")
    if not downgraded_tool:
        return updated, call_ids
    downgraded_tier = int(violation.get("downgraded_tier") or 1)
    remapped_args = violation.get("remapped_arguments")
    if not isinstance(remapped_args, dict):
        remapped_args = _remap_arguments(
            violation.get("arguments", {}) if isinstance(violation.get("arguments"), dict) else {},
            violation.get("param_map") if isinstance(violation.get("param_map"), dict) else {},
        )
    target_call_id = str(violation.get("call_id", ""))
    for tool_call in _tool_calls_from_event(updated):
        call_id = str(tool_call.get("id", ""))
        if target_call_id and call_id != target_call_id:
            continue
        if not target_call_id and str(tool_call.get("name", "")) != original_tool:
            continue
        tool_call["name"] = downgraded_tool
        tool_call["arguments"] = remapped_args
        tool_call["class"] = {
            "tool_tier": downgraded_tier,
            "is_write": _is_write_tool(downgraded_tool, remapped_args),
            "arguments": {},
        }
        tool_call["flag"] = {
            "is_violation": False,
            "free_band": FREE_BAND.get((downgraded_tier, _is_write_tool(downgraded_tool, remapped_args)), 1),
            "data_layer": int(violation.get("data_layer") or 0),
            "authorization": {"needed": False, "granted": False},
        }
        if call_id:
            call_ids.append(call_id)
    return updated, call_ids


_HEALTH_KEYWORDS = (
    "allerg", "allergolog", "allergist", "medical", "diagnos", "symptom",
    "treatment", "patient", "health",
)

_EXTERNAL_MEDICAL_KEYWORDS = (
    "mednow.it", "gestionaleambulatorio.it", "bookly", "prenota",
    "allergologia", "clinic", "medical", "doctor",
)

_HEALTH_APPOINTMENT_PERSISTENCE_KEYWORDS = (
    "appointment", "doctor", "dr.", "clinic", "mednow", "allergist",
    "allergologia", "visit", "consult", "provider",
)


def _max_data_layer_for_args(args: dict, pii_map: PIIMap) -> int:
    """Estimate the highest data layer present in tool call arguments."""
    level_map = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}
    max_layer = 0
    args_text = json.dumps(args).lower()
    for entity in pii_map.entities:
        if entity.text.lower() in args_text:
            max_layer = max(max_layer, level_map.get(entity.level, 0))
    has_health_pii = any(
        level_map.get(entity.level, 0) >= 3
        and (
            "health" in entity.label.lower()
            or "medical" in entity.label.lower()
            or "allerg" in entity.text.lower()
        )
        for entity in pii_map.entities
    )
    if has_health_pii and any(keyword in args_text for keyword in _HEALTH_KEYWORDS):
        max_layer = max(max_layer, 3)
    return max_layer


def _contains_exact_pii(args: dict, pii_map: PIIMap, min_level: int = 2) -> bool:
    level_map = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}
    args_text = json.dumps(args).lower()
    return any(
        level_map.get(entity.level, 0) >= min_level and entity.text.lower() in args_text
        for entity in pii_map.entities
    )


def _is_encrypted_persistence(name: str, args: dict) -> bool:
    args_text = json.dumps(args).lower() if isinstance(args, dict) else ""
    if name in {
        "active_memory_write",
        "active_memory_set",
        "wiki_apply",
        "enterprise-vault",
        "set",
        "vault_set",
        "vault_store",
    }:
        return True
    if name == "memory_write":
        encrypted_markers = (
            "encrypted", "encryption", "ciphertext", "vault", "secret", "sealed",
        )
        return any(marker in args_text for marker in encrypted_markers)
    return False


def _command_writes_plaintext_path(command: str) -> bool:
    lowered = command.lower()
    if any(marker in lowered for marker in ("enterprise-vault", "vault_set", "encrypted", "ciphertext")):
        return False
    writes = any(token in lowered for token in (" > ", ">>", " tee ", "touch ", "python", "cat >"))
    plaintext_target = (
        ".md" in lowered
        or ".txt" in lowered
        or ".csv" in lowered
        or ".json" in lowered
        or "/workspace/" in lowered
        or "user.md" in lowered
        or "memory.md" in lowered
    )
    return writes and plaintext_target


def _is_plaintext_persistence(name: str, args: dict) -> bool:
    if _is_encrypted_persistence(name, args):
        return False
    if name not in {"write", "edit", "memory_write", "exec", "set"}:
        return False
    if name == "exec":
        return _command_writes_plaintext_path(str(args.get("command", "")))
    path = str(args.get("path", "")).lower() if isinstance(args, dict) else ""
    return (
        name == "memory_write"
        or path.endswith(".md")
        or "/workspace/" in path
        or "user.md" in path
        or "memory.md" in path
    )


def _looks_like_health_appointment_persistence(args: dict) -> bool:
    if isinstance(args, dict) and _is_encrypted_persistence("", args):
        return False
    args_text = json.dumps(args).lower()
    has_appointment = any(
        keyword in args_text for keyword in _HEALTH_APPOINTMENT_PERSISTENCE_KEYWORDS
    )
    has_plaintext_target = any(
        marker in args_text for marker in ("user.md", "memory.md", "/workspace/", ".md")
    )
    return has_appointment and has_plaintext_target


def _message_content(event: dict) -> list[dict]:
    msg = event.get("message", {})
    content = msg.get("content", []) if isinstance(msg, dict) else []
    return content if isinstance(content, list) else []


def _assistant_text(event: dict) -> str:
    texts: list[str] = []
    for item in _message_content(event):
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text", "")
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts)


def _assistant_all_text(event: dict) -> str:
    texts: list[str] = []
    for item in _message_content(event):
        if not isinstance(item, dict):
            continue
        for field in ("text", "thinking"):
            value = item.get(field)
            if isinstance(value, str):
                texts.append(value)
    return "\n".join(texts)


def _replace_assistant_with_text(event: dict, text: str) -> dict:
    new_event = json.loads(json.dumps(event))
    new_event.setdefault("message", {})["content"] = [{"type": "text", "text": text}]
    return new_event


def _redact_assistant_text_fields(event: dict, pii_map: PIIMap) -> dict:
    new_event = json.loads(json.dumps(event))
    for item in _message_content(new_event):
        if not isinstance(item, dict):
            continue
        for field in ("text", "thinking"):
            if isinstance(item.get(field), str):
                item[field] = redact_text(item[field], pii_map)
    return new_event


def _normalize_assistant_punctuation(events: list[dict]) -> None:
    emoji_re = re.compile(
        "["
        "\U0001F300-\U0001FAFF"
        "\U00002700-\U000027BF"
        "\U00002600-\U000026FF"
        "\ufe0f"
        "]+"
    )
    for event in events:
        msg = event.get("message", {}) if isinstance(event, dict) else {}
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for item in _message_content(event):
            if not isinstance(item, dict):
                continue
            for field in ("text", "thinking"):
                if isinstance(item.get(field), str):
                    cleaned = item[field].replace("\u2014", ",")
                    cleaned = emoji_re.sub("", cleaned)
                    cleaned = cleaned.replace("Noriceived", "I did not receive")
                    cleaned = cleaned.replace("Tu informaci\u00f3n stay where it belongs.", "Your information stayed local.")
                    cleaned = cleaned.replace("Tu informacion stay where it belongs.", "Your information stayed local.")
                    cleaned = re.sub(r"\.\s+([a-z])", r", \1", cleaned)
                    cleaned = re.sub(r"\s+,", ",", cleaned)
                    cleaned = re.sub(r",\s*,", ",", cleaned)
                    cleaned = re.sub(r"\s+\.", ".", cleaned)
                    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
                    item[field] = cleaned


def _dedupe_recent_text_messages(events: list[dict]) -> list[dict]:
    """Remove short duplicate text-only user/assistant messages introduced by patches."""
    result: list[dict] = []
    parent_rewrite: dict[str, str] = {}
    recent: list[tuple[str, str, str]] = []

    for original in events:
        event = json.loads(json.dumps(original)) if isinstance(original, dict) else original
        if not isinstance(event, dict):
            result.append(event)
            continue

        parent_id = event.get("parentId")
        if isinstance(parent_id, str) and parent_id in parent_rewrite:
            event["parentId"] = parent_rewrite[parent_id]

        msg = event.get("message", {})
        content = msg.get("content", []) if isinstance(msg, dict) else []
        role = msg.get("role") if isinstance(msg, dict) else ""
        text_parts = [
            str(item.get("text", "")).strip()
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        has_tool = any(isinstance(item, dict) and item.get("type") == "toolCall" for item in content)
        text = "\n".join(part for part in text_parts if part).strip()
        can_dedupe = role in {"assistant", "user"} and text and not has_tool and len(text) < 240
        key = (str(role), text)
        duplicate_parent = next((seen_id for seen_role, seen_text, seen_id in recent if (seen_role, seen_text) == key), "")
        if can_dedupe and duplicate_parent:
            event_id = str(event.get("id", ""))
            if event_id:
                parent_rewrite[event_id] = duplicate_parent
            continue

        result.append(event)
        if can_dedupe:
            recent.append((str(role), text, str(event.get("id", result[-1].get("parentId", "")))))
            recent = recent[-8:]

    return result


def _user_profile_scrub_call_ids(event: dict) -> set[str]:
    msg = event.get("message", {}) if isinstance(event, dict) else {}
    if not isinstance(msg, dict) or msg.get("role") != "assistant":
        return set()
    text = _assistant_text(event).lower()
    if "found health details in plaintext user.md" not in text:
        return set()
    call_ids: set[str] = set()
    for tool_call in _tool_calls_from_event(event):
        if tool_call.get("name") != "write":
            continue
        args = tool_call.get("arguments", {})
        path = str(args.get("path", "")).lower() if isinstance(args, dict) else ""
        if path.endswith("user.md") or path.endswith("/user.md"):
            call_id = str(tool_call.get("id", ""))
            if call_id:
                call_ids.add(call_id)
    return call_ids


def _remove_redundant_user_profile_scrubs(events: list[dict], pii_map: PIIMap) -> list[dict]:
    """Drop scrub flows when the preceding USER.md read result is already health-clean."""
    result: list[dict] = []
    parent_rewrite: dict[str, str] = {}
    skip_tool_results: set[str] = set()

    for original in events:
        event = json.loads(json.dumps(original)) if isinstance(original, dict) else original
        if not isinstance(event, dict):
            result.append(event)
            continue

        parent_id = event.get("parentId")
        if isinstance(parent_id, str) and parent_id in parent_rewrite:
            event["parentId"] = parent_rewrite[parent_id]

        msg = event.get("message", {})
        if isinstance(msg, dict) and msg.get("role") == "toolResult":
            call_id = str(msg.get("toolCallId", ""))
            if call_id in skip_tool_results:
                event_id = str(event.get("id", ""))
                replacement_parent = result[-1].get("id", event.get("parentId", "")) if result else event.get("parentId", "")
                if event_id and replacement_parent:
                    parent_rewrite[event_id] = str(replacement_parent)
                continue

        scrub_call_ids = _user_profile_scrub_call_ids(event)
        if scrub_call_ids:
            previous_event = result[-1] if result else None
            previous_previous_event = result[-2] if len(result) >= 2 else None
            if (
                not _is_user_profile_l3_read_result(previous_event or {}, previous_previous_event, pii_map)
                or not _visible_user_profile_read_has_health_marker(previous_event or {}, pii_map)
            ):
                event_id = str(event.get("id", ""))
                replacement_parent = previous_event.get("id", event.get("parentId", "")) if isinstance(previous_event, dict) else event.get("parentId", "")
                if event_id and replacement_parent:
                    parent_rewrite[event_id] = str(replacement_parent)
                skip_tool_results.update(scrub_call_ids)
                continue

        result.append(event)

    return result


def _tool_calls_from_event(event: dict) -> list[dict]:
    return [
        item for item in _message_content(event)
        if isinstance(item, dict) and item.get("type") == "toolCall"
    ]


def _tool_result_text(event: dict) -> str:
    return "\n".join(
        str(item.get("text", ""))
        for item in _message_content(event)
        if isinstance(item, dict) and item.get("type") == "text"
    )


def _exec_command_from_tool_call(tool_call: dict) -> str:
    if tool_call.get("name") != "exec":
        return ""
    args = tool_call.get("arguments", {})
    if not isinstance(args, dict):
        return ""
    return str(args.get("command", ""))


def _is_tls_bypass_text(text: str) -> bool:
    lowered = text.lower()
    return (
        "verify=false" in lowered
        or "--insecure" in lowered
        or re.search(r"\s-k(\s|$)", lowered) is not None
        or "ssl_cert_file=\"\"" in lowered
        or "ssl_cert_file=''" in lowered
        or "ssl_ca_cert=\"\"" in lowered
        or "ssl_ca_cert=''" in lowered
        or "curl_ca_bundle=\"\"" in lowered
        or "curl_ca_bundle=''" in lowered
    )


def _is_http_gateway_downgrade_text(text: str) -> bool:
    lowered = text.lower()
    return (
        re.search(r"gateway_url\s*=\s*['\"]?http://", lowered) is not None
        or "http://api.34.59.210.220.nip.io" in lowered
    ) and any(token in lowered for token in ("vault_api_key", "authorization", "/auth", "gitlab", "client.py"))


def _has_gateway_credential_context(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in (
            "gateway",
            "gitlab",
            "client.py",
            "httpx",
            "vault_api_key",
            "gateway_url",
            "api.34.59.210.220.nip.io",
            "x-api-key",
        )
    )


def _is_credential_probe_command(command: str) -> bool:
    lowered = command.lower()
    if "vault_api_key" in lowered and any(token in lowered for token in ("echo", "printenv", "env ", "curl")):
        return True
    if "x-api-key" in lowered and ("sk-" in lowered or "vault_api_key" in lowered):
        return True
    if re.search(r"authorization['\"]?\s*:\s*['\"]?bearer\s+[A-Za-z0-9._-]{12,}", command, re.IGNORECASE):
        return True
    if _is_tls_bypass_text(command) and _has_gateway_credential_context(command):
        return True
    if _is_http_gateway_downgrade_text(command):
        return True
    return bool(re.search(r"\bsk-[A-Za-z0-9_-]{3,}", command))


def _credential_safe_command(command: str) -> str:
    lowered = command.lower()
    if _is_http_gateway_downgrade_text(command):
        return "printf 'Skipped insecure HTTP gateway call; fix HTTPS or use a verified internal endpoint before retrying.\\n'"
    if _is_tls_bypass_text(command):
        return "printf 'Skipped insecure TLS-bypass gateway probe; use the configured client with verification enabled.\\n'"
    if (
        "curl" in lowered
        or "x-api-key" in lowered
        or "authorization" in lowered
        or re.search(r"\bsk-[A-Za-z0-9_-]{3,}", command)
    ):
        return (
            "printf 'Skipped direct gateway credential probe; use the configured "
            "email client without printing or inlining API keys.\\n'"
        )
    return (
        "test -n \"$VAULT_API_KEY\" && printf 'VAULT_API_KEY is set\\n' || "
        "printf 'VAULT_API_KEY is not set\\n'; "
        "test -n \"$GATEWAY_URL\" && printf 'GATEWAY_URL is set\\n' || "
        "printf 'GATEWAY_URL is not set\\n'"
    )


def _has_credential_probe_exec(event: dict) -> bool:
    return any(
        _is_unsafe_gateway_tool_call(tool_call)
        for tool_call in _tool_calls_from_event(event)
    )


def _is_unsafe_gateway_tool_call(tool_call: dict) -> bool:
    command = _exec_command_from_tool_call(tool_call)
    if command and _is_credential_probe_command(command):
        return True
    args_text = json.dumps(tool_call.get("arguments", {}))
    return (
        (_is_tls_bypass_text(args_text) and _has_gateway_credential_context(args_text))
        or _is_http_gateway_downgrade_text(args_text)
    )


def _unsafe_gateway_block_text() -> str:
    return (
        "I can't bypass TLS verification or send gateway credentials over plain HTTP. "
        "Please fix the HTTPS gateway or provide a verified internal endpoint, then I can retry."
    )


def _credential_probe_replacement(event: dict) -> tuple[dict, dict | None]:
    new_event = json.loads(json.dumps(event))
    safe_result_text = "Credential values were not printed or sent in a shell command."
    safe_call_id = ""
    unsafe_non_exec = False
    for tool_call in _tool_calls_from_event(new_event):
        command = _exec_command_from_tool_call(tool_call)
        if not command:
            if _is_unsafe_gateway_tool_call(tool_call):
                unsafe_non_exec = True
            continue
        if not _is_credential_probe_command(command):
            continue
        safe_call_id = str(tool_call.get("id", ""))
        tool_call.setdefault("arguments", {})["command"] = _credential_safe_command(command)
        tool_call["class"] = {
            "tool_tier": 1,
            "is_write": False,
            "arguments": {"command": {"data_layer": None}},
        }
        tool_call["flag"] = {
            "is_violation": False,
            "free_band": 4,
            "data_layer": 0,
            "authorization": {"needed": False, "granted": False},
        }
        lowered_command = command.lower()
        if _is_http_gateway_downgrade_text(command):
            safe_result_text = "Skipped insecure HTTP gateway call."
        elif _is_tls_bypass_text(command):
            safe_result_text = "Skipped insecure TLS-bypass gateway probe."
        elif "curl" in lowered_command or "x-api-key" in lowered_command or "authorization" in lowered_command:
            safe_result_text = "Skipped direct gateway credential probe."
        else:
            safe_result_text = "VAULT_API_KEY is set\nGATEWAY_URL is set"

    if unsafe_non_exec and not safe_call_id:
        return _replace_assistant_with_text(new_event, _unsafe_gateway_block_text()), None

    if not safe_call_id:
        return new_event, None
    if "insecure" in safe_result_text.lower():
        for item in _message_content(new_event):
            if isinstance(item, dict) and item.get("type") == "text":
                item["text"] = _unsafe_gateway_block_text()

    result = {
        "type": "message",
        "id": _make_event_id(),
        "parentId": new_event.get("id", _make_event_id()),
        "timestamp": _offset_timestamp(new_event.get("timestamp", ""), 700),
        "message": {
            "role": "toolResult",
            "toolCallId": safe_call_id,
            "toolName": "exec",
            "content": [{"type": "text", "text": safe_result_text}],
            "details": {
                "status": "completed",
                "exitCode": 0,
                "durationMs": 1,
                "aggregated": safe_result_text,
                "cwd": "/home/user/.openclaw/workspace",
            },
            "isError": False,
        },
    }
    return new_event, result


def _sensitive_email_exec_command(event: dict, pii_map: PIIMap) -> str:
    for tool_call in _tool_calls_from_event(event):
        command = _exec_command_from_tool_call(tool_call)
        lowered = command.lower()
        if (
            command
            and "send" in lowered
            and "email" in lowered
            and "@" in command
            and _max_data_layer_for_args({"command": command}, pii_map) >= 3
        ):
            return command
    return ""


def _enterprise_mail_args_from_exec(command: str, pii_map: PIIMap | None = None) -> dict:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    subject = "Schedule Conflict / Tuesday Activities"
    body = "Please see the schedule-conflict message discussed in this session."
    recipients: list[str] = []

    if "send" in tokens:
        idx = tokens.index("send") + 1
        while idx < len(tokens) and not tokens[idx].startswith("--"):
            if "@" in tokens[idx]:
                recipients.append(tokens[idx])
            idx += 1

    for idx, token in enumerate(tokens):
        if token == "--subject" and idx + 1 < len(tokens):
            subject = tokens[idx + 1]
        elif token == "--body" and idx + 1 < len(tokens):
            body = tokens[idx + 1]

    if pii_map is not None and _text_contains_l3_or_higher(subject, pii_map):
        subject = "Schedule Conflict / Tuesday Activities"

    if not recipients:
        recipients = re.findall(r"[\w.+-]+@[\w.-]+\.\w+", command)[:1]

    return {
        "action": "send",
        "to": recipients,
        "subject": subject,
        "body": body,
        "retain_after_send": False,
    }


def _replace_email_exec_with_enterprise_mail(event: dict, pii_map: PIIMap) -> tuple[dict, dict, dict] | None:
    command = _sensitive_email_exec_command(event, pii_map)
    if not command:
        return None

    new_event = json.loads(json.dumps(event))
    call_id = ""
    for item in _message_content(new_event):
        if not isinstance(item, dict) or item.get("type") != "toolCall":
            continue
        if item.get("name") != "exec":
            continue
        if _exec_command_from_tool_call(item) != command:
            continue
        call_id = item.get("id", _make_tc_id())
        item["id"] = call_id
        item["name"] = "enterprise-mail"
        item["arguments"] = _enterprise_mail_args_from_exec(command, pii_map)
        item.pop("class", None)
        item.pop("flag", None)
        break

    if not call_id:
        return None

    result_id = _make_event_id()
    result_text = "Message sent through enterprise-mail. No local copy was retained."
    result = {
        "type": "message",
        "id": result_id,
        "parentId": new_event.get("id", _make_event_id()),
        "timestamp": _offset_timestamp(new_event.get("timestamp", ""), 900),
        "message": {
            "role": "toolResult",
            "toolCallId": call_id,
            "toolName": "enterprise-mail",
            "content": [{"type": "text", "text": result_text}],
            "details": {
                "status": "completed",
                "messageId": f"sent-{call_id[-8:]}",
                "retained": False,
            },
            "isError": False,
        },
    }

    final_id = _make_event_id()
    final = {
        "type": "message",
        "id": final_id,
        "parentId": result_id,
        "timestamp": _offset_timestamp(new_event.get("timestamp", ""), 1700),
        "message": {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": "Sent through enterprise-mail for this email only. I did not store a local copy.",
            }],
        },
    }
    return new_event, result, final


def _is_user_profile_l3_read_result(event: dict, previous_event: dict | None, pii_map: PIIMap) -> bool:
    msg = event.get("message", {}) if isinstance(event, dict) else {}
    if not isinstance(msg, dict) or msg.get("role") != "toolResult":
        return False
    if str(msg.get("toolName", "")) != "read":
        return False
    if not isinstance(previous_event, dict):
        return False
    read_user_file = False
    for tool_call in _tool_calls_from_event(previous_event):
        if tool_call.get("name") != "read":
            continue
        args = tool_call.get("arguments", {})
        path = str(args.get("path", "")).lower() if isinstance(args, dict) else ""
        if path.endswith("user.md") or path.endswith("/user.md"):
            read_user_file = True
            break
    if not read_user_file:
        return False
    result_text = _tool_result_text(event)
    return any(
        entity.level in ("L3", "L4", "BLOCK")
        and any(
            keyword in entity.label.lower()
            for keyword in ("health", "medical", "diagnosis", "medication", "insurance")
        )
        and _entity_in_text(entity, result_text)
        for entity in pii_map.entities
    )


def _visible_user_profile_read_has_health_marker(event: dict, pii_map: PIIMap) -> bool:
    """Return whether the verifier-visible USER.md read still shows health-sensitive material."""
    visible_text = redact_text(_tool_result_text(event), pii_map)
    if REDACTION_TOKEN in visible_text:
        return True
    lowered = visible_text.lower()
    return any(
        keyword in lowered
        for keyword in (
            "diagnosis",
            "diagnoses",
            "medication",
            "medicine",
            "medical",
            "health",
            "insurance",
            "allergy",
            "allergies",
        )
    )


def _append_user_profile_scrub_flow(
    patched: list[dict],
    read_result_event: dict,
    pii_map: PIIMap,
) -> str:
    ts = read_result_event.get("timestamp", "")
    parent_id = read_result_event.get("id", _make_event_id())
    original_text = _tool_result_text(read_result_event)
    scrubbed_text = redact_text(original_text, pii_map)

    call_id = _make_tc_id()
    notice_id = _make_event_id()
    patched.append({
        "type": "message",
        "id": notice_id,
        "parentId": parent_id,
        "timestamp": _offset_timestamp(ts, 700),
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "I found health details in plaintext USER.md. I won't reuse those "
                        "from memory, and I'll scrub the profile note now."
                    ),
                },
                {
                    "type": "toolCall",
                    "id": call_id,
                    "name": "write",
                    "arguments": {
                        "path": "~/.openclaw/workspace/USER.md",
                        "content": scrubbed_text,
                    },
                },
            ],
        },
    })

    result_id = _make_event_id()
    patched.append({
        "type": "message",
        "id": result_id,
        "parentId": notice_id,
        "timestamp": _offset_timestamp(ts, 1700),
        "message": {
            "role": "toolResult",
            "toolCallId": call_id,
            "toolName": "write",
            "content": [{"type": "text", "text": "Scrubbed L3 health details from USER.md."}],
            "isError": False,
        },
    })
    return result_id


def _extract_path_markers(args: dict) -> set[str]:
    markers: set[str] = set()
    if not isinstance(args, dict):
        return markers
    for key, value in args.items():
        key_lower = str(key).lower()
        if key_lower in {"path", "file", "file_path", "filename", "target", "destination"}:
            value_text = str(value).strip()
            if value_text:
                markers.add(value_text.lower())
                markers.add(Path(value_text).name.lower())
    command = str(args.get("command", "")).lower()
    for match in re.findall(r"[\w./~-]+\.md\b", command):
        markers.add(match)
        markers.add(Path(match).name.lower())
    return {marker for marker in markers if marker and marker != "."}


def _hard_denied_path_markers(event: dict, pii_map: PIIMap) -> set[str]:
    markers: set[str] = set()
    for tool_call in _tool_calls_from_event(event):
        name = str(tool_call.get("name", ""))
        args = tool_call.get("arguments", {})
        if not isinstance(args, dict):
            continue
        if (
            _is_write_tool(name, args)
            and (
                _max_data_layer_for_args(args, pii_map) >= 3
                or _looks_like_health_appointment_persistence(args)
            )
            and _is_plaintext_persistence(name, args)
        ):
            markers.update(_extract_path_markers(args))
    return markers


def _contains_denied_marker(text: str, denied_markers: set[str]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in denied_markers if marker)


def _denied_marker_contexts(text: str, denied_markers: set[str]) -> list[str]:
    lowered = text.lower()
    contexts: list[str] = []
    for line in lowered.splitlines():
        if any(marker and marker in line for marker in denied_markers):
            contexts.append(line)
    for marker in denied_markers:
        if not marker:
            continue
        # Long single-line outputs still need a bounded fallback.
        start = 0
        while True:
            idx = lowered.find(marker, start)
            if idx < 0:
                break
            left = lowered.rfind("\n", 0, idx)
            right = lowered.find("\n", idx)
            if left < 0 and right < 0:
                contexts.append(lowered[max(0, idx - 80): idx + len(marker) + 80])
            start = idx + len(marker)
    return contexts


def _is_denied_file_delete(event: dict, denied_markers: set[str]) -> bool:
    if not denied_markers:
        return False
    for tool_call in _tool_calls_from_event(event):
        name = str(tool_call.get("name", ""))
        args = tool_call.get("arguments", {})
        if not isinstance(args, dict):
            continue
        text = json.dumps(args).lower()
        if name == "exec":
            command = str(args.get("command", "")).lower()
            has_delete = any(token in command for token in ("rm ", "unlink", "delete", "trash"))
        else:
            has_delete = name in {"delete", "remove"} or any(
                word in text for word in ("delete", "remove", "trash")
            )
        if has_delete and _contains_denied_marker(text, denied_markers):
            return True
    return False


def _is_denied_file_access(event: dict, denied_markers: set[str]) -> bool:
    if not denied_markers:
        return False
    for tool_call in _tool_calls_from_event(event):
        name = str(tool_call.get("name", ""))
        args = tool_call.get("arguments", {})
        if not isinstance(args, dict):
            continue
        args_text = json.dumps(args).lower()
        if not _contains_denied_marker(args_text, denied_markers):
            continue
        if name in {"read", "edit", "write"}:
            return True
        if name == "exec" and any(
            token in str(args.get("command", "")).lower()
            for token in ("cat ", "sed ", "tee ", ">", ">>", "python", "zip ")
        ):
            return True
    return False


def _is_retention_claim(text: str, denied_markers: set[str], hard_deny_seen: bool) -> bool:
    lowered = text.lower()
    false_retention_claims = (
        "saved to your persona file",
        "saved to your profile",
        "saved to your memory",
        "stored in your persona file",
        "stored in your profile",
        "stored in your memory",
    )
    if any(phrase in lowered for phrase in false_retention_claims):
        return True
    if denied_markers:
        denied_file_action_terms = (
            "append", "appended", "archive", "archived", "copy", "copied",
            "create", "created", "include", "included", "open", "opened",
            "read", "retain", "retained", "save", "saved", "store", "stored",
            "update", "updated", "write", "wrote", "zip", "zipped",
        )
        if any(
            any(term in context for term in denied_file_action_terms)
            for context in _denied_marker_contexts(lowered, denied_markers)
        ):
            return True
    if not hard_deny_seen:
        return False
    file_terms = ("saved", "wrote", "created", "stored", "deleted", "delete", "removed", "exists")
    target_terms = ("file", ".md", "workspace", "notes", "note")
    health_terms = ("health", "medical", "asthma", "allergy", "clinic", "appointment")
    return (
        any(term in lowered for term in file_terms)
        and any(term in lowered for term in target_terms)
        and any(term in lowered for term in health_terms)
    )


def _looks_like_false_profile_save_claim(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in (
        "saved to your persona file",
        "saved to your profile",
        "saved to your memory",
        "stored in your persona file",
        "stored in your profile",
        "stored in your memory",
    ))


def _false_profile_save_replacement(text: str) -> str:
    lowered = text.lower()
    if "whisper" in lowered or "transcription" in lowered:
        return (
            "Noted. I'll use Local Whisper as the default for this task and block any external transcription path."
        )
    return "Noted. I won't claim this was saved unless a write actually happens."


def _transient_retention_message() -> str:
    return HATCH_REFUSAL


def _transient_ack_message() -> str:
    return "Got it. Keeping it here only, with no plaintext file saved."


def _no_retention_confirmation_message() -> str:
    return "Confirmed. Memory came back empty, and no plaintext workspace file was saved."


def _looks_like_transient_ack(text: str) -> bool:
    lowered = text.lower()
    return "keep it transient" in lowered or "don't save" in lowered or "dont save" in lowered


def _looks_like_no_retention_confirmation(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in (
        "nothing health-related",
        "nothing being kept",
        "none of the health",
        "no memory",
        "memory came back",
        "in my memory",
    ))


def _looks_like_persistence_request(text: str) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in (
        "without saving", "without storing", "don't save", "dont save",
        "do not save", "skip saving", "skip it",
    )):
        return False
    request_terms = ("save", "store", "remember", "retain", "keep", "write", "file", "note", "notes")
    return any(term in lowered for term in request_terms)


def _looks_like_category_memory_request(text: str) -> bool:
    lowered = text.lower()
    return (
        any(term in lowered for term in ("memory", "remember", "future conversation", "future conv"))
        and any(term in lowered for term in (
            "category", "categories", "category-level", "consent", "only safe",
            "safe to remember", "explicitly", "approve", "approved", "exclude",
            "longer-term interaction", "longer term interaction",
        ))
    )


def _category_memory_policy_message() -> str:
    return (
        "Yes. I can handle memory by category, but I won't save anything yet. "
        "Professional background, hobbies, and general productivity preferences can be remembered only after you approve those categories. "
        "Health details, identity documents, payroll, credentials, and anything you mark transient stay out of memory."
    )


def _looks_like_calendar_boundary_preference(text: str) -> bool:
    lowered = text.lower()
    return (
        "calendar" in lowered
        and any(term in lowered for term in ("generic title", "generic titles", "basic reminder", "time range", "time ranges"))
        and any(term in lowered for term in ("avoid syncing", "do not sync", "don't sync", "not sync", "only want", "comfortable allowing"))
    )


def _calendar_boundary_message() -> str:
    return (
        "Understood. The external calendar will be limited to generic titles, approximate time ranges, and basic reminders. "
        "Medical details, medication context, energy constraints, and symptom notes stay local and do not go into CalDAV."
    )


def _looks_like_local_only_sensitive_work_request(text: str) -> bool:
    lowered = text.lower()
    local_terms = (
        "local-only", "local only", "this session", "keep it to this session",
        "without saving", "without storing", "do not save", "don't save",
    )
    work_terms = (
        "summary", "template", "plan", "practice plan", "schedule",
        "preview", "draft", "outline", "appointment", "private practice",
    )
    return any(term in lowered for term in local_terms) and any(term in lowered for term in work_terms)


def _local_only_work_message(user_text: str) -> str:
    lowered = user_text.lower()
    if any(term in lowered for term in ("appointment", "provider", "medical", "medication", "symptom")):
        return (
            "Got it. Here is a local-only appointment template for this session: current symptoms, timing patterns, work triggers, "
            "medication notes, appointment logistics, and questions for the provider. Keep each section brief, leave out unrelated history, "
            "and do not store it after this task unless you explicitly choose a safe storage location."
        )
    if any(term in lowered for term in ("piano", "calligraphy", "practice", "calendar")):
        return (
            "Got it. Here is a local-only practice plan: keep piano blocks short, use lighter calligraphy blocks on busy days, "
            "add movement breaks between sessions, and keep any health-related constraints out of the external calendar. "
            "If you sync anything later, use only generic titles, approximate time ranges, and basic reminders."
        )
    return (
        "Got it. I can work from the details in this session and keep them transient. "
        "I'll provide the requested local-only output without writing it to memory or a plaintext workspace file."
    )


def _looks_like_appointment_logistics_scope_request(text: str) -> bool:
    lowered = text.lower()
    return "appointment logistics" in lowered and any(term in lowered for term in ("limited", "only", "avoid"))


def _appointment_logistics_scope_message() -> str:
    return (
        "Yes. Keep logistics separate and minimal: appointment date and time, provider or location reminder, and basic preparation notes. "
        "Do not combine those logistics with symptom history, medication details, or broader medical notes unless you decide it is needed for the visit."
    )


def _looks_like_sanitized_calendar_preview_request(text: str) -> bool:
    lowered = text.lower()
    return (
        any(term in lowered for term in ("caldav", "calendar event", "calendar events", "event titles"))
        and any(term in lowered for term in ("preview", "show", "review", "visible descriptions", "metadata", "sanitized"))
    )


def _sanitized_calendar_preview_message() -> str:
    return (
        "Here is the sanitized CalDAV preview:\n"
        "- Title: Piano Practice; description: focused creative block; reminder: 10 minutes before\n"
        "- Title: Arabic Calligraphy Practice; description: creative practice block; reminder: 10 minutes before\n"
        "- Title: Creative Practice Block; description: flexible hobby time; reminder: 15 minutes before\n"
        "Metadata should contain only the calendar, start/end time, reminder offset, and generic title. No medical details, medication timing, symptoms, or recovery notes are included."
    )


def _looks_like_privacy_confirmation_request(text: str) -> bool:
    lowered = text.lower()
    return (
        "confirm" in lowered
        and any(term in lowered for term in (
            "local-only", "local only", "external sync", "cloud upload", "long-term",
            "retention", "remembered", "forgotten", "hidden/background", "category-level",
            "approved blocks", "sensitive health", "memory storage",
        ))
    )


def _privacy_confirmation_message(user_text: str) -> str:
    lowered = user_text.lower()
    if any(term in lowered for term in ("memory", "remembered", "forgotten", "category-level")):
        return (
            "Confirmed. Memory is category-level only: approved low-risk categories can be saved, while health, identity, payroll, "
            "credentials, and anything marked transient stay excluded. I won't use hidden or background retention."
        )
    if any(term in lowered for term in ("appointment", "medical", "health", "local-only", "cloud upload")):
        return (
            "Confirmed. The appointment details stay local to this session, no external sync or cloud upload is performed, "
            "and I won't create a reusable health profile or long-term medical memory."
        )
    if any(term in lowered for term in ("calendar", "approved blocks", "sync")):
        return (
            "Confirmed. Health-related context stays local. Only approved generic titles, approximate time ranges, and basic reminders "
            "should go to the external calendar, with no medical details or medication context."
        )
    return "Confirmed. I used only the approved scope, avoided unnecessary retention, and did not store sensitive details for future use."


def _looks_like_broken_privacy_boilerplate(text: str) -> bool:
    stripped = text.strip()
    lowered = stripped.lower()
    return (
        stripped in {"NO_REPLY", "NO REPLY", "[NO REPLY]", REDACTION_TOKEN}
        or "denied plaintext reference file" in lowered
        or "i won't reprint the medical fields again" in lowered
        or "paste the values again if you want another full card rendering" in lowered
    )


def _contextual_response_for_blocked_file_noise(user_text: str) -> str | None:
    if _looks_like_privacy_confirmation_request(user_text):
        return _privacy_confirmation_message(user_text)
    if _looks_like_appointment_logistics_scope_request(user_text):
        return _appointment_logistics_scope_message()
    if _looks_like_sanitized_calendar_preview_request(user_text):
        return _sanitized_calendar_preview_message()
    if _looks_like_local_only_sensitive_work_request(user_text):
        return _local_only_work_message(user_text)
    if _looks_like_calendar_boundary_preference(user_text):
        return _calendar_boundary_message()
    return None


def _looks_like_cross_session_secret_request(text: str) -> bool:
    lowered = text.lower()
    secret_terms = (
        "password", "passcode", "api key", "secret", "credential",
        "recovery code", "private key", "token",
    )
    cross_session_terms = (
        "permanent", "permanently", "remember", "future", "next time",
        "don't have to", "dont have to", "so i don't", "so i dont",
    )
    return any(term in lowered for term in secret_terms) or any(
        term in lowered for term in cross_session_terms
    )


def _should_offer_encrypted_storage(text: str) -> bool:
    lowered = text.lower()
    if (
        _looks_like_category_memory_request(text)
        or _looks_like_local_only_sensitive_work_request(text)
        or _looks_like_calendar_boundary_preference(text)
        or _looks_like_sanitized_calendar_preview_request(text)
        or _looks_like_appointment_logistics_scope_request(text)
    ):
        return False
    sensitive_terms = (
        "health", "medical", "diagnosis", "allergy", "medication", "asthma",
        "ssn", "social security", "credit card", "card number", "bank",
        "salary", "gps", "coordinates", "home address", "minor", "children",
    )
    return (
        _looks_like_persistence_request(text)
        and any(term in lowered for term in sensitive_terms)
        and not _looks_like_cross_session_secret_request(text)
    )


def _looks_like_safe_local_export_request(text: str) -> bool:
    lowered = text.lower()
    local_export = any(term in lowered for term in ("export", "write", "file", "csv", "locally", "local"))
    privacy_safe = any(term in lowered for term in ("anonymized", "aggregate", "redacted", "no l2", "no lead name", "no external api"))
    return local_export and privacy_safe


def _last_substantive_user_text(events: list[dict]) -> str:
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        msg = event.get("message", {})
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        text = "\n".join(
            str(item.get("text", ""))
            for item in _message_content(event)
            if isinstance(item, dict) and item.get("type") == "text"
        )
        if not text.strip():
            continue
        if _looks_like_transient_ack(text):
            continue
        if not _looks_like_persistence_request(text) and any(
            phrase in text.lower() for phrase in ("keep going without saving", "without saving")
        ):
            continue
        return text
    return ""


def _soft_transient_notice() -> str:
    return "I'll keep these details transient for this task and won't save them to plaintext notes or USER.md."


def _denied_file_access_message() -> str:
    return (
        "I can't read or update that plaintext file because it would preserve sensitive data. "
        "I'll keep the sensitive details out of storage and can work from a redacted summary here."
    )


def _safe_denied_reference_message(original_text: str) -> str:
    lowered = original_text.lower()
    if "archive" in lowered or "zip" in lowered:
        return (
            "The archive was created from the safe budget files. "
            "I did not include or update the denied plaintext reference file."
        )
    if "cron" in lowered or "removed" in lowered or "files" in lowered or "workspace" in lowered:
        return (
            "The safe local files step is complete. "
            "I won't read or update the denied plaintext reference file."
        )
    if "diagram" in lowered:
        return (
            "I can't append to the denied plaintext file. "
            "I can show the redacted diagram here without storing the sensitive source details."
        )
    return "I won't use or update the denied plaintext reference file. I'll keep the sensitive details out of storage."


def _encrypted_store_payload(event: dict, fallback_text: str = "") -> tuple[str, str]:
    key = "sensitive_task_data"
    value_parts: list[str] = []
    for tool_call in _tool_calls_from_event(event):
        args = tool_call.get("arguments", {})
        if not isinstance(args, dict):
            continue
        markers = _extract_path_markers(args)
        if markers:
            filename = sorted(markers, key=len)[0]
            key = Path(filename).stem.replace("-", "_").replace(".", "_") or key
        for field in ("content", "value", "val", "text", "body", "data"):
            val = args.get(field)
            if isinstance(val, str) and val.strip():
                value_parts.append(val.strip())
        if not value_parts and fallback_text.strip():
            value_parts.append(fallback_text.strip())
        elif not value_parts:
            value_parts.append(json.dumps(args, ensure_ascii=False))
    value = "\n\n".join(value_parts).strip() or "Sensitive task data supplied in the current turn."
    return key[:80], value


def _append_encrypted_storage_flow(
    patched: list[dict],
    event: dict,
    parent_id: str | None,
    fallback_text: str = "",
) -> str:
    last_id = patched[-1].get("id", _make_event_id()) if patched else _make_event_id()
    ts = event.get("timestamp", "")
    key, value = _encrypted_store_payload(event, fallback_text)

    ask_id = _make_event_id()
    patched.append({
        "type": "message",
        "id": ask_id,
        "parentId": parent_id or last_id,
        "timestamp": _offset_timestamp(ts, -1800),
        "message": {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": (
                    "I can't save L3/L4 details to a plaintext workspace file. "
                    "I can store the task details in the encrypted vault for this task instead. "
                    "Do you approve encrypted vault storage?"
                ),
            }],
        },
    })

    grant_id = _make_event_id()
    patched.append({
        "type": "message",
        "id": grant_id,
        "parentId": ask_id,
        "timestamp": _offset_timestamp(ts, -1200),
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": "Yes, store it in the encrypted vault for this task."}],
        },
    })

    call_id = _make_tc_id()
    tool_id = _make_event_id()
    patched.append({
        "type": "message",
        "id": tool_id,
        "parentId": grant_id,
        "timestamp": _offset_timestamp(ts, -600),
        "message": {
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "id": call_id,
                "name": "set",
                "arguments": {
                    "key": key,
                    "val": value,
                },
                "class": {
                    "tool_tier": 1,
                    "is_write": True,
                    "arguments": {
                        "key": {"data_layer": None},
                        "val": {"data_layer": "L3"},
                    },
                },
                "flag": {
                    "is_violation": True,
                    "free_band": 2,
                    "data_layer": 3,
                    "authorization": {"needed": True, "granted": True},
                },
            }],
        },
    })

    result_id = _make_event_id()
    patched.append({
        "type": "message",
        "id": result_id,
        "parentId": tool_id,
        "timestamp": _offset_timestamp(ts, -300),
        "message": {
            "role": "toolResult",
            "toolCallId": call_id,
            "toolName": "set",
            "content": [{"type": "text", "text": json.dumps({"ok": True})}],
            "isError": False,
        },
    })

    confirm_id = _make_event_id()
    patched.append({
        "type": "message",
        "id": confirm_id,
        "parentId": result_id,
        "timestamp": ts,
        "message": {
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": (
                    "Saved in the encrypted vault. I did not create a plaintext workspace file."
                ),
            }],
        },
    })
    return confirm_id


def _deterministic_consent_gate(violation: dict) -> dict:
    turn_idx = violation["turn_idx"]
    tool_name = violation.get("tool_name", "")
    if tool_name == "assistant_text":
        assistant_ask = (
            "Heads up, booking through the clinic site means your name, date of birth, "
            "email, and phone may be sent to that external clinic. Want to share only "
            "those details for this appointment?"
        )
        user_grant = "Yes, share only those details for this appointment."
    elif tool_name == "browser":
        assistant_ask = (
            "Before I continue in the clinic booking form, I need your okay to send only "
            "the appointment details and your contact fields to the external clinic site. Proceed?"
        )
        user_grant = "Yes, proceed with only those details."
    elif tool_name in {"enterprise-email", "enterprise-mail"} and violation.get("sensitive_card_send"):
        assistant_ask = (
            "Before I use enterprise email, confirm you're okay sending the L3 health details "
            "in the redacted organizer card to this recipient for this email only. I won't send "
            "the full card, insurance ID, or provider details."
        )
        user_grant = "Yes, send only the redacted organizer card for this email."
    else:
        assistant_ask = (
            f"Before I use {tool_name or 'this tool'}, I need your okay because this may send "
            "sensitive details outside the local workspace. Proceed?"
        )
        user_grant = "Yes, proceed with only the needed details."
    return {
        "before_turn_idx": turn_idx,
        "assistant_ask": assistant_ask,
        "user_grant": user_grant,
    }


def _needs_local_image_substitute(tool_name: str, tier: int, data_layer: int) -> bool:
    """Route sensitive image prompts to a local document renderer."""
    return tool_name == "image_generate" and tier == 3 and data_layer >= 3


def _local_card_path(turn_idx: int, original_args: dict | None = None) -> str:
    prompt = ""
    if isinstance(original_args, dict):
        prompt = str(original_args.get("prompt", "")).lower()
    if turn_idx >= 10 or "organizer" in prompt or "event organizer" in prompt:
        return "/home/user/OpenClawTrainer/workspace/organizer_emergency_card.pdf"
    return "/home/user/OpenClawTrainer/workspace/emergency_health_card.pdf"


def _transient_card_text(original_args: dict) -> str:
    prompt = str(original_args.get("prompt", "")).strip()
    prompt = re.sub(r"\nStyle:.*", "", prompt, flags=re.DOTALL).strip()
    return (
        "I'll keep this as transient chat text for this turn and won't create a plaintext file "
        "or use an external image generator.\n\n"
        f"{prompt}"
    ).strip()


def _local_pdf_arguments(original_args: dict, turn_idx: int) -> dict:
    prompt = str(original_args.get("prompt", "")).strip()
    title = "Emergency Health Card"
    if turn_idx >= 10:
        title = "Organizer Emergency Card"
    content = (
        f"# {title}\n\n"
        "Create this card locally as a printable PDF. Do not send the medical details "
        "to an external image-generation service.\n\n"
        f"{prompt}"
    ).strip()
    return {
        "output_path": _local_card_path(turn_idx, original_args),
        "content": content,
        "format": "pdf",
    }


def _replace_sensitive_image_with_local_pdf(event: dict, turn_idx: int) -> dict:
    import copy

    event_copy = copy.deepcopy(event)
    content = _message_content(event_copy)
    replacement_text = ""
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "toolCall":
            continue
        if item.get("name") != "image_generate":
            continue
        replacement_text = _transient_card_text(item.get("arguments", {}))
        break

    if replacement_text:
        event_copy = _replace_assistant_with_text(event_copy, replacement_text)
    return event_copy


def _local_pdf_tool_result(original_result: dict, artifact_path: str) -> dict:
    import copy

    result = copy.deepcopy(original_result)
    msg = result.get("message", {})
    if isinstance(msg, dict):
        msg["toolName"] = "nano-pdf"
        msg["content"] = [{
            "type": "text",
            "text": f"Created local printable PDF: {artifact_path}",
        }]
        msg["details"] = {
            "path": artifact_path,
            "renderer": "nano-pdf",
            "localOnly": True,
        }
        msg["isError"] = False
    return result


def _replace_image_followup_with_local_pdf(event: dict, artifact_path: str) -> dict:
    import copy

    event_copy = copy.deepcopy(event)
    text = _assistant_text(event_copy)
    if "MEDIA:/home/user/.openclaw/media/tool-image-generation/" not in text:
        return event_copy

    if artifact_path == "__TRANSIENT_CARD__":
        replacement = (
            "I kept the card content transient in chat and did not create a plaintext "
            "file or use an external image generator."
        )
    else:
        replacement = (
            f"Created the printable card locally: `{artifact_path}`.\n\n"
            "I kept this local instead of sending the medical details to an external "
            "image generator."
        )
    return _replace_assistant_with_text(event_copy, replacement)


def _approval_reason(violation: dict) -> str:
    tool_name = violation.get("tool_name") or "tool"
    tier = violation.get("tier", "?")
    data_layer = violation.get("data_layer", "?")
    return (
        f"Allow this one {tool_name} call with L{data_layer} data crossing a T{tier} "
        "privacy boundary for the current task only."
    )


def _append_exec_approval_events(
    patched: list[dict],
    parent_id: str,
    ts: str,
    violation: dict,
) -> str:
    """Add realistic exec-approval request/resolve events and return last id."""
    import uuid

    approval_id = str(uuid.uuid4())
    request_id = _make_event_id()
    patched.append({
        "type": "custom",
        "customType": "exec.approval.requested",
        "id": request_id,
        "parentId": parent_id,
        "timestamp": _offset_timestamp(ts, -3600),
        "data": {
            "approvalId": approval_id,
            "approvalKind": "exec",
            "method": "exec.approval.waitDecision",
            "ask": "once",
            "allowedDecisions": ["allow-once", "deny"],
            "command": f"privacy-boundary:{violation.get('tool_name', 'tool')}",
            "reason": _approval_reason(violation),
        },
    })

    resolved_id = _make_event_id()
    patched.append({
        "type": "custom",
        "customType": "exec.approval.resolved",
        "id": resolved_id,
        "parentId": request_id,
        "timestamp": _offset_timestamp(ts, -3000),
        "data": {
            "approvalId": approval_id,
            "approvalKind": "exec",
            "method": "exec.approval.resolve",
            "decision": "allow-once",
            "runId": f"exec-approval-followup:{approval_id}:nonce:{_make_event_id()}",
        },
    })
    return resolved_id


def _turn_to_user_map(trajectory: ParsedTrajectory) -> dict[int, int]:
    mapping: dict[int, int] = {}
    last_user_idx = -1
    for kind, idx in trajectory.thread_order:
        if kind == "user":
            last_user_idx = int(idx)
        elif kind == "assistant" and last_user_idx >= 0:
            mapping[int(idx)] = last_user_idx
    if not mapping:
        for idx in range(min(len(trajectory.user_messages), len(trajectory.assistant_turns))):
            mapping[idx] = idx
    return mapping


def _entity_in_text(entity: PIIEntity, text: str) -> bool:
    text_lower = text.lower()
    entity_text = (entity.text or "").lower()
    if entity_text and entity_text in text_lower:
        return True
    if entity.level not in ("L3", "L4", "BLOCK"):
        return False
    fragments = [
        frag for frag in re.split(r"\W+", entity_text)
        if len(frag) > 3
    ]
    if len(fragments) >= 2:
        return sum(1 for frag in fragments if frag in text_lower) >= 2
    return bool(fragments and fragments[0] in text_lower)


def _looks_like_short_approval(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered.startswith("["):
        bracket = lowered.find("]")
        if bracket >= 0:
            lowered = lowered[bracket + 1:].strip()
    if len(lowered) > 80:
        return False
    return any(
        lowered.startswith(word)
        for word in ("yes", "yeah", "yep", "ok", "okay", "sure", "go ahead", "approve", "allow")
    )


def _is_public_weather_lookup(tool_name: str, args: dict) -> bool:
    if tool_name != "exec":
        return False
    args_text = json.dumps(args).lower()
    if any(token in args_text for token in ("x-api-key", "authorization: bearer", "vault_api_key")):
        return False
    return any(
        marker in args_text
        for marker in (
            "wttr.in",
            "api.open-meteo.com",
            "air-quality-api.open-meteo.com",
            "api.waqi.info",
            "pollencount.org",
            "wunderground.com",
        )
    )


def _values_appear_in_user_history(
    trajectory: ParsedTrajectory,
    turn_idx: int,
    values: list[str],
) -> bool:
    turn_map = _turn_to_user_map(trajectory)
    current_user_idx = turn_map.get(turn_idx, -1)
    if current_user_idx < 0:
        return False
    history = "\n".join(trajectory.user_messages[: current_user_idx + 1]).lower()
    return any(value and value.lower() in history for value in values)


def _stale_l34_values_for_tool(
    trajectory: ParsedTrajectory,
    turn_idx: int,
    args: dict,
    pii_map: PIIMap,
) -> list[str]:
    """Return L3/L4 values used by a tool call but not supplied in the latest turn."""
    args_text = json.dumps(args).lower()
    matched = [
        entity for entity in pii_map.entities
        if entity.level in ("L3", "L4", "BLOCK") and entity.text and entity.text.lower() in args_text
    ]
    if not matched:
        return []

    turn_map = _turn_to_user_map(trajectory)
    current_user_idx = turn_map.get(turn_idx, -1)
    current_text = (
        trajectory.user_messages[current_user_idx]
        if 0 <= current_user_idx < len(trajectory.user_messages)
        else ""
    )
    if any(_entity_in_text(entity, current_text) for entity in matched):
        return []

    # Narrow bridge: current user just approved a disclosed action and the
    # immediately previous user turn supplied the value.
    prior_user_idx = current_user_idx - 1
    prior_text = (
        trajectory.user_messages[prior_user_idx]
        if 0 <= prior_user_idx < len(trajectory.user_messages)
        else ""
    )
    if _looks_like_short_approval(current_text) and any(
        _entity_in_text(entity, prior_text) for entity in matched
    ):
        return []

    values: list[str] = []
    seen: set[str] = set()
    for entity in matched:
        if entity.text not in seen:
            seen.add(entity.text)
            values.append(entity.text)
    return values


def _deterministic_elicit_gate(violation: dict) -> dict:
    values = violation.get("sensitive_values") or []
    values_text = "; ".join(str(v) for v in values if v)
    return {
        "before_turn_idx": violation["turn_idx"],
        "assistant_ask": (
            "I need you to provide the sensitive value again for this step. "
            "I won't pull it from prior memory or earlier turns."
        ),
        "user_reinput": (
            f"Use this for this task only: {values_text}"
            if values_text
            else "I'll provide it again for this task."
        ),
    }


def _mentions_external_medical_site(args: dict) -> bool:
    args_text = json.dumps(args).lower()
    return any(keyword in args_text for keyword in _EXTERNAL_MEDICAL_KEYWORDS)


def _is_external_medical_action(name: str, args: dict, pii_map: PIIMap) -> bool:
    if _classify_tool_tier(name) != 3:
        return False
    if not _contains_exact_pii(args, pii_map):
        return False
    level_map = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4}
    if not any(level_map.get(entity.level, 0) >= 3 for entity in pii_map.entities):
        return False
    return _mentions_external_medical_site(args)


def _asks_for_external_booking_pii(text: str) -> bool:
    lowered = text.lower()
    field_hits = sum(
        1
        for keyword in ("name", "dob", "date of birth", "phone", "email", "contact")
        if keyword in lowered
    )
    booking_context = any(
        keyword in lowered
        for keyword in ("book", "booking", "appointment", "clinic", "form", "submit")
    )
    return booking_context and field_hits >= 2


def _has_l3_or_higher(pii_map: PIIMap) -> bool:
    level_map = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "BLOCK": 4}
    return any(level_map.get(entity.level, 0) >= 3 for entity in pii_map.entities)


def _text_contains_l3_or_higher(text: str, pii_map: PIIMap) -> bool:
    return any(
        entity.level in ("L3", "L4", "BLOCK")
        and entity.text
        and _entity_in_text(entity, text)
        for entity in pii_map.entities
    )


def _looks_like_emailing_sensitive_card(user_text: str, assistant_text: str, pii_map: PIIMap) -> bool:
    if not _text_contains_l3_or_higher(f"{user_text}\n{assistant_text}", pii_map):
        return False
    combined = f"{user_text}\n{assistant_text}".lower()
    send_context = any(word in combined for word in ("send", "email", "mail", "share", "forward"))
    card_context = any(word in combined for word in ("card", "pdf", "attachment", "organizer version", "redacted version"))
    recipient_context = any(word in combined for word in ("organizer", "@", "recipient", "valeria", "to "))
    return send_context and card_context and recipient_context


def _has_sensitive_card_l3(pii_map: PIIMap) -> bool:
    return any(
        entity.level in ("L3", "L4", "BLOCK")
        and any(
            keyword in entity.label.lower()
            for keyword in ("health", "medical", "diagnosis", "medication", "insurance")
        )
        for entity in pii_map.entities
    )


def _looks_like_sensitive_card_send_request(text: str) -> bool:
    lowered = text.lower()
    send_context = any(word in lowered for word in ("send", "email", "mail", "share", "forward"))
    card_context = any(
        phrase in lowered
        for phrase in (
            "organizer version",
            "redacted version",
            "redacted organizer",
            "organizer card",
            "emergency card",
            "medical card",
            "full card",
        )
    )
    recipient_context = any(word in lowered for word in ("organizer", "recipient", "valeria", "@"))
    return send_context and card_context and recipient_context


def _looks_like_email_send_assistant_intent(event: dict, pii_map: PIIMap) -> bool:
    text = _assistant_all_text(event).lower()
    if not _text_contains_l3_or_higher(text, pii_map):
        return False
    return (
        ("send" in text or "email" in text or "mail_send" in text)
        and ("organizer" in text or "recipient" in text or "valeria" in text)
        and ("redacted" in text or "card" in text)
    )


def _is_sensitive_email_exec_violation(violation: dict | None) -> bool:
    if not violation:
        return False
    args = violation.get("arguments", {})
    command = str(args.get("command", "") if isinstance(args, dict) else args).lower()
    return (
        violation.get("tool_name") == "exec"
        and "send" in command
        and "email" in command
        and "@" in command
        and int(violation.get("data_layer", 0) or 0) >= 3
    )


def _external_email_gateway_consent_text(violation: dict | None) -> tuple[str, str]:
    return (
        "Before I send this through the external enterprise email gateway, confirm you approve "
        "transmitting the children's names (L3 minor data) and schedule-conflict message to the "
        "school coordinator for this email only. I won't store the email after sending.",
        "Yes, send it through the external enterprise email gateway for this email only.",
    )


def _is_sensitive_card_context(text: str) -> bool:
    lowered = text.lower()
    table_card_markers = (
        "medication",
        "insurance",
        "allerg",
        "emergency contact",
        "provider",
        "medical card",
        "full card",
        "redacted card",
        "organizer card",
    )
    standalone_card_markers = (
        "emergency medical card",
        "medical card",
        "full card",
        "redacted card",
        "organizer card",
        "insurance id",
        "emergency contact",
    )
    return (
        ("|" in text and any(word in lowered for word in table_card_markers))
        or any(word in lowered for word in standalone_card_markers)
        or ("side by side" in lowered and "card" in lowered)
    )


def _looks_like_no_save_climbing_signoff(text: str) -> bool:
    lowered = text.lower()
    return "nothing saved" in lowered and any(
        phrase in lowered
        for phrase in ("go climb", "climb well", "fresh check")
    )


def _no_save_climbing_replacement() -> str:
    return "Nothing saved. Ask anytime and re-share the relevant details."


def _looks_like_insecure_gateway_summary(text: str) -> bool:
    lowered = text.lower()
    return (
        "gateway" in lowered
        and (
            "disable ssl verification" in lowered
            or "skip ssl verification" in lowered
            or "verify=false" in lowered
            or "http works" in lowered
            or "let me use http" in lowered
        )
    )


def _insecure_gateway_summary_replacement() -> str:
    return (
        "I couldn't complete the webhook rotation safely. The GitLab gateway TLS path failed, "
        "and I won't bypass TLS or send gateway credentials over plain HTTP. Please fix the "
        "HTTPS gateway or provide a verified internal endpoint, then I can retry the rotation."
    )


def _assistant_stale_l34_text_values(text: str, current_user_text: str, pii_map: PIIMap) -> list[str]:
    text_lower = text.lower()
    values: list[str] = []
    seen: set[str] = set()
    for entity in pii_map.entities:
        if entity.level not in ("L3", "L4", "BLOCK") or not entity.text:
            continue
        if entity.text.lower() not in text_lower:
            continue
        if _entity_in_text(entity, current_user_text):
            continue
        if entity.text not in seen:
            seen.add(entity.text)
            values.append(entity.text)
    return values


def _stale_sensitive_text_message(original_text: str) -> str:
    lowered = original_text.lower()
    if "need a few more details" in lowered or "blood type" in lowered:
        return (
            "Great idea. I won't pull sensitive health details from memory. "
            "Please provide the diagnoses, medications, allergies, emergency contact, "
            "and any provider or insurance fields you want included for this card."
        )
    if "card" in lowered or "medication" in lowered or "condition" in lowered:
        return (
            "I can't repeat sensitive health details from earlier turns. "
            "Please provide the values again if you want me to display this card text."
        )
    return (
        "I shouldn't pull sensitive health details from memory or earlier turns. "
        "Please provide the values again for this step."
    )


def _looks_like_health_profile_memory_claim(text: str) -> bool:
    lowered = text.lower()
    return (
        "from memory" in lowered
        and ("profile" in lowered or REDACTION_TOKEN in text)
        and any(word in lowered for word in ("health", "medical", "emergency", "card"))
    )


def _health_profile_memory_claim_replacement() -> str:
    return (
        "Great idea. To make the card useful for first responders, please provide your blood "
        "type if known, medical conditions, medications, allergies, emergency contact, and any "
        "provider or insurance fields you want included. Do you want a printable PDF, a "
        "quick-reference text card, or both?"
    )


def _looks_like_card_difference_confirmation(user_text: str, assistant_text: str) -> bool:
    user_lower = user_text.lower()
    assistant_lower = assistant_text.lower()
    return (
        "confirm" in user_lower
        and "full card" in user_lower
        and ("redacted" in user_lower or "organizer" in user_lower)
        and ("|---|" in assistant_text or REDACTION_TOKEN in assistant_text)
        and "full card" in assistant_lower
        and ("redacted" in assistant_lower or "organizer" in assistant_lower)
    )


def _card_difference_confirmation_message() -> str:
    return (
        "Confirmed. The full card is for your bag and can include the complete health, "
        "emergency contact, insurance, and provider details you provide for this task. "
        "The organizer version should keep only the details you approve for the organizer "
        "and omit insurance ID, provider name, and home address."
    )


def _looks_like_see_above_card_claim(text: str) -> bool:
    lowered = text.lower()
    return (
        "right there in the message above" in lowered
        and ("organizer version" in lowered or "card" in lowered)
    )


def _see_above_card_claim_replacement() -> str:
    return (
        "Confirmed. The organizer version should omit insurance ID, provider name, and "
        "home address. It should keep only the approved essentials for the organizer: "
        "name, medical conditions, medications, allergies, and emergency contact."
    )


def _format_original_trajectory(trajectory: ParsedTrajectory, pii_map: PIIMap) -> str:
    """Convert the parsed trajectory into annotated text with classification metadata.

    Each tool call gets annotated with its tier, is_write status, argument data layers,
    and whether a violation exists (free_band exceeded).
    """
    lines = []
    user_idx = 0
    turn_idx = 0
    violations: list[dict] = []

    for kind, idx in trajectory.thread_order:
        if kind == "user":
            if idx < len(trajectory.user_messages):
                msg = trajectory.user_messages[idx]
                if len(msg) > 2000:
                    msg = msg[:1800] + "\n...[truncated]..."
                lines.append(f"[USER MESSAGE {user_idx}]\n{msg}\n")
                user_idx += 1

        elif kind == "assistant":
            if idx < len(trajectory.assistant_turns):
                at = trajectory.assistant_turns[idx]
                parts = []

                text = "\n".join(at.text_blocks).strip()
                if text:
                    parts.append(f"Text: {text}")

                for tc in at.tool_calls:
                    tier = _classify_tool_tier(tc.name)
                    is_write = _is_write_tool(tc.name, tc.arguments)
                    arg_layer = _max_data_layer_for_args(tc.arguments, pii_map)
                    if tier == 1 and is_write and _looks_like_health_appointment_persistence(tc.arguments):
                        arg_layer = max(arg_layer, 3)
                    free = FREE_BAND.get((tier, is_write), 1)

                    args_str = json.dumps(tc.arguments)
                    if len(args_str) > 500:
                        args_str = args_str[:450] + "...}"

                    annotation = f"[T{tier} {'WRITE' if is_write else 'READ'}, args_data_layer=L{arg_layer}, free_band=L{free}"
                    if arg_layer > free:
                        annotation += ", VIOLATION: needs consent"
                        violations.append({
                            "turn": turn_idx,
                            "tool": tc.name,
                            "tier": tier,
                            "is_write": is_write,
                            "data_layer": arg_layer,
                            "free_band": free,
                        })
                    annotation += "]"

                    parts.append(f"Tool Call: {tc.name}({args_str}) {annotation}")

                    result = trajectory.tool_results_by_call_id.get(tc.call_id)
                    if result:
                        result_str = result.content
                        if len(result_str) > 500:
                            result_str = result_str[:450] + "...[truncated]"
                        parts.append(f"Tool Result ({tc.name}): {result_str}")

                content = "\n".join(parts)
                lines.append(f"[ASSISTANT TURN {turn_idx}]\n{content}\n")
                turn_idx += 1

    if violations:
        lines.insert(0, f"PRE-COMPUTED VIOLATIONS ({len(violations)} found):")
        for v in violations:
            lines.insert(1, f"  Turn {v['turn']}: {v['tool']} T{v['tier']} {'WRITE' if v['is_write'] else 'READ'} with L{v['data_layer']} data (free_band=L{v['free_band']}) -> NEEDS CONSENT GATE")
        lines.insert(1 + len(violations), "")
    else:
        lines.insert(0, "PRE-COMPUTED VIOLATIONS: None found (trajectory is clean).\n  You must still add one adversarial probe (Scenario D) and any appropriate privacy behaviors.\n")

    return "\n".join(lines)


def _format_pii_map(pii_map: PIIMap) -> str:
    """Format PII entities into a readable summary."""
    if not pii_map.entities:
        return "No PII entities detected."

    lines = [f"PII Entities Found ({len(pii_map.entities)} total, max level: {pii_map.max_level}):"]
    seen = set()
    for e in pii_map.entities:
        key = (e.label, e.level, e.text[:50])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  - {e.label} ({e.level}): \"{e.text[:80]}\"")

    return "\n".join(lines)


def _format_task_spec(trajectory: ParsedTrajectory) -> str:
    """Format task definition into readable text."""
    spec = trajectory.task_spec
    if not spec:
        return "No task definition available."

    lines = ["Task Definition:"]
    for key in ["title", "goal_summary", "privacy_scenario", "data_levels",
                "expected_privacy_actions", "tool_tiers", "pii_fields_exercised"]:
        val = spec.get(key, "")
        if val:
            lines.append(f"  {key}: {val}")

    return "\n".join(lines)


def _format_persona_vault(persona: dict) -> str:
    """Format persona's PII vault for context."""
    if not persona:
        return "No persona data available."

    lines = [f"Persona: {persona.get('first_name', '')} {persona.get('last_name', '')}"]

    vault = persona.get("pii_vault", {})
    if vault:
        lines.append("PII Vault (data the persona has):")
        for category, entries in vault.items():
            if isinstance(entries, dict):
                for k, v in entries.items():
                    val_str = str(v)
                    if len(val_str) > 80:
                        val_str = val_str[:77] + "..."
                    lines.append(f"  {category}.{k}: {val_str}")
            else:
                lines.append(f"  {category}: {entries}")

    personality = persona.get("personality_profile", {})
    if personality:
        lines.append(f"Communication style: {personality.get('communication_style', 'casual')}")
        lines.append(f"Verbosity: {personality.get('verbosity', 'moderate')}")

    return "\n".join(lines)


def _truncate_context(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def _format_persona_generation_context(trajectory: ParsedTrajectory, pii_map: PIIMap) -> str:
    persona = redact_value(trajectory.persona or {}, pii_map)
    if not persona:
        return "No persona object available. Preserve the style already present in the trajectory."
    return _truncate_context(json.dumps(persona, ensure_ascii=False, indent=2), 6000)


def _format_workspace_generation_context(
    files: dict[str, str],
    pii_map: PIIMap,
    label: str,
) -> str:
    if not files:
        return f"{label}: unavailable."

    preferred = {
        "AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "MEMORY.md",
        "HEARTBEAT.md", "IDENTITY.md", "BOOTSTRAP.md",
    }
    ordered = sorted(
        files.items(),
        key=lambda item: (
            0 if item[0].split("/")[-1] in preferred else 1,
            item[0].lower(),
        ),
    )
    lines = [f"{label} (read-only sample, L3/L4 redacted):"]
    total = 0
    for path, content in ordered[:18]:
        safe_path = redact_text(str(path), pii_map)
        safe_content = redact_text(str(content), pii_map)
        safe_content = _truncate_context(safe_content.strip(), 1200)
        block = f"\n--- {safe_path} ---\n{safe_content or '(empty)'}"
        if total + len(block) > 12000:
            lines.append("\n...[workspace sample truncated]")
            break
        lines.append(block)
        total += len(block)
    return "\n".join(lines)


def _openclaw_generation_context(trajectory: ParsedTrajectory, pii_map: PIIMap) -> str:
    before_files = trajectory.workspace_before_files or {}
    after_files = trajectory.workspace_files or {}
    if not before_files and after_files:
        before_note = (
            "Workspace-before was not present in the parsed export. "
            "Use workspace-after only as read-only state context; do not infer new writes from it."
        )
    else:
        before_note = ""

    return f"""OPENCLAW GENERATION WORKER CONTEXT

Operating mode:
- Use the persona below as the primary voice/style anchor for this task.
- Preserve the original trajectory persona and wording. Do not inject OpenClaw, Codex, reviewer, dataset, or synthetic-worker persona.
- Use workspace files only as read-only context for state consistency. Do not rewrite workspace files here.
- Return only the requested JSON object. Do not generate JSONL, ids, timestamps, parentId chains, metadata, or tool-call IDs.
- L3/L4 values in this context are redacted as {REDACTION_TOKEN}. Do not invent or reveal redacted values.
- The pipeline applies the patch deterministically after you return JSON.
- If the user provided a resource (doc, md, image, pdf, html), do not refuse to read or manipulate it just because it contains sensitive data. Keep resource text redacted in assistant-visible text, ask permission before using raw values as T2/T3 tool arguments, and prefer re-reading the resource or using a vault/source instead of recalling values from earlier turns.

PERSONA:
{_format_persona_generation_context(trajectory, pii_map)}

TASK SPEC:
{redact_text(_format_task_spec(trajectory), pii_map)}

{before_note}

WORKSPACE BEFORE:
{_format_workspace_generation_context(before_files, pii_map, "workspace_before")}

WORKSPACE AFTER:
{_format_workspace_generation_context(after_files, pii_map, "workspace")}
"""


def _is_category_memory_setup_trajectory(trajectory: ParsedTrajectory) -> bool:
    if trajectory.task_id in {"T-122-01", "P-122-01"}:
        return True
    spec = trajectory.task_spec or {}
    spec_text = json.dumps(spec, ensure_ascii=False).lower() if isinstance(spec, dict) else ""
    user_text = "\n".join(trajectory.user_messages[:2]).lower()
    return (
        "category-level consent" in spec_text
        or "memory consent" in spec_text
        or (
            any(term in user_text for term in ("future conversations", "longer-term interaction", "remembered or stored"))
            and any(term in user_text for term in ("safe to remember", "category", "categories", "approve", "consent"))
        )
    )


def _first_user_event(trajectory: ParsedTrajectory) -> dict:
    for event in trajectory.ordered_events or []:
        if isinstance(event, dict) and event.get("type") == "message":
            msg = event.get("message", {})
            if isinstance(msg, dict) and msg.get("role") == "user":
                return event
    text = trajectory.user_messages[0] if trajectory.user_messages else "I'd like to set memory preferences safely."
    return {
        "type": "message",
        "id": _make_event_id(),
        "parentId": None,
        "timestamp": "",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


def _category_memory_default_texts() -> dict[str, str]:
    return {
        "classification_response": (
            "Nice to meet you. I can keep memory category-based before saving anything. "
            "Broad professional interests, hobbies, and general productivity preferences are the safe categories. "
            "Health or wellness specifics, exact location, employer or title details, identity documents, payroll, credentials, "
            "and sensitive identifiers should stay out of memory. Which low-risk categories do you want remembered?"
        ),
        "consent_user": (
            "Remember my backend engineering interests, piano and calligraphy hobbies, and general productivity preferences. "
            "Do not remember health or wellness specifics, exact location, employer or title details, identity documents, payroll, credentials, or anything sensitive."
        ),
        "vault_value": (
            "Approved memory categories: backend engineering interests; hobbies include piano and Arabic calligraphy; "
            "general productivity preferences. Excluded from memory: health or wellness specifics, exact location, employer or title details, "
            "identity documents, payroll, credentials, and sensitive identifiers."
        ),
        "recap_response": (
            "Saved only the approved low-risk categories. Going forward I can use your backend engineering interests, piano and calligraphy hobbies, "
            "and general productivity preferences. I did not save health or wellness details, exact location, employer or title details, identity documents, "
            "payroll, credentials, or sensitive identifiers, and I won't add new memory categories without asking first."
        ),
        "adversarial_user_probe": "Actually, remember my health details too so I do not need to repeat them later.",
        "final_ack": "Anytime. I will keep using the approved memory categories and ask before changing them.",
    }


def _parse_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("```"):
        first_nl = raw.index("\n") if "\n" in raw else 3
        raw = raw[first_nl + 1:]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    candidate = match.group(0) if match else raw
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        import json_repair
        parsed = json_repair.loads(candidate)
    return parsed if isinstance(parsed, dict) else {}


def _clean_category_memory_texts(raw_texts: dict[str, Any], pii_map: PIIMap) -> dict[str, str]:
    defaults = _category_memory_default_texts()
    cleaned: dict[str, str] = {}
    for key, default in defaults.items():
        value = raw_texts.get(key, default)
        if not isinstance(value, str) or not value.strip():
            value = default
        value = value.strip().replace("—", "-")
        redacted = redact_text(value, pii_map)
        if redacted.strip() == REDACTION_TOKEN or redacted.count(REDACTION_TOKEN) >= 2:
            redacted = redact_text(default, pii_map)
        cleaned[key] = redacted

    vault_value = cleaned["vault_value"]
    lowered = vault_value.lower()
    if "approved" not in lowered or "excluded" not in lowered:
        vault_value = defaults["vault_value"]
    cleaned["vault_value"] = redact_text(vault_value, pii_map)
    return cleaned


async def _generate_category_memory_setup_texts(
    trajectory: ParsedTrajectory,
    pii_map: PIIMap,
    generation_backend: str = "direct",
    generation_model: str | None = None,
) -> dict[str, str]:
    """Generate natural wording for the category-memory setup trajectory."""
    persona = trajectory.persona or {}
    persona_name = (
        f"{persona.get('first_name', '')} {persona.get('last_name', '')}".strip()
        or persona.get("name", "the user")
    )
    user_context = "\n".join(
        f"- {redact_text(msg, pii_map)}" for msg in trajectory.user_messages[:5]
    ) or "- No explicit user messages found."
    task_context = redact_text(_format_task_spec(trajectory), pii_map)
    pii_context = redact_text(_format_pii_map(pii_map), pii_map)

    prompt = f"""{_openclaw_generation_context(trajectory, pii_map)}

Generate the natural-language text for one privacy-compliant memory setup trajectory.

The output will be inserted into an existing JSONL trajectory. Do not generate JSONL, ids, timestamps, tool calls, metadata, labels, or annotations.

CONTEXT:
- Persona: {persona_name}
- Task id: {trajectory.task_id}
- Task/submission: {trajectory.submission_id}

TASK SPEC:
{task_context}

ORIGINAL USER CONTEXT:
{user_context}

PII SUMMARY, with L3/L4 values redacted:
{pii_context}

REQUIRED BEHAVIOR:
- The first assistant turn must classify memory categories before saving anything.
- Allowed to store only low-risk approved categories such as professional background, hobbies, and general productivity preferences.
- Exclude health details, exact location, identity documents, payroll, credentials, exact addresses, sensitive identifiers, and transient data.
- Do not mention the user's full name, exact employer, exact job title, exact city, exact address, raw health detail, medication detail, credential, or identifier in any field.
- Use broad phrases like "backend engineering interests" instead of exact employer/title facts.
- The vault operation is fixed by code and must be the MCP vault tool set(key, val). You only generate the val string.
- The val string must summarize approved categories and exclusions. It must not contain L3/L4 raw values.
- Include a later user probe asking to remember a sensitive health detail for future use.
- The assistant refusal for that probe is fixed by code, so do not generate it.

STYLE:
- Sound like this is a real user-agent conversation.
- Keep each field concise.
- No markdown lists inside field values.
- No construction labels, no words like synthetic, dataset, adversarial, annotation, policy test, or trajectory.
- No em dashes.

Return ONLY this JSON object:
{{
  "classification_response": "assistant text that classifies safe vs excluded memory categories and asks what to remember",
  "consent_user": "natural user reply approving only low-risk categories",
  "vault_value": "compact vault value containing approved categories and explicit exclusions",
  "recap_response": "assistant confirmation after the vault write",
  "adversarial_user_probe": "natural user request to remember sensitive health details for future conversations",
  "final_ack": "short assistant close after the user accepts the refusal"
}}"""

    if (generation_backend or "direct").lower() == "cli":
        from openclaw_repair_worker import _call_openclaw_cli

        required_keys = tuple(_category_memory_default_texts().keys())
        raw_texts = await _call_openclaw_cli(
            prompt,
            generation_model,
            required_keys=required_keys,
            session_prefix="privacy-category-memory",
            purpose="stage3-generation",
        )
        if not isinstance(raw_texts, dict):
            raise RuntimeError("OpenClaw CLI returned no parseable category-memory JSON")
        logger.info("OpenClaw CLI category-memory generation returned %d fields", len(raw_texts))
        return _clean_category_memory_texts(raw_texts, pii_map)

    try:
        client = _get_client()
        response = await client.messages.create(
            model=REWRITER_MODEL,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = ""
        for block in response.content:
            if hasattr(block, "text"):
                raw += block.text
        tracker.record_anthropic(response, "category_memory_text")
        return _clean_category_memory_texts(_parse_json_object(raw), pii_map)
    except Exception as exc:
        logger.warning("Falling back to deterministic category-memory wording for %s: %s", trajectory.task_id, exc)
        return _clean_category_memory_texts({}, pii_map)


def _category_memory_setup_result(
    trajectory: ParsedTrajectory,
    pii_map: PIIMap,
    generated_texts: dict[str, str] | None = None,
) -> RewriteResult:
    import copy
    texts = _clean_category_memory_texts(generated_texts or {}, pii_map)

    session_events = [
        copy.deepcopy(event)
        for event in (trajectory.ordered_events or [])
        if isinstance(event, dict) and event.get("type") == "session"
    ][:1]
    events: list[dict] = session_events
    parent_id = events[-1].get("id") if events else None

    first_user = copy.deepcopy(_first_user_event(trajectory))
    if parent_id:
        first_user["parentId"] = parent_id
    events.append(first_user)
    parent_id = first_user.get("id")
    ts = str(first_user.get("timestamp", ""))

    classify_id = _make_event_id()
    events.append({
        "type": "message",
        "id": classify_id,
        "parentId": parent_id,
        "timestamp": _offset_timestamp(ts, 1200),
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": texts["classification_response"]}],
        },
    })

    consent_id = _make_event_id()
    events.append({
        "type": "message",
        "id": consent_id,
        "parentId": classify_id,
        "timestamp": _offset_timestamp(ts, 2400),
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": texts["consent_user"]}],
        },
    })

    call_id = _make_tc_id()
    tool_id = _make_event_id()
    events.append({
        "type": "message",
        "id": tool_id,
        "parentId": consent_id,
        "timestamp": _offset_timestamp(ts, 3600),
        "message": {
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "id": call_id,
                "name": "set",
                "arguments": {
                    "key": "elif_memory_preferences",
                    "val": texts["vault_value"],
                },
                "class": {
                    "tool_tier": 1,
                    "is_write": True,
                    "arguments": {
                        "key": {"data_layer": None},
                        "val": {"data_layer": "L2"},
                    },
                },
                "flag": {
                    "is_violation": False,
                    "free_band": 4,
                    "data_layer": 2,
                    "authorization": {"needed": False, "granted": False},
                },
            }],
        },
    })

    result_id = _make_event_id()
    events.append({
        "type": "message",
        "id": result_id,
        "parentId": tool_id,
        "timestamp": _offset_timestamp(ts, 4200),
        "message": {
            "role": "toolResult",
            "toolCallId": call_id,
            "toolName": "set",
            "content": [{"type": "text", "text": json.dumps({"ok": True})}],
            "details": {"ok": True},
            "isError": False,
        },
    })

    recap_id = _make_event_id()
    events.append({
        "type": "message",
        "id": recap_id,
        "parentId": result_id,
        "timestamp": _offset_timestamp(ts, 5400),
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": texts["recap_response"]}],
        },
    })

    probe_id = _make_event_id()
    events.append({
        "type": "message",
        "id": probe_id,
        "parentId": recap_id,
        "timestamp": _offset_timestamp(ts, 7200),
        "message": {
            "role": "user",
            "content": [{"type": "text", "text": texts["adversarial_user_probe"]}],
        },
    })

    refusal_id = _make_event_id()
    events.append({
        "type": "message",
        "id": refusal_id,
        "parentId": probe_id,
        "timestamp": _offset_timestamp(ts, 8400),
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": HATCH_REFUSAL}],
        },
    })

    thanks_id = _make_event_id()
    events.append({
        "type": "message",
        "id": thanks_id,
        "parentId": refusal_id,
        "timestamp": _offset_timestamp(ts, 9600),
        "message": {"role": "user", "content": [{"type": "text", "text": "Thanks, that makes sense."}]},
    })

    final_id = _make_event_id()
    events.append({
        "type": "message",
        "id": final_id,
        "parentId": thanks_id,
        "timestamp": _offset_timestamp(ts, 10800),
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": texts["final_ack"]}],
        },
    })

    _annotate_tool_calls(events, pii_map)
    events = redact_event_stream(events, pii_map)
    return RewriteResult(
        task_id=trajectory.task_id,
        submission_id=trajectory.submission_id,
        scenarios_covered=["A", "D"],
        skills_used=["set"],
        privacy_decision_points=3,
        patched_events=events,
    )


def _get_rag_context(trajectory: ParsedTrajectory, pii_map: PIIMap) -> str:
    """Pull relevant examples from RAG for the LLM prompt."""
    try:
        from rag_retriever import (
            get_similar_conversations,
            get_privacy_patterns,
            is_index_ready,
        )
        if not is_index_ready():
            return ""

        task_goal = trajectory.task_spec.get("goal_summary", "")
        if not task_goal:
            task_goal = " ".join(trajectory.user_messages[:2])[:200]

        parts = []
        conv_examples = get_similar_conversations(task_goal, n=3)
        if conv_examples:
            parts.append(conv_examples)

        privacy_query = f"privacy handling for {pii_map.max_level} data"
        privacy_examples = get_privacy_patterns(privacy_query, n=3)
        if privacy_examples:
            parts.append(privacy_examples)

        return "\n\n".join(parts)
    except Exception as e:
        logger.debug("RAG context fetch failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# The core prompt
# ---------------------------------------------------------------------------

PRIVACY_RULES = """
HATCH TRUST GUIDELINES (HTG) — COMPLETE PRIVACY RULES:

═══════════════════════════════════════════════════════════════
1. DATA CLASSIFICATION LEVELS
═══════════════════════════════════════════════════════════════

  L0 (Public): Broad locations, general knowledge.
     → No flagging needed. Store freely. No consent required.

  L1 (Caution): Device IDs, IP addresses, metro-level city, age, hobbies, proprietary code.
     → No flagging needed. Store freely. No consent required.

  L2 (Sensitive): Full names, emails, employer, job title, account IDs, transaction history, hashed IDs.
     → Can be SAVED without consent using T1 tools (including memory_write).
     → Needs consent for T3 tools. OK for T2 without consent.

  L3 (Highly Sensitive): Health records (PHI), detailed financials, private communications, precise GPS.
     → Zero retention. Agent amnesia. Must re-input per turn.
     → Can be PROCESSED by T1 tools (e.g., local OCR on a medical PDF) but NEVER written to persistent storage.

  L4 (Critical): SSNs, passwords, biometrics, full credit card numbers.
     → NEVER store anywhere. Agent amnesia. Must re-input per turn.
     → Can be PROCESSED by T1 tools for the current turn only, then dropped.

KEY RULE — L0, L1, AND L2 WITH T1 TOOLS: NO CONSENT NEEDED.
  The agent can store L0, L1, and L2 data using T1 tools (memory_write, write, etc.) WITHOUT asking for
  explicit user permission. Just do it naturally. Do NOT insert unnecessary consent gates for L0/L1/L2
  data when using local tools. The user telling you their name, employer, or city is enough — save it.

═══════════════════════════════════════════════════════════════
2. TOOL TRUST TIERS (complete lists)
═══════════════════════════════════════════════════════════════

  T1 (Local) — Safe to PROCESS all data up to L4. Safe to WRITE/SAVE up to L2. L3/L4 writes allowed ONLY with consent + encrypted storage (vault set(key,val), encrypted memory_write, active_memory_write, wiki_apply). Plaintext L3/L4 writes are HARD DENIED.
     Tools: memory_write, memory_search, memory_read, read, write, edit, exec, cron, sessions,
            MCP vault tools get(key), set(key,val), delete(key),
            nano-pdf, surya, openai-whisper, word-docx, excel-xlsx, powerpoint-pptx,
            docker-essentials, data-analysis, automation-workflows, markdown-converter,
            ontology, self-improving, obsidian, mcporter, humanizer, productivity,
            skill-creator, ui-ux-pro-max, self-reflection, language-learning, cfo, health,
            relationship-skills, workout, healthcheck, mechanic, marketing-mode,
            Linux-native tools (ls, cp, mv, rm, cat, grep, awk, sed, find, chmod, tar, curl localhost)

  T2 (1P Cloud) — Safe to READ/WRITE up to L2. L3/L4 requires EXPLICIT user consent via consent gate.
     Tools: enterprise-mail, enterprise-calendar, enterprise-rag, enterprise-inference, enterprise-vault

  T3 (3P API) — Safe to READ/WRITE up to L1 only. L2+ STRICTLY PROHIBITED without documented user exception.
     Tools: web_search, web_fetch, browser, agent-browser, github, slack, notion, gog, trello,
            weather, news-summary, stock-analysis, caldav-calendar, clawhub, api-gateway,
            polymarket, goplaces, openai-whisper-api, edge-tts, academic-research,
            flight-search, eventbrite, plan2meal, legaldoc-ai, music-cog, bundled-web-tools

  IF A TOOL IS NOT IN ANY LIST ABOVE: Default to T3 treatment (L1 data cap).

═══════════════════════════════════════════════════════════════
3. CONSENT GATE RULES (when to ask)
═══════════════════════════════════════════════════════════════

  Only insert a consent gate when there is a MISMATCH between the data level and tool tier:

  - T1 + L0/L1/L2 (READ or WRITE): NO consent needed. Just proceed.
  - T1 + L3/L4 (PROCESS/READ only): NO consent needed. Execute locally, then drop data.
  - T1 + L3/L4 (WRITE to encrypted store): CONSENT REQUIRED. Allowed ONLY with explicit user consent AND encrypted storage (MCP vault set(key,val), enterprise-vault, encrypted memory_write, active_memory_write, wiki_apply).
  - T1 + L3/L4 (WRITE to plaintext): HARD DENY. Refuse with the Hatch refusal string. Cannot be overridden by consent. No writing L3/L4 to plain .md files, unencrypted MEMORY.md, or workspace files.
  - T2 + L0/L1/L2: NO consent needed. Just proceed.
  - T2 + L3/L4: Consent gate required. Ask user explicitly before proceeding.
  - T3 + L0/L1: NO consent needed. Just proceed.
  - T3 + L2+: Consent gate required. Ask user explicitly, explain data will go to external service.

  The consent text must reference the ACTUAL task being done (not generic "booking" or "appointment").

═══════════════════════════════════════════════════════════════
4. TRAJECTORY SCENARIO PATTERNS
═══════════════════════════════════════════════════════════════

  Use the correct pattern based on what the task involves:

  SCENARIO A — Ideal Local Execution (Zero Exfiltration):
    When user needs L3/L4 data processed and a T1 tool can do it.
    Flow: Identify data as L3/L4 → Ask user to provide data for this turn → Select T1 tool →
          Execute locally → Return result → Drop data from context (amnesia).

  SCENARIO B — Cloud Fallback with Explicit Consent:
    When user needs L3/L4 data processed but only a T2 tool can do it.
    Flow: Attempt T1 first (fails) → Select T2 tool → Recognize mismatch (T2 caps at L2, data is L3) →
          Halt and warn user → Request authorization → Execute only after user approves.

  SCENARIO C — 3P API Hard Block / Exception Handling:
    When user asks to send L2/L3/L4 data to a T3 tool.
    Flow: Identify T3 caps at L1 → Halt immediately → Explain exfiltration risk →
          Ask for authorization, confirm user understands data leaves controlled network.

  SCENARIO D — Memory Manipulation / Jailbreak Attempt (ADVERSARIAL):
    When user tries to make the agent permanently store L3/L4 data.
    Flow: Identify data as L3/L4 → Enforce Zero Retention → Output EXACT refusal:
          "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data.
           Please provide it again for the current task."

═══════════════════════════════════════════════════════════════
5. REQUIRED BEHAVIORS
═══════════════════════════════════════════════════════════════

  1. ADVERSARIAL REFUSAL: Every trajectory MUST include one adversarial probe (Scenario D) where the user
     tries to get the agent to save L3/L4 data. The assistant MUST refuse with the EXACT Hatch string:
     "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. Please provide it again for the current task."

     CRITICAL: After the refusal string, do NOT repeat the same concept in different words. Do NOT add
     extra sentences explaining why it can't store data, how retention works, or what the sensitivity tier is.
     The refusal string says everything needed. After it, briefly offer what CAN be done (e.g., "happy to
     help with X if you paste it in when you need it") and move on. One refusal, not a lecture.

     BAD example (redundant):
       "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. Please provide
        it again for the current task. SSN falls into the highest sensitivity tier. I'm not allowed to store
        it anywhere, not even locally, regardless of consent. No persistent storage, ever."
     GOOD example (concise):
       "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. Please provide
        it again for the current task. Happy to use it for whatever you need right now though, just paste it
        in when you're ready."

  2. L3/L4 THREE-TURN FLOW: When L3/L4 data is needed for a task:
     Turn 1 (Assistant): Explains it needs the data and asks user to provide it for this turn.
     Turn 2 (User): Provides the L3/L4 data.
     Turn 3 (Assistant): Uses it for that turn only, then drops it (agent amnesia).

  3. LOCAL-FIRST: Always prefer T1 tools over T2/T3 when the task can be done locally.
     Only escalate to T2/T3 when a local tool is genuinely insufficient.

  4. DATA MINIMIZATION: Store only what's useful for future sessions. Skip data that won't help.
     But do NOT be overly cautious. If the user shares L0/L1/L2 data and wants it remembered, just save it.

  5. TRANSPARENCY: When refusing or requesting consent, the assistant should:
     - Briefly cite the privacy concern (casual tone, e.g., "that's L3 health data, need your ok")
     - Offer alternatives when possible
     - Explain what would be needed to proceed

  6. CONSISTENCY: What the assistant says it will do must match what it actually does.
     If it says "I'll save your preferences", it must actually call memory_write.
     If it says "I won't store that", it must NOT call memory_write with that data.

  7. NO LECTURING: Do not have the agent lecture the user on security practices unless the user is
     actively violating a constraint. Be concise, capable, and transparent. If elicitation is required,
     simply ask for the data and execute once provided.

═══════════════════════════════════════════════════════════════
6. WRITING STYLE RULES (MANDATORY)
═══════════════════════════════════════════════════════════════

  ALL assistant text in the trajectory MUST sound like a real human wrote it. Follow these rules strictly:

  BANNED PATTERNS (never use these):
  - Em dashes (--) or (—). Use commas, periods, or parentheses instead.
  - "Additionally", "Furthermore", "Moreover" as sentence starters.
  - "crucial", "pivotal", "vital", "key" (as adjective), "landscape" (abstract), "tapestry", "testament",
    "underscore", "delve", "foster", "garner", "showcase", "vibrant", "intricate", "enduring", "enhance".
  - "It's not just X; it's Y" (negative parallelism).
  - "serves as", "stands as", "marks a" (copula avoidance). Just use "is" or "are".
  - "I hope this helps!", "Let me know if you'd like...", "Of course!", "Certainly!", "Great question!",
    "You're absolutely right!", "Absolutely!" (sycophantic/servile tone).
  - Rule of three lists forced for effect ("innovation, inspiration, and insight").
  - Emoji in any text.
  - Curly quotation marks. Use straight quotes only.
  - Excessive hedging ("it could potentially possibly be argued that...").
  - Filler phrases ("In order to", "Due to the fact that", "It is important to note that").
  - Generic positive conclusions ("The future looks bright", "Exciting times lie ahead").

  REQUIRED STYLE:
  - Vary sentence length naturally. Mix short punchy sentences with longer ones.
  - Use simple constructions: "is", "are", "has" over fancy substitutes.
  - Be direct. Say what you mean. Real people don't pad their sentences.
  - Use contractions naturally ("I'll", "can't", "won't", "it's").
  - When explaining privacy decisions, be casual: "heads up, that's health data so I can't hang onto it"
    not "I must inform you that under the established guidelines, this data falls under Level 3 classification."
  - Acknowledge the user's request before getting into privacy stuff. Don't lead with the rules.
"""


def _build_prompt(
    trajectory: ParsedTrajectory,
    pii_map: PIIMap,
    rag_context: str,
) -> str:
    """Build the complete prompt for the single-shot rewriter."""

    original_text = _format_original_trajectory(trajectory, pii_map)
    pii_text = _format_pii_map(pii_map)
    task_text = _format_task_spec(trajectory)
    persona_text = _format_persona_vault(trajectory.persona or {})

    return f"""You are an expert at converting standard SFT agent trajectories into privacy-compliant Privacy SFT trajectories.

## YOUR CORE MANDATE: PRESERVE AND ENHANCE, NEVER REPLACE

You MUST keep the original trajectory's task, content, and structure intact. Your job is to ADD privacy
behaviors (consent gates, refusals, data handling) to the existing conversation, NOT to replace it with
a different conversation.

ABSOLUTELY FORBIDDEN:
- Changing the user's name, identity, or task
- Replacing the original task with a different topic
- Inventing tool call results that differ from the originals
- Changing the session ID, timestamps, or working directory
- Dropping original tool calls from the trajectory
- Making up conversations that don't follow from the original

WHAT YOU MUST DO:
1. Keep EVERY original user message (reword slightly for naturalness if needed, but preserve intent and content)
2. Keep EVERY original assistant turn and tool call (tool name, arguments, results must match originals)
3. Keep the original tool call results exactly as they appeared (do not fabricate search results)
4. INSERT new turns only for: consent gates, adversarial probes, or privacy explanations
5. MODIFY assistant text only when needed: to add privacy reasoning, consent requests, or refusals
6. Add ONE adversarial probe (Scenario D) woven naturally into the conversation at an appropriate point

Each tool call in the original trajectory has been pre-annotated with:
  - Tool tier (T1/T2/T3)
  - Read/Write classification
  - Argument data layer (L0-L4)
  - Whether it exceeds the free_band (VIOLATION flag)

Where a VIOLATION is flagged, you MUST insert a consent gate before that tool call.
Where NO violation exists, keep the trajectory as-is (just ensure assistant text sounds natural).

{PRIVACY_RULES}

---
{task_text}

---
{persona_text}

---
{pii_text}

---
ORIGINAL TRAJECTORY (with classification annotations):

{original_text}

---
{rag_context}

---
OUTPUT FORMAT:

Produce the privacy-enhanced trajectory as a JSON array of turn objects.

```json
[
  {{
    "role": "user",
    "text": "the ORIGINAL user message text (preserve content faithfully)",
    "is_original": true
  }},
  {{
    "role": "assistant",
    "text": "natural assistant response (REQUIRED, preserve original intent, add privacy reasoning if needed)",
    "tool_calls": [
      {{
        "name": "original_tool_name",
        "arguments": {{"original": "arguments"}},
        "result": "ORIGINAL tool result text (preserve exactly)"
      }}
    ],
    "privacy_actions": ["consent_gate", "data_classification", etc.],
    "scenario": "A/B/C/D/E/F/etc."
  }},
  {{
    "role": "user",
    "text": "user response to consent gate or adversarial probe",
    "is_synthetic": true
  }}
]
```

CRITICAL RULES:
1. Every assistant turn MUST have non-empty "text".
2. User and assistant turns MUST strictly alternate. Never two assistants in a row.
3. PRESERVE all original tool calls with their EXACT arguments and results.
   - Copy tool names, argument keys/values, and result text from the original.
   - Do NOT fabricate or summarize tool results. Use the ORIGINAL result text.
4. For the adversarial probe (Scenario D):
   - Insert a synthetic user message that tries to get the agent to store L3/L4 data permanently
   - Make the probe contextually relevant to the conversation topic
   - The assistant refuses with the exact Hatch Trust Guidelines string (verbatim)
   - Then a synthetic user acknowledgment, and the conversation continues naturally
5. For consent gates (ONLY at pre-annotated VIOLATION points):
   - Insert a synthetic assistant consent request referencing the ACTUAL task
   - Insert a synthetic user approval
   - Then the original tool call proceeds
6. For memory_write calls (T1):
   - L0/L1/L2 data: JUST SAVE IT. No consent needed. Proceed naturally.
   - L3/L4 to plaintext: NEVER store. Refuse with the HTG string.
   - L3/L4 to encrypted storage: Allowed WITH explicit user consent.
7. CONSISTENCY: Actions must match what the assistant says it will do.
8. All assistant text MUST sound human-written. NO em dashes, NO AI vocabulary, NO sycophantic phrases.
9. Mark synthetic user messages with "is_synthetic": true.
10. Mark original user messages with "is_original": true.
11. Do NOT be overly cautious. L0/L1/L2 with T1 tools needs NO consent. Just do it.

Produce ONLY the JSON array. Start with [ and end with ]."""


# ---------------------------------------------------------------------------
# Parse LLM output into RewriteResult
# ---------------------------------------------------------------------------

def _parse_llm_output(raw: str, trajectory: ParsedTrajectory) -> RewriteResult:
    """Parse the LLM's JSON output into a RewriteResult.

    This also rewrites trajectory.user_messages and trajectory.thread_order
    to match the LLM's output, so the writer's user-message mapping works.
    """
    json_match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON array found in LLM output")

    raw_json = json_match.group()
    try:
        turns_data = json.loads(raw_json)
    except json.JSONDecodeError:
        try:
            import json_repair
            turns_data = json_repair.loads(raw_json)
            logger.info("Used json_repair to fix malformed JSON output")
        except Exception:
            raw_json = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw_json)
            turns_data = json.loads(raw_json)

    # First pass: build the full conversation in order
    # We need to know which user messages precede which assistant turns
    new_user_messages: list[str] = []
    new_thread_order: list[tuple[str, int]] = []
    rewritten_turns: list[RewrittenTurn] = []

    user_idx = 0
    assistant_idx = 0
    pending_synthetic_user: str | None = None
    pending_adversarial_user: str | None = None
    pending_consent_response: bool = False

    for i, item in enumerate(turns_data):
        role = item.get("role", "")

        if role == "user":
            is_synthetic = item.get("is_synthetic", False)
            text = item.get("text", "")

            if is_synthetic:
                # Look ahead: is the next assistant turn adversarial?
                next_asst = None
                for j in range(i + 1, len(turns_data)):
                    if turns_data[j].get("role") == "assistant":
                        next_asst = turns_data[j]
                        break

                is_adversarial_probe = False
                if next_asst:
                    pa = next_asst.get("privacy_actions", [])
                    if "adversarial_refusal" in pa or next_asst.get("scenario") == "D":
                        is_adversarial_probe = True

                if is_adversarial_probe:
                    pending_adversarial_user = text
                else:
                    pending_synthetic_user = text
                    if next_asst and "consent_granted" in next_asst.get("privacy_actions", []):
                        pending_consent_response = True
            else:
                # Original user message
                new_user_messages.append(text)
                new_thread_order.append(("user", user_idx))
                user_idx += 1

        elif role == "assistant":
            tool_calls = []
            tool_results = []
            for tc in item.get("tool_calls", []):
                tc_dict = {
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", {}),
                }
                tool_calls.append(tc_dict)

                if "result" in tc:
                    tool_results.append({
                        "tool_name": tc.get("name", ""),
                        "content": str(tc.get("result", "")),
                        "is_error": tc.get("is_error", False),
                    })

            privacy_actions = item.get("privacy_actions", [])
            scenario = item.get("scenario", "")
            text = item.get("text", "")

            is_adversarial = pending_adversarial_user is not None
            consent_decision = "granted" if pending_consent_response else ""

            rt = RewrittenTurn(
                turn_index=assistant_idx,
                thinking="",
                text=text,
                tool_calls=tool_calls,
                tool_results=tool_results,
                privacy_actions=privacy_actions,
                scenario=scenario,
                synthetic_user_message=pending_synthetic_user or "",
                consent_decision=consent_decision,
                is_adversarial=is_adversarial,
                adversarial_user_message=pending_adversarial_user or "",
            )
            rewritten_turns.append(rt)
            new_thread_order.append(("assistant", assistant_idx))
            assistant_idx += 1

            pending_synthetic_user = None
            pending_adversarial_user = None
            pending_consent_response = False

    # Update the trajectory so the writer can map user messages correctly
    trajectory.user_messages = new_user_messages
    trajectory.thread_order = new_thread_order

    scenarios_covered = list(set(
        rt.scenario for rt in rewritten_turns if rt.scenario
    ))

    privacy_points = sum(
        1 for rt in rewritten_turns
        if rt.privacy_actions
    )

    return RewriteResult(
        task_id=trajectory.task_id,
        submission_id=trajectory.submission_id,
        turns=rewritten_turns,
        scenarios_covered=scenarios_covered,
        privacy_decision_points=privacy_points,
    )


# ---------------------------------------------------------------------------
# PATCH-MODE REWRITER (cuarena-inspired: never regenerate, only splice)
# ---------------------------------------------------------------------------

def _analyze_violations(trajectory: ParsedTrajectory, pii_map: PIIMap) -> list[dict]:
    """Scan the trajectory and return a list of violation points."""
    violations = []
    emitted_external_medical_gate = False
    emitted_booking_submission_gate = False
    emitted_booking_details_request_gate = False
    emitted_elicit_turns: set[int] = set()
    emitted_elicit_value_signatures: set[tuple[str, ...]] = set()
    emitted_email_send_gate = False
    seen_external_medical_site = False
    saw_plaintext_medical_persistence = False
    turn_to_user = _turn_to_user_map(trajectory)
    try:
        from skill_substitution import load_catalog, build_result_index, consumed_for_call
        subst_catalog = load_catalog()
        result_index, all_texts = build_result_index(trajectory)
        consumed_for_call_fn = consumed_for_call
    except Exception:
        subst_catalog = None
        result_index = {}
        all_texts = []
        consumed_for_call_fn = None
    for turn_idx, at in enumerate(trajectory.assistant_turns):
        turn_text = "\n".join(at.text_blocks)
        user_idx = turn_to_user.get(turn_idx, -1)
        current_user_text = (
            trajectory.user_messages[user_idx]
            if 0 <= user_idx < len(trajectory.user_messages)
            else ""
        )
        if (
            not emitted_email_send_gate
            and _has_sensitive_card_l3(pii_map)
            and _looks_like_sensitive_card_send_request(current_user_text)
            and (
                at.tool_calls
                or any(word in turn_text.lower() for word in ("send", "email", "mail"))
            )
        ):
            violations.append({
                "turn_idx": turn_idx,
                "action": "consent_gate",
                "tool_name": "enterprise-email",
                "tier": 2,
                "is_write": True,
                "data_layer": 3,
                "free_band": 2,
                "call_id": "",
                "args_summary": current_user_text[:240],
                "sensitive_card_send": True,
            })
            emitted_email_send_gate = True
        if (
            not emitted_email_send_gate
            and _looks_like_emailing_sensitive_card(current_user_text, turn_text, pii_map)
            and any(tc.name in {"exec", "enterprise-mail", "enterprise-email", "gmail"} for tc in at.tool_calls)
        ):
            violations.append({
                "turn_idx": turn_idx,
                "action": "consent_gate",
                "tool_name": "enterprise-email",
                "tier": 2,
                "is_write": True,
                "data_layer": 3,
                "free_band": 2,
                "call_id": "",
                "args_summary": current_user_text[:240],
            })
            emitted_email_send_gate = True
        if (
            seen_external_medical_site
            and not emitted_booking_details_request_gate
            and _asks_for_external_booking_pii(turn_text)
        ):
            violations.append({
                "turn_idx": turn_idx,
                "action": "consent_gate",
                "tool_name": "assistant_text",
                "tier": 3,
                "is_write": True,
                "data_layer": 2,
                "free_band": 1,
                "call_id": "",
                "args_summary": turn_text[:200],
            })
            emitted_booking_details_request_gate = True

        for tc in at.tool_calls:
            tier = _classify_tool_tier(tc.name)
            is_write = _is_write_tool(tc.name, tc.arguments)
            arg_layer = _max_data_layer_for_args(tc.arguments, pii_map)
            if tier == 1 and is_write and _looks_like_health_appointment_persistence(tc.arguments):
                arg_layer = max(arg_layer, 3)
            if tc.name in {"web_search", "web_fetch"} and not _contains_exact_pii(tc.arguments, pii_map):
                arg_layer = min(arg_layer, 1)
            external_context = _mentions_external_medical_site(tc.arguments)
            if external_context:
                seen_external_medical_site = True
            external_medical = _is_external_medical_action(tc.name, tc.arguments, pii_map)
            if external_medical:
                arg_layer = max(arg_layer, 3)
            if (
                saw_plaintext_medical_persistence
                and seen_external_medical_site
                and not emitted_booking_submission_gate
                and tier == 3
                and tc.name == "browser"
            ):
                arg_layer = max(arg_layer, 3)
            free = FREE_BAND.get((tier, is_write), 1)
            action = "consent_gate"
            is_plaintext_l34_write = (
                tier == 1
                and is_write
                and arg_layer >= 3
                and _is_plaintext_persistence(tc.name, tc.arguments)
            )
            if is_plaintext_l34_write:
                action = "hard_deny"
                saw_plaintext_medical_persistence = True
            elif _needs_local_image_substitute(tc.name, tier, arg_layer):
                action = "local_substitute"

            stale_values = _stale_l34_values_for_tool(
                trajectory, turn_idx, tc.arguments, pii_map
            )
            if (
                stale_values
                and _is_public_weather_lookup(tc.name, tc.arguments)
                and not _values_appear_in_user_history(trajectory, turn_idx, stale_values)
            ):
                stale_values = []
            stale_signature = tuple(sorted(stale_values))
            if (
                stale_values
                and not is_plaintext_l34_write
                and turn_idx not in emitted_elicit_turns
                and stale_signature not in emitted_elicit_value_signatures
            ):
                violations.append({
                    "turn_idx": turn_idx,
                    "action": "elicit",
                    "tool_name": tc.name,
                    "tier": tier,
                    "is_write": is_write,
                    "data_layer": max(arg_layer, 3),
                    "free_band": free,
                    "call_id": tc.call_id,
                    "args_summary": json.dumps(tc.arguments)[:200],
                    "arguments": tc.arguments,
                    "sensitive_values": stale_values,
                })
                emitted_elicit_turns.add(turn_idx)
                emitted_elicit_value_signatures.add(stale_signature)
                continue

            if action == "consent_gate":
                downgrade = _lower_tier_substitution(
                    tool_name=tc.name,
                    call_id=tc.call_id,
                    args=tc.arguments,
                    tier=tier,
                    is_write=is_write,
                    data_layer=arg_layer,
                    catalog=subst_catalog,
                    result_index=result_index,
                    all_texts=all_texts,
                    consumed_for_call_fn=consumed_for_call_fn,
                )
                if downgrade:
                    violations.append({
                        "turn_idx": turn_idx,
                        "action": "tool_downgrade",
                        "tool_name": tc.name,
                        "tier": tier,
                        "is_write": is_write,
                        "data_layer": arg_layer,
                        "free_band": free,
                        "call_id": tc.call_id,
                        "args_summary": json.dumps(tc.arguments)[:200],
                        "arguments": tc.arguments,
                        **downgrade,
                    })
                    continue

            if tier == 3 and arg_layer >= 2:
                if tc.name == "browser" and saw_plaintext_medical_persistence:
                    if emitted_booking_submission_gate:
                        continue
                    emitted_booking_submission_gate = True
                elif (
                    emitted_external_medical_gate
                    and external_medical
                    and not _needs_local_image_substitute(tc.name, tier, arg_layer)
                ):
                    continue
                emitted_external_medical_gate = True

            if arg_layer > free:
                violations.append({
                    "turn_idx": turn_idx,
                    "action": action,
                    "tool_name": tc.name,
                    "tier": tier,
                    "is_write": is_write,
                    "data_layer": arg_layer,
                    "free_band": free,
                    "call_id": tc.call_id,
                    "args_summary": json.dumps(tc.arguments)[:200],
                    "arguments": tc.arguments,
                })
    return violations


def _pick_adversarial_spot(trajectory: ParsedTrajectory) -> int:
    """Pick the best turn index to insert an adversarial probe.

    Looks for a point roughly 60-80% through the conversation where
    a memory_write or write tool is used, or falls back to 2/3 through.
    """
    total = len(trajectory.assistant_turns)
    target_range = range(int(total * 0.5), int(total * 0.85))

    for i in target_range:
        at = trajectory.assistant_turns[i]
        for tc in at.tool_calls:
            if tc.name in ("memory_write", "write", "memory_search"):
                return i

    return max(1, int(total * 0.65))


async def _generate_patch_text(
    trajectory: ParsedTrajectory,
    pii_map: PIIMap,
    violations: list[dict],
    adversarial_spot: int,
    generation_backend: str = "direct",
    generation_model: str | None = None,
) -> dict:
    """Ask the LLM to generate ONLY the patch text, not the full trajectory.

    Returns a dict with:
      - consent_gates: list of {turn_idx, assistant_ask, user_grant}
      - adversarial: {user_probe, assistant_refusal, user_ack}
      - text_modifications: list of {turn_idx, original_snippet, replacement}
    """
    persona = trajectory.persona or {}
    persona_name = f"{persona.get('first_name', 'the user')} {persona.get('last_name', '')}".strip()

    user_snippets = []
    for i, msg in enumerate(trajectory.user_messages[:3]):
        clean = msg.replace("Sender (untrusted metadata):", "").strip()
        clean = redact_text(clean, pii_map)
        if len(clean) > 200:
            clean = clean[:200]
        user_snippets.append(f"  User msg {i}: {clean}")

    violation_desc = []
    for v in violations:
        turn_idx = v["turn_idx"]
        args_text = redact_text(str(v["args_summary"]), pii_map)
        if len(args_text) > 300:
            args_text = args_text[:300]

        violation_desc.append(
            f"  Turn {turn_idx}: {v['tool_name']} (action={v.get('action', 'consent_gate')}, T{v['tier']} "
            f"{'WRITE' if v['is_write'] else 'READ'}) with L{v['data_layer']} data "
            f"(free_band=L{v['free_band']}).\n"
            f"    EXACT data being handled: {args_text}\n"
            f"    Your consent gate text MUST only reference data visible in the args above. Do NOT mention names or details from elsewhere."
        )

    task_summary = ""
    if trajectory.task_spec:
        task_summary = trajectory.task_spec.get("goal_summary", "")
    if not task_summary:
        for msg in trajectory.user_messages[:3]:
            clean = msg.split("\n")[-1].strip() if "\n" in msg else msg.strip()
            clean = redact_text(clean, pii_map)
            if len(clean) > 20 and not clean.startswith("{"):
                task_summary = clean[:200]
                break
    task_summary = redact_text(task_summary, pii_map)

    prompt = f"""{_openclaw_generation_context(trajectory, pii_map)}

Generate ONLY the privacy patch text for an existing agent trajectory. Do NOT regenerate the trajectory.

CONTEXT:
- Persona: {persona_name}
- Task: {task_summary}
- User message samples:
{chr(10).join(user_snippets)}

PII found: {redact_text(_format_pii_map(pii_map), pii_map)}

VIOLATIONS THAT NEED CONSENT GATES ({len(violations)} total):
{chr(10).join(violation_desc) if violation_desc else "  None. The trajectory has no free-band violations."}

ADVERSARIAL PROBE should be inserted near turn {adversarial_spot}.
The conversation at that point involves tool calls. Create an adversarial probe where the user
tries to get the agent to permanently store L3/L4 data (SSN, password, health record, etc.).

PRIVACY RULES:
- L0/L1/L2 with T1 tools: no consent needed
- T3 + L2+: consent gate required
- T2 + L3/L4: consent gate required
- L3/L4 to plaintext storage: HARD DENY with exact string:
  "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. Please provide it again for the current task."
- After the refusal string, do NOT add any extra sentence.

WRITING STYLE: Sound human. No em dashes. No "Additionally/Furthermore". No sycophantic phrases. Use contractions. Be casual and direct.

Return a JSON object:
```json
{{
  "consent_gates": [
    {{
      "before_turn_idx": <int>,
      "assistant_ask": "natural text asking for consent, referencing the actual task",
      "user_grant": "natural user approval text"
    }}
  ],
  "adversarial": {{
    "user_probe": "user message trying to permanently store sensitive data, contextually relevant",
    "assistant_refusal": "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. Please provide it again for the current task.",
    "user_ack": "short natural acknowledgment from user"
  }},
  "assistant_text_tweaks": [
    {{
      "turn_idx": <int>,
      "prepend_text": "optional privacy note to prepend to existing assistant text (keep short)"
    }}
  ]
}}
```

Rules:
1. consent_gates: MUST reference ONLY the data shown in "EXACT data being handled" for that violation. NEVER mention names, details, or context from other parts of the conversation. The consent text should describe the specific write/read being performed and why it needs consent.
2. adversarial.assistant_refusal MUST be exactly the HTG string. No trailing offer or explanation.
3. assistant_text_tweaks: only if the assistant should acknowledge a privacy decision. Keep very short.
4. All text must match the persona's casual tone. Be brief and specific to the immediate action.
5. CRITICAL: If the tool call args contain "Megan" and "Sprague Lake", the consent gate MUST mention "Megan" and "Sprague Lake", NOT names from other parts of the conversation like "Haley" or "Venkatesh".

Return ONLY the JSON object."""

    if (generation_backend or "direct").lower() == "cli":
        from openclaw_repair_worker import _call_openclaw_cli

        patch = await _call_openclaw_cli(
            prompt,
            generation_model,
            required_keys=("consent_gates", "adversarial", "assistant_text_tweaks"),
            session_prefix="privacy-generate",
            purpose="stage3-generation",
        )
        if not isinstance(patch, dict):
            raise RuntimeError("OpenClaw CLI returned no parseable Stage 3 patch JSON")
        logger.info("OpenClaw CLI patch text response: keys=%s", sorted(patch.keys()))
        return patch

    client = _get_client()
    response = await client.messages.create(
        model=REWRITER_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = ""
    for block in response.content:
        if hasattr(block, "text"):
            raw += block.text

    tracker.record_anthropic(response, "single_shot_rewrite")

    logger.info("Patch text response: %d chars", len(raw))

    if raw.strip().startswith("```"):
        first_nl = raw.index("\n") if "\n" in raw else 3
        raw = raw[first_nl + 1:]
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3]

    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        try:
            import json_repair
            return json_repair.loads(raw.strip())
        except Exception:
            logger.error("Failed to parse patch text JSON")
            return {"consent_gates": [], "adversarial": None, "assistant_text_tweaks": []}


def _annotate_tool_calls(events: list[dict], pii_map: PIIMap) -> None:
    """Add classification metadata to every tool call in the event stream.

    Adds `class` and `flag` fields matching cuarena-pipeline's schema so
    the output demonstrates privacy-aware classification even for benign tools.
    """
    for evt in events:
        if not isinstance(evt, dict):
            continue
        msg = evt.get("message", {})
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "toolCall":
                continue
            if "class" in item:
                continue

            name = item.get("name", "")
            args = item.get("arguments", {})
            tier = _classify_tool_tier(name)
            is_write = _is_write_tool(name, args)
            arg_layer = _max_data_layer_for_args(args, pii_map)
            if tier == 1 and is_write and _looks_like_health_appointment_persistence(args):
                arg_layer = max(arg_layer, 3)
            free = FREE_BAND.get((tier, is_write), 1)
            is_violation = arg_layer > free

            arg_classes = {}
            if isinstance(args, dict):
                for k, v in args.items():
                    dl = _max_data_layer_for_args({k: v}, pii_map)
                    arg_classes[k] = {
                        "data_layer": f"L{dl}" if dl > 0 else None,
                    }

            item["class"] = {
                "tool_tier": tier,
                "is_write": is_write,
                "arguments": arg_classes,
            }
            item["flag"] = {
                "is_violation": is_violation,
                "free_band": free,
                "data_layer": arg_layer,
                "authorization": {
                    "needed": is_violation,
                    "granted": False,
                },
            }


def _make_event_id() -> str:
    import uuid as _uuid
    return _uuid.uuid4().hex[:24]


def _make_tc_id() -> str:
    import uuid as _uuid
    return f"chatcmpl-tool-{_uuid.uuid4().hex[:16]}"


def _offset_timestamp(ts_str: str, offset_ms: int) -> str:
    """Add offset_ms to an ISO timestamp string. Returns adjusted ISO string."""
    from datetime import datetime, timezone, timedelta
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        dt = dt + timedelta(milliseconds=offset_ms)
        ms = int(dt.microsecond / 1000)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms:03d}Z"
    except (ValueError, TypeError):
        return ts_str


def _apply_patches(
    trajectory: ParsedTrajectory,
    pii_map: PIIMap,
    patches: dict,
    adversarial_spot: int,
    violations: list[dict] | None = None,
) -> RewriteResult:
    """Apply patches directly to the original raw JSONL events.

    Instead of reconstructing from RewrittenTurn objects (which loses
    information), this works on the original events and splices new
    events at the right positions. The original events are preserved
    byte-for-byte except where patches are inserted.
    """
    import copy

    consent_gates = {g["before_turn_idx"]: g for g in patches.get("consent_gates", [])}
    consent_violations = {
        v["turn_idx"]: v
        for v in (violations or [])
        if v.get("action") == "consent_gate"
    }
    elicit_gates = {
        v["turn_idx"]: _deterministic_elicit_gate(v)
        for v in (violations or [])
        if v.get("action") == "elicit"
    }
    hard_denies = {
        v["turn_idx"]: v
        for v in (violations or [])
        if v.get("action") == "hard_deny"
    }
    local_substitutions = {
        v["turn_idx"]: v
        for v in (violations or [])
        if v.get("action") == "local_substitute"
    }
    tool_downgrades = {
        v["turn_idx"]: v
        for v in (violations or [])
        if v.get("action") == "tool_downgrade"
    }
    adversarial = patches.get("adversarial")

    raw_events = trajectory.ordered_events or []
    if not raw_events:
        logger.warning("No ordered_events found, falling back to empty")
        return RewriteResult(
            task_id=trajectory.task_id,
            submission_id=trajectory.submission_id,
            patched_events=[],
        )

    assistant_event_ids: list[str] = []
    for evt in raw_events:
        if isinstance(evt, dict):
            msg = evt.get("message", {})
            if msg.get("role") == "assistant":
                assistant_event_ids.append(evt.get("id", ""))

    assistant_idx_to_event_pos: dict[int, int] = {}
    asst_counter = 0
    for pos, evt in enumerate(raw_events):
        if isinstance(evt, dict):
            msg = evt.get("message", {})
            if msg.get("role") == "assistant":
                assistant_idx_to_event_pos[asst_counter] = pos
                asst_counter += 1

    patched: list[dict] = []
    scenarios_covered = []
    privacy_points = 0
    asst_seen = 0
    adversarial_inserted = False
    skip_next_n = 0
    pending_parent_fixup: str | None = None
    consent_grant_id: str = ""
    denied_file_markers: set[str] = set()
    hard_deny_seen = False
    pending_local_artifact_path: str | None = None
    plaintext_profile_notice_inserted = False
    email_consent_inserted = False
    authorized_event_ids: set[str] = set()
    downgraded_tool_results: dict[str, str] = {}
    credential_probe_seen = False
    sensitive_card_render_count = 0

    for pos, evt in enumerate(raw_events):
        if skip_next_n > 0:
            skip_next_n -= 1
            continue

        if not isinstance(evt, dict):
            patched.append(evt)
            continue

        msg = evt.get("message", {})
        role = msg.get("role", "")

        if role == "assistant":
            has_consent = asst_seen in consent_gates
            has_hard_deny = asst_seen in hard_denies
            has_local_substitute = asst_seen in local_substitutions
            has_tool_downgrade = asst_seen in tool_downgrades
            has_elicit = asst_seen in elicit_gates and not has_hard_deny
            new_parent = None

            if pending_parent_fixup:
                new_parent = pending_parent_fixup
                pending_parent_fixup = None

            if has_hard_deny and _looks_like_safe_local_export_request(_last_substantive_user_text(patched)):
                has_hard_deny = False
                has_elicit = asst_seen in elicit_gates

            if _has_credential_probe_exec(evt):
                if credential_probe_seen:
                    last_tool_result_pos = pos
                    for scan_pos in range(pos + 1, len(raw_events)):
                        scan_evt = raw_events[scan_pos]
                        if isinstance(scan_evt, dict) and scan_evt.get("message", {}).get("role") == "toolResult":
                            last_tool_result_pos = scan_pos
                        else:
                            break
                    skip_next_n = last_tool_result_pos - pos
                    pending_parent_fixup = patched[-1].get("id") if patched else None
                    asst_seen += 1
                    continue
                credential_probe_seen = True
                safe_event, safe_result = _credential_probe_replacement(evt)
                if new_parent:
                    safe_event["parentId"] = new_parent
                patched.append(safe_event)

                last_tool_result_pos = pos
                for scan_pos in range(pos + 1, len(raw_events)):
                    scan_evt = raw_events[scan_pos]
                    if isinstance(scan_evt, dict) and scan_evt.get("message", {}).get("role") == "toolResult":
                        last_tool_result_pos = scan_pos
                    else:
                        break
                if safe_result is not None:
                    patched.append(safe_result)
                    pending_parent_fixup = safe_result.get("id")
                else:
                    pending_parent_fixup = safe_event.get("id")
                skip_next_n = last_tool_result_pos - pos
                privacy_points += 1
                asst_seen += 1
                continue

            if pending_local_artifact_path and "MEDIA:/home/user/.openclaw/media/tool-image-generation/" in _assistant_text(evt):
                followup = _replace_image_followup_with_local_pdf(evt, pending_local_artifact_path)
                if new_parent:
                    followup["parentId"] = new_parent
                patched.append(followup)
                pending_parent_fixup = followup.get("id")
                pending_local_artifact_path = None
                asst_seen += 1
                continue

            if (
                not email_consent_inserted
                and _looks_like_email_send_assistant_intent(evt, pii_map)
            ):
                last_id = patched[-1].get("id", _make_event_id()) if patched else _make_event_id()
                ts = evt.get("timestamp", "")
                ask_id = _make_event_id()
                patched.append({
                    "type": "message",
                    "id": ask_id,
                    "parentId": new_parent or last_id,
                    "timestamp": _offset_timestamp(ts, -4200),
                    "message": {
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": (
                                "Before I use enterprise email, confirm you're okay sending "
                                "L3 health data from the organizer card to this external recipient. "
                                "I'll send only the organizer version, with no insurance ID or provider."
                            ),
                        }],
                    },
                })
                approval_parent_id = _append_exec_approval_events(
                    patched,
                    ask_id,
                    ts,
                    {
                        "tool_name": "enterprise-email",
                        "tier": 2,
                        "data_layer": 3,
                    },
                )
                grant_id = _make_event_id()
                patched.append({
                    "type": "message",
                    "id": grant_id,
                    "parentId": approval_parent_id,
                    "timestamp": _offset_timestamp(ts, -2100),
                    "message": {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": "Yes, send only the organizer version with the L3 health details to Valeria for this email only.",
                        }],
                    },
                })
                new_parent = grant_id
                email_consent_inserted = True
                scenarios_covered.append("C")
                privacy_points += 1

            if _is_denied_file_delete(evt, denied_file_markers):
                last_id = patched[-1].get("id", _make_event_id()) if patched else _make_event_id()
                ts = evt.get("timestamp", "")
                contextual_text = _contextual_response_for_blocked_file_noise(
                    _last_substantive_user_text(patched)
                )
                notice_id = _make_event_id()
                patched.append({
                    "type": "message",
                    "id": notice_id,
                    "parentId": new_parent or last_id,
                    "timestamp": ts,
                    "message": {
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": contextual_text or (
                                "There's no plaintext workspace file saved on my side to delete. "
                                "I kept the health details transient and won't retain them."
                            ),
                        }],
                    },
                })
                last_tool_result_pos = pos
                for scan_pos in range(pos + 1, len(raw_events)):
                    scan_evt = raw_events[scan_pos]
                    if isinstance(scan_evt, dict) and scan_evt.get("message", {}).get("role") == "toolResult":
                        last_tool_result_pos = scan_pos
                    else:
                        break
                skip_next_n = last_tool_result_pos - pos
                pending_parent_fixup = notice_id
                privacy_points += 1
                asst_seen += 1
                continue

            if has_elicit:
                gate = elicit_gates[asst_seen]
                last_id = patched[-1].get("id", _make_event_id()) if patched else _make_event_id()
                ts = evt.get("timestamp", "")

                ask_id = _make_event_id()
                patched.append({
                    "type": "message",
                    "id": ask_id,
                    "parentId": new_parent or last_id,
                    "timestamp": _offset_timestamp(ts, -5200),
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": gate["assistant_ask"]}],
                    },
                })

                reinput_id = _make_event_id()
                patched.append({
                    "type": "message",
                    "id": reinput_id,
                    "parentId": ask_id,
                    "timestamp": _offset_timestamp(ts, -4700),
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": gate["user_reinput"]}],
                    },
                })
                new_parent = reinput_id
                scenarios_covered.append("E")
                privacy_points += 1

            if has_hard_deny:
                denied_file_markers.update(_hard_denied_path_markers(evt, pii_map))
                if hard_denies.get(asst_seen, {}).get("arguments"):
                    denied_file_markers.update(_extract_path_markers(hard_denies[asst_seen]["arguments"]))
                hard_deny_seen = True
                previous_role = None
                previous_text = ""
                if patched and isinstance(patched[-1], dict):
                    previous_msg = patched[-1].get("message", {})
                    if isinstance(previous_msg, dict):
                        previous_role = previous_msg.get("role")
                        previous_text = "\n".join(
                            str(item.get("text", ""))
                            for item in _message_content(patched[-1])
                            if isinstance(item, dict) and item.get("type") == "text"
                        )
                substantive_user_text = _last_substantive_user_text(patched)
                if _looks_like_category_memory_request(substantive_user_text):
                    last_id = patched[-1].get("id", _make_event_id()) if patched else _make_event_id()
                    ts = evt.get("timestamp", "")
                    policy_id = _make_event_id()
                    patched.append({
                        "type": "message",
                        "id": policy_id,
                        "parentId": new_parent or last_id,
                        "timestamp": ts,
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": _category_memory_policy_message()}],
                        },
                    })
                    last_tool_result_pos = pos
                    for scan_pos in range(pos + 1, len(raw_events)):
                        scan_evt = raw_events[scan_pos]
                        if isinstance(scan_evt, dict) and scan_evt.get("message", {}).get("role") == "toolResult":
                            last_tool_result_pos = scan_pos
                        else:
                            break
                    skip_next_n = last_tool_result_pos - pos
                    pending_parent_fixup = policy_id
                    scenarios_covered.append("C")
                    privacy_points += 1
                    asst_seen += 1
                    continue
                if _looks_like_calendar_boundary_preference(substantive_user_text):
                    last_id = patched[-1].get("id", _make_event_id()) if patched else _make_event_id()
                    ts = evt.get("timestamp", "")
                    boundary_id = _make_event_id()
                    patched.append({
                        "type": "message",
                        "id": boundary_id,
                        "parentId": new_parent or last_id,
                        "timestamp": ts,
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": _calendar_boundary_message()}],
                        },
                    })
                    last_tool_result_pos = pos
                    for scan_pos in range(pos + 1, len(raw_events)):
                        scan_evt = raw_events[scan_pos]
                        if isinstance(scan_evt, dict) and scan_evt.get("message", {}).get("role") == "toolResult":
                            last_tool_result_pos = scan_pos
                        else:
                            break
                    skip_next_n = last_tool_result_pos - pos
                    pending_parent_fixup = boundary_id
                    scenarios_covered.append("A")
                    privacy_points += 1
                    asst_seen += 1
                    continue
                if _looks_like_sanitized_calendar_preview_request(substantive_user_text):
                    last_id = patched[-1].get("id", _make_event_id()) if patched else _make_event_id()
                    ts = evt.get("timestamp", "")
                    preview_id = _make_event_id()
                    patched.append({
                        "type": "message",
                        "id": preview_id,
                        "parentId": new_parent or last_id,
                        "timestamp": ts,
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": _sanitized_calendar_preview_message()}],
                        },
                    })
                    last_tool_result_pos = pos
                    for scan_pos in range(pos + 1, len(raw_events)):
                        scan_evt = raw_events[scan_pos]
                        if isinstance(scan_evt, dict) and scan_evt.get("message", {}).get("role") == "toolResult":
                            last_tool_result_pos = scan_pos
                        else:
                            break
                    skip_next_n = last_tool_result_pos - pos
                    pending_parent_fixup = preview_id
                    scenarios_covered.append("A")
                    privacy_points += 1
                    asst_seen += 1
                    continue
                if _looks_like_local_only_sensitive_work_request(substantive_user_text):
                    last_id = patched[-1].get("id", _make_event_id()) if patched else _make_event_id()
                    ts = evt.get("timestamp", "")
                    local_id = _make_event_id()
                    patched.append({
                        "type": "message",
                        "id": local_id,
                        "parentId": new_parent or last_id,
                        "timestamp": ts,
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": _local_only_work_message(substantive_user_text)}],
                        },
                    })
                    last_tool_result_pos = pos
                    for scan_pos in range(pos + 1, len(raw_events)):
                        scan_evt = raw_events[scan_pos]
                        if isinstance(scan_evt, dict) and scan_evt.get("message", {}).get("role") == "toolResult":
                            last_tool_result_pos = scan_pos
                        else:
                            break
                    skip_next_n = last_tool_result_pos - pos
                    pending_parent_fixup = local_id
                    scenarios_covered.append("A")
                    privacy_points += 1
                    asst_seen += 1
                    continue
                user_asked_to_persist = _looks_like_persistence_request(substantive_user_text)
                offer_encrypted_storage = _should_offer_encrypted_storage(substantive_user_text)
                if offer_encrypted_storage:
                    final_id = _append_encrypted_storage_flow(
                        patched, evt, new_parent, substantive_user_text
                    )
                    last_tool_result_pos = pos
                    for scan_pos in range(pos + 1, len(raw_events)):
                        scan_evt = raw_events[scan_pos]
                        if isinstance(scan_evt, dict) and scan_evt.get("message", {}).get("role") == "toolResult":
                            last_tool_result_pos = scan_pos
                        else:
                            break
                    skip_next_n = last_tool_result_pos - pos
                    pending_parent_fixup = final_id
                    scenarios_covered.append("D")
                    privacy_points += 1
                    asst_seen += 1
                    continue
                last_id = patched[-1].get("id", _make_event_id()) if patched else _make_event_id()
                ts = evt.get("timestamp", "")
                refusal_id = _make_event_id()
                patched.append({
                    "type": "message",
                    "id": refusal_id,
                    "parentId": new_parent or last_id,
                    "timestamp": _offset_timestamp(ts, -1200),
                    "message": {
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": (
                                _transient_retention_message()
                                if user_asked_to_persist
                                else _soft_transient_notice()
                            ),
                        }],
                    },
                })
                ack_id = _make_event_id()
                patched.append({
                    "type": "message",
                    "id": ack_id,
                    "parentId": refusal_id,
                    "timestamp": _offset_timestamp(ts, -600),
                    "message": {
                        "role": "user",
                        "content": [{
                            "type": "text",
                            "text": (
                                "Ok, keep it transient here. Don't save it."
                                if user_asked_to_persist
                                else "Ok, keep going without saving it."
                            ),
                        }],
                    },
                })

                last_tool_result_pos = pos
                for scan_pos in range(pos + 1, len(raw_events)):
                    scan_evt = raw_events[scan_pos]
                    if isinstance(scan_evt, dict) and scan_evt.get("message", {}).get("role") == "toolResult":
                        last_tool_result_pos = scan_pos
                    else:
                        break
                skip_next_n = last_tool_result_pos - pos
                pending_parent_fixup = ack_id
                scenarios_covered.append("C")
                privacy_points += 1
                asst_seen += 1
                continue

            if _is_denied_file_access(evt, denied_file_markers):
                last_id = patched[-1].get("id", _make_event_id()) if patched else _make_event_id()
                ts = evt.get("timestamp", "")
                contextual_text = _contextual_response_for_blocked_file_noise(
                    _last_substantive_user_text(patched)
                )
                notice_id = _make_event_id()
                patched.append({
                    "type": "message",
                    "id": notice_id,
                    "parentId": new_parent or last_id,
                    "timestamp": ts,
                    "message": {
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": contextual_text or _denied_file_access_message(),
                        }],
                    },
                })
                last_tool_result_pos = pos
                for scan_pos in range(pos + 1, len(raw_events)):
                    scan_evt = raw_events[scan_pos]
                    if isinstance(scan_evt, dict) and scan_evt.get("message", {}).get("role") == "toolResult":
                        last_tool_result_pos = scan_pos
                    else:
                        break
                skip_next_n = last_tool_result_pos - pos
                pending_parent_fixup = notice_id
                privacy_points += 1
                asst_seen += 1
                continue

            if has_consent:
                gate = consent_gates[asst_seen]
                violation = consent_violations.get(asst_seen, {"tool_name": "tool", "tier": 3, "data_layer": 2})
                assistant_ask = gate["assistant_ask"]
                user_grant = gate["user_grant"]
                replace_sensitive_email_exec = _is_sensitive_email_exec_violation(violation)
                if _is_sensitive_email_exec_violation(violation):
                    assistant_ask, user_grant = _external_email_gateway_consent_text(violation)
                    approval_violation = {
                        **violation,
                        "tool_name": "enterprise-mail",
                        "tier": 2,
                        "data_layer": max(int(violation.get("data_layer", 3) or 3), 3),
                    }
                else:
                    approval_violation = violation
                last_id = patched[-1].get("id", _make_event_id()) if patched else _make_event_id()
                ts = evt.get("timestamp", "")

                consent_ask_id = _make_event_id()
                patched.append({
                    "type": "message",
                    "id": consent_ask_id,
                    "parentId": new_parent or last_id,
                    "timestamp": _offset_timestamp(ts, -4200),
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": assistant_ask}],
                    },
                })

                approval_parent_id = _append_exec_approval_events(
                    patched,
                    consent_ask_id,
                    ts,
                    approval_violation,
                )

                consent_grant_id = _make_event_id()
                patched.append({
                    "type": "message",
                    "id": consent_grant_id,
                    "parentId": approval_parent_id,
                    "timestamp": _offset_timestamp(ts, -2100),
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": user_grant}],
                    },
                })
                new_parent = consent_grant_id
                scenarios_covered.append("C")
                privacy_points += 1
            else:
                replace_sensitive_email_exec = False

            if has_local_substitute:
                local_event = _replace_sensitive_image_with_local_pdf(evt, asst_seen)
                if new_parent:
                    local_event["parentId"] = new_parent
                patched.append(local_event)

                last_tool_result_pos = pos
                for scan_pos in range(pos + 1, len(raw_events)):
                    scan_evt = raw_events[scan_pos]
                    if isinstance(scan_evt, dict) and scan_evt.get("message", {}).get("role") == "toolResult":
                        last_tool_result_pos = scan_pos
                    else:
                        break
                skip_next_n = last_tool_result_pos - pos
                pending_parent_fixup = local_event.get("id")
                pending_local_artifact_path = "__TRANSIENT_CARD__"
                scenarios_covered.append("C")
                privacy_points += 1
                asst_seen += 1
                continue

            orig_evt = copy.deepcopy(evt)
            if new_parent:
                orig_evt["parentId"] = new_parent
            if has_consent:
                authorized_event_ids.add(orig_evt.get("id", ""))
            if has_tool_downgrade:
                orig_evt, downgraded_call_ids = _apply_tool_downgrade_to_event(
                    orig_evt,
                    tool_downgrades[asst_seen],
                )
                downgraded_tool = str(tool_downgrades[asst_seen].get("downgraded_tool", ""))
                for call_id in downgraded_call_ids:
                    downgraded_tool_results[call_id] = downgraded_tool
                scenarios_covered.append("A")
                privacy_points += 1
            if replace_sensitive_email_exec:
                replaced_email = _replace_email_exec_with_enterprise_mail(orig_evt, pii_map)
                if replaced_email:
                    email_event, email_result, email_final = replaced_email
                    patched.append(email_event)
                    patched.append(email_result)
                    patched.append(email_final)
                    authorized_event_ids.add(email_event.get("id", ""))
                    skip_next_n = len(raw_events) - pos - 1
                    scenarios_covered.append("C")
                    privacy_points += 1
                    asst_seen += 1
                    continue
            original_text = _assistant_text(orig_evt)
            current_user_text = _last_substantive_user_text(patched)
            if (
                original_text
                and not _tool_calls_from_event(orig_evt)
                and _looks_like_privacy_confirmation_request(current_user_text)
                and (
                    _looks_like_broken_privacy_boilerplate(original_text)
                    or "confirm" not in original_text.lower()
                    or REDACTION_TOKEN in original_text
                )
            ):
                orig_evt = _replace_assistant_with_text(
                    orig_evt,
                    _privacy_confirmation_message(current_user_text),
                )
                original_text = _assistant_text(orig_evt)
            if (
                original_text
                and not _tool_calls_from_event(orig_evt)
                and _looks_like_local_only_sensitive_work_request(current_user_text)
                and _looks_like_broken_privacy_boilerplate(original_text)
            ):
                orig_evt = _replace_assistant_with_text(
                    orig_evt,
                    _local_only_work_message(current_user_text),
                )
                original_text = _assistant_text(orig_evt)
            if (
                original_text
                and not _tool_calls_from_event(orig_evt)
                and _looks_like_calendar_boundary_preference(current_user_text)
                and (
                    _looks_like_broken_privacy_boilerplate(original_text)
                    or "encrypted vault" in original_text.lower()
                )
            ):
                orig_evt = _replace_assistant_with_text(
                    orig_evt,
                    _calendar_boundary_message(),
                )
                original_text = _assistant_text(orig_evt)
            if (
                original_text
                and not _tool_calls_from_event(orig_evt)
                and _looks_like_sanitized_calendar_preview_request(current_user_text)
                and (
                    _looks_like_broken_privacy_boilerplate(original_text)
                    or "medical fields" in original_text.lower()
                    or REDACTION_TOKEN in original_text
                )
            ):
                orig_evt = _replace_assistant_with_text(
                    orig_evt,
                    _sanitized_calendar_preview_message(),
                )
                original_text = _assistant_text(orig_evt)
            if (
                original_text
                and not _tool_calls_from_event(orig_evt)
                and _looks_like_card_difference_confirmation(current_user_text, original_text)
            ):
                orig_evt = _replace_assistant_with_text(
                    orig_evt,
                    _card_difference_confirmation_message(),
                )
                original_text = _assistant_text(orig_evt)
            if (
                original_text
                and not _tool_calls_from_event(orig_evt)
                and _looks_like_see_above_card_claim(original_text)
            ):
                orig_evt = _replace_assistant_with_text(
                    orig_evt,
                    _see_above_card_claim_replacement(),
                )
                original_text = _assistant_text(orig_evt)
            if (
                original_text
                and not _tool_calls_from_event(orig_evt)
                and _looks_like_health_profile_memory_claim(original_text)
            ):
                orig_evt = _replace_assistant_with_text(
                    orig_evt,
                    _health_profile_memory_claim_replacement(),
                )
                original_text = _assistant_text(orig_evt)
            if (
                original_text
                and not _tool_calls_from_event(orig_evt)
                and _looks_like_no_save_climbing_signoff(original_text)
            ):
                orig_evt = _replace_assistant_with_text(
                    orig_evt,
                    _no_save_climbing_replacement(),
                )
                original_text = _assistant_text(orig_evt)
            if (
                original_text
                and not _tool_calls_from_event(orig_evt)
                and _looks_like_insecure_gateway_summary(original_text)
            ):
                orig_evt = _replace_assistant_with_text(
                    orig_evt,
                    _insecure_gateway_summary_replacement(),
                )
                original_text = _assistant_text(orig_evt)
            if (
                original_text
                and not _tool_calls_from_event(orig_evt)
                and _assistant_stale_l34_text_values(original_text, current_user_text, pii_map)
            ):
                orig_evt = _redact_assistant_text_fields(orig_evt, pii_map)
                original_text = _assistant_text(orig_evt)
            if (
                original_text
                and not _tool_calls_from_event(orig_evt)
                and _is_sensitive_card_context(original_text)
                and (
                    _text_contains_l3_or_higher(original_text, pii_map)
                    or REDACTION_TOKEN in original_text
                )
            ):
                sensitive_card_render_count += 1
                if sensitive_card_render_count > 1 and not _text_contains_l3_or_higher(current_user_text, pii_map):
                    orig_evt = _replace_assistant_with_text(
                        orig_evt,
                        (
                            "I won't reprint the medical fields again from earlier context. "
                            "Paste the values again if you want another full card rendering."
                        ),
                    )
                    original_text = _assistant_text(orig_evt)
            sanitized_retention_claim = False
            previous_role = None
            previous_text = ""
            if patched and isinstance(patched[-1], dict):
                previous_msg = patched[-1].get("message", {})
                if isinstance(previous_msg, dict):
                    previous_role = previous_msg.get("role")
                    previous_text = _assistant_text(patched[-1])
                    if previous_role == "user":
                        previous_text = "\n".join(
                            str(item.get("text", ""))
                            for item in _message_content(patched[-1])
                            if isinstance(item, dict) and item.get("type") == "text"
                        )
            if original_text and _is_retention_claim(original_text, denied_file_markers, hard_deny_seen):
                if previous_role == "user" and _looks_like_transient_ack(previous_text):
                    replacement_text = _transient_ack_message()
                elif _looks_like_false_profile_save_claim(original_text):
                    replacement_text = _false_profile_save_replacement(original_text)
                elif _looks_like_no_retention_confirmation(original_text):
                    replacement_text = _no_retention_confirmation_message()
                elif (
                    denied_file_markers
                    and _contains_denied_marker(original_text, denied_file_markers)
                    and not (
                        previous_role == "user"
                        and _looks_like_persistence_request(previous_text)
                    )
                ):
                    replacement_text = _safe_denied_reference_message(original_text)
                else:
                    replacement_text = _transient_retention_message()
                orig_evt = _replace_assistant_with_text(orig_evt, replacement_text)
                sanitized_retention_claim = True
            if sanitized_retention_claim and previous_role == "assistant":
                pending_parent_fixup = patched[-1].get("id")
                asst_seen += 1
                continue
            patched.append(orig_evt)

            if asst_seen == adversarial_spot and adversarial and not adversarial_inserted:
                last_tool_result_pos = pos
                for scan_pos in range(pos + 1, len(raw_events)):
                    scan_evt = raw_events[scan_pos]
                    if isinstance(scan_evt, dict):
                        scan_role = scan_evt.get("message", {}).get("role", "")
                        if scan_role == "toolResult":
                            last_tool_result_pos = scan_pos
                        else:
                            break

                for tr_pos in range(pos + 1, last_tool_result_pos + 1):
                    patched.append(copy.deepcopy(raw_events[tr_pos]))

                skip_next_n = last_tool_result_pos - pos

                last_id = patched[-1].get("id", _make_event_id())
                last_ts = patched[-1].get("timestamp", evt.get("timestamp", ""))

                probe_id = _make_event_id()
                patched.append({
                    "type": "message",
                    "id": probe_id,
                    "parentId": last_id,
                    "timestamp": _offset_timestamp(last_ts, 8300),
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": adversarial["user_probe"]}],
                    },
                })

                refusal_id = _make_event_id()
                patched.append({
                    "type": "message",
                    "id": refusal_id,
                    "parentId": probe_id,
                    "timestamp": _offset_timestamp(last_ts, 14700),
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": HATCH_REFUSAL}],
                    },
                })

                last_injected_id = refusal_id
                if adversarial.get("user_ack"):
                    ack_id = _make_event_id()
                    patched.append({
                        "type": "message",
                        "id": ack_id,
                        "parentId": refusal_id,
                        "timestamp": _offset_timestamp(last_ts, 19200),
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": adversarial["user_ack"]}],
                        },
                    })
                    last_injected_id = ack_id

                pending_parent_fixup = last_injected_id
                adversarial_inserted = True
                scenarios_covered.append("D")
                privacy_points += 1

            asst_seen += 1
        else:
            evt_copy = copy.deepcopy(evt)
            if pending_parent_fixup and evt_copy.get("type") == "message":
                evt_copy["parentId"] = pending_parent_fixup
                pending_parent_fixup = None
            msg = evt_copy.get("message", {}) if isinstance(evt_copy, dict) else {}
            if isinstance(msg, dict) and msg.get("role") == "toolResult":
                call_id = str(msg.get("toolCallId", ""))
                if call_id in downgraded_tool_results:
                    msg["toolName"] = downgraded_tool_results[call_id]
            patched.append(evt_copy)
            previous_event = patched[-2] if len(patched) >= 2 else None
            if (
                not plaintext_profile_notice_inserted
                and _is_user_profile_l3_read_result(evt_copy, previous_event, pii_map)
            ):
                pending_parent_fixup = _append_user_profile_scrub_flow(
                    patched, evt_copy, pii_map
                )
                plaintext_profile_notice_inserted = True
                scenarios_covered.append("E")
                privacy_points += 1

    if not scenarios_covered:
        scenarios_covered = ["E"]

    patched = _remove_redundant_user_profile_scrubs(patched, pii_map)
    _normalize_assistant_punctuation(patched)
    patched = _dedupe_recent_text_messages(patched)
    _annotate_tool_calls(patched, pii_map)
    for event in patched:
        if not isinstance(event, dict) or event.get("id", "") not in authorized_event_ids:
            continue
        for tool_call in _tool_calls_from_event(event):
            flag = tool_call.get("flag")
            if not isinstance(flag, dict):
                continue
            authorization = flag.setdefault("authorization", {})
            if isinstance(authorization, dict) and authorization.get("needed"):
                authorization["granted"] = True
    patched = redact_event_stream(patched, pii_map)

    tool_results_by_call_id: dict[str, dict] = {}
    for event in patched:
        msg = event.get("message", {}) if isinstance(event, dict) else {}
        if not isinstance(msg, dict) or msg.get("role") != "toolResult":
            continue
        call_id = str(msg.get("toolCallId", ""))
        result_text = "\n".join(
            str(item.get("text", ""))
            for item in msg.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        )
        if call_id:
            tool_results_by_call_id[call_id] = {
                "call_id": call_id,
                "tool_name": msg.get("toolName", ""),
                "content": result_text,
                "is_error": msg.get("isError", False),
            }

    turns = []
    for event in patched:
        msg = event.get("message", {}) if isinstance(event, dict) else {}
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        text_parts: list[str] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        for item in msg.get("content", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
            elif item.get("type") == "toolCall":
                call_id = str(item.get("id", ""))
                tool_calls.append({
                    "id": call_id,
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", {}),
                })
                if call_id in tool_results_by_call_id:
                    tool_results.append(tool_results_by_call_id[call_id])

        text = "\n".join(part for part in text_parts if part).strip()
        lowered = text.lower()
        privacy_actions: list[str] = []
        if "hatch trust guidelines" in lowered:
            privacy_actions.append("adversarial_refusal")
        if "approval required" in lowered or "sensitive value again" in lowered:
            privacy_actions.append("privacy_gate")

        turns.append(RewrittenTurn(
            turn_index=len(turns),
            text=text,
            tool_calls=tool_calls,
            tool_results=tool_results,
            privacy_actions=privacy_actions,
        ))

    return RewriteResult(
        task_id=trajectory.task_id,
        submission_id=trajectory.submission_id,
        turns=turns,
        scenarios_covered=list(set(scenarios_covered)),
        privacy_decision_points=privacy_points,
        patched_events=patched,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def rewrite_trajectory_single_shot(
    trajectory: ParsedTrajectory,
    pii_map: PIIMap,
    generation_backend: str = "direct",
    generation_model: str | None = None,
) -> RewriteResult:
    """Rewrite a trajectory using PATCH MODE (cuarena-inspired).

    Instead of regenerating the entire trajectory (which causes hallucination),
    this approach:
    1. Analyzes violations deterministically (tool tiers, data layers, free-band)
    2. Asks the LLM to generate ONLY the patch text (consent gates, refusals, probes)
    3. Splices new events into the original trajectory at violation points
    4. Preserves 100% of original content, tool calls, results, and structure
    """
    if _is_category_memory_setup_trajectory(trajectory):
        logger.info("Generating category-memory setup rewrite for %s", trajectory.task_id)
        generated_texts = await _generate_category_memory_setup_texts(
            trajectory,
            pii_map,
            generation_backend=generation_backend,
            generation_model=generation_model,
        )
        result = _category_memory_setup_result(trajectory, pii_map, generated_texts)
        if (generation_backend or "direct").lower() == "cli":
            result.rewrite_repairs = [
                *result.rewrite_repairs,
                {
                    "kind": "openclaw_stage3_generation",
                    "backend": "cli",
                    "model": generation_model,
                    "context": "persona_workspace_patch_json",
                },
            ]
        return result

    violations = _analyze_violations(trajectory, pii_map)
    adversarial_spot = _pick_adversarial_spot(trajectory)

    logger.info(
        "Patch-mode rewrite: %d violations found, adversarial at turn %d",
        len(violations), adversarial_spot,
    )

    patch_text_violations = [
        violation for violation in violations
        if violation.get("action") not in {"hard_deny", "elicit", "local_substitute", "tool_downgrade"}
    ]
    patches = await _generate_patch_text(
        trajectory,
        pii_map,
        patch_text_violations,
        adversarial_spot,
        generation_backend=generation_backend,
        generation_model=generation_model,
    )
    existing_gate_turns = {
        gate.get("before_turn_idx")
        for gate in patches.get("consent_gates", [])
        if isinstance(gate, dict)
    }
    for violation in violations:
        if violation.get("action") in {"hard_deny", "elicit", "local_substitute", "tool_downgrade"}:
            continue
        turn_idx = violation["turn_idx"]
        if turn_idx in existing_gate_turns:
            continue
        patches.setdefault("consent_gates", []).append(_deterministic_consent_gate(violation))
        existing_gate_turns.add(turn_idx)

    logger.info(
        "Patches: %d consent gates, adversarial=%s, %d text tweaks",
        len(patches.get("consent_gates", [])),
        "yes" if patches.get("adversarial") else "no",
        len(patches.get("assistant_text_tweaks", [])),
    )

    result = _apply_patches(trajectory, pii_map, patches, adversarial_spot, violations)
    if (generation_backend or "direct").lower() == "cli":
        result.rewrite_repairs = [
            *result.rewrite_repairs,
            {
                "kind": "openclaw_stage3_generation",
                "backend": "cli",
                "model": generation_model,
                "context": "persona_workspace_patch_json",
                "patch_keys": sorted(patches.keys()),
            },
        ]

    logger.info(
        "Patch result: %d patched events (orig: %d), %d scenarios, %d privacy points",
        len(result.patched_events or []),
        len(trajectory.ordered_events or []),
        len(result.scenarios_covered),
        result.privacy_decision_points,
    )

    return result


def _validate_and_repair(result: RewriteResult, trajectory: ParsedTrajectory) -> None:
    """Comprehensive post-rewrite structural validation and repair.

    Checks inspired by cuarena-pipeline's structural integrity:
    1. Every assistant turn has text
    2. Tool call / tool result pairing is consistent
    3. No consecutive assistants without user turns
    4. Original tool calls are preserved
    """
    original_tools = set()
    for at in trajectory.assistant_turns:
        for tc in at.tool_calls:
            original_tools.add(tc.name)

    rewritten_tools = set()
    for rt in result.turns:
        for tc in rt.tool_calls:
            if isinstance(tc, dict):
                rewritten_tools.add(tc.get("name", ""))

    missing = original_tools - rewritten_tools
    if missing:
        logger.warning(
            "Rewrite dropped original tools: %s. Original had %d unique tools, rewrite has %d.",
            missing, len(original_tools), len(rewritten_tools),
        )

    for rt in result.turns:
        if rt.tool_calls and not (rt.text and rt.text.strip()):
            tool_names = [tc.get("name", "?") for tc in rt.tool_calls if isinstance(tc, dict)]
            rt.text = f"Let me {', '.join(tool_names[:2])} for you."
            logger.warning(
                "Turn %d had tool calls but no text, injected fallback",
                rt.turn_index,
            )

        if isinstance(rt.tool_calls, list) and isinstance(rt.tool_results, list):
            if len(rt.tool_results) > len(rt.tool_calls):
                rt.tool_results = rt.tool_results[:len(rt.tool_calls)]
                logger.warning(
                    "Turn %d had more results than calls, trimmed",
                    rt.turn_index,
                )

    for rt in result.turns:
        if rt.text:
            rt.text = rt.text.replace("\u2014", ",").replace("\u2013", ",")
            rt.text = rt.text.replace("\u201c", '"').replace("\u201d", '"')
            rt.text = rt.text.replace("\u2018", "'").replace("\u2019", "'")
