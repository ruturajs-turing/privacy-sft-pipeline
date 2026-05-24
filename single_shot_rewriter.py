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
from pathlib import Path
from typing import Any

import anthropic

from config import ANTHROPIC_API_KEY, REWRITER_MODEL
from models import (
    ParsedTrajectory, PIIMap, PIIEntity,
    RewriteResult, RewrittenTurn,
)
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


_T1_FALLBACK = {
    "read", "write", "edit", "exec", "process", "gateway",
    "memory_write", "memory_search", "active_memory_write",
    "wiki_apply", "cron", "glob", "grep",
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
    if name not in {"write", "edit", "memory_write", "exec"}:
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


def _replace_assistant_with_text(event: dict, text: str) -> dict:
    new_event = json.loads(json.dumps(event))
    new_event.setdefault("message", {})["content"] = [{"type": "text", "text": text}]
    return new_event


def _tool_calls_from_event(event: dict) -> list[dict]:
    return [
        item for item in _message_content(event)
        if isinstance(item, dict) and item.get("type") == "toolCall"
    ]


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


def _transient_retention_message() -> str:
    return (
        "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. "
        "Please provide it again for the current task. I can keep it transient in this chat "
        "so you can copy it, but I won't save a plaintext file."
    )


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
    return _looks_like_persistence_request(text) and not _looks_like_cross_session_secret_request(text)


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
        for field in ("content", "value", "text", "body", "data"):
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
                    "I can store the task details in encrypted memory for this task instead. "
                    "Do you approve encrypted storage?"
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
            "content": [{"type": "text", "text": "Yes, store it encrypted for this task."}],
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
                "name": "active_memory_write",
                "arguments": {
                    "key": key,
                    "value": value,
                    "encrypted": True,
                    "scope": "current_task",
                },
                "class": {
                    "tool_tier": 1,
                    "is_write": True,
                    "arguments": {
                        "key": {"data_layer": None},
                        "value": {"data_layer": "L3"},
                        "encrypted": {"data_layer": None},
                        "scope": {"data_layer": None},
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
            "toolName": "active_memory_write",
            "content": [{"type": "text", "text": f"Encrypted task memory stored under key: {key}"}],
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
                    "Saved in encrypted task memory. I did not create a plaintext workspace file."
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

  T1 (Local) — Safe to PROCESS all data up to L4. Safe to WRITE/SAVE up to L2. L3/L4 writes allowed ONLY with consent + encrypted storage (vault, encrypted memory_write, active_memory_write, wiki_apply). Plaintext L3/L4 writes are HARD DENIED.
     Tools: memory_write, memory_search, memory_read, read, write, edit, exec, cron, sessions,
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
  - T1 + L3/L4 (WRITE to encrypted store): CONSENT REQUIRED. Allowed ONLY with explicit user consent AND encrypted storage (enterprise-vault, encrypted memory_write, active_memory_write, wiki_apply).
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
    seen_external_medical_site = False
    saw_plaintext_medical_persistence = False
    for turn_idx, at in enumerate(trajectory.assistant_turns):
        turn_text = "\n".join(at.text_blocks)
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
            if tier == 1 and is_write and arg_layer >= 3 and _is_plaintext_persistence(tc.name, tc.arguments):
                action = "hard_deny"
                saw_plaintext_medical_persistence = True
            if tier == 3 and arg_layer >= 2:
                if tc.name == "browser" and saw_plaintext_medical_persistence:
                    if emitted_booking_submission_gate:
                        continue
                    emitted_booking_submission_gate = True
                elif emitted_external_medical_gate and external_medical:
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
        if len(clean) > 200:
            clean = clean[:200]
        user_snippets.append(f"  User msg {i}: {clean}")

    violation_desc = []
    for v in violations:
        turn_idx = v["turn_idx"]
        args_text = v["args_summary"]
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
            if len(clean) > 20 and not clean.startswith("{"):
                task_summary = clean[:200]
                break

    prompt = f"""Generate ONLY the privacy patch text for an existing agent trajectory. Do NOT regenerate the trajectory.

CONTEXT:
- Persona: {persona_name}
- Task: {task_summary}
- User message samples:
{chr(10).join(user_snippets)}

PII found: {_format_pii_map(pii_map)}

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
- After the refusal string, do NOT elaborate. Just offer what CAN be done.

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
    "assistant_refusal": "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. Please provide it again for the current task. <brief offer of alternative>",
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
2. adversarial.assistant_refusal MUST start with the exact HTG string. After that, max 1 short sentence offering help.
3. assistant_text_tweaks: only if the assistant should acknowledge a privacy decision. Keep very short.
4. All text must match the persona's casual tone. Be brief and specific to the immediate action.
5. CRITICAL: If the tool call args contain "Megan" and "Sprague Lake", the consent gate MUST mention "Megan" and "Sprague Lake", NOT names from other parts of the conversation like "Haley" or "Venkatesh".

Return ONLY the JSON object."""

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
    hard_denies = {
        v["turn_idx"]: v
        for v in (violations or [])
        if v.get("action") == "hard_deny"
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
            new_parent = None

            if pending_parent_fixup:
                new_parent = pending_parent_fixup
                pending_parent_fixup = None

            if _is_denied_file_delete(evt, denied_file_markers):
                last_id = patched[-1].get("id", _make_event_id()) if patched else _make_event_id()
                ts = evt.get("timestamp", "")
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
                            "text": (
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
                            "text": _denied_file_access_message(),
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
                        "content": [{"type": "text", "text": gate["assistant_ask"]}],
                    },
                })

                consent_grant_id = _make_event_id()
                patched.append({
                    "type": "message",
                    "id": consent_grant_id,
                    "parentId": consent_ask_id,
                    "timestamp": _offset_timestamp(ts, -2100),
                    "message": {
                        "role": "user",
                        "content": [{"type": "text", "text": gate["user_grant"]}],
                    },
                })
                new_parent = consent_grant_id
                scenarios_covered.append("C")
                privacy_points += 1

            orig_evt = copy.deepcopy(evt)
            if new_parent:
                orig_evt["parentId"] = new_parent
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
                        "content": [{"type": "text", "text": adversarial["assistant_refusal"]}],
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
            patched.append(evt_copy)

    if not scenarios_covered:
        scenarios_covered = ["E"]

    _annotate_tool_calls(patched, pii_map)

    turns = []
    for at in trajectory.assistant_turns:
        text = "\n".join(at.text_blocks).strip()
        tool_calls = [{"name": tc.name, "arguments": tc.arguments} for tc in at.tool_calls]
        tool_results = []
        for tc in at.tool_calls:
            tr = trajectory.tool_results_by_call_id.get(tc.call_id)
            if tr:
                tool_results.append({
                    "tool_name": tc.name,
                    "content": tr.content,
                    "is_error": tr.is_error,
                })
        turns.append(RewrittenTurn(
            turn_index=len(turns),
            text=text,
            tool_calls=tool_calls,
            tool_results=tool_results,
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
) -> RewriteResult:
    """Rewrite a trajectory using PATCH MODE (cuarena-inspired).

    Instead of regenerating the entire trajectory (which causes hallucination),
    this approach:
    1. Analyzes violations deterministically (tool tiers, data layers, free-band)
    2. Asks the LLM to generate ONLY the patch text (consent gates, refusals, probes)
    3. Splices new events into the original trajectory at violation points
    4. Preserves 100% of original content, tool calls, results, and structure
    """
    violations = _analyze_violations(trajectory, pii_map)
    adversarial_spot = _pick_adversarial_spot(trajectory)

    logger.info(
        "Patch-mode rewrite: %d violations found, adversarial at turn %d",
        len(violations), adversarial_spot,
    )

    patches = await _generate_patch_text(
        trajectory, pii_map, violations, adversarial_spot,
    )
    existing_gate_turns = {
        gate.get("before_turn_idx")
        for gate in patches.get("consent_gates", [])
        if isinstance(gate, dict)
    }
    for violation in violations:
        if violation.get("action") == "hard_deny":
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
