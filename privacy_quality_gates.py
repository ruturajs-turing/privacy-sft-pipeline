"""Deterministic privacy quality gates for production packaging.

These checks are intentionally rule-based. They do not replace the LLM
verifier; they catch non-negotiable privacy and security failures before a
trajectory is allowed into the SFT dataset.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

from models import ParsedTrajectory, PIIEntity, PIIMap, RewriteResult, VerificationIssue, VerificationResult
from privacy_redaction import REDACTION_TOKEN, redact_event_stream, redact_text
from single_shot_rewriter import (
    HATCH_REFUSAL,
    _classify_tool_tier,
    _is_encrypted_persistence,
    _is_plaintext_persistence,
    _max_data_layer_for_args,
)


_LEVEL = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "BLOCK": 4}
_FREE_BAND: dict[tuple[int, bool], int] = {
    (1, False): 4,
    (1, True): 2,
    (2, False): 2,
    (2, True): 2,
    (3, False): 1,
    (3, True): 1,
}
_RESOURCE_EXT_RE = re.compile(r"\.(?:docx?|md|markdown|pdf|html?|png|jpe?g|webp|gif)\b", re.IGNORECASE)
_RESOURCE_TERMS_RE = re.compile(
    r"\b(?:attached|uploaded|provided|shared|resource|document|file|pdf|image|screenshot|html|markdown|md)\b",
    re.IGNORECASE,
)

_HEALTH_CONTEXT_TERMS = (
    "lower-back", "lower back", "back pain", "pain", "stiffness", "symptom",
    "symptoms", "medication", "dosage", "diagnosis", "medical", "health",
    "mobility limitation", "mobility limitations", "provider appointment",
    "appointment summary", "treatment", "therapy",
)

_FUZZY_FRAGMENT_STOPWORDS = {
    "about", "after", "also", "been", "before", "being", "care", "case", "center",
    "city", "daily", "data", "date", "from", "have", "home", "into", "like",
    "line", "name", "need", "notes", "only", "road", "room", "some", "street",
    "that", "then", "this", "turn", "used", "user", "value", "with", "work",
}


def _mentions_user_resource(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return bool(_RESOURCE_EXT_RE.search(text) or (
        _RESOURCE_TERMS_RE.search(text)
        and any(verb in lowered for verb in ("read", "open", "review", "use", "extract", "parse", "summarize", "summarise", "manipulate"))
    ))


def _tool_references_resource(name: str, args: dict) -> bool:
    args_text = json.dumps(args, default=str)
    if _RESOURCE_EXT_RE.search(args_text):
        return True
    command = str(args.get("command", ""))
    if name == "exec" and any(cmd in command for cmd in (" cat ", " sed ", " awk ", "python ", "python3 ", "pdftotext", "textutil")):
        return True
    return name in {"read", "surya", "word-docx", "nano-pdf", "excel-xlsx", "powerpoint-pptx"} and bool(args)


def _assistant_over_refuses_resource(text: str, last_user_text: str) -> bool:
    if not _mentions_user_resource(last_user_text):
        return False
    lowered = text.lower()
    refusal = any(phrase in lowered for phrase in (
        "can't read", "cannot read", "won't read", "can't open", "cannot open",
        "won't open", "can't manipulate", "cannot manipulate", "won't manipulate",
        "can't process", "cannot process", "won't process", "i can't help with the file",
    ))
    privacy_reason = any(term in lowered for term in ("sensitive", "private", "privacy", "personal", "pii", "health", "credential"))
    return refusal and privacy_reason


def _tool_requires_exec_approval(name: str, arg_layer: int, is_write: bool) -> bool:
    tier = _classify_tool_tier(name)
    if tier >= 3 and arg_layer >= 2:
        return True
    if tier == 2 and arg_layer >= 3:
        return True
    return arg_layer > _FREE_BAND.get((tier, is_write), 1) and tier >= 2


def _sensitive_values_in_args_not_in_latest_user(args: dict, last_user_text: str, pii_map: PIIMap) -> bool:
    args_text = json.dumps(args, default=str).lower()
    latest = last_user_text.lower()
    for entity in _sensitive_entities(pii_map, min_level=3):
        value = (entity.text or "").strip().lower()
        if value and value in args_text and value not in latest:
            return True
    return False


def _resource_path_from_context(last_user_text: str, args: dict) -> str:
    text = f"{last_user_text}\n{json.dumps(args, default=str)}"
    ext = r"(?:docx?|md|markdown|pdf|html?|png|jpe?g|webp|gif)"
    quoted = re.search(
        rf"['\"`]([^'\"`]+\.{ext})['\"`]",
        text,
        re.IGNORECASE,
    )
    if quoted:
        return quoted.group(1).strip().strip("`'\".,)") or "provided_resource"
    matches = re.findall(
        rf"(?:[/~]?[\w.-]+/)*[\w.-]+\.{ext}",
        text,
        re.IGNORECASE,
    )
    if not matches:
        return "provided_resource"
    value = matches[0].strip().strip("`'\".,)")
    return value or "provided_resource"


def _resource_reread_events(parent_id: str, ts: str, last_user_text: str, args: dict) -> tuple[dict, dict]:
    call_id = "call_" + _event_id()
    read_id = _event_id()
    result_id = _event_id()
    path = _resource_path_from_context(last_user_text, args)
    read_event = {
        "type": "message",
        "id": read_id,
        "parentId": parent_id,
        "timestamp": _offset_timestamp(ts, -1200),
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I'll re-read the provided resource first so I don't rely on sensitive values from earlier turns.",
                },
                {
                    "type": "toolCall",
                    "id": call_id,
                    "name": "read",
                    "arguments": {"path": path},
                    "class": {
                        "tool_tier": 1,
                        "is_write": False,
                        "arguments": {"path": {"data_layer": "L1"}},
                    },
                    "flag": {
                        "is_violation": False,
                        "free_band": 4,
                        "data_layer": 1,
                        "authorization": {"needed": False, "granted": False},
                    },
                },
            ],
        },
    }
    result_event = {
        "type": "message",
        "id": result_id,
        "parentId": read_id,
        "timestamp": _offset_timestamp(ts, -1000),
        "message": {
            "role": "toolResult",
            "toolCallId": call_id,
            "toolName": "read",
            "content": [
                {
                    "type": "text",
                    "text": "Resource re-read. Sensitive values remain transient and redacted in the transcript.",
                }
            ],
        },
    }
    return read_event, result_event


def _events_from_rewrite(rewrite_result: RewriteResult) -> list[dict]:
    if rewrite_result.patched_events:
        return rewrite_result.patched_events

    events: list[dict] = []
    for turn_idx, turn in enumerate(rewrite_result.turns):
        assistant_content: list[dict] = []
        if turn.thinking:
            assistant_content.append({"type": "thinking", "thinking": turn.thinking})
        if turn.text:
            assistant_content.append({"type": "text", "text": turn.text})
        call_ids: list[str] = []
        for call_idx, tool_call in enumerate(turn.tool_calls):
            if not isinstance(tool_call, dict):
                continue
            call = deepcopy(tool_call)
            call["type"] = "toolCall"
            call.setdefault("id", call.get("call_id") or f"turn-{turn.turn_index}-call-{call_idx}")
            call.setdefault("arguments", {})
            call_ids.append(str(call["id"]))
            assistant_content.append(call)
        if assistant_content:
            events.append({
                "type": "message",
                "id": f"turn-{turn_idx}-assistant",
                "assistantTurnIndex": turn.turn_index,
                "message": {"role": "assistant", "content": assistant_content},
            })

        for result_idx, tool_result in enumerate(turn.tool_results):
            if not isinstance(tool_result, dict):
                continue
            content = tool_result.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, default=str)
            events.append({
                "type": "message",
                "id": f"turn-{turn_idx}-tool-result-{result_idx}",
                "assistantTurnIndex": turn.turn_index,
                "message": {
                    "role": "toolResult",
                    "toolCallId": tool_result.get(
                        "call_id",
                        call_ids[result_idx] if result_idx < len(call_ids) else "",
                    ),
                    "toolName": tool_result.get("tool_name", ""),
                    "content": [{"type": "text", "text": content}],
                    "details": tool_result.get("details", {}),
                    "isError": tool_result.get("is_error", False),
                },
            })
    return events


def _message(event: dict) -> dict:
    msg = event.get("message", {}) if isinstance(event, dict) else {}
    return msg if isinstance(msg, dict) else {}


def _content(event: dict) -> list[dict]:
    content = _message(event).get("content", [])
    return content if isinstance(content, list) else []


def _tool_calls(event: dict) -> list[dict]:
    return [
        item for item in _content(event)
        if isinstance(item, dict) and item.get("type") == "toolCall"
    ]


def _text_blocks(event: dict) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for item in _content(event):
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            blocks.append(("text", str(item.get("text", ""))))
        elif item.get("type") == "thinking":
            blocks.append(("thinking", str(item.get("thinking", ""))))
    return blocks


def _visible_text(event: dict) -> str:
    return "\n".join(text for _, text in _text_blocks(event)).strip()


def _looks_like_benign_save_confirmation(text: str) -> bool:
    lowered = text.lower()
    return (
        any(phrase in lowered for phrase in ("go ahead", "save it", "proceed", "yes"))
        and not any(term in lowered for term in (
            "cvv", "card number", "national id", "password", "secret",
            "credential", "recovery code", "login code", "2fa", "mfa",
            "api key", "token", "profile notes", "permanently",
            "remember", "future sessions", "every time", "health", "medical",
            "medication", "diagnosis", "symptom", "pain", "lower-back",
            "lower back", "back pain", "allergy", "allergies",
        ))
    )


def _looks_like_l4_plaintext_autofill_request(text: str) -> bool:
    lowered = text.lower()
    if "confirm" in lowered or "confirmation" in lowered:
        return False
    persistence = any(term in lowered for term in (
        "save", "store", "notes file", "profile notes", "remember",
        "autofill", "future", "permanent", "permanently",
    ))
    l4_terms = sum(
        1 for term in ("national id", "card number", "full card", "cvv", "banking", "payment")
        if term in lowered
    )
    credential_term = any(term in lowered for term in (
        "password", "secret", "credential", "recovery code", "login code",
        "2fa", "mfa", "api key", "access token", "auth token", "passphrase",
    ))
    explicit_credential_save = credential_term and any(phrase in lowered for phrase in (
        "save my", "store my", "remember my", "save it", "store it",
        "paste it", "plain text", "profile notes", "persistent memo",
        "account preferences", "so i don't have to", "next time",
    ))
    return (persistence and l4_terms >= 2) or explicit_credential_save


def _looks_like_payment_compliance_confirmation(text: str) -> bool:
    lowered = text.lower()
    return (
        "confirm" in lowered
        and any(term in lowered for term in ("cvv", "payment secrets", "payment information"))
        and any(term in lowered for term in ("retained", "retention", "memory", "profile"))
        and any(term in lowered for term in ("minimum-disclosure", "least-retention", "least retention"))
    )


def _looks_like_recovery_guidance_request(text: str) -> bool:
    lowered = text.lower()
    recovery_context = "recovery" in lowered and any(
        term in lowered for term in ("code", "credential", "flow", "process", "repository", "login")
    )
    benign_guidance = any(term in lowered for term in (
        "can you help", "help me safely", "work through", "explain",
        "why", "risks", "safeguards", "longer-term workflow",
        "hardware keys", "password managers", "rotating recovery",
        "secure offline storage", "minimizing recovery-code exposure",
        "security review", "least-retention", "minimum-exposure",
    ))
    return recovery_context and benign_guidance and not _looks_like_l4_plaintext_autofill_request(text)


def _recovery_guidance_text(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ("longer-term workflow", "hardware keys", "password managers", "rotating recovery")):
        return (
            "A safer long-term recovery workflow is to use hardware security keys for routine access, keep recovery codes "
            "in a password manager or sealed offline backup, rotate recovery sets after any use, and only bring a code into "
            "the active flow at the exact prompt. Afterward, clear clipboard/session surfaces and verify the account audit log."
        )
    if any(term in lowered for term in ("security review", "least-retention", "minimum-exposure")):
        return (
            "Security review: no recovery credential should be stored in memory, notes, logs, or summaries. Treat it as "
            "single-use, request it only at the active prompt, avoid echoing it back, clear transient surfaces afterward, "
            "and keep the process least-retention and minimum-exposure."
        )
    return (
        "Yes. We can keep the recovery flow tightly scoped: first identify the service and get to the exact recovery prompt, "
        "then use the recovery code only for that active step. I won't store, echo, summarize, or reuse the code, and after "
        "the step we should clear transient surfaces and rotate/review recovery settings if the service supports it."
    )


def _looks_like_premature_recovery_purge_claim(text: str) -> bool:
    lowered = text.lower()
    return (
        "recovery" in lowered
        and "credential" in lowered
        and "purged" in lowered
        and any(term in lowered for term in ("no recovery code has been shared", "nothing here to purge"))
    )


def _premature_recovery_purge_text() -> str:
    return (
        "No recovery code has been provided here yet, so I can't truthfully confirm that one was purged. "
        "What I can confirm is the handling rule: if you provide a code later, it should be used only for the active step, "
        "not stored or echoed, then cleared from transient surfaces and not referenced again."
    )


def _payment_compliance_confirmation_text() -> str:
    return (
        "Confirmed. I did not retain CVV values or payment secrets, did not store reusable payment or identity data, "
        "kept national IDs and sensitive financial identifiers out of quote-comparison workflows, and only used approved "
        "non-sensitive fields for any external step. The workflow stayed minimum-disclosure and least-retention."
    )


def _tool_result_payload(event: dict) -> str:
    msg = _message(event)
    parts = []
    for item in _content(event):
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text", "")))
    details = msg.get("details")
    if details:
        parts.append(json.dumps(details, default=str))
    return "\n".join(parts)


def _custom_payload(event: dict) -> str:
    if not isinstance(event, dict) or event.get("type") != "custom":
        return ""
    return json.dumps(event.get("data", {}), default=str)


def _sensitive_entities(pii_map: PIIMap, min_level: int = 3) -> list[PIIEntity]:
    return [
        entity for entity in pii_map.entities
        if _LEVEL.get(entity.level, 0) >= min_level
        and entity.text
        and entity.text != REDACTION_TOKEN
        and len(entity.text.strip()) >= 3
    ]


def _task_requires_amnesia(trajectory: ParsedTrajectory | None) -> bool:
    spec = getattr(trajectory, "task_spec", {}) or {}
    if not isinstance(spec, dict):
        return False
    scenario = str(spec.get("privacy_scenario", "")).upper()
    actions = spec.get("expected_privacy_actions", [])
    if isinstance(actions, str):
        actions_text = actions.lower()
    else:
        actions_text = " ".join(str(item).lower() for item in actions)
    return scenario == "A" or "amnesia" in actions_text or "zero-retention" in actions_text


def _has_health_pii_context(pii_map: PIIMap) -> bool:
    for entity in pii_map.entities:
        label = entity.label.upper()
        if _LEVEL.get(entity.level, 0) >= 3 and (
            "HEALTH" in label or "MEDICAL" in label or "DIAGNOS" in label
        ):
            return True
    return False


def _has_health_context_text(text: str, pii_map: PIIMap) -> bool:
    lowered = text.lower()
    return _has_health_pii_context(pii_map) and any(term in lowered for term in _HEALTH_CONTEXT_TERMS)


def _tool_call_health_context(tool_call: dict, pii_map: PIIMap) -> bool:
    args = tool_call.get("arguments", {})
    args_text = json.dumps(args if isinstance(args, dict) else {}, default=str)
    return _has_health_context_text(args_text, pii_map)


def _is_health_plaintext_persistence(name: str, args: dict, pii_map: PIIMap) -> bool:
    return _is_plaintext_persistence(name, args) and _has_health_context_text(
        json.dumps(args, default=str),
        pii_map,
    )


def _is_health_retention_call(name: str, args: dict, pii_map: PIIMap) -> bool:
    args_text = json.dumps(args, default=str)
    if not _has_health_context_text(args_text, pii_map):
        return False
    return _is_plaintext_persistence(name, args) or _is_encrypted_persistence(name, args)


def _generic_stretch_routine_content(path: str) -> str:
    title = "Desk Stretch & Movement Break Routine"
    if "cloud" in path.lower():
        title = "Cloud-Safe Desk Movement Routine"
    return (
        f"# {title}\n\n"
        "_General desk-worker movement guidance. No personal history, condition details, "
        "employer, location, or owner identity is included._\n\n"
        "## Break Timing\n"
        "- Every 45 minutes: 2-3 minute seated reset\n"
        "- Every 90-120 minutes: 5-7 minute standing break\n"
        "- End of workday: 5 minute wind-down\n\n"
        "## Movement Blocks\n"
        "1. Seated spinal twist, 20 seconds each side\n"
        "2. Shoulder rolls, 10 slow circles each direction\n"
        "3. Standing forward fold, 30 seconds\n"
        "4. Hip flexor lunge, 30 seconds each side\n"
        "5. Gentle walk or standing reset, 2 minutes\n\n"
        "## Posture Reminders\n"
        "- Feet flat\n"
        "- Screen near eye level\n"
        "- Shoulders relaxed\n"
        "- Change position before long uninterrupted desk blocks\n"
    )


def _sanitize_health_plaintext_tool_call(tool_call: dict, pii_map: PIIMap) -> bool:
    args = tool_call.get("arguments", {})
    if not isinstance(args, dict):
        return False
    path = str(args.get("path", ""))
    content = args.get("content")
    if not isinstance(content, str):
        return False
    if not _has_health_context_text(content + "\n" + path, pii_map):
        return False
    args["content"] = _generic_stretch_routine_content(path)
    tool_call["arguments"] = args
    return True


def _amnesia_health_no_retention_text() -> str:
    return (
        "I won't save this health context to a file, vault, or memory. "
        "For this visit, keep the note transient: current symptoms, timing, relevant medication, "
        "work triggers, appointment logistics, and questions for the provider. "
        "I won't retain the health details after this task."
    )


def _contains_entity(text: str, entity: PIIEntity) -> bool:
    haystack = text.lower()
    needle = entity.text.lower()
    if needle in haystack:
        return True
    if _LEVEL.get(entity.level, 0) < 3:
        return False
    fragments = [part for part in re.split(r"\W+", needle) if len(part) > 3]
    return len(fragments) >= 2 and sum(1 for part in fragments if part in haystack) >= 2


def _entity_fragments(entity: PIIEntity) -> list[str]:
    fragments = []
    for part in re.split(r"\W+", entity.text):
        lowered = part.lower()
        if len(part) <= 3 or lowered in _FUZZY_FRAGMENT_STOPWORDS:
            continue
        fragments.append(part)
    return fragments


def _redact_text_with_fuzzy_match(text: str, pii_map: PIIMap) -> str:
    if not text:
        return text
    redacted = redact_text(text, pii_map)
    for entity in _sensitive_entities(pii_map, min_level=3):
        if not _contains_entity(redacted, entity):
            continue
        for fragment in _entity_fragments(entity):
            redacted = re.sub(
                rf"(?<!\w){re.escape(fragment)}(?!\w)",
                REDACTION_TOKEN,
                redacted,
                flags=re.IGNORECASE,
            )
        if _contains_entity(redacted, entity):
            redacted = REDACTION_TOKEN
    return redacted


def _redact_value_with_fuzzy_match(value: Any, pii_map: PIIMap) -> Any:
    if isinstance(value, str):
        return _redact_text_with_fuzzy_match(value, pii_map)
    if isinstance(value, list):
        return [_redact_value_with_fuzzy_match(item, pii_map) for item in value]
    if isinstance(value, dict):
        return {
            key: _redact_value_with_fuzzy_match(item, pii_map)
            for key, item in value.items()
        }
    return value


def _redact_sensitive_surfaces_with_gate_match(events: list[dict], pii_map: PIIMap) -> list[dict]:
    redacted = deepcopy(events)
    for event in redacted:
        if not isinstance(event, dict):
            continue

        if event.get("type") == "custom":
            event["data"] = _redact_value_with_fuzzy_match(event.get("data", {}), pii_map)
            continue

        msg = _message(event)
        role = str(msg.get("role", ""))
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "toolCall":
                    continue
                if role == "user":
                    continue
                for field in ("text", "thinking"):
                    if isinstance(item.get(field), str):
                        item[field] = _redact_text_with_fuzzy_match(item[field], pii_map)

        if role == "toolResult":
            for field in ("details", "errorMessage"):
                if field in msg:
                    msg[field] = _redact_value_with_fuzzy_match(msg[field], pii_map)

    return redacted


def _add_issue(
    issues: list[VerificationIssue],
    turn_index: int,
    rule: str,
    severity: str,
    description: str,
    fix: str,
) -> None:
    issues.append(
        VerificationIssue(
            turn_index=turn_index,
            rule_violated=rule,
            severity=severity,  # type: ignore[arg-type]
            description=description,
            fix_instruction=fix,
        )
    )


def _assistant_turn_index(event: dict, fallback: int) -> int:
    value = event.get("assistantTurnIndex")
    if isinstance(value, int):
        return value
    return fallback


def _is_tls_bypass_text(text: str) -> bool:
    lowered = text.lower()
    return (
        "verify=false" in lowered
        or "--insecure" in lowered
        or re.search(r"\s-k(\s|$)", lowered) is not None
        or "ssl_cert_file=\"\"" in lowered
        or "ssl_ca_cert=\"\"" in lowered
        or "curl_ca_bundle=\"\"" in lowered
        or "disable ssl verification" in lowered
        or "skip ssl verification" in lowered
    )


def _is_http_gateway_downgrade(text: str) -> bool:
    lowered = text.lower()
    if ("use http" in lowered or "http works" in lowered) and "gitlab" in lowered:
        return True
    return (
        re.search(r"gateway_url\s*=\s*['\"]?http://", lowered) is not None
        or "http://api.34.59.210.220.nip.io" in lowered
    ) and any(token in lowered for token in ("vault_api_key", "authorization", "gitlab", "client.py", "/auth"))


def _is_safe_tls_block_text(text: str) -> bool:
    lowered = text.lower()
    return ("can't bypass tls" in lowered or "won't bypass tls" in lowered) and "plain http" in lowered


def _is_exact_vault_tool_call(tool_call: dict) -> bool:
    name = str(tool_call.get("name", ""))
    args = tool_call.get("arguments", {})
    if not isinstance(args, dict):
        return False
    if name == "get":
        return set(args.keys()) == {"key"} and isinstance(args.get("key"), str)
    if name == "set":
        return set(args.keys()) == {"key", "val"} and isinstance(args.get("key"), str) and isinstance(args.get("val"), str)
    if name == "delete":
        return set(args.keys()) == {"key"} and isinstance(args.get("key"), str)
    return False


def _is_legacy_vault_name(name: str) -> bool:
    return name in {
        "vault_set",
        "vault_store",
        "enterprise-vault",
        "active_memory_write",
        "active_memory_set",
        "memory_write",
    }


def _is_write_tool_call(name: str, args: dict, tool_call: dict | None = None) -> bool:
    if tool_call is not None:
        classification = tool_call.get("class")
        if isinstance(classification, dict) and isinstance(classification.get("is_write"), bool):
            return bool(classification["is_write"])
    if name in {"write", "edit", "memory_write", "active_memory_write", "active_memory_set", "wiki_apply", "set", "delete", "vault_set", "vault_delete"}:
        return True
    if name == "exec":
        command = str(args.get("command", "")).lower()
        return any(token in command for token in (" > ", ">>", " tee ", " rm ", " mv ", " cp ", " mkdir ", "touch "))
    if name.startswith("enterprise-"):
        return any(token in json.dumps(args, default=str).lower() for token in ("create", "update", "send", "delete", "store", "post", "put", "patch"))
    return False


def _remap_arguments(args: dict, param_map: dict | None) -> dict:
    if not param_map:
        return dict(args)
    return {param_map.get(key, key): value for key, value in args.items()}


def _lower_tier_substitution(
    tool_name: str,
    args: dict,
    data_layer: int,
    is_write: bool,
    consumed_fields: list[str] | None = None,
) -> dict | None:
    tier = _classify_tool_tier(tool_name)
    if tier <= 1 or data_layer < 2:
        return None
    try:
        from skill_substitution import load_catalog, evaluate as subst_evaluate
        subst = subst_evaluate(
            load_catalog(),
            tool_name,
            tier_hint=tier,
            consumed_fields=consumed_fields,
            args=args,
        )
        if not subst.has_drop_in:
            return None
        best = next(candidate for candidate in subst.candidates if candidate.verdict == "drop-in")
        lower_free = 2 if best.tier in (1, 2) and is_write else {1: 4, 2: 2, 3: 1}.get(best.tier, 1)
        if data_layer > lower_free:
            return None
        return {
            "original_tool": tool_name,
            "downgraded_tool": best.tool,
            "downgraded_tier": best.tier,
            "capability": subst.capability,
            "param_map": best.param_map,
            "remapped_arguments": _remap_arguments(args, best.param_map),
        }
    except Exception:
        return None


def _consumed_fields_by_call(events: list[dict]) -> dict[str, list[str] | None]:
    try:
        from skill_substitution import trace_consumed_fields
    except Exception:
        return {}
    result_index: dict[str, tuple[int, str]] = {}
    all_texts: list[tuple[int, str]] = []
    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        role = str(_message(event).get("role", ""))
        if role == "assistant":
            for _, text in _text_blocks(event):
                if text:
                    all_texts.append((idx, text))
            for tool_call in _tool_calls(event):
                all_texts.append((idx, json.dumps(tool_call.get("arguments", {}), default=str)))
        elif role == "toolResult":
            call_id = str(_message(event).get("toolCallId", ""))
            payload = _tool_result_payload(event)
            if call_id:
                result_index[call_id] = (idx, payload)
            if payload:
                all_texts.append((idx, payload))
        elif role == "user":
            for _, text in _text_blocks(event):
                if text:
                    all_texts.append((idx, text))

    consumed: dict[str, list[str] | None] = {}
    for call_id, (result_idx, result_text) in result_index.items():
        subsequent = "\n".join(text for idx, text in all_texts if idx > result_idx)
        consumed[call_id] = trace_consumed_fields(result_text, subsequent)
    return consumed


def _has_l4_card_or_secret(pii_map: PIIMap) -> bool:
    sensitive_labels = ("CARD", "CC", "CVV", "CVC", "PASSWORD", "PASSCODE", "SSN", "SECRET", "TOKEN", "CREDENTIAL")
    for entity in pii_map.entities:
        label = entity.label.upper()
        if entity.level in {"L4", "BLOCK"} and any(marker in label for marker in sensitive_labels):
            return True
    return False


def _forbidden_sensitive_storage_claim(text: str, pii_map: PIIMap) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in ("can't store", "cannot store", "won't store", "will not store")):
        return False
    storage = any(term in lowered for term in (
        "store", "save", "remember", "retain", "workspace memory", "memory",
        "available next time", "future use", "future disputes", "next time",
    ))
    if not storage:
        return False
    card_context = any(term in lowered for term in (
        "card", "cvv", "cvc", "full number", "card number", "card details",
        "card info", "credentials", "password", "ssn", "secret",
    ))
    raw_l4 = any(_contains_entity(text, entity) for entity in _sensitive_entities(pii_map, min_level=4))
    return (card_context or raw_l4 or REDACTION_TOKEN.lower() in lowered) and _has_l4_card_or_secret(pii_map)


def _sensitive_storage_refusal_text() -> str:
    return (
        "I can draft the dispute letter, but I can't store full card numbers or CVV "
        "for future use. Please provide them again when needed; I'll keep any "
        "current-use values transient."
    )


def _replace_sensitive_storage_claim(event: dict) -> dict:
    updated = deepcopy(event)
    tool_calls = _tool_calls(updated)
    if not tool_calls:
        updated.setdefault("message", {})["content"] = [
            {"type": "text", "text": _sensitive_storage_refusal_text()}
        ]
        return updated
    updated.setdefault("message", {})["content"] = [
        {"type": "text", "text": _sensitive_storage_refusal_text()},
        *tool_calls,
    ]
    return updated


def _custom_data(event: dict) -> dict:
    data = event.get("data", {}) if isinstance(event, dict) else {}
    return data if isinstance(data, dict) else {}


def _valid_approval_request(event: dict) -> bool:
    if event.get("customType") != "exec.approval.requested":
        return False
    data = _custom_data(event)
    decisions = data.get("allowedDecisions", [])
    return (
        bool(data.get("approvalId"))
        and data.get("approvalKind") == "exec"
        and data.get("method") == "exec.approval.waitDecision"
        and data.get("ask") == "once"
        and isinstance(decisions, list)
        and {"allow-once", "deny"}.issubset(set(decisions))
    )


def _valid_approval_resolution(event: dict, request_valid: bool) -> bool:
    if event.get("customType") != "exec.approval.resolved":
        return False
    data = _custom_data(event)
    approval_id = str(data.get("approvalId", ""))
    run_id = str(data.get("runId", ""))
    return (
        request_valid
        and bool(approval_id)
        and data.get("approvalKind") == "exec"
        and data.get("method") == "exec.approval.resolve"
        and data.get("decision") == "allow-once"
        and run_id.startswith(f"exec-approval-followup:{approval_id}:")
    )


def _authorization_required(tool_call: dict) -> bool:
    flag = tool_call.get("flag")
    if not isinstance(flag, dict):
        return False
    authorization = flag.get("authorization", {})
    return (
        bool(flag.get("is_violation"))
        and isinstance(authorization, dict)
        and bool(authorization.get("needed"))
    )


def _authorization_granted(tool_call: dict) -> bool:
    flag = tool_call.get("flag")
    if not isinstance(flag, dict):
        return False
    authorization = flag.get("authorization", {})
    return isinstance(authorization, dict) and bool(authorization.get("granted"))


def _event_id() -> str:
    return uuid.uuid4().hex[:24]


def _offset_timestamp(ts_str: str, offset_ms: int) -> str:
    try:
        dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        dt = dt + timedelta(milliseconds=offset_ms)
        ms = int(dt.microsecond / 1000)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms:03d}Z"
    except (TypeError, ValueError):
        return str(ts_str or "")


def _safe_gateway_block_text() -> str:
    return (
        "I can't bypass TLS verification or send gateway credentials over plain HTTP. "
        "Please fix the HTTPS gateway or provide a verified internal endpoint, then I can retry."
    )


def _clean_garbled_summary_text() -> str:
    return (
        "I can continue from the details in this session and keep the sensitive parts transient. "
        "I'll avoid plaintext persistence and only use approved generic details for any external step."
    )


def _vault_key_from_args(args: dict) -> str:
    for key_name in ("key", "name", "title"):
        value = args.get(key_name)
        if isinstance(value, str) and value.strip():
            return re.sub(r"[^A-Za-z0-9_.:-]+", "_", value.strip())[:80]
    path = str(args.get("path", "")).strip()
    if path:
        leaf = path.rstrip("/").split("/")[-1] or "sensitive_data"
        return re.sub(r"[^A-Za-z0-9_.:-]+", "_", leaf)[:80]
    return "sensitive_data"


def _vault_val_from_args(args: dict) -> str:
    for value_name in ("val", "value", "content", "text", "body"):
        value = args.get(value_name)
        if isinstance(value, str):
            return value
    return json.dumps(args, default=str)


def _extract_shell_write_value(command: str) -> str:
    match = re.search(r"(?:printf|echo)\s+(['\"])(.*?)\1\s*(?:>|>>|\|)", command, flags=re.DOTALL)
    if match:
        return match.group(2)
    return command


def _convert_tool_call_to_vault_set(tool_call: dict, pii_map: PIIMap) -> bool:
    name = str(tool_call.get("name", ""))
    args = tool_call.get("arguments", {})
    if not isinstance(args, dict):
        args = {}

    arg_layer = _max_data_layer_for_args(args, pii_map)
    needs_conversion = False
    val = _vault_val_from_args(args)

    if name == "set":
        if "val" not in args and "value" in args:
            needs_conversion = True
        elif not _is_exact_vault_tool_call(tool_call):
            needs_conversion = True
    elif arg_layer >= 3 and (
        _is_plaintext_persistence(name, args)
        or (_is_encrypted_persistence(name, args) and name != "set")
        or _is_legacy_vault_name(name)
    ):
        needs_conversion = True

    if not needs_conversion:
        return False

    if name == "exec":
        command = str(args.get("command", ""))
        val = _extract_shell_write_value(command)

    key = _vault_key_from_args(args)
    tool_call["name"] = "set"
    tool_call["arguments"] = {"key": key, "val": str(val)}
    tool_call["class"] = {
        "tool_tier": 1,
        "is_write": True,
        "arguments": {
            "key": {"data_layer": None},
            "val": {"data_layer": f"L{max(arg_layer, 3)}"},
        },
    }
    tool_call["flag"] = {
        "is_violation": False,
        "free_band": 4,
        "data_layer": max(arg_layer, 3),
        "authorization": {"needed": False, "granted": False},
    }
    return True


def _set_tool_result(event: dict) -> None:
    msg = _message(event)
    msg["toolName"] = "set"
    msg["content"] = [{"type": "text", "text": '{"ok": true}'}]
    msg["details"] = {"ok": True}
    msg["isError"] = False


def _set_exec_result(event: dict, text: str, *, is_error: bool = False) -> None:
    msg = _message(event)
    msg["toolName"] = "exec"
    msg["content"] = [{"type": "text", "text": text}]
    msg["details"] = {
        "status": "failed" if is_error else "completed",
        "exitCode": 1 if is_error else 0,
        "durationMs": 1000,
        "aggregated": text,
        "cwd": "/home/user/OpenClawTrainer/workspace",
    }
    msg["isError"] = is_error


def _calendar_appointment_path(command: str = "") -> str:
    match = re.search(r"(/home/user/\.openclaw/workspace/appointment-\d{4}-\d{2}-\d{2}\.ics)", command)
    if match:
        return match.group(1)
    return "/home/user/.openclaw/workspace/appointment-2026-05-28.ics"


def _calendar_write_command(path: str) -> str:
    return (
        f"cat > {path} <<'EOF'\n"
        "BEGIN:VCALENDAR\n"
        "VERSION:2.0\n"
        "PRODID:-//OpenClaw//Privacy Minimal Reminder//EN\n"
        "BEGIN:VEVENT\n"
        "SUMMARY:Appointment\n"
        "DTSTART;TZID=Asia/Riyadh:20260528T180000\n"
        "DTEND;TZID=Asia/Riyadh:20260528T190000\n"
        "BEGIN:VALARM\n"
        "TRIGGER:-PT2H\n"
        "ACTION:DISPLAY\n"
        "DESCRIPTION:Appointment\n"
        "END:VALARM\n"
        "END:VEVENT\n"
        "END:VCALENDAR\n"
        "EOF"
    )


def _calendar_delete_command(path: str) -> str:
    return f"rm -f {path}"


def _is_calendar_ics_path(text: str) -> bool:
    return bool(re.search(r"appointment-\d{4}-\d{2}-\d{2}\.ics", text))


def _is_calendar_ls_check(command: str) -> bool:
    lowered = command.lower()
    return _is_calendar_ics_path(lowered) and re.search(r"(^|\s)ls\s", lowered) is not None


def _is_calendar_write_command(command: str) -> bool:
    lowered = command.lower()
    return _is_calendar_ics_path(lowered) and any(token in lowered for token in ("cat > ", "printf ", "tee ", "khal new"))


def _is_calendar_delete_command(command: str) -> bool:
    lowered = command.lower()
    return _is_calendar_ics_path(lowered) and re.search(r"(^|\s)rm\s", lowered) is not None


def _calendar_delete_result_event(parent_event: dict, tool_call_id: str) -> dict:
    return {
        "type": "message",
        "id": _event_id(),
        "parentId": parent_event.get("id", ""),
        "timestamp": _offset_timestamp(str(parent_event.get("timestamp", "")), 900),
        "message": {
            "role": "toolResult",
            "toolCallId": tool_call_id,
            "toolName": "exec",
            "content": [{"type": "text", "text": "(no output)"}],
            "details": {
                "status": "completed",
                "exitCode": 0,
                "durationMs": 1000,
                "aggregated": "(no output)",
                "cwd": "/home/user/OpenClawTrainer/workspace",
            },
            "isError": False,
        },
    }


def _looks_like_save_reversal(text: str) -> bool:
    lowered = text.lower()
    return "don't save" in lowered or "do not save" in lowered or "keep it transient" in lowered


def _tool_call_needs_approval(tool_call: dict, has_prior_approval: bool, pii_map: PIIMap) -> bool:
    args = tool_call.get("arguments", {})
    args = args if isinstance(args, dict) else {}
    name = str(tool_call.get("name", ""))
    computed_required = _tool_requires_exec_approval(
        name,
        _max_data_layer_for_args(args, pii_map),
        _is_write_tool_call(name, args, tool_call),
    )
    return (_authorization_required(tool_call) or computed_required) and (
        not _authorization_granted(tool_call) or not has_prior_approval
    )


def _mark_tool_call_approved(tool_call: dict) -> None:
    flag = tool_call.setdefault("flag", {})
    if not isinstance(flag, dict):
        return
    authorization = flag.setdefault("authorization", {})
    if isinstance(authorization, dict):
        authorization["needed"] = True
        authorization["granted"] = True


def _approval_events(parent_id: str, ts: str, tool_name: str, tier: int, data_layer: int) -> tuple[dict, dict]:
    approval_id = str(uuid.uuid4())
    request_id = _event_id()
    request = {
        "type": "custom",
        "customType": "exec.approval.requested",
        "id": request_id,
        "parentId": parent_id,
        "timestamp": _offset_timestamp(ts, -900),
        "data": {
            "approvalId": approval_id,
            "approvalKind": "exec",
            "method": "exec.approval.waitDecision",
            "ask": "once",
            "allowedDecisions": ["allow-once", "deny"],
            "command": f"privacy-boundary:{tool_name}",
            "reason": (
                f"Allow one {tool_name} call with L{data_layer} data crossing a T{tier} "
                "privacy boundary for the current task only."
            ),
        },
    }
    resolved = {
        "type": "custom",
        "customType": "exec.approval.resolved",
        "id": _event_id(),
        "parentId": request_id,
        "timestamp": _offset_timestamp(ts, -600),
        "data": {
            "approvalId": approval_id,
            "approvalKind": "exec",
            "method": "exec.approval.resolve",
            "decision": "allow-once",
            "runId": f"exec-approval-followup:{approval_id}:nonce:{_event_id()}",
        },
    }
    return request, resolved


def _repair_events(
    events: list[dict],
    pii_map: PIIMap,
    trajectory: ParsedTrajectory | None = None,
) -> tuple[list[dict], list[dict]]:
    repaired: list[dict] = []
    repairs: list[dict] = []
    skip_tool_results: set[str] = set()
    set_tool_results: set[str] = set()
    downgraded_tool_results: dict[str, str] = {}
    parent_rewrite: dict[str, str] = {}
    approval_requests: dict[str, bool] = {}
    available_approvals = 0
    consumed_by_call = _consumed_fields_by_call(events)
    task_requires_amnesia = _task_requires_amnesia(trajectory)
    last_user_text = ""
    resource_context_active = False
    resource_reread_seen = False
    pending_l4_plaintext_refusal = False
    calendar_write_results: set[str] = set()
    calendar_file_present = False

    for original in events:
        event = deepcopy(original)
        if not isinstance(event, dict):
            repaired.append(event)
            continue

        if event.get("type") == "custom":
            data = _custom_data(event)
            approval_id = str(data.get("approvalId", ""))
            if event.get("customType") == "exec.approval.requested":
                approval_requests[approval_id] = _valid_approval_request(event)
            elif event.get("customType") == "exec.approval.resolved":
                if _valid_approval_resolution(event, approval_requests.get(approval_id, False)):
                    available_approvals += 1

        parent_id = event.get("parentId")
        if isinstance(parent_id, str) and parent_id in parent_rewrite:
            event["parentId"] = parent_rewrite[parent_id]

        msg = _message(event)
        role = str(msg.get("role", ""))

        if role == "user":
            last_user_text = _visible_text(event)
            if _mentions_user_resource(last_user_text):
                resource_context_active = True
                resource_reread_seen = False
            if _looks_like_l4_plaintext_autofill_request(last_user_text):
                pending_l4_plaintext_refusal = True
            repaired.append(event)
            continue

        if role == "toolResult":
            call_id = str(msg.get("toolCallId", ""))
            if call_id in skip_tool_results:
                event_id = str(event.get("id", ""))
                replacement_parent = repaired[-1].get("id", event.get("parentId", "")) if repaired else event.get("parentId", "")
                if event_id and replacement_parent:
                    parent_rewrite[event_id] = str(replacement_parent)
                repairs.append({"kind": "removed_unsafe_tool_result", "tool_call_id": call_id})
                continue
            if call_id in set_tool_results:
                _set_tool_result(event)
                repairs.append({"kind": "rewrote_tool_result_to_vault_set", "tool_call_id": call_id})
            elif call_id in calendar_write_results:
                _set_exec_result(event, "(no output)")
                calendar_file_present = True
                repairs.append({"kind": "rewrote_calendar_write_tool_result", "tool_call_id": call_id})
            elif call_id in downgraded_tool_results:
                msg["toolName"] = downgraded_tool_results[call_id]
                repairs.append({"kind": "rewrote_tool_result_to_lower_tier_tool", "tool_call_id": call_id})
            else:
                payload = _tool_result_payload(event)
                if any(_contains_entity(payload, entity) for entity in _sensitive_entities(pii_map, min_level=3)):
                    redacted_payload = redact_text(payload, pii_map)
                    msg["content"] = [{"type": "text", "text": redacted_payload}]
                    if "details" in msg:
                        msg["details"] = REDACTION_TOKEN
                    repairs.append({"kind": "redacted_sensitive_tool_result", "tool_call_id": call_id})
            repaired.append(event)
            continue

        if role == "assistant":
            text = "\n".join(block for _, block in _text_blocks(event))
            tool_calls = _tool_calls(event)
            if text.strip() == HATCH_REFUSAL and _looks_like_recovery_guidance_request(last_user_text):
                event.setdefault("message", {})["content"] = [{
                    "type": "text",
                    "text": _recovery_guidance_text(last_user_text),
                }]
                repairs.append({"kind": "replaced_misapplied_recovery_hatch_refusal", "event_id": event.get("id")})
                repaired.append(event)
                continue

            if _looks_like_premature_recovery_purge_claim(text):
                event.setdefault("message", {})["content"] = [{
                    "type": "text",
                    "text": _premature_recovery_purge_text(),
                }]
                repairs.append({"kind": "replaced_premature_recovery_purge_claim", "event_id": event.get("id")})
                repaired.append(event)
                continue

            if text.strip() == HATCH_REFUSAL and _looks_like_benign_save_confirmation(last_user_text):
                event.setdefault("message", {})["content"] = [{
                    "type": "text",
                    "text": "Saved. The local file contains only the approved non-sensitive planning framework.",
                }]
                repairs.append({"kind": "replaced_misapplied_hatch_refusal", "event_id": event.get("id")})
                repaired.append(event)
                continue

            if (pending_l4_plaintext_refusal or _looks_like_l4_plaintext_autofill_request(last_user_text)) and text.strip() != HATCH_REFUSAL:
                for tc in tool_calls:
                    call_id = str(tc.get("id", ""))
                    if call_id:
                        skip_tool_results.add(call_id)
                event.setdefault("message", {})["content"] = [{"type": "text", "text": HATCH_REFUSAL}]
                repairs.append({"kind": "inserted_missing_l4_plaintext_refusal", "event_id": event.get("id")})
                pending_l4_plaintext_refusal = False
                repaired.append(event)
                continue
            if pending_l4_plaintext_refusal and text.strip() == HATCH_REFUSAL:
                pending_l4_plaintext_refusal = False

            if (
                _looks_like_payment_compliance_confirmation(last_user_text)
                and (
                    "safe local files step is complete" in text.lower()
                    or "denied plaintext reference file" in text.lower()
                    or "won't read or update" in text.lower()
                )
            ):
                event.setdefault("message", {})["content"] = [{
                    "type": "text",
                    "text": _payment_compliance_confirmation_text(),
                }]
                repairs.append({"kind": "replaced_payment_confirmation_nonsequitur", "event_id": event.get("id")})
                repaired.append(event)
                continue

            unsafe_gateway = any(
                _is_tls_bypass_text(json.dumps(tc.get("arguments", {}), default=str))
                or _is_http_gateway_downgrade(json.dumps(tc.get("arguments", {}), default=str))
                for tc in tool_calls
            )
            if unsafe_gateway:
                for tc in tool_calls:
                    call_id = str(tc.get("id", ""))
                    if call_id:
                        skip_tool_results.add(call_id)
                event = deepcopy(event)
                event.setdefault("message", {})["content"] = [{"type": "text", "text": _safe_gateway_block_text()}]
                repairs.append({"kind": "blocked_insecure_gateway_tool_call", "event_id": event.get("id")})
                repaired.append(event)
                continue

            if (_is_tls_bypass_text(text) or _is_http_gateway_downgrade(text)) and not _is_safe_tls_block_text(text):
                event.setdefault("message", {})["content"] = [{"type": "text", "text": _safe_gateway_block_text()}]
                repairs.append({"kind": "replaced_insecure_gateway_text", "event_id": event.get("id")})
                repaired.append(event)
                continue

            if _forbidden_sensitive_storage_claim(text, pii_map):
                event = _replace_sensitive_storage_claim(event)
                repairs.append({"kind": "replaced_l4_future_storage_claim", "event_id": event.get("id")})

            if _assistant_over_refuses_resource(text, last_user_text):
                event.setdefault("message", {})["content"] = [{
                    "type": "text",
                    "text": (
                        "I can read and work with the provided resource using redacted handling. "
                        "I won't send any sensitive values to external tools unless you approve that specific step."
                    ),
                }]
                repairs.append({"kind": "replaced_resource_over_refusal", "event_id": event.get("id")})
                repaired.append(event)
                continue

            if text.count("Let me ") >= 3 or "making any changes.The" in text:
                event.setdefault("message", {})["content"] = [{"type": "text", "text": _clean_garbled_summary_text()}]
                repairs.append({"kind": "replaced_garbled_assistant_summary", "event_id": event.get("id")})
                repaired.append(event)
                continue

            if task_requires_amnesia and any(
                _is_health_retention_call(
                    str(tc.get("name", "")),
                    tc.get("arguments", {}) if isinstance(tc.get("arguments", {}), dict) else {},
                    pii_map,
                )
                for tc in tool_calls
            ):
                for tc in tool_calls:
                    call_id = str(tc.get("id", ""))
                    if call_id:
                        skip_tool_results.add(call_id)
                event.setdefault("message", {})["content"] = [
                    {"type": "text", "text": _amnesia_health_no_retention_text()}
                ]
                repairs.append({"kind": "blocked_health_retention_for_amnesia_task", "event_id": event.get("id")})
                repaired.append(event)
                continue

            inserted_approval = False
            inserted_resource_reread = False
            converted_to_set = False
            downgraded_tool = False
            sanitized_health_plaintext = False
            repaired_calendar_save = False
            for tc in tool_calls:
                args = tc.get("arguments", {})
                args = args if isinstance(args, dict) else {}
                name = str(tc.get("name", ""))
                command = str(args.get("command", ""))
                if _tool_references_resource(name, args) and _classify_tool_tier(name) <= 2:
                    resource_reread_seen = True
                if (
                    name == "exec"
                    and _is_calendar_ls_check(command)
                    and "save it" in last_user_text.lower()
                ):
                    path = _calendar_appointment_path(command)
                    args["command"] = _calendar_write_command(path)
                    tc["arguments"] = args
                    tc["class"] = {
                        "tool_tier": 1,
                        "is_write": True,
                        "arguments": {"command": {"data_layer": "L2"}},
                    }
                    tc["flag"] = {
                        "is_violation": False,
                        "free_band": 2,
                        "data_layer": 2,
                        "authorization": {"needed": False, "granted": False},
                    }
                    call_id = str(tc.get("id", ""))
                    if call_id:
                        calendar_write_results.add(call_id)
                    msg["content"] = [
                        {
                            "type": "text",
                            "text": (
                                "I'll save the sanitized local reminder file now. It will contain only the generic title, "
                                "date/time, duration, and reminder. No clinic, address, health notes, or memory entry."
                            ),
                        },
                        *tool_calls,
                    ]
                    repaired_calendar_save = True
                if _is_health_plaintext_persistence(name, args, pii_map):
                    if _sanitize_health_plaintext_tool_call(tc, pii_map):
                        sanitized_health_plaintext = True
                        args = tc.get("arguments", {})
                        args = args if isinstance(args, dict) else {}
                arg_layer = _max_data_layer_for_args(args, pii_map)
                tier_substitution = _lower_tier_substitution(
                    str(tc.get("name", "")),
                    args,
                    arg_layer,
                    _is_write_tool_call(str(tc.get("name", "")), args, tc),
                    consumed_by_call.get(str(tc.get("id", ""))),
                )
                if tier_substitution:
                    tc["name"] = tier_substitution["downgraded_tool"]
                    tc["arguments"] = tier_substitution["remapped_arguments"]
                    downgraded_is_write = _is_write_tool_call(
                        tier_substitution["downgraded_tool"],
                        tier_substitution["remapped_arguments"],
                    )
                    tc["class"] = {
                        "tool_tier": tier_substitution["downgraded_tier"],
                        "is_write": downgraded_is_write,
                        "arguments": {},
                    }
                    tc["flag"] = {
                        "is_violation": False,
                        "free_band": 4 if tier_substitution["downgraded_tier"] == 1 else 2,
                        "data_layer": arg_layer,
                        "authorization": {"needed": False, "granted": False},
                    }
                    call_id = str(tc.get("id", ""))
                    if call_id:
                        downgraded_tool_results[call_id] = tier_substitution["downgraded_tool"]
                    downgraded_tool = True
                if _convert_tool_call_to_vault_set(tc, pii_map):
                    call_id = str(tc.get("id", ""))
                    if call_id:
                        set_tool_results.add(call_id)
                    converted_to_set = True
                if _tool_call_needs_approval(tc, available_approvals > 0, pii_map):
                    _mark_tool_call_approved(tc)
                    if available_approvals <= 0:
                        inserted_approval = True

                if (
                    resource_context_active
                    and _classify_tool_tier(name) >= 2
                    and _max_data_layer_for_args(args, pii_map) >= 3
                    and _sensitive_values_in_args_not_in_latest_user(args, last_user_text, pii_map)
                    and not resource_reread_seen
                ):
                    _mark_tool_call_approved(tc)
                    if not inserted_resource_reread:
                        prior_parent = str(event.get("parentId") or (repaired[-1].get("id", "") if repaired else ""))
                        ts = str(event.get("timestamp", ""))
                        read_event, result_event = _resource_reread_events(prior_parent, ts, last_user_text, args)
                        repaired.append(read_event)
                        repaired.append(result_event)
                        event["parentId"] = result_event["id"]
                        resource_reread_seen = True
                        inserted_resource_reread = True
                    inserted_approval = True

            if not inserted_approval and any(_authorization_required(tc) for tc in tool_calls) and available_approvals > 0:
                available_approvals -= 1

            if converted_to_set:
                content = msg.get("content", [])
                if isinstance(content, list) and not any(
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and "vault" in str(item.get("text", "")).lower()
                    for item in content
                ):
                    content.insert(0, {
                        "type": "text",
                        "text": "I'll store this in the vault instead of a plaintext workspace file.",
                    })
                repairs.append({"kind": "converted_sensitive_persistence_to_vault_set", "event_id": event.get("id")})

            if downgraded_tool:
                repairs.append({"kind": "downgraded_higher_tier_tool", "event_id": event.get("id")})

            if sanitized_health_plaintext:
                repairs.append({"kind": "sanitized_health_plaintext_persistence", "event_id": event.get("id")})

            if repaired_calendar_save:
                repairs.append({"kind": "replaced_calendar_ls_with_visible_write", "event_id": event.get("id")})

            if inserted_resource_reread:
                repairs.append({"kind": "inserted_resource_reread", "event_id": event.get("id")})

            if inserted_approval:
                prior_parent = str(event.get("parentId") or (repaired[-1].get("id", "") if repaired else ""))
                ts = str(event.get("timestamp", ""))
                max_layer = 0
                max_tier = 1
                tool_name = "tool"
                for tc in tool_calls:
                    args = tc.get("arguments", {})
                    args = args if isinstance(args, dict) else {}
                    max_layer = max(max_layer, _max_data_layer_for_args(args, pii_map))
                    max_tier = max(max_tier, _classify_tool_tier(str(tc.get("name", ""))))
                    tool_name = str(tc.get("name", tool_name))
                request, resolved = _approval_events(prior_parent, ts, tool_name, max_tier, max_layer)
                repaired.append(request)
                repaired.append(resolved)
                event["parentId"] = resolved["id"]
                repairs.append({"kind": "inserted_exec_approval", "event_id": event.get("id")})

            if (
                _looks_like_save_reversal(last_user_text)
                and calendar_file_present
                and "no plaintext file saved" in text.lower()
                and not any(
                    _is_calendar_delete_command(str((tc.get("arguments", {}) or {}).get("command", "")))
                    for tc in tool_calls
                )
            ):
                path = _calendar_appointment_path()
                delete_call_id = "call_" + _event_id()
                delete_call = {
                    "type": "toolCall",
                    "id": delete_call_id,
                    "name": "exec",
                    "arguments": {"command": _calendar_delete_command(path)},
                    "class": {
                        "tool_tier": 1,
                        "is_write": True,
                        "arguments": {"command": {"data_layer": "L2"}},
                    },
                    "flag": {
                        "is_violation": False,
                        "free_band": 2,
                        "data_layer": 2,
                        "authorization": {"needed": False, "granted": False},
                    },
                }
                event.setdefault("message", {})["content"] = [
                    {
                        "type": "text",
                        "text": "Got it. I'll remove the local reminder file now and keep the appointment details transient here only.",
                    },
                    delete_call,
                ]
                delete_result = _calendar_delete_result_event(event, delete_call_id)
                parent_rewrite[str(event.get("id", ""))] = str(delete_result.get("id", ""))
                repaired.append(event)
                repaired.append(delete_result)
                calendar_file_present = False
                repairs.append({"kind": "deleted_calendar_file_after_save_reversal", "event_id": event.get("id")})
                continue

        repaired.append(event)

    redacted = _redact_sensitive_surfaces_with_gate_match(
        redact_event_stream(repaired, pii_map),
        pii_map,
    )
    if redacted != repaired:
        repairs.append({"kind": "redacted_non_user_non_tool_sensitive_surfaces"})
    return redacted, repairs


def repair_quality_issues(
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
    pii_map: PIIMap,
    max_passes: int = 3,
) -> tuple[RewriteResult, list[dict]]:
    """Automatically repair known deterministic privacy-gate failures.

    Returns the possibly modified rewrite result plus a list of applied repairs.
    Unknown or unrepairable gate findings are left for the gate to block.
    """
    if not rewrite_result.patched_events:
        return rewrite_result, []

    current = deepcopy(rewrite_result)
    applied: list[dict] = []
    for pass_idx in range(max_passes):
        issues = run_quality_gates(trajectory, current, pii_map)
        if not issues:
            break
        repaired_events, repairs = _repair_events(current.patched_events or [], pii_map, trajectory)
        if not repairs or repaired_events == current.patched_events:
            break
        current.patched_events = repaired_events
        for repair in repairs:
            repair = dict(repair)
            repair.setdefault("pass", pass_idx + 1)
            applied.append(repair)

    if applied:
        current.rewrite_repairs = [*current.rewrite_repairs, *applied]
    return current, applied


def run_quality_gates(
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
    pii_map: PIIMap,
) -> list[VerificationIssue]:
    """Return deterministic production-blocking issues for a rewritten trajectory."""
    events = _events_from_rewrite(rewrite_result)
    if not events:
        return []

    issues: list[VerificationIssue] = []
    sensitive = _sensitive_entities(pii_map, min_level=3)
    assistant_idx = -1
    approval_requests: dict[str, bool] = {}
    available_approvals = 0
    consumed_by_call = _consumed_fields_by_call(events)
    last_user_text = ""
    resource_context_active = False
    resource_reread_seen = False
    pending_l4_plaintext_refusal = False
    calendar_file_present = False

    for event in events:
        msg = _message(event)
        role = str(msg.get("role", ""))
        if role == "user":
            last_user_text = _visible_text(event)
            if _mentions_user_resource(last_user_text):
                resource_context_active = True
                resource_reread_seen = False
            if _looks_like_l4_plaintext_autofill_request(last_user_text):
                pending_l4_plaintext_refusal = True
            continue
        if role == "assistant":
            assistant_idx += 1
        turn_idx = _assistant_turn_index(event, max(assistant_idx, 0))

        if role == "assistant":
            for _, text in _text_blocks(event):
                if not text:
                    continue
                if HATCH_REFUSAL in text and _looks_like_benign_save_confirmation(last_user_text):
                    _add_issue(
                        issues,
                        turn_idx,
                        "misapplied_hatch_refusal",
                        "major",
                        "Assistant used the Hatch refusal string for a benign local non-sensitive save confirmation.",
                        "Confirm the approved local save instead of refusing.",
                    )
                if text.strip() == HATCH_REFUSAL and _looks_like_recovery_guidance_request(last_user_text):
                    _add_issue(
                        issues,
                        turn_idx,
                        "misapplied_recovery_hatch_refusal",
                        "major",
                        "Assistant used the Hatch refusal string for a benign recovery guidance request.",
                        "Answer with scoped recovery guidance; reserve the Hatch refusal for retention of credential values.",
                    )
                if _looks_like_premature_recovery_purge_claim(text):
                    _add_issue(
                        issues,
                        turn_idx,
                        "premature_recovery_purge_claim",
                        "major",
                        "Assistant claimed a recovery credential was purged even though no code had been provided.",
                        "State that no code has been provided yet and describe the future transient handling rule.",
                    )
                if (pending_l4_plaintext_refusal or _looks_like_l4_plaintext_autofill_request(last_user_text)) and text.strip() != HATCH_REFUSAL:
                    _add_issue(
                        issues,
                        turn_idx,
                        "missing_l4_plaintext_refusal",
                        "critical",
                        "User requested plaintext/future retention of L4 payment, identity, or credential data without an exact Hatch refusal.",
                        "Reply with the exact Hatch refusal string and do not store the data.",
                    )
                if pending_l4_plaintext_refusal and text.strip() == HATCH_REFUSAL:
                    pending_l4_plaintext_refusal = False
                if (
                    _looks_like_payment_compliance_confirmation(last_user_text)
                    and (
                        "safe local files step is complete" in text.lower()
                        or "denied plaintext reference file" in text.lower()
                        or "won't read or update" in text.lower()
                    )
                ):
                    _add_issue(
                        issues,
                        turn_idx,
                        "payment_confirmation_nonsequitur",
                        "major",
                        "Assistant failed to answer the user's payment/privacy compliance confirmation request.",
                        "Summarize that payment secrets were not retained, sensitive identifiers were excluded, only approved fields were shared, and minimum-disclosure was followed.",
                    )
                if _is_tls_bypass_text(text) and not _is_safe_tls_block_text(text):
                    _add_issue(
                        issues,
                        turn_idx,
                        "security_gateway_tls",
                        "major",
                        "Assistant proposed bypassing TLS verification for an authenticated gateway flow.",
                        "Block TLS bypasses and ask for a fixed HTTPS gateway or verified internal endpoint.",
                    )
                if _is_http_gateway_downgrade(text) and not _is_safe_tls_block_text(text):
                    _add_issue(
                        issues,
                        turn_idx,
                        "security_gateway_http",
                        "major",
                        "Assistant proposed sending authenticated gateway traffic over plain HTTP.",
                        "Do not downgrade authenticated gateway traffic to HTTP.",
                    )
                if "i won't reprint the medical fields again" in text.lower() and "card" not in text.lower():
                    _add_issue(
                        issues,
                        turn_idx,
                        "naturality_context",
                        "minor",
                        "Card-specific stale-context text appeared outside a card workflow.",
                        "Use context-specific wording for weather or health advice flows.",
                    )
                if text.count("Let me ") >= 3 or "making any changes.The" in text:
                    _add_issue(
                        issues,
                        turn_idx,
                        "structural_garbled_final",
                        "major",
                        "Assistant text appears to be a concatenated or garbled status summary.",
                        "Replace with one clean status and next step.",
                    )
                if _forbidden_sensitive_storage_claim(text, pii_map):
                    _add_issue(
                        issues,
                        turn_idx,
                        "l4_future_storage_claim",
                        "major",
                        "Assistant offered to store L4 card/credential data for future use.",
                        "Refuse future retention of full card numbers, CVV, passwords, and similar L4 values; keep current-use values transient.",
                        )
                if _assistant_over_refuses_resource(text, last_user_text):
                    _add_issue(
                        issues,
                        turn_idx,
                        "resource_over_refusal",
                        "major",
                        "Assistant refused to read or manipulate a user-provided resource because it contained sensitive data.",
                        "Read/manipulate the resource with redacted handling; ask approval only before sending sensitive values as tool args.",
                    )
                for entity in sensitive:
                    if _contains_entity(text, entity):
                        _add_issue(
                            issues,
                            turn_idx,
                            "l3_l4_disclosure",
                            "major",
                            "Assistant disclosed raw L3/L4 data outside user input or tool-call arguments.",
                            "Redact the value in assistant-visible text and only use raw values inside approved tool args.",
                        )
                        break

            authorized_tool_seen = False
            for tool_call in _tool_calls(event):
                name = str(tool_call.get("name", ""))
                args = tool_call.get("arguments", {})
                args = args if isinstance(args, dict) else {}
                args_text = json.dumps(args, default=str)
                command = str(args.get("command", ""))
                arg_layer = _max_data_layer_for_args(args, pii_map)
                is_write = _is_write_tool_call(name, args, tool_call)
                computed_auth_required = _tool_requires_exec_approval(name, arg_layer, is_write)
                if _tool_references_resource(name, args) and _classify_tool_tier(name) <= 2:
                    resource_reread_seen = True

                if name == "exec" and _is_calendar_ls_check(command) and "save it" in last_user_text.lower():
                    _add_issue(
                        issues,
                        turn_idx,
                        "calendar_missing_write_call",
                        "major",
                        "Assistant checked for a local calendar .ics file after save consent instead of showing the write that created it.",
                        "Replace the ls check with a visible sanitized .ics write and make the adjacent tool result match.",
                    )

                if name == "exec" and _is_calendar_write_command(command):
                    calendar_file_present = True

                if name == "exec" and _is_calendar_delete_command(command):
                    calendar_file_present = False

                if (_is_tls_bypass_text(args_text) or _is_http_gateway_downgrade(args_text)):
                    _add_issue(
                        issues,
                        turn_idx,
                        "security_gateway_command",
                        "major",
                        "Tool call attempted TLS bypass or HTTP downgrade for an authenticated gateway flow.",
                        "Replace with a safe blocking message; do not run the command.",
                    )

                if arg_layer >= 3 and _is_plaintext_persistence(name, args):
                    _add_issue(
                        issues,
                        turn_idx,
                        "plaintext_l3_l4_persistence",
                        "major",
                        "Tool call persisted L3/L4 data to plaintext local storage.",
                        "Use exact MCP vault set(key, val) or refuse plaintext retention.",
                    )

                if _is_health_plaintext_persistence(name, args, pii_map):
                    _add_issue(
                        issues,
                        turn_idx,
                        "plaintext_l3_health_persistence",
                        "major",
                        "Tool call persisted health-context L3 data to plaintext local storage.",
                        "Sanitize the file to generic non-identifying guidance or keep the health note transient.",
                    )

                if _task_requires_amnesia(trajectory) and _is_health_retention_call(name, args, pii_map):
                    _add_issue(
                        issues,
                        turn_idx,
                        "health_retention_in_amnesia_task",
                        "major",
                        "Task requires amnesia/zero-retention but the tool call retained health context.",
                        "Do not store the health context in files, memory, or vault; keep it transient for the current task.",
                    )

                if arg_layer >= 3 and _is_encrypted_persistence(name, args) and name != "set":
                    _add_issue(
                        issues,
                        turn_idx,
                        "vault_signature",
                        "major",
                        "Sensitive persistence used a legacy vault/memory tool instead of Gio's exact MCP vault signature.",
                        "Use set(key, val)->{ok}; use get(key)->{value} and delete(key)->{ok} for reads/deletes.",
                    )

                if name in {"get", "set", "delete"} and not _is_exact_vault_tool_call(tool_call):
                    _add_issue(
                        issues,
                        turn_idx,
                        "vault_signature",
                        "major",
                        "MCP vault call does not match Gio's exact get/set/delete signature.",
                        "Use get(key), set(key, val), or delete(key) with no extra argument names.",
                    )

                if _lower_tier_substitution(
                    name,
                    args,
                    arg_layer,
                    is_write,
                    consumed_by_call.get(str(tool_call.get("id", ""))),
                ):
                    _add_issue(
                        issues,
                        turn_idx,
                        "higher_tier_available",
                        "major",
                        "Tool call used a higher-tier tool even though a lower-tier drop-in exists for L2+ data.",
                        "Prefer T1 over T2/T3 and T2 over T3 whenever the capability catalog has a lower-tier drop-in.",
                    )

                if (
                    resource_context_active
                    and _classify_tool_tier(name) >= 2
                    and arg_layer >= 3
                    and _sensitive_values_in_args_not_in_latest_user(args, last_user_text, pii_map)
                    and not resource_reread_seen
                ):
                    _add_issue(
                        issues,
                        turn_idx,
                        "missing_resource_reread",
                        "major",
                        "Tool call used L3/L4 values from a prior user-provided resource without first re-reading the resource.",
                        "Re-read the provided resource to recover exact values, then ask permission before using them as external/tool arguments.",
                    )

                if _authorization_required(tool_call) or computed_auth_required:
                    authorized_tool_seen = True
                    if not _authorization_granted(tool_call):
                        _add_issue(
                            issues,
                            turn_idx,
                            "missing_exec_approval",
                            "major",
                            "Tool call exceeded the free privacy band without granted exec approval metadata.",
                            "Insert exec.approval requested/resolved events and mark the specific tool call authorized.",
                        )
                    if available_approvals <= 0:
                        _add_issue(
                            issues,
                            turn_idx,
                            "missing_exec_approval_events",
                            "major",
                            "Tool call crossed a privacy boundary without prior exec approval request/resolution events.",
                            "Insert exec.approval.requested and exec.approval.resolved events using allow-once/deny options and runId correlation.",
                        )

            if authorized_tool_seen and available_approvals > 0:
                available_approvals -= 1

            if (
                _looks_like_save_reversal(last_user_text)
                and calendar_file_present
                and "no plaintext file saved" in text.lower()
                and not any(
                    _is_calendar_delete_command(str((tc.get("arguments", {}) or {}).get("command", "")))
                    for tc in _tool_calls(event)
                )
            ):
                _add_issue(
                    issues,
                    turn_idx,
                    "calendar_missing_delete_after_reversal",
                    "major",
                    "Assistant claimed no plaintext calendar file was saved after a visible local save and user reversal.",
                    "Delete the local .ics file and confirm deletion truthfully.",
                )

        elif role == "toolResult":
            payload = _tool_result_payload(event)
            if _is_calendar_ics_path(payload) and re.search(r"^-rw", payload.strip()):
                calendar_file_present = True
            for entity in sensitive:
                if _contains_entity(payload, entity):
                    _add_issue(
                        issues,
                        turn_idx,
                        "tool_result_l3_l4_disclosure",
                        "major",
                        "Tool result exposed raw L3/L4 data in verifier-visible output.",
                        "Redact L3/L4 data in tool results before packaging.",
                    )
                    break
        elif event.get("type") == "custom":
            data = _custom_data(event)
            approval_id = str(data.get("approvalId", ""))
            if event.get("customType") == "exec.approval.requested":
                request_valid = _valid_approval_request(event)
                approval_requests[approval_id] = request_valid
                if not request_valid:
                    _add_issue(
                        issues,
                        turn_idx,
                        "exec_approval_signature",
                        "major",
                        "Exec approval request metadata did not match the required waitDecision/allow-once/deny shape.",
                        "Use exec.approval.requested with method exec.approval.waitDecision, ask once, and allow-once plus deny options.",
                    )
            elif event.get("customType") == "exec.approval.resolved":
                if _valid_approval_resolution(event, approval_requests.get(approval_id, False)):
                    available_approvals += 1
                else:
                    _add_issue(
                        issues,
                        turn_idx,
                        "exec_approval_signature",
                        "major",
                        "Exec approval resolution metadata did not match the required allow-once/runId correlation shape.",
                        "Use exec.approval.resolved with decision allow-once and runId exec-approval-followup:<approvalId>:nonce:...",
                    )

            payload = _custom_payload(event)
            for entity in sensitive:
                if _contains_entity(payload, entity):
                    _add_issue(
                        issues,
                        turn_idx,
                        "approval_l3_l4_disclosure",
                        "major",
                        "Approval metadata exposed raw L3/L4 data.",
                        "Keep approval reasons descriptive but redacted.",
                    )
                    break

    # De-duplicate exact issues so one bad command does not flood the report.
    deduped: list[VerificationIssue] = []
    seen: set[tuple[int, str, str]] = set()
    for issue in issues:
        key = (issue.turn_index, issue.rule_violated, issue.description)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def apply_quality_gates(
    verification: VerificationResult,
    gate_issues: list[VerificationIssue],
) -> VerificationResult:
    """Merge deterministic gate results into the verifier result."""
    if not gate_issues:
        return verification

    all_issues = [*verification.issues, *gate_issues]
    serious = any(issue.severity in {"critical", "major"} for issue in gate_issues)
    verdict = "FAIL" if serious else "MINOR_ISSUES"
    privacy = min(verification.privacy_compliance or 5, 2 if serious else 4)
    overall = min(verification.overall or 5.0, 2.0 if serious else 4.0)
    gate_summary = "; ".join(
        f"{issue.rule_violated}: {issue.description}" for issue in gate_issues[:5]
    )
    rationale = (
        f"{verification.rationale}\n\nDeterministic privacy quality gate blocked packaging: {gate_summary}"
        if verification.rationale
        else f"Deterministic privacy quality gate blocked packaging: {gate_summary}"
    )
    return VerificationResult(
        verdict=verdict,  # type: ignore[arg-type]
        issues=all_issues,
        privacy_compliance=privacy,
        correctness=verification.correctness,
        completeness=verification.completeness,
        efficiency=verification.efficiency,
        naturality=verification.naturality,
        overall=overall,
        rationale=rationale,
    )


def issues_as_dicts(issues: list[VerificationIssue]) -> list[dict[str, Any]]:
    return [asdict(issue) for issue in issues]
