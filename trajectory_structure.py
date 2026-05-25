"""OpenClaw JSONL trajectory structure checks.

The trainer exports are append-only event streams.  Privacy rewrites should
splice turns into that stream, not reinterpret assistant/tool chains as a new
dialogue format.  These checks intentionally allow the normal pattern:

    assistant(toolCall) -> toolResult -> assistant(toolCall) -> toolResult

because that is how OpenClaw continues after tool results.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


VALID_MESSAGE_ROLES = {"user", "assistant", "toolResult"}


def _event_id(event: dict[str, Any]) -> str:
    value = event.get("id")
    return value if isinstance(value, str) else ""


def _message_role(event: dict[str, Any]) -> str | None:
    if event.get("type") != "message":
        return None
    msg = event.get("message")
    if not isinstance(msg, dict):
        return None
    role = msg.get("role")
    return role if isinstance(role, str) else None


def _content(event: dict[str, Any]) -> list[dict[str, Any]]:
    msg = event.get("message")
    if not isinstance(msg, dict):
        return []
    content = msg.get("content")
    return content if isinstance(content, list) else []


def _is_approval_custom_event(event: dict[str, Any]) -> bool:
    custom_type = event.get("customType")
    return isinstance(custom_type, str) and custom_type.startswith("exec.approval.")


def validate_event_stream(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return structural issues in an OpenClaw JSONL event stream.

    This is deliberately schema-light.  It catches the failure modes that make
    a trajectory unusable for SFT/RLHF without rejecting valid trainer exports:
    missing ids, broken parent links, invalid roles, consecutive assistant
    messages without an intervening tool result/user, and orphan tool results.
    """
    issues: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    seen_tool_call_ids: set[str] = set()
    previous_message_role: str | None = None

    for idx, event in enumerate(events):
        if not isinstance(event, dict):
            issues.append({"index": idx, "code": "non_object_event"})
            continue

        event_id = _event_id(event)
        if not event_id:
            issues.append({"index": idx, "code": "missing_event_id"})
        elif event_id in seen_event_ids:
            issues.append({"index": idx, "code": "duplicate_event_id", "id": event_id})

        parent = event.get("parentId")
        if (
            isinstance(parent, str)
            and parent
            and parent not in seen_event_ids
            and event.get("type") not in {"model_change"}
        ):
            issues.append({
                "index": idx,
                "code": "parent_not_seen",
                "id": event_id,
                "parentId": parent,
            })

        if event.get("type") == "custom" and _is_approval_custom_event(event):
            previous_message_role = None

        if event.get("type") == "message":
            msg = event.get("message")
            if not isinstance(msg, dict):
                issues.append({"index": idx, "code": "message_not_object", "id": event_id})
                continue

            role = _message_role(event)
            if role not in VALID_MESSAGE_ROLES:
                issues.append({
                    "index": idx,
                    "code": "invalid_message_role",
                    "id": event_id,
                    "role": role,
                })
                role = None

            if role == "assistant" and previous_message_role == "assistant":
                issues.append({
                    "index": idx,
                    "code": "consecutive_assistant_messages",
                    "id": event_id,
                })

            if role == "assistant":
                has_text_or_tool = False
                for item in _content(event):
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "text" and str(item.get("text", "")).strip():
                        has_text_or_tool = True
                    if item.get("type") == "toolCall":
                        has_text_or_tool = True
                        tc_id = item.get("id")
                        if not isinstance(tc_id, str) or not tc_id:
                            issues.append({
                                "index": idx,
                                "code": "tool_call_missing_id",
                                "event_id": event_id,
                            })
                        elif tc_id in seen_tool_call_ids:
                            issues.append({
                                "index": idx,
                                "code": "duplicate_tool_call_id",
                                "toolCallId": tc_id,
                            })
                        else:
                            seen_tool_call_ids.add(tc_id)
                if not has_text_or_tool:
                    issues.append({
                        "index": idx,
                        "code": "empty_assistant_message",
                        "id": event_id,
                    })

            if role == "toolResult":
                tc_id = msg.get("toolCallId")
                if not isinstance(tc_id, str) or not tc_id:
                    issues.append({
                        "index": idx,
                        "code": "tool_result_missing_tool_call_id",
                        "id": event_id,
                    })
                elif tc_id not in seen_tool_call_ids:
                    issues.append({
                        "index": idx,
                        "code": "orphan_tool_result",
                        "id": event_id,
                        "toolCallId": tc_id,
                    })

            if role:
                previous_message_role = role

        if event_id:
            seen_event_ids.add(event_id)

    return issues


def select_active_branch(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the reachable parentId branch ending at the final event.

    OpenClaw session files are append-only and can contain abandoned branches
    from retries, model fallbacks, or manual branch changes. The export path
    walks parentId links from the leaf before packaging a trajectory; privacy
    rewriting needs the same shape or old branch tool call ids can appear twice.
    """
    has_session = any(
        isinstance(event, dict) and event.get("type") == "session"
        for event in events
    )
    if not has_session:
        return deepcopy(events)

    event_by_id: dict[str, dict[str, Any]] = {}
    leaf_id = ""
    for event in events:
        if not isinstance(event, dict):
            continue
        event_id = _event_id(event)
        if not event_id:
            continue
        event_by_id[event_id] = event
        if event.get("type") != "session":
            leaf_id = event_id

    if not leaf_id or leaf_id not in event_by_id:
        return deepcopy(events)

    branch_ids: set[str] = set()
    current_id: str | None = leaf_id
    while current_id and current_id in event_by_id and current_id not in branch_ids:
        branch_ids.add(current_id)
        parent = event_by_id[current_id].get("parentId")
        if isinstance(parent, str) and parent and parent not in event_by_id:
            return deepcopy(events)
        current_id = parent if isinstance(parent, str) and parent else None

    if not branch_ids:
        return deepcopy(events)

    # Synthetic or already-linear datasets sometimes have sparse parentIds.
    # If following the final leaf would throw away most of the conversation,
    # keep the stream and let parent repair make it usable.
    non_session_count = sum(
        1
        for event in events
        if isinstance(event, dict) and event.get("type") != "session" and _event_id(event)
    )
    if len(branch_ids) < max(2, non_session_count // 2):
        return deepcopy(events)

    return [
        deepcopy(event)
        for event in events
        if isinstance(event, dict)
        and (event.get("type") == "session" or _event_id(event) in branch_ids)
    ]


def drop_empty_assistant_messages(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove aborted assistant stubs and relink their children to the stub parent."""
    parent_redirects: dict[str, str | None] = {}
    kept: list[dict[str, Any]] = []

    for original in events:
        if not isinstance(original, dict):
            continue
        event = deepcopy(original)
        parent = event.get("parentId")
        if isinstance(parent, str) and parent in parent_redirects:
            redirected = parent_redirects[parent]
            if redirected is None:
                event.pop("parentId", None)
            else:
                event["parentId"] = redirected

        role = _message_role(event)
        event_id = _event_id(event)
        if role == "assistant" and event_id:
            content = _content(event)
            has_visible_content = any(
                isinstance(item, dict)
                and (
                    item.get("type") == "toolCall"
                    or (
                        item.get("type") in {"text", "image"}
                        and str(item.get("text") or item.get("url") or "").strip()
                    )
                )
                for item in content
            )
            if not has_visible_content:
                parent_value = event.get("parentId")
                parent_redirects[event_id] = parent_value if isinstance(parent_value, str) else None
                continue

        kept.append(event)

    return kept


def _assistant_has_tool_call(event: dict[str, Any]) -> bool:
    return any(
        isinstance(item, dict) and item.get("type") == "toolCall"
        for item in _content(event)
    )


def _assistant_text(event: dict[str, Any]) -> str:
    return "\n".join(
        str(item.get("text", ""))
        for item in _content(event)
        if isinstance(item, dict) and item.get("type") == "text"
    ).strip()


def _is_denied_plaintext_boilerplate(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in (
        "denied plaintext",
        "plaintext file because it would preserve sensitive data",
        "won't read or update the denied",
        "can't read or update that plaintext file",
        "can't append to the denied plaintext file",
    ))


def _replace_text_content(event: dict[str, Any], text: str) -> dict[str, Any]:
    updated = deepcopy(event)
    content = _content(updated)
    replaced = False
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            item["text"] = text
            replaced = True
            break
    if not replaced:
        content.append({"type": "text", "text": text})
    return updated


def collapse_consecutive_text_assistant_messages(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse text-only assistant duplicates introduced by privacy patching.

    Tool-call assistant messages are never merged because their paired
    toolResult events depend on exact ordering. Text-only denial boilerplate can
    appear beside a safe final answer when a blocked plaintext-file write is
    removed; keep the substantive answer and drop duplicate boilerplate.
    """
    kept: list[dict[str, Any]] = []
    for original in events:
        if not isinstance(original, dict):
            continue
        event = deepcopy(original)
        if (
            kept
            and _message_role(event) == "assistant"
            and _message_role(kept[-1]) == "assistant"
            and not _assistant_has_tool_call(event)
            and not _assistant_has_tool_call(kept[-1])
        ):
            previous_text = _assistant_text(kept[-1])
            current_text = _assistant_text(event)
            if previous_text == current_text:
                continue
            if _is_denied_plaintext_boilerplate(previous_text):
                kept[-1] = event
                continue
            if _is_denied_plaintext_boilerplate(current_text):
                continue
            merged_text = "\n\n".join(part for part in (previous_text, current_text) if part)
            kept[-1] = _replace_text_content(kept[-1], merged_text)
            continue
        kept.append(event)
    return kept


def drop_orphan_tool_results(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop toolResult events whose toolCallId is not present earlier in stream."""
    kept: list[dict[str, Any]] = []
    seen_tool_call_ids: set[str] = set()

    for original in events:
        if not isinstance(original, dict):
            continue
        event = deepcopy(original)

        if _message_role(event) == "assistant":
            for item in _content(event):
                if isinstance(item, dict) and item.get("type") == "toolCall":
                    tc_id = item.get("id")
                    if isinstance(tc_id, str) and tc_id:
                        seen_tool_call_ids.add(tc_id)

        if _message_role(event) == "toolResult":
            msg = event.get("message", {})
            tc_id = msg.get("toolCallId") if isinstance(msg, dict) else None
            if not isinstance(tc_id, str) or tc_id not in seen_tool_call_ids:
                continue

        kept.append(event)

    return kept


def repair_parent_chain(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fill missing/broken parent links without changing event order or tool grouping."""
    repaired = deepcopy(events)
    previous_id: str | None = None
    seen_ids: set[str] = set()
    for event in repaired:
        if not isinstance(event, dict):
            continue
        if event.get("type") == "session":
            event.pop("parentId", None)
        elif event.get("type") == "model_change" and previous_id is None:
            event["parentId"] = None
        else:
            parent = event.get("parentId")
            parent_is_valid = isinstance(parent, str) and parent in seen_ids
            if previous_id and not parent_is_valid:
                event["parentId"] = previous_id
            elif not previous_id and not parent_is_valid:
                event["parentId"] = None
        event_id = _event_id(event)
        if event_id:
            seen_ids.add(event_id)
            previous_id = event_id
    return repaired


def normalize_event_stream(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a linear, SFT-ready OpenClaw event stream."""
    branch = select_active_branch(events)
    pruned = drop_empty_assistant_messages(branch)
    collapsed = collapse_consecutive_text_assistant_messages(pruned)
    without_orphans = drop_orphan_tool_results(collapsed)
    return repair_parent_chain(without_orphans)
