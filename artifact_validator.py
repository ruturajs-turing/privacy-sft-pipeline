"""Validate delivered trajectory artifacts for obvious synthetic fingerprints."""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

FINGERPRINT_KEYS = {"is_adversarial", "attack_type", "is_synthetic", "_synthetic"}
REDACTION_TOKEN = "__PII_REDACTED__"

SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")
API_KEY_RE = re.compile(r"\b(?:sk|pk|rk|xox[baprs]|gh[pousr])-[A-Za-z0-9_-]{8,}\b", re.IGNORECASE)
SECRET_ASSIGNMENT_RE = re.compile(
    r"\b(?:password|passcode|api[_ -]?key|secret|token|credential|cvv|cvc|ssn)\b\s*[:=]\s*"
    r"(?!__PII_REDACTED__)([^\s,;]{3,})",
    re.IGNORECASE,
)
HEALTH_MEMORY_RE = re.compile(
    r"\b(?:diagnosis|condition|medication|allergy|allergies|symptom|patient|health)\b\s*[:=]\s*"
    r"(?!__PII_REDACTED__)([^,\n;]{3,})",
    re.IGNORECASE,
)
CARD_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def _ts_to_ms(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return int(dt.timestamp() * 1000)


def _contains_fingerprint_key(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FINGERPRINT_KEYS:
                return key
            found = _contains_fingerprint_key(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _contains_fingerprint_key(item)
            if found:
                return found
    return None


def _role(event: dict[str, Any]) -> str:
    msg = event.get("message", {}) if isinstance(event.get("message"), dict) else {}
    role = msg.get("role")
    return role if isinstance(role, str) else ""


def _tool_calls(event: dict[str, Any]) -> list[dict[str, Any]]:
    msg = event.get("message", {}) if isinstance(event.get("message"), dict) else {}
    content = msg.get("content", [])
    if not isinstance(content, list):
        return []
    return [
        item for item in content
        if isinstance(item, dict) and item.get("type") == "toolCall"
    ]


def validate_events(events: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    seen_event_ids: set[str] = set()
    seen_tool_ids: set[str] = set()
    previous_timestamp: int | None = None
    previous_message_role = ""
    milliseconds: list[int] = []

    for idx, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            issues.append(f"line {idx}: event is not an object")
            continue

        event_id = event.get("id")
        if isinstance(event_id, str):
            if "synth" in event_id.lower():
                issues.append(f"line {idx}: event id contains synth")
            if event_id in seen_event_ids:
                issues.append(f"line {idx}: duplicate event id {event_id}")
        elif event.get("type") != "session":
            issues.append(f"line {idx}: missing event id")

        parent_id = event.get("parentId")
        if parent_id not in (None, "") and isinstance(parent_id, str):
            if parent_id not in seen_event_ids:
                issues.append(f"line {idx}: parentId does not reference a prior event")

        fingerprint_key = _contains_fingerprint_key(event)
        if fingerprint_key:
            issues.append(f"line {idx}: construction metadata key leaked: {fingerprint_key}")

        ts_ms = _ts_to_ms(event.get("timestamp") or event.get("createdAt"))
        if ts_ms is not None:
            milliseconds.append(ts_ms % 1000)
            if previous_timestamp is not None and ts_ms < previous_timestamp:
                issues.append(f"line {idx}: timestamp moved backwards")
            previous_timestamp = ts_ms

        role = _role(event)
        if role == "assistant" and previous_message_role == "assistant":
            issues.append(f"line {idx}: consecutive assistant message")
        if role == "assistant":
            has_visible_content = False
            msg = event.get("message", {}) if isinstance(event.get("message"), dict) else {}
            content = msg.get("content", []) if isinstance(msg, dict) else []
            if isinstance(content, list):
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
                issues.append(f"line {idx}: empty assistant message")
        if role in {"assistant", "user", "toolResult"}:
            previous_message_role = role

        for tool_call in _tool_calls(event):
            tool_id = tool_call.get("id")
            if not isinstance(tool_id, str) or not tool_id:
                issues.append(f"line {idx}: tool call missing id")
                continue
            if "synth" in tool_id.lower():
                issues.append(f"line {idx}: tool call id contains synth")
            if tool_id in seen_tool_ids:
                issues.append(f"line {idx}: duplicate tool call id {tool_id}")
            seen_tool_ids.add(tool_id)

        if role == "toolResult":
            msg = event.get("message", {})
            tool_ref = msg.get("toolCallId") if isinstance(msg, dict) else None
            if not isinstance(tool_ref, str) or not tool_ref:
                issues.append(f"line {idx}: toolResult missing toolCallId")
            elif tool_ref not in seen_tool_ids:
                issues.append(f"line {idx}: toolResult references unknown toolCallId")

        if isinstance(event_id, str):
            seen_event_ids.add(event_id)

    if len(milliseconds) >= 4 and len(set(milliseconds)) <= 1:
        issues.append("all event timestamps have the same millisecond value")

    return issues


def _luhn_valid(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for idx, char in enumerate(digits):
        value = ord(char) - ord("0")
        if idx % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _safe_memory_line(line: str) -> bool:
    lowered = line.lower()
    return (
        REDACTION_TOKEN.lower() in lowered
        or "exclude" in lowered
        or "excluded" in lowered
        or "do not" in lowered
        or "don't" in lowered
        or "not save" in lowered
        or "not retain" in lowered
    )


def validate_memory_text(text: str) -> list[str]:
    """Validate redacted memory/workspace-memory content without needing a PII map."""
    issues: list[str] = []
    if REDACTION_TOKEN in text:
        scan_text = text.replace(REDACTION_TOKEN, "")
    else:
        scan_text = text

    if SSN_RE.search(scan_text):
        issues.append("memory file contains unredacted SSN-like value")
    if IBAN_RE.search(scan_text):
        issues.append("memory file contains unredacted IBAN-like value")
    if API_KEY_RE.search(scan_text):
        issues.append("memory file contains unredacted API-key-like value")
    if SECRET_ASSIGNMENT_RE.search(scan_text):
        issues.append("memory file contains unredacted credential assignment")

    for match in CARD_CANDIDATE_RE.finditer(scan_text):
        digits = re.sub(r"\D", "", match.group(0))
        line_start = scan_text.rfind("\n", 0, match.start()) + 1
        line_end = scan_text.find("\n", match.end())
        if line_end == -1:
            line_end = len(scan_text)
        line = scan_text[line_start:line_end].lower()
        card_context = any(token in line for token in ("card", "credit", "debit", "pan", "cc "))
        if 13 <= len(digits) <= 19 and (_luhn_valid(digits) or card_context):
            issues.append("memory file contains unredacted payment-card-like value")
            break

    for line_no, line in enumerate(scan_text.splitlines(), start=1):
        if _safe_memory_line(line):
            continue
        if HEALTH_MEMORY_RE.search(line):
            issues.append(f"line {line_no}: memory file contains unredacted health profile value")

    return issues


def _is_memory_file(path: Path) -> bool:
    parts = path.parts
    name = path.name
    if name == "MEMORY.md":
        return True
    return name.endswith(".md") and "memory" in {part.lower() for part in parts}


def validate_memory_file(path: Path) -> list[str]:
    try:
        text = path.read_text(errors="replace")
    except OSError as exc:
        return [f"could not read memory file: {exc}"]
    return validate_memory_text(text)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def iter_trajectory_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("trajectory.jsonl")))
    return files


def iter_memory_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            if _is_memory_file(path):
                files.append(path)
        elif path.is_dir():
            for candidate in sorted(path.rglob("*.md")):
                if _is_memory_file(candidate):
                    files.append(candidate)
    return files


def main(argv: list[str]) -> int:
    paths = [Path(arg) for arg in argv[1:]]
    if not paths:
        print("usage: python3 artifact_validator.py <trajectory.jsonl|output-dir> [...]")
        return 2

    failures = 0
    checked = 0
    memory_checked = 0
    for file_path in iter_trajectory_files(paths):
        checked += 1
        issues = validate_events(load_jsonl(file_path))
        if issues:
            failures += 1
            print(f"FAIL {file_path}")
            for issue in issues[:25]:
                print(f"  - {issue}")
            if len(issues) > 25:
                print(f"  ... {len(issues) - 25} more")
        else:
            print(f"PASS {file_path}")

    for file_path in iter_memory_files(paths):
        memory_checked += 1
        issues = validate_memory_file(file_path)
        if issues:
            failures += 1
            print(f"FAIL {file_path}")
            for issue in issues[:25]:
                print(f"  - {issue}")
            if len(issues) > 25:
                print(f"  ... {len(issues) - 25} more")

    print(f"checked={checked} memory_checked={memory_checked} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
