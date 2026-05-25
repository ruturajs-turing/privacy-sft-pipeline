"""Deterministic redaction helpers for delivery artifacts.

Training trajectories may keep sensitive values in user-provided text and in
tool-call arguments because those are the explicit data-flow surfaces. The same
values must not leak through assistant prose, tool results, metadata, or memory
files after the agent has "recalled" them.
"""
from __future__ import annotations

import re
import json
from typing import Any

from models import PIIEntity, PIIMap

REDACTION_TOKEN = "__PII_REDACTED__"

_LEVEL_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "BLOCK": 5}


def sensitive_entity_texts(pii_map: PIIMap, min_level: int = 3) -> list[str]:
    """Return unique sensitive entity values, longest first."""
    values = [entity.text for entity in sensitive_entities(pii_map, min_level=min_level)]
    values.sort(key=len, reverse=True)
    return values


def sensitive_entities(pii_map: PIIMap, min_level: int = 3) -> list[PIIEntity]:
    """Return unique sensitive entities, longest text first."""
    seen: set[str] = set()
    entities: list[PIIEntity] = []
    for entity in pii_map.entities:
        if _LEVEL_ORDER.get(entity.level, 0) < min_level:
            continue
        value = (entity.text or "").strip()
        if len(value) < 2 or value in seen:
            continue
        seen.add(value)
        entities.append(entity)
    entities.sort(key=lambda entity: len(entity.text or ""), reverse=True)
    return entities


_ADDRESS_SUFFIX_PATTERNS = {
    "avenue": r"(?:avenue|ave\.?)",
    "ave": r"(?:avenue|ave\.?)",
    "street": r"(?:street|st\.?)",
    "st": r"(?:street|st\.?)",
    "road": r"(?:road|rd\.?)",
    "rd": r"(?:road|rd\.?)",
    "boulevard": r"(?:boulevard|blvd\.?)",
    "blvd": r"(?:boulevard|blvd\.?)",
    "drive": r"(?:drive|dr\.?)",
    "dr": r"(?:drive|dr\.?)",
    "lane": r"(?:lane|ln\.?)",
    "ln": r"(?:lane|ln\.?)",
}


def _flexible_word_pattern(parts: list[str]) -> str:
    rendered: list[str] = []
    for part in parts:
        rendered.append(_ADDRESS_SUFFIX_PATTERNS.get(part.lower(), re.escape(part)))
    return r"(?<!\w)" + r"[\s._:/-]*".join(rendered) + r"(?!\w)"


def _patterns_for_value(value: str) -> list[re.Pattern[str]]:
    patterns = [re.compile(re.escape(value), re.IGNORECASE)]

    # Match common punctuation/spacing changes introduced by assistant prose,
    # for example "Metoprolol 25mg. twice daily" when the classifier saw
    # "Metoprolol 25mg twice daily".
    parts = re.findall(r"[A-Za-z]+|\d+(?:\.\d+)?", value)
    if len(parts) >= 2:
        patterns.append(re.compile(_flexible_word_pattern(parts), re.IGNORECASE))

    # Catch common formatted variants of IDs/cards/tokens that differ only in
    # spaces or dashes, while avoiding broad digit matching for short numbers.
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 8:
        digit_pattern = r"(?<!\d)" + r"[\s-]*".join(map(re.escape, digits)) + r"(?!\d)"
        patterns.append(re.compile(digit_pattern))

    return patterns


def _patterns_for_medication_value(value: str) -> list[re.Pattern[str]]:
    match = re.match(r"\s*([A-Za-z][A-Za-z-]{2,})\b", value)
    if not match:
        return []
    medication = re.escape(match.group(1))
    dose = r"(?:\s+\d+(?:\.\d+)?\s*(?:mcg|mg|g|ml|units?)\.?)?"
    frequency_word = (
        r"once|twice|three\s+times|four\s+times|daily|nightly|weekly|monthly|"
        r"morning|evening|bedtime|prn|bid|tid|qid|as\s+needed"
    )
    frequency = rf"(?:\s+(?:{frequency_word}))*"
    return [re.compile(rf"\b{medication}\b{dose}{frequency}", re.IGNORECASE)]


_PROFILE_LINE_TERMS = (
    "allerg", "diagnos", "health", "inhaler", "medical", "medication",
    "respiratory", "sensitive", "symptom", "trigger", "treatment", "uses",
)


def _redact_profile_lines(text: str) -> str:
    """Collapse sensitive-looking profile bullets to avoid partial leaks."""
    lines: list[str] = []
    changed = False
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        newline = line[len(body):]
        lowered = body.lower()
        looks_like_profile_bullet = bool(re.match(r"^\s*[-*]\s+", body))
        sensitive_profile_line = (
            any(term in lowered for term in _PROFILE_LINE_TERMS)
            and (
                REDACTION_TOKEN.lower() in lowered
                or (looks_like_profile_bullet and ":" in body)
            )
        )
        if sensitive_profile_line:
            if looks_like_profile_bullet:
                changed = True
                continue
            bold_label = re.match(r"^(.+?:\*\*)\s*.*$", body)
            if bold_label:
                body = bold_label.group(1) + f" {REDACTION_TOKEN}"
            elif ":" in body:
                body = body.split(":", 1)[0] + f": {REDACTION_TOKEN}"
            else:
                body = REDACTION_TOKEN
            changed = True
        lines.append(body + newline)
    return "".join(lines) if changed else text


def _patterns_for_entity(entity: PIIEntity) -> list[re.Pattern[str]]:
    patterns = _patterns_for_value(entity.text)
    if "HEALTH_MEDICATION" in entity.label:
        patterns.extend(_patterns_for_medication_value(entity.text))
    return patterns


def redact_text(text: str, pii_map: PIIMap, min_level: int = 3) -> str:
    """Replace L3/L4 entity values in text with the standard redaction token."""
    if not text:
        return text
    redacted = text
    for entity in sensitive_entities(pii_map, min_level=min_level):
        for pattern in _patterns_for_entity(entity):
            redacted = pattern.sub(REDACTION_TOKEN, redacted)
    return _redact_profile_lines(redacted)


def redact_value(value: Any, pii_map: PIIMap, min_level: int = 3) -> Any:
    """Recursively redact strings inside JSON-like values."""
    if isinstance(value, str):
        return redact_text(value, pii_map, min_level=min_level)
    if isinstance(value, list):
        return [redact_value(item, pii_map, min_level=min_level) for item in value]
    if isinstance(value, dict):
        return {
            key: redact_value(item, pii_map, min_level=min_level)
            for key, item in value.items()
        }
    return value


def redact_file_map(files: dict[str, str], pii_map: PIIMap, min_level: int = 3) -> dict[str, str]:
    """Redact sensitive values in workspace file names and contents."""
    return {
        redact_text(name, pii_map, min_level=min_level): redact_text(content, pii_map, min_level=min_level)
        for name, content in files.items()
    }


def redact_event_stream(events: list[dict], pii_map: PIIMap, min_level: int = 3) -> list[dict]:
    """Redact sensitive values outside user text and tool-call arguments."""
    redacted = json.loads(json.dumps(events))
    for evt in redacted:
        if not isinstance(evt, dict):
            continue

        if evt.get("type") == "custom":
            evt["data"] = redact_value(evt.get("data", {}), pii_map, min_level=min_level)
            continue

        msg = evt.get("message", {})
        if not isinstance(msg, dict):
            continue

        role = msg.get("role", "")
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
                        item[field] = redact_text(item[field], pii_map, min_level=min_level)

        if role == "toolResult":
            for field in ("details", "errorMessage"):
                if field in msg:
                    msg[field] = redact_value(msg[field], pii_map, min_level=min_level)

    return redacted
