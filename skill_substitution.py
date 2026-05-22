"""Skill substitution — capability-aware lower-tier tool replacement.

Ported from cuarena-pipeline feat/skill-substitution (Vitor, 328f11d).

Four-layer architecture (cheap to expensive):
  Layer 1   — Capability lookup            (deterministic, microseconds)
  Layer 2   — Structural I/O field check   (deterministic, microseconds)
  Layer 2.5 — Per-capability scope guards  (deterministic, microseconds)
  Layer 3   — LLM verdict                  (reserved, not wired)

Given a tool call, returns whether a lower-tier alternative exists and classifies
the substitution as drop-in, breaks-downstream, or out-of-scope.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from config import DATA_DIR


# ---------------------------------------------------------------------------
# Pydantic models (from catalog.py)
# ---------------------------------------------------------------------------

Tier = Literal[1, 2, 3]
Verdict = Literal["drop-in", "breaks-downstream", "out-of-scope"]


class Returns(BaseModel):
    shape: str
    fields_consumable: list[str] = Field(default_factory=list)
    notes: str = ""


class CapabilityEntry(BaseModel):
    tier: Tier
    skill: str
    tool: str
    params: dict[str, str] = Field(default_factory=dict)
    param_map: dict[str, str] = Field(default_factory=dict)
    returns: Returns
    side_effects: list[str] = Field(default_factory=list)
    scope: str
    constraints: list[str] = Field(default_factory=list)
    verified: bool
    source: str


class Capability(BaseModel):
    description: str
    skills: list[CapabilityEntry]


class Catalog(BaseModel):
    version: int
    updated: str
    capabilities: dict[str, Capability]


class Candidate(BaseModel):
    skill: str
    tool: str
    tier: int
    verdict: Verdict
    reason: str
    param_map: dict[str, str] = Field(default_factory=dict)


class SubstitutionResult(BaseModel):
    tool_name: str
    skill: Optional[str] = None
    tool_tier: Optional[int] = None
    capability: Optional[str] = None
    candidates: List[Candidate] = Field(default_factory=list)
    has_drop_in: bool = False
    consumed_fields: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

CATALOG_PATH = DATA_DIR / "capabilities.json"
REVERSE_PATH = DATA_DIR / "skill_capabilities.json"

_catalog_cache: Optional[Catalog] = None


def load_catalog(path: Path = CATALOG_PATH) -> Catalog:
    global _catalog_cache
    if _catalog_cache is None:
        _catalog_cache = Catalog.model_validate_json(path.read_text(encoding="utf-8"))
    return _catalog_cache


def build_reverse_index(catalog: Catalog) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for cap_id, cap in catalog.capabilities.items():
        for entry in cap.skills:
            if cap_id not in out[entry.skill]:
                out[entry.skill].append(cap_id)
    return {k: out[k] for k in sorted(out)}


# ---------------------------------------------------------------------------
# Scope checks (Layer 2.5) — per-capability deterministic guards
# ---------------------------------------------------------------------------

Checker = Callable[[dict, CapabilityEntry], Optional[Tuple[str, str]]]
CHECKERS: dict[str, list[Checker]] = {}


def register(capability_id: str) -> Callable[[Checker], Checker]:
    def deco(fn: Checker) -> Checker:
        CHECKERS.setdefault(capability_id, []).append(fn)
        return fn
    return deco


def run_checks(
    capability_id: str, args: dict | None, alt: CapabilityEntry,
) -> tuple[str, str] | None:
    if args is None:
        return None
    for check in CHECKERS.get(capability_id, []):
        result = check(args, alt)
        if result is not None:
            return result
    return None


@register("notes.write")
def _notes_write_notion_specific(args: dict, alt: CapabilityEntry) -> tuple[str, str] | None:
    """Reject substitution to obsidian when Notion-specific args are present."""
    if alt.skill != "obsidian":
        return None
    if args.get("parent"):
        return (
            "out-of-scope",
            f"Notion-specific `parent={args['parent']!r}` — "
            "obsidian has no nested-page hierarchy",
        )
    if args.get("properties"):
        return (
            "out-of-scope",
            "Notion database `properties` — obsidian notes "
            "are flat Markdown with no database-style typed fields",
        )
    return None


@register("mail.send")
def _mail_send_corp_policy(args: dict, alt: CapabilityEntry) -> tuple[str, str] | None:
    """Block substitution to enterprise-mail for non-corp recipients."""
    if alt.skill != "enterprise-mail":
        return None
    recipients = args.get("to") or []
    if not isinstance(recipients, list):
        return None
    for r in recipients:
        if not isinstance(r, str) or "@" not in r:
            continue
        if not r.endswith("@corp.com"):
            return (
                "out-of-scope",
                f"recipient {r!r} is external — corp-mail policy blocks substituting "
                "to enterprise-mail",
            )
    return None


# ---------------------------------------------------------------------------
# Consumed fields tracker (from checker.py)
# ---------------------------------------------------------------------------

_COMMON_WORDS = frozenset({
    "open", "ok", "true", "false", "yes", "no", "done", "error",
    "null", "none", "name", "id", "user", "data", "value", "key", "type",
})


def _looks_like_real_value(value) -> bool:
    if value in (None, True, False, ""):
        return False
    s = str(value)
    if not s:
        return False
    if not s.isalnum():
        return True
    if s.isdigit():
        return len(s) >= 2
    if len(s) < 4:
        return False
    return s.lower() not in _COMMON_WORDS


def _appears_word_bounded(value: str, text: str) -> bool:
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(value) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, text) is not None


def trace_consumed_fields(result_text: str, subsequent_text: str) -> list[str] | None:
    """Return top-level result fields whose values appear in subsequent text.

    Returns None when the result isn't a JSON object — caller falls back to
    strict field-coverage mode.
    """
    try:
        obj = json.loads(result_text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    consumed: list[str] = []
    for key, value in obj.items():
        if isinstance(value, (str, int, float)):
            if _looks_like_real_value(value) and _appears_word_bounded(str(value), subsequent_text):
                consumed.append(key)
        elif isinstance(value, list):
            for item in value:
                if (isinstance(item, (str, int, float))
                        and _looks_like_real_value(item)
                        and _appears_word_bounded(str(item), subsequent_text)):
                    consumed.append(key)
                    break
    return consumed


# ---------------------------------------------------------------------------
# Evaluator (Layers 1, 2, 2.5)
# ---------------------------------------------------------------------------

def _find_entry(
    catalog: Catalog, tool_name: str, tier_hint: int | None = None,
) -> tuple[str, CapabilityEntry] | None:
    hits: list[tuple[str, CapabilityEntry]] = []
    for cap_id, cap in catalog.capabilities.items():
        for entry in cap.skills:
            if entry.tool == tool_name:
                hits.append((cap_id, entry))
    if not hits:
        return None
    if tier_hint is not None:
        for cap_id, e in hits:
            if e.tier == tier_hint:
                return cap_id, e
    return hits[0]


def _classify(
    orig: CapabilityEntry,
    alt: CapabilityEntry,
    consumed_fields: list[str] | None = None,
    args: dict | None = None,
    capability_id: str | None = None,
) -> tuple[Verdict, str]:
    """Layer 2: structural I/O check + Layer 2.5: scope guards."""
    alt_fields = set(alt.returns.fields_consumable)

    if consumed_fields is not None:
        needed = set(consumed_fields)
        missing = sorted(needed - alt_fields)
        if missing:
            base: tuple[Verdict, str] = (
                "breaks-downstream",
                f"trajectory consumes fields not in alt return: {missing}",
            )
        else:
            base = (
                "drop-in",
                f"alt return covers fields actually consumed by trajectory: "
                f"{sorted(needed) or '[]'}",
            )
    else:
        orig_fields = set(orig.returns.fields_consumable)
        missing = sorted(orig_fields - alt_fields)
        if missing:
            base = (
                "breaks-downstream",
                f"alt return missing fields exposed by orig: {missing}",
            )
        else:
            base = ("drop-in", "alt return covers orig's consumable fields")

    # Layer 2.5: per-capability scope guards
    if base[0] == "drop-in" and capability_id is not None and args is not None:
        override = run_checks(capability_id, args, alt)
        if override is not None:
            return override
    return base


def evaluate(
    catalog: Catalog,
    tool_name: str,
    tier_hint: int | None = None,
    consumed_fields: list[str] | None = None,
    args: dict | None = None,
) -> SubstitutionResult:
    """Main entry point — evaluate a tool call for lower-tier substitution.

    Returns a SubstitutionResult with candidates sorted by tier (lowest first).
    If has_drop_in is True, at least one candidate can replace the tool.
    """
    hit = _find_entry(catalog, tool_name, tier_hint=tier_hint)
    if hit is None:
        return SubstitutionResult(tool_name=tool_name)
    cap_id, orig = hit

    candidates: list[Candidate] = []
    for alt in catalog.capabilities[cap_id].skills:
        if alt.tier >= orig.tier:
            continue
        verdict, reason = _classify(
            orig, alt,
            consumed_fields=consumed_fields,
            args=args,
            capability_id=cap_id,
        )
        candidates.append(Candidate(
            skill=alt.skill,
            tool=alt.tool,
            tier=alt.tier,
            verdict=verdict,
            reason=reason,
            param_map=alt.param_map,
        ))
    candidates.sort(key=lambda c: c.tier)
    return SubstitutionResult(
        tool_name=tool_name,
        skill=orig.skill,
        tool_tier=orig.tier,
        capability=cap_id,
        candidates=candidates,
        has_drop_in=any(c.verdict == "drop-in" for c in candidates),
        consumed_fields=consumed_fields,
    )


# ---------------------------------------------------------------------------
# Trajectory-level helpers for consumed-fields analysis
# ---------------------------------------------------------------------------

def build_result_index(trajectory) -> tuple[dict[str, tuple[int, str]], list[tuple[int, str]]]:
    """Build a tool result index from a ParsedTrajectory.

    Returns:
      result_index: tool_call_id -> (turn_index, result_text)
      all_texts: list of (turn_index, text_content) from all turns
    """
    result_index: dict[str, tuple[int, str]] = {}
    all_texts: list[tuple[int, str]] = []

    for turn in trajectory.assistant_turns:
        idx = turn.turn_index

        if turn.text:
            all_texts.append((idx, turn.text))
        if hasattr(turn, 'thinking') and turn.thinking:
            all_texts.append((idx, turn.thinking))

        for tc in turn.tool_calls:
            args_text = json.dumps(tc.arguments or {}, ensure_ascii=False)
            all_texts.append((idx, args_text))

            if tc.call_id and tc.call_id in trajectory.tool_results_by_call_id:
                result = trajectory.tool_results_by_call_id[tc.call_id]
                result_text = result.content if hasattr(result, 'content') else str(result)
                result_index[tc.call_id] = (idx, result_text)
                all_texts.append((idx, result_text))

    for msg_idx, msg in enumerate(trajectory.user_messages):
        all_texts.append((msg_idx, msg))

    return result_index, all_texts


def consumed_for_call(
    tc_call_id: str | None,
    result_index: dict[str, tuple[int, str]],
    all_texts: list[tuple[int, str]],
) -> list[str] | None:
    """Get consumed fields for a specific tool call."""
    if not tc_call_id or tc_call_id not in result_index:
        return None
    result_idx, result_text = result_index[tc_call_id]
    subsequent = "\n".join(t for (idx, t) in all_texts if idx > result_idx)
    return trace_consumed_fields(result_text, subsequent)
