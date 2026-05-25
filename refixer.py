"""Stage 4b: Refix loop — parse verifier issues and apply targeted fixes.

Takes a MINOR_ISSUES trajectory, categorizes the verifier's per-turn issues,
applies deterministic fixes where possible, and falls back to Claude for
natural-language fixes. Returns a patched RewriteResult ready for re-verification.
"""
from __future__ import annotations

import copy
import json
import logging
import re

import anthropic
import json_repair

from config import ANTHROPIC_API_KEY, REFIX_LLM_MODEL
from models import (
    AssistantTurn,
    ParsedTrajectory,
    PIIMap,
    RewriteResult,
    RewrittenTurn,
    VerificationIssue,
    VerificationResult,
)
from assembler import (
    HATCH_REFUSAL,
    _EMAIL_PATTERN,
    _minimize_write_content as _minimize_content,
)
from privacy_registry import (
    get_tool_tier,
    get_lower_tier_equivalent,
    is_persistence_path,
    _LEVEL_ORDER,
    _max_pii_level_in_content,
)
from token_tracker import tracker

logger = logging.getLogger(__name__)

# ── Issue categorization ─────────────────────────────────────────────────

_DETERMINISTIC_RULES = frozenset({
    "structural", "pii_leak", "rule_1", "rule_2",
})

_LLM_RULES = frozenset({
    "rule_5", "rule_6", "naturality",
})


def categorize_issues(
    issues: list[VerificationIssue],
) -> tuple[list[VerificationIssue], list[VerificationIssue]]:
    """Split issues into deterministic and LLM-assisted buckets."""
    deterministic: list[VerificationIssue] = []
    llm_assisted: list[VerificationIssue] = []

    for issue in issues:
        rule = issue.rule_violated.lower().strip()
        if rule in _DETERMINISTIC_RULES:
            deterministic.append(issue)
        elif rule in _LLM_RULES:
            llm_assisted.append(issue)
        else:
            # rule_3, rule_4, unknown — attempt deterministic first
            if issue.severity == "minor":
                deterministic.append(issue)
            else:
                llm_assisted.append(issue)

    deterministic.sort(key=lambda i: (
        0 if i.rule_violated == "structural" else
        1 if i.rule_violated == "pii_leak" else
        2 if i.rule_violated == "rule_1" else 3
    ))

    return deterministic, llm_assisted


# ── Deterministic fixers ─────────────────────────────────────────────────

def _fix_structural(
    turn: RewrittenTurn,
    issue: VerificationIssue,
    trajectory: ParsedTrajectory,
) -> bool:
    """Restore malformed tool calls from the original trajectory."""
    if not turn.tool_calls:
        return False

    original_turn = _find_original_turn(turn.turn_index, trajectory)
    if not original_turn:
        return False

    orig_by_id = {tc.call_id: tc for tc in original_turn.tool_calls}
    fixed = False
    for i, tc in enumerate(turn.tool_calls):
        if not isinstance(tc, dict):
            continue
        cid = tc.get("id", "")
        name = tc.get("name", "")
        args = tc.get("arguments", {})

        if not isinstance(args, dict) or not args:
            if cid in orig_by_id and orig_by_id[cid].name == name:
                tc["arguments"] = copy.deepcopy(orig_by_id[cid].arguments)
                fixed = True
            elif i < len(original_turn.tool_calls) and original_turn.tool_calls[i].name == name:
                tc["arguments"] = copy.deepcopy(original_turn.tool_calls[i].arguments)
                fixed = True

        if name == "edit" and isinstance(args, dict):
            edits = args.get("edits", [])
            if not edits or not isinstance(edits, list):
                if cid in orig_by_id:
                    tc["arguments"] = copy.deepcopy(orig_by_id[cid].arguments)
                    fixed = True

    # Remove tool calls that remain invalid after repair
    turn.tool_calls = [tc for tc in turn.tool_calls if isinstance(tc, dict) and _is_valid_tool_call_dict(tc)]
    return fixed


def _fix_pii_leak(
    turn: RewrittenTurn,
    issue: VerificationIssue,
    pii_map: PIIMap,
) -> bool:
    """Strip PII from the turn's text and tool arguments."""
    changed = False

    if turn.text:
        original = turn.text
        for ent in pii_map.entities:
            if ent.label == "ID_FULL_NAME":
                turn.text = re.sub(re.escape(ent.text), "the user", turn.text, flags=re.IGNORECASE)
                for part in ent.text.split():
                    if len(part) > 2 and part[0].isupper():
                        turn.text = re.sub(r'\b' + re.escape(part) + r'\b', 'the user', turn.text, flags=re.IGNORECASE)
            elif ent.label == "ID_EMAIL":
                turn.text = re.sub(re.escape(ent.text), "[email]", turn.text, flags=re.IGNORECASE)
            elif ent.label == "ID_PHONE":
                turn.text = turn.text.replace(ent.text, "[phone]")
            elif ent.label.startswith("HEALTH"):
                turn.text = re.sub(re.escape(ent.text), "[health info]", turn.text, flags=re.IGNORECASE)
            elif ent.label in ("ID_SSN", "ID_GOV"):
                turn.text = turn.text.replace(ent.text, "[REDACTED]")
            elif ent.label.startswith("EMP_"):
                turn.text = re.sub(re.escape(ent.text), "[work info]", turn.text, flags=re.IGNORECASE)
        turn.text = _EMAIL_PATTERN.sub("[email]", turn.text)
        turn.text = re.sub(r'\+?\d[\d\s\-]{8,}\d', '[phone]', turn.text)
        if turn.text != original:
            changed = True

    for tc in turn.tool_calls:
        if not isinstance(tc, dict):
            continue
        args = tc.get("arguments", {})
        if not isinstance(args, dict):
            continue
        content = args.get("content", "")
        if content and isinstance(content, str):
            minimized = _minimize_content(content, pii_map.entities)
            if minimized != content:
                args["content"] = minimized
                changed = True

    return changed


def _fix_classification(
    turn: RewrittenTurn,
    issue: VerificationIssue,
    pii_map: PIIMap,
) -> bool:
    """No-op — thinking blocks removed in v2 assembler. Classification is implicit."""
    return False


def _fix_over_collection(
    turn: RewrittenTurn,
    issue: VerificationIssue,
    pii_map: PIIMap,
) -> bool:
    """Remove unnecessary tool calls or strip extra fields."""
    if not turn.tool_calls:
        return False

    before_count = len(turn.tool_calls)

    # If the issue mentions a specific tool call, remove it
    desc_lower = (issue.description + " " + issue.fix_instruction).lower()
    if "remove" in desc_lower or "unnecessary" in desc_lower or "extra" in desc_lower:
        persistence_tcs = []
        other_tcs = []
        for tc in turn.tool_calls:
            if not isinstance(tc, dict):
                continue
            path = (tc.get("arguments") or {}).get("path", "") or ""
            if is_persistence_path(path) or tc.get("name") in ("memory_write", "active_memory_write", "set", "vault_set"):
                persistence_tcs.append(tc)
            else:
                other_tcs.append(tc)

        if persistence_tcs and other_tcs:
            turn.tool_calls = other_tcs
        elif persistence_tcs and not other_tcs:
            # Minimize instead of removing
            for tc in persistence_tcs:
                args = tc.get("arguments", {})
                if "content" in args and isinstance(args["content"], str):
                    args["content"] = _minimize_content(args["content"], pii_map.entities)
            turn.tool_calls = persistence_tcs

    return len(turn.tool_calls) != before_count


# ── LLM-assisted fixers ──────────────────────────────────────────────────

async def _fix_with_llm(
    turn: RewrittenTurn,
    issues: list[VerificationIssue],
    trajectory: ParsedTrajectory,
    pii_map: PIIMap,
) -> bool:
    """Use Claude to rewrite the turn's text based on verifier feedback."""
    if not issues:
        return False

    fix_instructions = "\n".join(
        f"- [{i.rule_violated}/{i.severity}] {i.description}"
        + (f" Fix: {i.fix_instruction}" if i.fix_instruction else "")
        for i in issues
    )

    # Build minimal context
    user_msg = ""
    if turn.turn_index < len(trajectory.user_messages):
        user_msg = trajectory.user_messages[turn.turn_index][:300]

    current_state = json.dumps({
        "text": turn.text[:500],
        "tool_calls": [tc.get("name", "?") for tc in turn.tool_calls if isinstance(tc, dict)],
        "privacy_actions": turn.privacy_actions,
    }, indent=2)

    prompt = f"""Fix this assistant turn based on verifier feedback.

USER MESSAGE (context): {user_msg}

CURRENT TURN STATE:
{current_state}

VERIFIER ISSUES TO FIX:
{fix_instructions}

RULES:
- Write natural, casual text — no thinking blocks, no template classification
- If the issue is about missing alternatives, add a concrete one
- If the issue is about local-first, suggest a T1 local tool
- If the issue is about naturality, rewrite for natural conversational flow
- Keep the Hatch refusal string when refusing L3/L4: "{HATCH_REFUSAL}"
- Do NOT add PII to the response
- Keep it concise (2-3 sentences max)

Return JSON: {{"text": "fixed assistant text"}}"""

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    try:
        response = await client.messages.create(
            model=REFIX_LLM_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        tracker.record_anthropic(response, "refixer")
    except Exception as e:
        logger.warning("Refix LLM call failed for turn %d: %s", turn.turn_index, e)
        return False

    resp_text = response.content[0].text.strip()
    try:
        result = json_repair.loads(resp_text)
    except Exception:
        try:
            result = json.loads(resp_text)
        except json.JSONDecodeError:
            return False

    if not isinstance(result, dict):
        return False

    changed = False
    new_text = result.get("text", "")
    if new_text and new_text != turn.text:
        turn.text = new_text
        changed = True

    return changed


# ── Helpers ──────────────────────────────────────────────────────────────

def _find_original_turn(
    turn_index: int, trajectory: ParsedTrajectory
) -> AssistantTurn | None:
    for t in trajectory.assistant_turns:
        if t.turn_index == turn_index:
            return t
    return None


def _is_valid_tool_call_dict(tc: dict) -> bool:
    """Validate a tool call dict has required fields."""
    name = tc.get("name", "")
    args = tc.get("arguments", {})
    if not name or not isinstance(args, dict):
        return False
    if name == "edit":
        edits = args.get("edits", [])
        return bool(args.get("path") or args.get("file_path")) and isinstance(edits, list) and len(edits) > 0
    if name == "write":
        return bool(args.get("path") or args.get("file_path")) and bool(args.get("content"))
    return True


def _extract_level_from_text(text: str) -> str:
    """Extract a PII level (L0-L4) from verifier text."""
    match = re.search(r'\b(L[0-4])\b', text)
    return match.group(1) if match else ""


# ── Main entry point ─────────────────────────────────────────────────────

async def refix_trajectory(
    trajectory: ParsedTrajectory,
    pii_map: PIIMap,
    rewrite_result: RewriteResult,
    verification: VerificationResult,
) -> RewriteResult:
    """Apply targeted fixes to a MINOR_ISSUES trajectory.

    Returns a new RewriteResult with the fixes applied in-place.
    """
    result = copy.deepcopy(rewrite_result)
    issues = verification.issues

    if not issues:
        return result

    deterministic, llm_needed = categorize_issues(issues)
    turns_by_idx = {t.turn_index: t for t in result.turns}
    fixes_applied = 0

    # Pass 1: deterministic fixes
    for issue in deterministic:
        turn = turns_by_idx.get(issue.turn_index)
        if not turn:
            continue

        rule = issue.rule_violated.lower().strip()
        fixed = False

        if rule == "structural":
            fixed = _fix_structural(turn, issue, trajectory)
        elif rule == "pii_leak":
            fixed = _fix_pii_leak(turn, issue, pii_map)
        elif rule == "rule_1":
            fixed = _fix_classification(turn, issue, pii_map)
        elif rule == "rule_2":
            fixed = _fix_over_collection(turn, issue, pii_map)
        else:
            # Generic minor issue — try PII strip + classification
            _fix_pii_leak(turn, issue, pii_map)
            fixed = _fix_classification(turn, issue, pii_map)

        if fixed:
            fixes_applied += 1

    # Pass 2: LLM-assisted fixes (grouped by turn)
    llm_by_turn: dict[int, list[VerificationIssue]] = {}
    for issue in llm_needed:
        llm_by_turn.setdefault(issue.turn_index, []).append(issue)

    for turn_idx, turn_issues in llm_by_turn.items():
        turn = turns_by_idx.get(turn_idx)
        if not turn:
            continue
        fixed = await _fix_with_llm(turn, turn_issues, trajectory, pii_map)
        if fixed:
            fixes_applied += 1

    logger.info(
        "Refix applied %d fixes (%d deterministic, %d LLM) across %d issues",
        fixes_applied, len(deterministic), len(llm_needed), len(issues),
    )

    # Recompute summary
    result.scenarios_covered = list(set(t.scenario for t in result.turns if t.scenario))
    result.skills_used = list(set(
        tc["name"] for t in result.turns for tc in t.tool_calls if isinstance(tc, dict)
    ))
    result.privacy_decision_points = sum(
        1 for t in result.turns
        if any(a in t.privacy_actions for a in ("consent_gate", "hard_deny", "local_first", "classify"))
    )

    return result
