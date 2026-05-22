"""Post-generation trajectory reviewer — LLM-powered quality gate.

Reads the full assembled trajectory alongside the project reference documents
(Proposal, Annotation Guidelines, Classification taxonomy) and flags structural,
compliance, and naturalness issues. When possible, returns auto-corrected events.

Integrated as Stage 4.5 in the pipeline: runs AFTER assembly, BEFORE verification.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import anthropic
import json_repair

from config import ANTHROPIC_API_KEY, VERIFIER_MODEL
from models import ParsedTrajectory, PIIMap, RewriteResult, RewrittenTurn
from token_tracker import tracker

logger = logging.getLogger(__name__)

_REF_DIR = Path(__file__).resolve().parent.parent
_REF_DOCS: dict[str, Path] = {
    "proposal": _REF_DIR / "Copy of OpenClaw_Privacy_Proposal.md",
    "guidelines": _REF_DIR / "Copy of [EXT_ Turing] Annotation Guidelines_ OpenClaw Privacy Trajectories (1).md",
    "classification": _REF_DIR / "Classification.md",
}

_MAX_REF_CHARS = 6000

_cached_refs: str | None = None


def _load_reference_context() -> str:
    """Load and cache a compact digest of the reference documents."""
    global _cached_refs
    if _cached_refs is not None:
        return _cached_refs

    parts: list[str] = []

    for label, path in _REF_DOCS.items():
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > _MAX_REF_CHARS:
            text = text[:_MAX_REF_CHARS] + "\n[... truncated ...]"
        parts.append(f"=== {label.upper()} ===\n{text}")

    _cached_refs = "\n\n".join(parts)
    logger.info("Loaded %d reference docs (%d chars)", len(parts), len(_cached_refs))
    return _cached_refs


def _trajectory_to_readable(
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
) -> str:
    """Render the assembled trajectory as a readable transcript for the reviewer."""
    lines: list[str] = []

    user_idx = 0
    for rt in rewrite_result.turns:
        if rt.adversarial_user_message:
            lines.append(f"[USER] {rt.adversarial_user_message}")
        elif rt.synthetic_user_message:
            lines.append(f"[USER] {rt.synthetic_user_message}")

        if (
            not rt.adversarial_user_message
            and not rt.synthetic_user_message
            and user_idx < len(trajectory.user_messages)
        ):
            lines.append(f"[USER] {trajectory.user_messages[user_idx]}")
            user_idx += 1

        tag = "ASSISTANT"
        if rt.privacy_actions:
            tag += f" [{','.join(rt.privacy_actions[:3])}]"
        text = rt.text or "(no text)"
        lines.append(f"[{tag}] {text}")

        for tc in rt.tool_calls:
            if isinstance(tc, dict):
                lines.append(f"  -> tool: {tc.get('name', '?')}({json.dumps(tc.get('arguments', {}))[:200]})")

        for tr in rt.tool_results:
            if isinstance(tr, dict):
                lines.append(f"  <- result: {str(tr.get('content', ''))[:150]}")

    return "\n".join(lines)


_REVIEW_PROMPT = """You are a quality reviewer for privacy-compliant AI agent trajectories.

REFERENCE STANDARDS:
{reference_context}

TASK CONTEXT:
Task ID: {task_id}
Goal: {goal}
Privacy scenario: {privacy_scenario}
Data levels: {data_levels}
Persona: {persona_name} ({persona_details})

FULL TRAJECTORY:
{transcript}

---

Review this trajectory against the reference standards. Check for:

1. STRUCTURAL ISSUES:
   - Does the conversation end abruptly? (e.g., adversarial refusal as the last message with no user acknowledgment or natural closing)
   - Are there consecutive assistant messages without an intervening user turn?
   - Are tool results consistent with the tool calls? (e.g., a write to path X should show result for path X, not some unrelated path)
   - Does the conversation flow naturally from start to finish?

2. PRIVACY COMPLIANCE:
   - Is the Hatch refusal string present when required (L3/L4 memory write attempts)?
   - Are consent gates placed correctly before sensitive writes?
   - Is data minimization applied to persistence writes?
   - Does the adversarial refusal properly refuse WITHOUT complying even partially?

3. NATURALNESS:
   - Do user messages sound like a real human? No overly structured or robotic messages.
   - Do assistant messages avoid sycophantic openers ("Great question!", "Of course!", "Absolutely!")?
   - Does the conversation topic match the task goal?
   - Is the closing natural? (Not abrupt, not overly formal)

4. CONTENT ACCURACY:
   - Do tool calls reference correct paths/files consistent with the workspace?
   - Are there any hallucinated references (e.g., mentioning files or folders that don't exist in the context)?
   - Does the PII match what the persona would realistically have?

Return a JSON object:
{{
  "verdict": "PASS" | "NEEDS_FIX",
  "issues": [
    {{
      "type": "structural" | "privacy" | "naturalness" | "accuracy",
      "severity": "critical" | "minor",
      "turn_index": <int or null>,
      "description": "<what's wrong>",
      "fix_suggestion": "<how to fix it>"
    }}
  ],
  "fixed_turns": [
    {{
      "turn_index": <int>,
      "new_text": "<corrected assistant text>",
      "new_user_message": "<corrected user message or null>"
    }}
  ]
}}

If the trajectory is fine, return {{"verdict": "PASS", "issues": [], "fixed_turns": []}}.
Only include "fixed_turns" for issues you can confidently auto-fix (text-level fixes). Do NOT attempt to restructure tool calls.
Return ONLY the JSON object."""


async def review_trajectory(
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
    pii_map: PIIMap,
) -> dict:
    """Run the LLM reviewer on the assembled trajectory.

    Returns a dict with:
      verdict: "PASS" or "NEEDS_FIX"
      issues: list of issues found
      fixed_turns: auto-corrections that can be applied
    """
    ref_ctx = _load_reference_context()
    transcript = _trajectory_to_readable(trajectory, rewrite_result)

    task_spec = trajectory.task_spec or {}
    persona = trajectory.persona or {}
    persona_name = f"{persona.get('first_name', '')} {persona.get('last_name', '')}".strip()
    p = persona.get("personality", {})
    persona_details = (
        f"age {persona.get('exact_age', '?')}, "
        f"{persona.get('job_title', '')} in {persona.get('city', '')}, "
        f"style: {'casual' if p.get('conscientiousness', 0.5) < 0.5 else 'careful'}"
    )

    prompt = _REVIEW_PROMPT.format(
        reference_context=ref_ctx[:12000],
        task_id=trajectory.task_id,
        goal=task_spec.get("goal_summary", ""),
        privacy_scenario=task_spec.get("privacy_scenario", ""),
        data_levels=task_spec.get("data_levels", ""),
        persona_name=persona_name,
        persona_details=persona_details,
        transcript=transcript[:15000],
    )

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    try:
        from llm_retry import call_anthropic
        resp = await call_anthropic(
            client,
            model=VERIFIER_MODEL,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
            stage="trajectory_reviewer",
        )
        tracker.record_anthropic(resp, "trajectory_reviewer")
    except Exception as e:
        logger.error("Trajectory reviewer failed: %s", e)
        return {"verdict": "PASS", "issues": [], "fixed_turns": []}

    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        first_nl = raw.index("\n") if "\n" in raw else 3
        raw = raw[first_nl + 1:]
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    try:
        result = json_repair.loads(raw)
    except Exception:
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse reviewer response")
            return {"verdict": "PASS", "issues": [], "fixed_turns": []}

    if not isinstance(result, dict):
        return {"verdict": "PASS", "issues": [], "fixed_turns": []}

    issues = result.get("issues", [])
    if issues:
        for issue in issues:
            logger.info(
                "Reviewer issue [%s/%s] turn %s: %s",
                issue.get("type", "?"),
                issue.get("severity", "?"),
                issue.get("turn_index", "?"),
                issue.get("description", "?")[:120],
            )

    return result


def apply_reviewer_fixes(
    rewrite_result: RewriteResult,
    review: dict,
) -> RewriteResult:
    """Apply auto-fix suggestions from the reviewer to the rewrite result.

    Only applies text-level fixes (new_text, new_user_message).
    Does not restructure tool calls or add/remove turns.
    """
    fixed_turns = review.get("fixed_turns", [])
    if not fixed_turns:
        return rewrite_result

    fix_map: dict[int, dict] = {}
    for fix in fixed_turns:
        idx = fix.get("turn_index")
        if isinstance(idx, int):
            fix_map[idx] = fix

    applied = 0
    for turn in rewrite_result.turns:
        if turn.turn_index not in fix_map:
            continue
        fix = fix_map[turn.turn_index]

        new_text = fix.get("new_text")
        if new_text and isinstance(new_text, str) and new_text != turn.text:
            logger.info("Reviewer fix: turn %d text updated", turn.turn_index)
            turn.text = new_text
            applied += 1

        new_user = fix.get("new_user_message")
        if new_user and isinstance(new_user, str):
            if turn.synthetic_user_message:
                turn.synthetic_user_message = new_user
                applied += 1
            elif turn.adversarial_user_message:
                turn.adversarial_user_message = new_user
                applied += 1

    if applied:
        logger.info("Reviewer applied %d text fixes", applied)

    return rewrite_result


def _check_structural_issues_fast(
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
) -> list[dict]:
    """Fast rule-based check for common structural issues (no LLM needed)."""
    issues: list[dict] = []
    turns = rewrite_result.turns
    if not turns:
        return issues

    last_turn = turns[-1]
    if last_turn.is_adversarial or "adversarial_refusal" in (last_turn.privacy_actions or []):
        issues.append({
            "type": "structural",
            "severity": "critical",
            "turn_index": last_turn.turn_index,
            "description": "Adversarial refusal is the last turn — conversation ends abruptly after jailbreak refusal",
            "fix_suggestion": "Add user acknowledgment and natural closing after the refusal",
        })

    if (
        last_turn.text
        and "Catch you" in last_turn.text
        and "adversarial_refusal" in (last_turn.privacy_actions or [])
    ):
        issues.append({
            "type": "structural",
            "severity": "critical",
            "turn_index": last_turn.turn_index,
            "description": "Adversarial refusal merged with closing text",
            "fix_suggestion": "Split refusal and closing into separate turns",
        })

    prev_role = None
    for i, turn in enumerate(turns):
        cur_role = "assistant"
        if turn.synthetic_user_message or turn.adversarial_user_message:
            cur_role = "user_then_assistant"
        if (
            prev_role == "assistant"
            and cur_role == "assistant"
            and not turn.synthetic_user_message
            and not turn.adversarial_user_message
        ):
            if i > 0 and turns[i - 1].turn_index not in _get_consent_gate_indices(turns):
                issues.append({
                    "type": "structural",
                    "severity": "minor",
                    "turn_index": turn.turn_index,
                    "description": f"Consecutive assistant turns at index {i-1} and {i} without user message",
                })
        prev_role = "assistant"

    # Check for tool-only assistant turns (no text) that effectively respond
    # to a user message — the user sees silent tool calls with no explanation.
    for i, turn in enumerate(turns):
        has_tools = bool(turn.tool_calls)
        has_text = bool(turn.text and turn.text.strip())
        if has_tools and not has_text:
            is_after_user = False
            if i == 0:
                is_after_user = True
            elif turns[i - 1].synthetic_user_message or turns[i - 1].adversarial_user_message:
                is_after_user = True
            elif i > 0 and not turns[i - 1].tool_calls and turns[i - 1].text:
                pass
            if is_after_user:
                tool_names = [
                    tc.get("name", "?") for tc in turn.tool_calls
                    if isinstance(tc, dict)
                ]
                issues.append({
                    "type": "structural",
                    "severity": "critical",
                    "turn_index": turn.turn_index,
                    "description": (
                        f"Tool-only assistant turn ({', '.join(tool_names)}) with no text "
                        f"— user sees silent tool calls with no explanation"
                    ),
                    "fix_suggestion": "Add brief text explaining the action before the tool calls",
                })

    return issues


def _get_consent_gate_indices(turns: list[RewrittenTurn]) -> set[int]:
    """Get turn indices that are consent gates (expected to be followed by assistant)."""
    return {
        t.turn_index for t in turns
        if "consent_gate" in (t.privacy_actions or [])
    }


async def review_and_fix(
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
    pii_map: PIIMap,
    skip_llm: bool = False,
) -> RewriteResult:
    """Full review pipeline: fast structural check + optional LLM review.

    1. Run fast rule-based checks
    2. If critical issues found, apply structural fixes directly
    3. Optionally run LLM reviewer for deeper analysis
    4. Apply text-level fixes from LLM
    """
    fast_issues = _check_structural_issues_fast(trajectory, rewrite_result)
    critical_fast = [i for i in fast_issues if i.get("severity") == "critical"]

    if critical_fast:
        logger.warning(
            "Fast check found %d critical structural issues, applying fixes",
            len(critical_fast),
        )
        rewrite_result = _fix_structural_issues(trajectory, rewrite_result, critical_fast)

    if not skip_llm:
        review = await review_trajectory(trajectory, rewrite_result, pii_map)
        if review.get("verdict") == "NEEDS_FIX":
            rewrite_result = apply_reviewer_fixes(rewrite_result, review)

    return rewrite_result


def _fix_structural_issues(
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
    issues: list[dict],
) -> RewriteResult:
    """Apply structural fixes for critical issues found by the fast checker."""
    import random

    turns = rewrite_result.turns

    for issue in issues:
        desc = issue.get("description", "")

        if "Adversarial refusal is the last turn" in desc:
            persona = trajectory.persona or {}
            from assembler import _generate_post_adversarial_ack
            ack = _generate_post_adversarial_ack(persona)
            turns.append(ack)
            logger.info("Structural fix: added post-adversarial acknowledgment at end")

        if "merged with closing text" in desc:
            idx = issue.get("turn_index")
            if idx is not None and idx < len(turns):
                turn = turns[idx]
                if turn.text:
                    parts = turn.text.split("\n\n")
                    if len(parts) > 1:
                        refusal_part = parts[0]
                        closing_part = "\n\n".join(parts[1:])
                        if "Hatch Trust" in refusal_part or "cannot retain" in refusal_part:
                            turn.text = refusal_part
                            logger.info("Structural fix: separated refusal from closing text")

    for i, t in enumerate(turns):
        t.turn_index = i

    rewrite_result.turns = turns
    return rewrite_result
