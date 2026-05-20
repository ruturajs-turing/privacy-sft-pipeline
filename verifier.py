"""Stage 4: GPT-5.4 verification — checks HTG compliance, scores, flags issues."""
from __future__ import annotations

import asyncio
import json
import logging

import json_repair
import openai

from config import MAX_CONCURRENT_TASKS, OPENAI_API_KEY, VERIFIER_MODEL
from models import (
    ParsedTrajectory,
    PIIMap,
    RewriteResult,
    RewrittenTurn,
    VerificationIssue,
    VerificationResult,
)
from prompts.verifier_system import build_verifier_system
from token_tracker import tracker

logger = logging.getLogger(__name__)

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    return _semaphore


def _build_trajectory_for_verification(
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
    pii_map: PIIMap,
) -> str:
    """Format the rewritten trajectory as a readable document for the verifier."""
    lines = []

    # Header
    lines.append(f"# Trajectory: {trajectory.task_id} (submission: {trajectory.submission_id})")
    lines.append("")

    # PII Summary
    lines.append("## PII Detected")
    if pii_map.entities:
        for e in pii_map.entities[:20]:
            lines.append(f"  - \"{e.text}\" → {e.label} ({e.level}) [engines: {', '.join(e.engines)}]")
        lines.append(f"  Max level: {pii_map.max_level}")
    else:
        lines.append("  None (L0 only)")
    lines.append("")

    # Persona
    if trajectory.persona:
        p = trajectory.persona
        lines.append(f"## Persona: {p.get('name', '')} ({p.get('persona_id', '')})")
        lines.append(f"  Profession: {p.get('profession', '')}")
        lines.append(f"  Country: {p.get('country', '')}")
        lines.append("")

    # Conversation — use thread_order for proper interleaving
    lines.append("## Conversation Turns")
    lines.append("")
    lines.append("NOTE: This is a multi-step conversation where a single user message")
    lines.append("can trigger multiple assistant turns (tool calls → results → follow-up).")
    lines.append("Consecutive assistant turns between user messages are normal multi-step execution.")
    lines.append("")

    # Build user-before-turn mapping (same logic as writer)
    user_before_turn: dict[int, int] = {}
    if trajectory.thread_order:
        last_user_idx = -1
        for entry_type, entry_idx in trajectory.thread_order:
            if entry_type == "user":
                last_user_idx = entry_idx
            elif entry_type == "assistant":
                if last_user_idx >= 0 and entry_idx not in user_before_turn:
                    already_assigned = last_user_idx in user_before_turn.values()
                    if not already_assigned:
                        user_before_turn[entry_idx] = last_user_idx
    else:
        for i in range(min(len(trajectory.user_messages), len(rewrite_result.turns))):
            user_before_turn[i] = i

    for rt in rewrite_result.turns:
        # Emit user message only when thread_order says it appeared
        if rt.turn_index in user_before_turn:
            u_idx = user_before_turn[rt.turn_index]
            if u_idx < len(trajectory.user_messages):
                lines.append(f"### User Message {u_idx + 1}:")
                lines.append(trajectory.user_messages[u_idx])
                lines.append("")

        lines.append(f"### Assistant Turn {rt.turn_index + 1}:")
        lines.append(f"**Thinking:** {rt.thinking[:500]}{'...' if len(rt.thinking) > 500 else ''}")
        lines.append(f"**Text:** {rt.text}")

        if rt.tool_calls:
            lines.append("**Tool Calls:**")
            for tc in rt.tool_calls:
                if isinstance(tc, dict):
                    lines.append(f"  - {tc.get('name', '?')}({json.dumps(tc.get('arguments', {}))[:200]})")
            lines.append("")

        if rt.tool_results:
            lines.append("**Tool Results:**")
            for tr in rt.tool_results:
                if isinstance(tr, dict):
                    content = tr.get("content", "")[:150]
                    lines.append(f"  - [{tr.get('tool_name', '?')}]: {content}")
            lines.append("")

        lines.append(f"**Privacy Actions:** {', '.join(rt.privacy_actions)}")
        lines.append(f"**Scenario:** {rt.scenario}")
        lines.append("")

    return "\n".join(lines)


async def verify_trajectory(
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
    pii_map: PIIMap,
) -> VerificationResult:
    """Run GPT-5.4 verification on a rewritten trajectory."""
    sem = _get_semaphore()
    system_prompt = build_verifier_system()
    trajectory_doc = _build_trajectory_for_verification(trajectory, rewrite_result, pii_map)

    user_prompt = f"""Verify this rewritten privacy trajectory for HTG compliance.

{trajectory_doc}

---

Evaluate against all 6 HTG rules, check structural integrity, detect PII leaks, and score on the 5 dimensions.
Return the JSON verdict as specified."""

    async with sem:
        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        try:
            response = await client.chat.completions.create(
                model=VERIFIER_MODEL,
                max_completion_tokens=4096,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            tracker.record_openai(response, "verifier")
        except openai.RateLimitError:
            await asyncio.sleep(30)
            response = await client.chat.completions.create(
                model=VERIFIER_MODEL,
                max_completion_tokens=4096,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            tracker.record_openai(response, "verifier")

    resp_text = response.choices[0].message.content.strip()
    try:
        result = json_repair.loads(resp_text)
    except Exception:
        try:
            result = json.loads(resp_text)
        except json.JSONDecodeError:
            logger.error("Verifier returned unparseable response for %s", trajectory.task_id)
            return VerificationResult(
                verdict="FAIL",
                issues=[VerificationIssue(
                    turn_index=-1,
                    rule_violated="structural",
                    severity="critical",
                    description="Verifier response unparseable",
                )],
                rationale="Verifier output could not be parsed",
            )

    # Parse issues
    issues = []
    for issue_data in result.get("issues", []):
        issues.append(VerificationIssue(
            turn_index=issue_data.get("turn_index", -1),
            rule_violated=issue_data.get("rule_violated", "unknown"),
            severity=issue_data.get("severity", "minor"),
            description=issue_data.get("description", ""),
            fix_instruction=issue_data.get("fix_instruction", ""),
        ))

    scores = result.get("scores", {})
    privacy_compliance = scores.get("privacy_compliance", 0)
    correctness = scores.get("correctness", 0)
    completeness = scores.get("completeness", 0)
    efficiency = scores.get("efficiency", 0)
    naturality = scores.get("naturality", 0)
    overall = (privacy_compliance + correctness + completeness + efficiency + naturality) / 5.0

    return VerificationResult(
        verdict=result.get("verdict", "FAIL"),
        issues=issues,
        privacy_compliance=privacy_compliance,
        correctness=correctness,
        completeness=completeness,
        efficiency=efficiency,
        naturality=naturality,
        overall=overall,
        rationale=result.get("rationale", ""),
    )
