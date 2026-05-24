"""Stage 4: Claude Opus 4.7 verification — checks HTG compliance, scores, flags issues."""
from __future__ import annotations

import asyncio
import json
import logging

import anthropic
import json_repair

from config import ANTHROPIC_API_KEY, MAX_CONCURRENT_TASKS, VERIFIER_MODEL
from models import (
    ParsedTrajectory,
    PIIMap,
    RewriteResult,
    RewrittenTurn,
    VerificationIssue,
    VerificationResult,
)
from prompts.verifier_system import build_verifier_system
from task_context import get_task_definition, build_verifier_rubric
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
    lines.append("NOTE: Some user messages are **synthetic** (consent simulation or adversarial jailbreak")
    lines.append("drills). They are labeled in headings below. They are intentional training content,")
    lines.append("not accidental PII injection by the assistant.")
    lines.append("")

    if rewrite_result.patched_events is not None:
        for turn_no, event in enumerate(
            e for e in rewrite_result.patched_events
            if isinstance(e, dict) and e.get("type") == "message"
        ):
            msg = event.get("message", {})
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "unknown")
            lines.append(f"### {role.title()} Event {turn_no + 1}:")
            content = msg.get("content", [])
            if isinstance(content, str):
                lines.append(content)
            elif isinstance(content, list):
                text_parts = []
                tool_calls = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        text_parts.append(str(block.get("text", "")))
                    elif block.get("type") == "toolCall":
                        args_str = json.dumps(block.get("arguments", {}))
                        if len(args_str) > 400:
                            args_str = args_str[:400] + "..."
                        tool_calls.append(f"  - {block.get('name', '?')}({args_str})")
                if text_parts:
                    lines.append("**Text:** " + "\n".join(t for t in text_parts if t.strip()))
                if tool_calls:
                    lines.append("**Tool Calls:**")
                    lines.extend(tool_calls)
            if role == "toolResult":
                result_text = ""
                content = msg.get("content", [])
                if isinstance(content, list):
                    result_text = " ".join(
                        str(block.get("text", ""))
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                if len(result_text) > 300:
                    result_text = result_text[:300] + "..."
                lines.append(f"**Tool Result:** {msg.get('toolName', '?')}: {result_text}")
            lines.append("")
        return "\n".join(lines)

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

    emitted_user_indices: set[int] = set()
    for rt in rewrite_result.turns:
        # Emit user message only when thread_order says it appeared (once per turn_index)
        if rt.turn_index in user_before_turn and rt.turn_index not in emitted_user_indices:
            u_idx = user_before_turn[rt.turn_index]
            if u_idx < len(trajectory.user_messages):
                lines.append(f"### User Message {u_idx + 1}:")
                lines.append(trajectory.user_messages[u_idx])
                lines.append("")
            emitted_user_indices.add(rt.turn_index)

        # If this turn has a synthetic user consent message, show it before the execution
        if rt.synthetic_user_message:
            lines.append(f"### [Synthetic User Consent Response] (decision: {rt.consent_decision}):")
            lines.append(rt.synthetic_user_message)
            lines.append("")

        # Adversarial drill: show the synthetic jailbreak user line before the assistant refusal
        if rt.is_adversarial and rt.adversarial_user_message:
            atk = (rt.attack_type or "unknown").strip()
            lines.append(
                f"### [Synthetic Adversarial User Message] (attack_type: {atk}) — "
                "RLHF/SFT drill, not a real end-user message:"
            )
            lines.append(rt.adversarial_user_message)
            lines.append("")

        lines.append(f"### Assistant Turn {rt.turn_index + 1}:")
        if rt.is_adversarial:
            lines.append("**(This turn responds to the synthetic adversarial user message above.)**")

        if rt.text:
            lines.append(f"**Text:** {rt.text}")

        if rt.tool_calls:
            lines.append("**Tool Calls:**")
            for tc in rt.tool_calls:
                if isinstance(tc, dict):
                    args_str = json.dumps(tc.get('arguments', {}))
                    if len(args_str) > 400:
                        args_str = args_str[:400] + "..."
                    lines.append(f"  - {tc.get('name', '?')}({args_str})")
            lines.append("")

        if rt.tool_results:
            lines.append("**Tool Results:**")
            for tr in rt.tool_results:
                if isinstance(tr, dict):
                    content = tr.get("content", "")[:300]
                    lines.append(f"  - [{tr.get('tool_name', '?')}]: {content}")
            lines.append("")

        if rt.privacy_actions:
            lines.append(f"**Privacy Actions:** {', '.join(rt.privacy_actions)}")
        lines.append("")

    return "\n".join(lines)


async def verify_trajectory(
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
    pii_map: PIIMap,
) -> VerificationResult:
    """Run Claude Opus 4.7 verification on a rewritten trajectory."""
    sem = _get_semaphore()
    system_prompt = build_verifier_system()
    trajectory_doc = _build_trajectory_for_verification(trajectory, rewrite_result, pii_map)

    # Inject task-specific rubric if available
    task_def = get_task_definition(trajectory.task_id)
    rubric_block = ""
    if task_def:
        rubric_block = f"""
## Task-Specific Rubric (from task definition)
{build_verifier_rubric(task_def)}

Use this rubric as ground truth for what the trajectory SHOULD accomplish and which
privacy actions are expected. Flag deviations as issues.
---
"""

    user_prompt = f"""Verify this rewritten privacy trajectory for HTG compliance.

{rubric_block}{trajectory_doc}

---

Evaluate against all 6 HTG rules, check structural integrity, detect PII leaks, and score on the 5 dimensions.
Use the task-specific rubric above (if present) to verify that expected privacy actions were correctly applied.
Return the JSON verdict as specified."""

    from llm_retry import call_anthropic
    async with sem:
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        response = await call_anthropic(
            client,
            model=VERIFIER_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            stage="verifier",
        )
        tracker.record_anthropic(response, "verifier")

    resp_text = response.content[0].text.strip()

    # Claude may wrap JSON in markdown fences — strip them
    if resp_text.startswith("```"):
        first_newline = resp_text.index("\n") if "\n" in resp_text else 3
        resp_text = resp_text[first_newline + 1:]
        if resp_text.endswith("```"):
            resp_text = resp_text[:-3].strip()

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
