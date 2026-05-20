"""RLHF Preference Pair Generator — Stage 6 of the Privacy SFT Pipeline.

Takes completed SFT trajectories and generates step-level preference pairs
by identifying privacy-critical decision points and using Claude Opus 4.6
to produce suboptimal rejected alternatives.

Incorporates:
- Decision tree from the RLHF diagram (R correct/wrong, A correct/wrong branches)
- COVERT-style augmentations (distractor tools, ambiguous queries, noisy outputs)
- Critical Step Optimization (focus on high-impact decision points)
- Privacy Collapse prevention (balanced over-refusal representation)
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from anthropic import AsyncAnthropic

from config import ANTHROPIC_API_KEY
from models import (
    ParsedTrajectory, PIIMap, PIIEntity, RewriteResult, RewrittenTurn,
    RLHFPair, RejectedStep, PrivacyViolation, RLHFBatchReport,
)
from tool_tiers import (
    TOOL_REGISTRY, LEVEL_ORDER, TIER_MAX_LEVELS,
    get_tool_tier, get_tool_max_level, get_distractors,
    get_tools_by_tier, get_alternative_tool, level_exceeds,
)
from prompts.rlhf_system import RLHF_SYSTEM_PROMPT, build_rlhf_user_prompt

logger = logging.getLogger("rlhf_generator")

ALTERNATIVES_PER_STEP = 9
MIN_CRITICALITY_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Decision Point Identification
# ---------------------------------------------------------------------------

@dataclass
class DecisionPoint:
    """A privacy-critical decision point in the trajectory."""
    turn_index: int
    tool_call: dict
    tool_name: str
    tool_tier: str
    data_level: str
    decision_branch: str  # R_correct, A_correct, A_wrong, R_wrong
    criticality: float    # 0-1
    scenario: str
    context_messages: list[dict]
    chosen_thinking: str
    chosen_tool_response: str
    chosen_assistant_response: str


def identify_decision_points(
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
    pii_map: PIIMap,
) -> list[DecisionPoint]:
    """Scan rewritten trajectory for privacy-critical tool call steps.

    Uses the diagram logic:
    - L(T(t)) < L(u) → tool tier cannot handle the data level → refusal expected
    - L(T(t)) >= L(u) → tool tier CAN handle → accept expected
    """
    points: list[DecisionPoint] = []

    for turn in rewrite_result.turns:
        if not turn.tool_calls:
            continue

        turn_pii = _get_pii_for_turn(turn.turn_index, pii_map)
        max_data_level = _max_level(turn_pii) if turn_pii else "L0"

        for tc in turn.tool_calls:
            tool_name = tc.get("name", "")
            tool_tier = get_tool_tier(tool_name)
            tool_max = get_tool_max_level(tool_name)

            has_privacy_action = bool(turn.privacy_actions)
            did_refuse = any(
                kw in turn.text.lower()
                for kw in ["cannot", "i'm unable", "privacy", "hatch trust", "not permitted"]
            )

            if level_exceeds(max_data_level, tool_tier):
                branch = "R_correct" if did_refuse else "A_wrong"
            else:
                branch = "A_correct" if not did_refuse else "R_wrong"

            criticality = _compute_criticality(
                max_data_level, tool_tier, branch, has_privacy_action
            )

            if criticality < MIN_CRITICALITY_THRESHOLD:
                continue

            context = _build_context(trajectory, rewrite_result, turn.turn_index)

            point = DecisionPoint(
                turn_index=turn.turn_index,
                tool_call=tc,
                tool_name=tool_name,
                tool_tier=tool_tier,
                data_level=max_data_level,
                decision_branch=branch,
                criticality=criticality,
                scenario=turn.scenario or "A",
                context_messages=context,
                chosen_thinking=turn.thinking,
                chosen_tool_response=_get_tool_response_text(tc, turn),
                chosen_assistant_response=turn.text,
            )
            points.append(point)

    points.sort(key=lambda p: p.criticality, reverse=True)
    return points


def _get_pii_for_turn(turn_index: int, pii_map: PIIMap) -> list[PIIEntity]:
    """Get PII entities relevant to a specific turn."""
    return [e for e in pii_map.entities if e.turn_index == turn_index or e.turn_index == -1]


def _max_level(entities: list[PIIEntity]) -> str:
    """Get the highest data level from a list of PII entities."""
    if not entities:
        return "L0"
    levels = [e.level for e in entities if e.level in LEVEL_ORDER]
    if not levels:
        return "L0"
    return max(levels, key=lambda l: LEVEL_ORDER.get(l, 0))


def _compute_criticality(
    data_level: str, tool_tier: str, branch: str, has_privacy_action: bool
) -> float:
    """Score how critical this decision point is (0-1).

    Higher scores mean more important for RLHF training.
    """
    score = 0.0

    level_val = LEVEL_ORDER.get(data_level, 0)
    score += level_val * 0.15  # L4=0.6, L3=0.45, L2=0.3, L1=0.15

    if branch in ("R_correct", "A_wrong"):
        score += 0.3  # Tier mismatch situations are highly critical
    elif branch == "R_wrong":
        score += 0.2  # Over-refusals are important for balance

    tier_risk = {"T3": 0.15, "T2": 0.10, "T1": 0.05}
    score += tier_risk.get(tool_tier, 0.05)

    if has_privacy_action:
        score += 0.1

    return min(score, 1.0)


def _build_context(
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
    up_to_turn: int,
) -> list[dict]:
    """Build conversation context up to (but not including) the target turn."""
    context = []
    for i, msg in enumerate(trajectory.user_messages):
        if i >= up_to_turn:
            break
        context.append({"role": "user", "content": msg})
        if i < len(rewrite_result.turns) and i < up_to_turn:
            t = rewrite_result.turns[i]
            context.append({"role": "assistant", "content": t.text or t.thinking})
    return context[-6:]  # Last 3 exchanges max for prompt efficiency


def _get_tool_response_text(tool_call: dict, turn: RewrittenTurn) -> str:
    """Extract tool response text for a specific tool call from the turn."""
    call_id = tool_call.get("call_id", tool_call.get("id", ""))
    for tr in turn.tool_results:
        if tr.get("call_id") == call_id or tr.get("tool_name") == tool_call.get("name"):
            return tr.get("content", "")[:500]
    return "(tool response not captured)"


# ---------------------------------------------------------------------------
# Rejected Alternative Generation (LLM-based)
# ---------------------------------------------------------------------------

async def generate_rejected_alternatives(
    point: DecisionPoint,
    client: AsyncAnthropic,
    num_alternatives: int = ALTERNATIVES_PER_STEP,
) -> list[RejectedStep]:
    """Call Claude Opus 4.6 to generate rejected alternatives for a decision point."""

    same_tier = get_tools_by_tier(point.tool_tier)[:5]
    higher_tiers = []
    lower_tiers = []

    if point.tool_tier == "T1":
        higher_tiers = get_tools_by_tier("T2")[:3] + get_tools_by_tier("T3")[:3]
    elif point.tool_tier == "T2":
        higher_tiers = get_tools_by_tier("T3")[:5]
        lower_tiers = get_tools_by_tier("T1")[:5]
    else:
        lower_tiers = get_tools_by_tier("T1")[:3] + get_tools_by_tier("T2")[:3]

    distractors = get_distractors(point.tool_name)

    context_str = "\n".join(
        f"[{m['role']}]: {m['content'][:200]}" for m in point.context_messages
    )

    chosen_tc_str = json.dumps(point.tool_call, indent=2) if point.tool_call else "null (refusal)"
    target_tokens = max(100, len(point.chosen_thinking.split()) + len(point.chosen_assistant_response.split()))

    user_prompt = build_rlhf_user_prompt(
        turn_index=point.turn_index,
        data_level=point.data_level,
        tool_tier=point.tool_tier,
        decision_branch=point.decision_branch,
        scenario=point.scenario,
        context=context_str or "(session start)",
        chosen_thinking=point.chosen_thinking[:500],
        chosen_tool_call=chosen_tc_str,
        chosen_tool_response=point.chosen_tool_response[:300],
        chosen_assistant_response=point.chosen_assistant_response[:300],
        same_tier_tools=same_tier,
        higher_tier_tools=higher_tiers,
        lower_tier_tools=lower_tiers,
        distractor_tools=distractors,
        num_alternatives=num_alternatives,
        target_tokens=target_tokens,
    )

    try:
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8192,
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            system=RLHF_SYSTEM_PROMPT,
            temperature=0.9,
        )
        text = response.content[0].text
        alternatives = _parse_alternatives_response(text)
        return alternatives

    except Exception as e:
        logger.error("Failed to generate alternatives for turn %d: %s", point.turn_index, e)
        return []


def _parse_alternatives_response(text: str) -> list[RejectedStep]:
    """Parse the LLM response into RejectedStep objects."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]

    try:
        items = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                items = json.loads(text[start:end])
            except json.JSONDecodeError:
                logger.warning("Failed to parse RLHF alternatives JSON")
                return []
        else:
            return []

    steps = []
    for item in items:
        if not isinstance(item, dict):
            continue
        violation_rule = item.get("privacy_violation_rule")
        severity = item.get("severity", "minor")
        failure_mode = item.get("failure_mode", "hallucination")

        has_violation = violation_rule is not None and failure_mode not in ("over_refusal", "hallucination")

        step = RejectedStep(
            thinking=item.get("thinking", ""),
            tool_call=item.get("tool_call"),
            tool_response=item.get("tool_response"),
            assistant_response=item.get("assistant_response", ""),
            failure_mode=failure_mode,
            privacy_violation=PrivacyViolation(
                flag=has_violation,
                rule=violation_rule,
                severity=severity if has_violation else None,
            ),
            reward_score=0.0,  # Will be filled by reward_scorer
            perturbation_type=item.get("perturbation_type", ""),
        )
        steps.append(step)

    return steps


# ---------------------------------------------------------------------------
# Main RLHF Generation Orchestrator
# ---------------------------------------------------------------------------

async def generate_rlhf_pairs(
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
    pii_map: PIIMap,
    max_pairs_per_trajectory: int = 50,
) -> list[RLHFPair]:
    """Generate RLHF preference pairs for an entire trajectory.

    1. Identify all privacy-critical decision points
    2. For each (sorted by criticality), generate rejected alternatives via LLM
    3. Assemble into RLHFPair objects
    """
    if not ANTHROPIC_API_KEY:
        logger.error("ANTHROPIC_API_KEY not set, skipping RLHF generation")
        return []

    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    decision_points = identify_decision_points(trajectory, rewrite_result, pii_map)
    logger.info(
        "Found %d decision points (criticality >= %.1f)",
        len(decision_points), MIN_CRITICALITY_THRESHOLD
    )

    if not decision_points:
        return []

    budget_per_point = max(3, max_pairs_per_trajectory // len(decision_points))
    budget_per_point = min(budget_per_point, ALTERNATIVES_PER_STEP)

    pairs: list[RLHFPair] = []
    sem = asyncio.Semaphore(3)

    async def _generate_for_point(point: DecisionPoint) -> RLHFPair | None:
        async with sem:
            alternatives = await generate_rejected_alternatives(
                point, client, num_alternatives=budget_per_point
            )
            if not alternatives:
                return None

            chosen = {
                "thinking": point.chosen_thinking,
                "tool_call": point.tool_call,
                "tool_response": {"content": point.chosen_tool_response, "is_error": False},
                "assistant_response": point.chosen_assistant_response,
                "reward_score": 1.0,
            }

            return RLHFPair(
                task_id=trajectory.task_id,
                submission_id=trajectory.submission_id,
                turn_index=point.turn_index,
                step_criticality=point.criticality,
                pair_level="step",
                context=point.context_messages,
                chosen=chosen,
                rejected=alternatives,
                decision_branch=point.decision_branch,
                data_level_involved=point.data_level,
                tool_tier_involved=point.tool_tier,
            )

    tasks = [_generate_for_point(p) for p in decision_points]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for result in results:
        if isinstance(result, RLHFPair):
            pairs.append(result)
        elif isinstance(result, Exception):
            logger.error("RLHF generation error: %s", result)

    logger.info("Generated %d RLHF pairs with %d total rejected alternatives",
                len(pairs), sum(len(p.rejected) for p in pairs))
    return pairs


def compute_batch_report(
    task_id: str,
    submission_id: str,
    pairs: list[RLHFPair],
) -> RLHFBatchReport:
    """Compute statistics for a batch of RLHF pairs."""
    report = RLHFBatchReport(task_id=task_id, submission_id=submission_id)

    all_rejected: list[RejectedStep] = []
    for p in pairs:
        all_rejected.extend(p.rejected)

    report.total_pairs = len(all_rejected)

    for step in all_rejected:
        mode = step.failure_mode or "unknown"
        report.pairs_by_failure_mode[mode] = report.pairs_by_failure_mode.get(mode, 0) + 1

    for p in pairs:
        level = p.pair_level
        report.pairs_by_level[level] = report.pairs_by_level.get(level, 0) + 1

    if all_rejected:
        report.avg_reward_score = sum(s.reward_score for s in all_rejected) / len(all_rejected)
        over_refusals = sum(1 for s in all_rejected if s.failure_mode == "over_refusal")
        violations = sum(1 for s in all_rejected if s.privacy_violation.flag)
        report.over_refusal_ratio = over_refusals / len(all_rejected)
        report.violation_ratio = violations / len(all_rejected)

    return report
