"""Process Reward Model (PRM) — scores rejected alternatives on a 0-1 continuous scale.

Uses GPT-5.4 to assign nuanced reward scores to each rejected alternative,
enabling weighted DPO or KTO training rather than binary preference.
"""
from __future__ import annotations

import asyncio
import json
import logging

from openai import AsyncOpenAI

from config import OPENAI_API_KEY
from models import RLHFPair, RejectedStep
from prompts.rlhf_scorer import REWARD_SCORER_SYSTEM, build_scorer_prompt

logger = logging.getLogger("reward_scorer")

SCORER_MODEL = "gpt-5.4"
MAX_CONCURRENT_SCORES = 10


async def score_rejected_alternatives(
    pairs: list[RLHFPair],
    task_description: str = "",
) -> list[RLHFPair]:
    """Score all rejected alternatives in a batch of RLHF pairs using GPT-5.4 as PRM.

    Modifies pairs in-place by setting reward_score on each RejectedStep.
    Returns the same list for chaining.
    """
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set, using heuristic scoring")
        _apply_heuristic_scores(pairs)
        return pairs

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    sem = asyncio.Semaphore(MAX_CONCURRENT_SCORES)

    async def _score_single(pair: RLHFPair, step: RejectedStep) -> None:
        async with sem:
            score = await _call_scorer(client, pair, step, task_description)
            step.reward_score = score

    tasks = []
    for pair in pairs:
        for step in pair.rejected:
            tasks.append(_score_single(pair, step))

    if tasks:
        logger.info("Scoring %d rejected alternatives with %s...", len(tasks), SCORER_MODEL)
        await asyncio.gather(*tasks, return_exceptions=True)

    scored = sum(1 for p in pairs for s in p.rejected if s.reward_score > 0)
    logger.info("Scored %d/%d alternatives", scored, len(tasks))
    return pairs


async def _call_scorer(
    client: AsyncOpenAI,
    pair: RLHFPair,
    step: RejectedStep,
    task_description: str,
) -> float:
    """Call GPT-5.4 to score a single rejected alternative."""
    chosen = pair.chosen
    correct_action = "refuse/elicit" if pair.decision_branch in ("R_correct",) else "accept and proceed"

    chosen_summary = (
        f"Thinking: {chosen.get('thinking', '')[:200]}\n"
        f"Tool: {json.dumps(chosen.get('tool_call', {}))[:200]}\n"
        f"Response: {chosen.get('assistant_response', '')[:200]}"
    )

    user_prompt = build_scorer_prompt(
        task_description=task_description or f"Task {pair.task_id}",
        data_level=pair.data_level_involved,
        correct_tier=pair.tool_tier_involved,
        correct_action=correct_action,
        chosen_summary=chosen_summary,
        step_thinking=step.thinking[:300],
        step_tool_call=json.dumps(step.tool_call)[:300] if step.tool_call else "null (refusal)",
        step_tool_response=json.dumps(step.tool_response)[:200] if step.tool_response else "null",
        step_assistant_response=step.assistant_response[:300],
        failure_mode=step.failure_mode,
    )

    try:
        response = await client.chat.completions.create(
            model=SCORER_MODEL,
            max_completion_tokens=256,
            temperature=0.1,
            messages=[
                {"role": "system", "content": REWARD_SCORER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content.strip()
        return _parse_score(text)

    except Exception as e:
        logger.warning("Scoring failed for turn %d: %s, using heuristic", pair.turn_index, e)
        return _heuristic_score(step)


def _parse_score(text: str) -> float:
    """Parse the JSON response from GPT-5.4 to extract the score."""
    try:
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
        data = json.loads(text)
        score = float(data.get("score", 0.5))
        return max(0.0, min(1.0, score))
    except (json.JSONDecodeError, ValueError, TypeError):
        start = text.find('"score"')
        if start >= 0:
            after = text[start + 7:]
            for ch in after:
                if ch in "0123456789.":
                    end = after.find(ch)
                    num_str = ""
                    for c in after[end:]:
                        if c in "0123456789.":
                            num_str += c
                        else:
                            break
                    if num_str:
                        try:
                            return max(0.0, min(1.0, float(num_str)))
                        except ValueError:
                            pass
                    break
        return 0.5


def _heuristic_score(step: RejectedStep) -> float:
    """Fallback heuristic scoring when API is unavailable."""
    mode = step.failure_mode
    severity = step.privacy_violation.severity if step.privacy_violation else None

    SCORE_MAP = {
        ("wrong_tool_tier_down", "critical"): 0.05,
        ("wrong_tool_tier_down", "major"): 0.15,
        ("wrong_tool_tier_up", "critical"): 0.1,
        ("wrong_tool_tier_up", "major"): 0.2,
        ("wrong_param_higher", "critical"): 0.1,
        ("wrong_param_higher", "major"): 0.25,
        ("missing_elicitation", "critical"): 0.15,
        ("missing_elicitation", "major"): 0.3,
        ("missing_consent", "critical"): 0.2,
        ("missing_consent", "major"): 0.35,
        ("ambient_pii_leak", "critical"): 0.1,
        ("ambient_pii_leak", "major"): 0.25,
        ("memory_violation", "critical"): 0.05,
        ("memory_violation", "major"): 0.15,
        ("over_refusal", None): 0.6,
        ("over_refusal", "minor"): 0.65,
        ("hallucination", None): 0.4,
        ("hallucination", "minor"): 0.45,
    }

    key = (mode, severity)
    if key in SCORE_MAP:
        return SCORE_MAP[key]

    if step.privacy_violation and step.privacy_violation.flag:
        if severity == "critical":
            return 0.1
        elif severity == "major":
            return 0.25
        return 0.4

    if mode == "over_refusal":
        return 0.6

    return 0.4


def _apply_heuristic_scores(pairs: list[RLHFPair]) -> None:
    """Apply heuristic scores to all rejected alternatives when API is unavailable."""
    for pair in pairs:
        for step in pair.rejected:
            step.reward_score = _heuristic_score(step)
