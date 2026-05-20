"""Stage 3: Claude Opus 4.6 per-turn rewriting engine with HTG compliance."""
from __future__ import annotations

import asyncio
import json
import logging
import secrets

import anthropic
import json_repair

from config import ANTHROPIC_API_KEY, MAX_CONCURRENT_TASKS, REWRITER_MODEL
from models import AssistantTurn, ParsedTrajectory, PIIMap, RewriteResult, RewrittenTurn
from prompts.rewriter_system import build_rewriter_system
from prompts.rewriter_turn import build_turn_prompt
from token_tracker import tracker

logger = logging.getLogger(__name__)

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    return _semaphore


def _determine_scenario(trajectory: ParsedTrajectory, pii_map: PIIMap, turn: AssistantTurn) -> str:
    """Determine which privacy scenario (A-N) best fits this turn."""
    has_persistence = any(
        tc.name in ("memory_write", "active_memory_write", "active_memory_set", "wiki_apply")
        for tc in turn.tool_calls
    )
    has_t3 = any(
        tc.name in ("web_search", "web_fetch", "gmail", "github", "google-drive",
                     "slack", "notion", "browser", "chromium", "api-gateway")
        for tc in turn.tool_calls
    )
    has_t2 = any(
        tc.name.startswith("enterprise-")
        for tc in turn.tool_calls
    )
    has_t1_local = any(
        tc.name in ("write", "read", "exec", "code-analysis", "data-analysis")
        for tc in turn.tool_calls
    )

    if has_persistence and (pii_map.has_l3 or pii_map.has_l4):
        return "D"  # Memory/jailbreak refusal
    if pii_map.has_l4 and has_persistence:
        return "D"
    if has_t3 and (pii_map.has_l3 or pii_map.has_l4):
        return "C"  # T3 hard block
    if has_t2 and pii_map.has_l3:
        return "B"  # Cloud fallback + consent
    if has_t3 and pii_map.max_level in ("L2",):
        return "C"  # T3 with L2
    if pii_map.has_l3 and has_t1_local:
        return "A"  # Ideal local execution
    if pii_map.has_l4:
        return "L"  # Ephemeral credential handling

    # Default based on PII level
    if pii_map.has_l3:
        return "A" if not has_t3 else "B"
    if pii_map.max_level == "L2":
        return "F"  # Implicit recognition

    # From task spec
    task_scenario = trajectory.task_spec.get("privacy_scenario", "")
    if task_scenario and len(task_scenario) == 1:
        return task_scenario

    return "A"


async def _rewrite_single_turn(
    trajectory: ParsedTrajectory,
    turn: AssistantTurn,
    pii_map: PIIMap,
    system_prompt: str,
) -> RewrittenTurn:
    """Rewrite a single assistant turn using Claude."""
    sem = _get_semaphore()

    scenario = _determine_scenario(trajectory, pii_map, turn)
    turn_context = _build_context_for_turn(trajectory, turn)

    user_prompt = build_turn_prompt(
        trajectory=trajectory,
        turn=turn,
        turn_context=turn_context,
        pii_map=pii_map,
        scenario_hint=scenario,
    )

    async with sem:
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        try:
            response = await client.messages.create(
                model=REWRITER_MODEL,
                max_tokens=8192,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            tracker.record_anthropic(response, "rewriter")
        except anthropic.RateLimitError:
            await asyncio.sleep(30)
            response = await client.messages.create(
                model=REWRITER_MODEL,
                max_tokens=8192,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            tracker.record_anthropic(response, "rewriter")

    # Parse response
    resp_text = response.content[0].text.strip()
    try:
        result = json_repair.loads(resp_text)
    except Exception:
        try:
            result = json.loads(resp_text)
        except json.JSONDecodeError:
            result = None

    # Handle non-dict results (list, None, parse errors)
    if not isinstance(result, dict):
        logger.warning("Turn %d: non-dict rewriter response (type=%s)", turn.turn_index, type(result).__name__)
        result = {
            "thinking": f"[Parse recovery] Raw response was not a JSON object. Original preserved.",
            "text": "\n".join(turn.text_blocks) if turn.text_blocks else "",
            "tool_calls": [],
            "tool_results": [],
            "privacy_actions": ["parse_error"],
            "scenario": scenario,
        }

    return RewrittenTurn(
        turn_index=turn.turn_index,
        thinking=result.get("thinking", ""),
        text=result.get("text", ""),
        tool_calls=result.get("tool_calls", []),
        tool_results=result.get("tool_results", []),
        privacy_actions=result.get("privacy_actions", []),
        scenario=result.get("scenario", scenario),
    )


def _build_context_for_turn(trajectory: ParsedTrajectory, turn: AssistantTurn) -> str:
    """Build surrounding context for a turn (preceding turns summary)."""
    lines = []
    for at in trajectory.assistant_turns:
        if at.turn_index >= turn.turn_index:
            break
        summary = (at.text_blocks[0][:100] + "...") if at.text_blocks else "(tool calls only)"
        tools = ", ".join(tc.name for tc in at.tool_calls) if at.tool_calls else "none"
        lines.append(f"  Turn {at.turn_index + 1}: {summary} [tools: {tools}]")

    if not lines:
        return "PRECEDING TURNS: This is the first assistant turn."
    return "PRECEDING TURNS:\n" + "\n".join(lines)


def _detect_workspace_path_from_trajectory(trajectory: ParsedTrajectory) -> str:
    """Detect the workspace path used across the entire trajectory."""
    for at in trajectory.assistant_turns:
        for tc in at.tool_calls:
            args = tc.arguments
            for key in ("path", "file_path", "command"):
                val = args.get(key, "")
                if isinstance(val, str):
                    if "/home/user/.openclaw/workspace" in val:
                        return "/home/user/.openclaw/workspace"
                    if "/home/user/OpenClawTrainer/workspace" in val:
                        return "/home/user/OpenClawTrainer/workspace"
    return "/home/user/.openclaw/workspace"


async def rewrite_trajectory(
    trajectory: ParsedTrajectory,
    pii_map: PIIMap,
) -> RewriteResult:
    """Rewrite all assistant turns in a trajectory for privacy compliance."""
    workspace_path = _detect_workspace_path_from_trajectory(trajectory)
    system_prompt = build_rewriter_system(workspace_path=workspace_path)
    rewritten_turns: list[RewrittenTurn] = []

    # Process turns sequentially to maintain context coherence
    for turn in trajectory.assistant_turns:
        rewritten = await _rewrite_single_turn(trajectory, turn, pii_map, system_prompt)
        rewritten_turns.append(rewritten)

    scenarios_covered = list(set(t.scenario for t in rewritten_turns if t.scenario))
    all_actions = []
    for t in rewritten_turns:
        all_actions.extend(t.privacy_actions)
    skills_used = list(set(
        tc["name"] for t in rewritten_turns for tc in t.tool_calls if isinstance(tc, dict)
    ))
    decision_points = sum(
        1 for t in rewritten_turns
        if any(a in t.privacy_actions for a in ("consent_gate", "refuse", "local_first", "vault_op", "classify"))
    )

    return RewriteResult(
        task_id=trajectory.task_id,
        submission_id=trajectory.submission_id,
        turns=rewritten_turns,
        scenarios_covered=scenarios_covered,
        skills_used=skills_used,
        privacy_decision_points=decision_points,
    )


async def refix_turns(
    trajectory: ParsedTrajectory,
    pii_map: PIIMap,
    rewrite_result: RewriteResult,
    turn_indices: list[int],
    fix_instructions: list[str],
) -> RewriteResult:
    """Re-fix specific turns based on verifier feedback."""
    workspace_path = _detect_workspace_path_from_trajectory(trajectory)
    system_prompt = build_rewriter_system(workspace_path=workspace_path)

    for idx, instruction in zip(turn_indices, fix_instructions):
        if idx >= len(trajectory.assistant_turns):
            continue

        turn = trajectory.assistant_turns[idx]
        scenario = _determine_scenario(trajectory, pii_map, turn)
        turn_context = _build_context_for_turn(trajectory, turn)

        # Include the previous rewrite attempt and fix instruction
        prev_attempt = ""
        for rt in rewrite_result.turns:
            if rt.turn_index == idx:
                prev_attempt = json.dumps({
                    "thinking": rt.thinking,
                    "text": rt.text,
                    "tool_calls": rt.tool_calls,
                    "privacy_actions": rt.privacy_actions,
                }, indent=2)
                break

        user_prompt = build_turn_prompt(
            trajectory=trajectory,
            turn=turn,
            turn_context=turn_context,
            pii_map=pii_map,
            scenario_hint=scenario,
        )
        user_prompt += f"\n\n---\nPREVIOUS ATTEMPT (needs fixing):\n{prev_attempt}\n\nFIX INSTRUCTION: {instruction}"

        sem = _get_semaphore()
        async with sem:
            client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
            response = await client.messages.create(
                model=REWRITER_MODEL,
                max_tokens=8192,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            tracker.record_anthropic(response, "rewriter_refix")

        resp_text = response.content[0].text.strip()
        try:
            result = json_repair.loads(resp_text)
        except Exception:
            result = json.loads(resp_text)

        fixed_turn = RewrittenTurn(
            turn_index=idx,
            thinking=result.get("thinking", ""),
            text=result.get("text", ""),
            tool_calls=result.get("tool_calls", []),
            tool_results=result.get("tool_results", []),
            privacy_actions=result.get("privacy_actions", []),
            scenario=result.get("scenario", scenario),
        )

        # Replace the turn in the result
        for i, t in enumerate(rewrite_result.turns):
            if t.turn_index == idx:
                rewrite_result.turns[i] = fixed_turn
                break

    # Recompute summary
    rewrite_result.scenarios_covered = list(set(t.scenario for t in rewrite_result.turns if t.scenario))
    rewrite_result.skills_used = list(set(
        tc["name"] for t in rewrite_result.turns for tc in t.tool_calls if isinstance(tc, dict)
    ))
    rewrite_result.privacy_decision_points = sum(
        1 for t in rewrite_result.turns
        if any(a in t.privacy_actions for a in ("consent_gate", "refuse", "local_first", "vault_op", "classify"))
    )

    return rewrite_result
