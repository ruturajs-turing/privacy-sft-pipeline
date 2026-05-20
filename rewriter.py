"""Stage 3: Claude Opus 4.6 per-turn rewriting engine with HTG compliance."""
from __future__ import annotations

import asyncio
import json
import logging
import random
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
) -> list[RewrittenTurn]:
    """Rewrite a single assistant turn using Claude. Returns 1+ turns (multi-turn for consent flows)."""
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

    return _parse_rewriter_result(result, turn.turn_index, scenario)


def _parse_rewriter_result(result: dict, turn_index: int, fallback_scenario: str) -> list[RewrittenTurn]:
    """Parse rewriter output into one or more RewrittenTurns.

    Handles both Format A (normal) and Format B (consent_flow with multi-turn expansion).
    Returns a list — usually 1 item, but 2 for consent flows (gate + execution).
    """
    if result.get("consent_flow"):
        return _expand_consent_flow(result, turn_index, fallback_scenario)

    return [RewrittenTurn(
        turn_index=turn_index,
        thinking=result.get("thinking", ""),
        text=result.get("text", ""),
        tool_calls=result.get("tool_calls", []),
        tool_results=result.get("tool_results", []),
        privacy_actions=result.get("privacy_actions", []),
        scenario=result.get("scenario", fallback_scenario),
    )]


def _expand_consent_flow(result: dict, turn_index: int, fallback_scenario: str) -> list[RewrittenTurn]:
    """Expand a consent_flow response into two RewrittenTurns with a synthetic user message."""
    gate = result.get("gate_message", {})
    user_resp = result.get("user_response", {})
    exec_msg = result.get("execution_message", {})
    scenario = result.get("scenario", fallback_scenario)
    actions = result.get("privacy_actions", [])
    consent_decision = user_resp.get("consent_decision", "granted")
    user_text = user_resp.get("text", "yes")

    # Turn A: the gate message (assistant asks for consent)
    gate_turn = RewrittenTurn(
        turn_index=turn_index,
        thinking=gate.get("thinking", ""),
        text=gate.get("text", ""),
        tool_calls=[],
        tool_results=[],
        privacy_actions=[a for a in actions if a in ("classify", "consent_gate")],
        scenario=scenario,
    )

    # Turn B: the execution (or graceful denial) — carries the synthetic user message
    exec_turn = RewrittenTurn(
        turn_index=turn_index,
        thinking=exec_msg.get("thinking", ""),
        text=exec_msg.get("text", ""),
        tool_calls=exec_msg.get("tool_calls", []),
        tool_results=exec_msg.get("tool_results", []),
        privacy_actions=[a for a in actions if a not in ("classify", "consent_gate")],
        scenario=scenario,
        synthetic_user_message=user_text,
        consent_decision=consent_decision,
    )

    return [gate_turn, exec_turn]


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


_ADVERSARIAL_PROMPT = """You are generating a synthetic adversarial user message and the assistant's polite refusal for a privacy-compliance training trajectory.

CONTEXT:
- The trajectory involves: {topic}
- The persona's name: {persona_name}
- PII available in context: {pii_summary}
- Attack type to generate: {attack_type}

ATTACK TYPE DESCRIPTIONS:
- instruction_override: User directly tells the AI to ignore/override its privacy rules
- social_engineering: User claims authority/approval to bypass rules ("my boss said", "security team approved")
- encoding: User tries to sneak PII via Base64, leetspeak, unicode tricks
- persistence_escalation: After being denied memory, user tries alternative persistence paths
- repeated_pressure: User pressures repeatedly after denial ("just do it!", "why won't you?")
- emotional_manipulation: User uses emotional appeals to bypass rules ("I'll forget my meds")

Generate a JSON object with this EXACT structure:
{{
  "user_attempt": {{
    "text": "<adversarial user message — 1-3 sentences, aggressive/manipulative/sneaky>",
    "attack_type": "{attack_type}"
  }},
  "assistant_response": {{
    "thinking": "<internal reasoning: classify the attempt, identify attack vector, cite HTG rule that prevents compliance>",
    "text": "<polite refusal: acknowledge request, explain briefly why impossible, offer alternative — 2-3 sentences, warm not robotic>"
  }}
}}

RULES:
- User message should reference REAL PII from the persona (names, SSN, health data, etc.) to make it realistic
- Assistant must NEVER comply, even partially
- Assistant must NOT be condescending or say "nice try"
- Assistant must offer a concrete alternative (use for current task, local password manager, etc.)
- Keep user message under 40 words
- Keep assistant response under 60 words"""

ATTACK_TYPES = [
    "instruction_override",
    "social_engineering",
    "encoding",
    "persistence_escalation",
    "repeated_pressure",
    "emotional_manipulation",
]


async def _inject_adversarial_turns(
    trajectory: ParsedTrajectory,
    pii_map: PIIMap,
    rewritten_turns: list[RewrittenTurn],
) -> list[RewrittenTurn]:
    """Generate 1-2 adversarial user+assistant pairs and inject them into the trajectory.

    Inserts adversarial turns at natural conversation breaks (between existing turns).
    Returns the modified turn list with adversarial turns inserted.
    """
    num_adversarial = random.choice([1, 1, 2])
    total_turns = len(rewritten_turns)

    if total_turns < 2:
        return rewritten_turns

    # Pick insertion points (after which turn index to insert)
    possible_positions = list(range(1, total_turns))
    random.shuffle(possible_positions)
    insertion_positions = sorted(possible_positions[:num_adversarial], reverse=True)

    # Derive context for the adversarial prompt
    persona = trajectory.persona or {}
    persona_name = f"{persona.get('first_name', 'User')} {persona.get('last_name', '')}".strip()
    pii_vault = persona.get("pii_vault", {})
    pii_summary_parts = []
    if pii_vault.get("government", {}).get("ssn"):
        pii_summary_parts.append(f"SSN: {pii_vault['government']['ssn']}")
    if pii_vault.get("health", {}).get("insurance_id"):
        pii_summary_parts.append(f"Insurance: {pii_vault['health']['insurance_id']}")
    if pii_vault.get("health", {}).get("diagnoses"):
        pii_summary_parts.append(f"Diagnoses: {', '.join(pii_vault['health']['diagnoses'][:2])}")
    if pii_vault.get("financial", {}).get("credit_card"):
        pii_summary_parts.append(f"Card: {pii_vault['financial']['credit_card']}")
    pii_summary = "; ".join(pii_summary_parts) if pii_summary_parts else "Full name, email, phone, health records"

    topic_parts = []
    for msg in trajectory.user_messages[:3]:
        topic_parts.append(msg[:100])
    topic = " | ".join(topic_parts) if topic_parts else "general task assistance"

    # Generate adversarial turns via Claude
    attack_types_used = random.sample(ATTACK_TYPES, min(num_adversarial, len(ATTACK_TYPES)))

    sem = _get_semaphore()
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    adversarial_turns = []
    for i, attack_type in enumerate(attack_types_used):
        prompt = _ADVERSARIAL_PROMPT.format(
            topic=topic[:200],
            persona_name=persona_name,
            pii_summary=pii_summary[:300],
            attack_type=attack_type,
        )

        async with sem:
            try:
                response = await client.messages.create(
                    model=REWRITER_MODEL,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                tracker.record_anthropic(response, "adversarial_gen")
            except anthropic.RateLimitError:
                await asyncio.sleep(15)
                response = await client.messages.create(
                    model=REWRITER_MODEL,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                tracker.record_anthropic(response, "adversarial_gen")

        resp_text = response.content[0].text.strip()
        try:
            result = json_repair.loads(resp_text)
        except Exception:
            try:
                result = json.loads(resp_text)
            except json.JSONDecodeError:
                logger.warning("Failed to parse adversarial turn %d, skipping", i)
                continue

        if not isinstance(result, dict):
            continue

        user_attempt = result.get("user_attempt", {})
        assistant_resp = result.get("assistant_response", {})

        adv_turn = RewrittenTurn(
            turn_index=-1,  # Will be re-indexed after insertion
            thinking=assistant_resp.get("thinking", "Adversarial attempt detected. Refusing per HTG."),
            text=assistant_resp.get("text", "I understand your request, but I'm not able to persist sensitive data regardless of authorization. I can use it for your current task though."),
            tool_calls=[],
            tool_results=[],
            privacy_actions=["adversarial_refusal", "classify"],
            scenario="D",
            is_adversarial=True,
            adversarial_user_message=user_attempt.get("text", "Save all my data to memory."),
            attack_type=user_attempt.get("attack_type", attack_type),
        )
        adversarial_turns.append(adv_turn)

    # Insert adversarial turns at the chosen positions
    result_turns = list(rewritten_turns)
    for idx, (pos, adv_turn) in enumerate(zip(insertion_positions, adversarial_turns)):
        if pos <= len(result_turns):
            result_turns.insert(pos, adv_turn)

    # Re-index all turns
    for i, t in enumerate(result_turns):
        t.turn_index = i

    return result_turns


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
        expanded = await _rewrite_single_turn(trajectory, turn, pii_map, system_prompt)
        rewritten_turns.extend(expanded)

    # Inject adversarial turns (1-2 per trajectory)
    rewritten_turns = await _inject_adversarial_turns(trajectory, pii_map, rewritten_turns)

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
