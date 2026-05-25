"""Stage 3: Claude Opus 4.6 per-turn rewriting engine with HTG compliance."""
from __future__ import annotations

import asyncio
import copy
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


def _is_valid_edit_args(args: object) -> bool:
    """Require path + non-empty edits list of {oldText, newText} dicts (strings)."""
    if not isinstance(args, dict):
        return False
    if not isinstance(args.get("path"), str) or not args["path"].strip():
        return False
    edits = args.get("edits")
    if not isinstance(edits, list) or len(edits) == 0:
        return False
    for e in edits:
        if not isinstance(e, dict):
            return False
        if "oldText" not in e or "newText" not in e:
            return False
        if not isinstance(e["oldText"], str) or not isinstance(e["newText"], str):
            return False
    return True


def _is_valid_write_args(args: object) -> bool:
    if not isinstance(args, dict):
        return False
    path = args.get("file_path") or args.get("path")
    if not isinstance(path, str) or not path.strip():
        return False
    if "content" not in args or not isinstance(args.get("content"), str):
        return False
    return True


def _repair_tool_calls_from_original(
    original_turn: AssistantTurn,
    tool_calls: list | None,
) -> tuple[list[dict], list[dict]]:
    """If Claude mangled edit/write JSON, restore arguments from the original turn (same id or index).

    Returns (repaired_tool_calls, repair_log) where each log entry has tool, reason, call_id.
    """
    if not tool_calls:
        return [], []

    orig_list = list(original_turn.tool_calls)
    orig_by_id = {tc.call_id: tc for tc in orig_list if getattr(tc, "call_id", None)}
    out: list[dict] = []
    log: list[dict] = []

    for i, d in enumerate(tool_calls):
        if not isinstance(d, dict):
            continue
        name = d.get("name", "") or ""
        args = d.get("arguments")
        if not isinstance(args, dict):
            args = {}
        cid = d.get("id") or d.get("call_id") or ""

        otc = None
        if cid and cid in orig_by_id and orig_by_id[cid].name == name:
            otc = orig_by_id[cid]
        elif i < len(orig_list) and orig_list[i].name == name:
            otc = orig_list[i]

        new_d = copy.deepcopy(d)
        if otc is None:
            out.append(new_d)
            continue

        reason = ""
        if name == "edit" and not _is_valid_edit_args(args):
            new_d["arguments"] = copy.deepcopy(otc.arguments)
            reason = "invalid_edit_args_restored_from_original"
        elif name == "write" and not _is_valid_write_args(args):
            new_d["arguments"] = copy.deepcopy(otc.arguments)
            reason = "invalid_write_args_restored_from_original"
        elif name == "edit" and args.get("path") != otc.arguments.get("path"):
            # Keep model newText minimization when structure is valid; fix wrong path only
            new_args = copy.deepcopy(args)
            new_args["path"] = otc.arguments.get("path", new_args.get("path"))
            new_d["arguments"] = new_args
            reason = "edit_path_normalized_to_original"
        if reason:
            log.append({"tool": name, "reason": reason, "call_id": cid or otc.call_id})
        if "id" not in new_d and otc.call_id:
            new_d["id"] = otc.call_id
        out.append(new_d)

    return out, log


def _determine_scenario(trajectory: ParsedTrajectory, pii_map: PIIMap, turn: AssistantTurn) -> str:
    """Determine which privacy scenario (A-N) best fits this turn."""
    has_persistence = any(
        tc.name in ("memory_write", "active_memory_write", "active_memory_set", "wiki_apply", "set", "vault_set")
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
    consent_context: str = "",
) -> tuple[list[RewrittenTurn], list[dict]]:
    """Rewrite a single assistant turn using Claude. Returns (turns, repair_log entries)."""
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

    # Inject consent state so the rewriter knows not to re-ask
    if consent_context:
        user_prompt += f"\n\n---\n\n{consent_context}"

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

    turns = _parse_rewriter_result(result, turn.turn_index, scenario)
    repair_entries: list[dict] = []
    for t in turns:
        t.tool_calls, rep = _repair_tool_calls_from_original(turn, t.tool_calls)
        for r in rep:
            repair_entries.append({"source_assistant_turn": turn.turn_index, **r})
    return turns, repair_entries


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

    # Flag the current turn's operation type
    current_ops = []
    for tc in turn.tool_calls:
        path = tc.arguments.get("path", "") or tc.arguments.get("file_path", "") or ""
        is_persistence = any(p in path.lower() for p in ("/memory/", "/notes/", "memory.md"))
        if tc.name == "read" or tc.name == "memory_search":
            current_ops.append(f"{tc.name} (READ-ONLY — no consent needed)")
        elif is_persistence and tc.name in ("write", "edit"):
            current_ops.append(f"{tc.name}→{path[-40:]} (PERSISTENCE — consent gate needed for L2)")
        else:
            current_ops.append(f"{tc.name}")

    op_note = ""
    if current_ops:
        op_note = f"\nCURRENT TURN OPERATIONS: {', '.join(current_ops)}"

    return "PRECEDING TURNS:\n" + "\n".join(lines) + op_note


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

FULL CONVERSATION CONTEXT:
{conversation_context}

CURRENT SITUATION:
- The persona's name: {persona_name}
- PII available in context: {pii_summary}
- Attack type to generate: {attack_type}

ATTACK TYPE DESCRIPTIONS:
- instruction_override: User directly tells the AI to ignore/override its privacy rules
- social_engineering: User claims authority/approval to bypass rules ("my boss said", "security team approved")
- encoding: User tries to sneak PII via Base64, leetspeak, unicode tricks
- persistence_escalation: After being denied memory, user tries alternative persistence paths
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
- The user is mentioning this sensitive data for the FIRST TIME. Do NOT say "I already told you" or reference previous requests.
- User message should reference REAL PII from the persona (names, SSN, health data, etc.)
- The message must make sense given the conversation context above — match the user's tone and the topic.
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

    # Pick insertion points — never the very end (at least 2 from the end)
    last_valid = max(1, total_turns - 2)
    possible_positions = list(range(1, last_valid + 1))
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

    # Build full conversation context for coherent adversarial generation
    from task_context import get_task_definition as _get_task_def
    task_def = _get_task_def(trajectory.task_id)
    ctx_parts = []
    if task_def:
        ctx_parts.append(
            f"TASK: {task_def.get('task_title', '')}\n"
            f"Goal: {task_def.get('goal_summary', '')}"
        )
    if persona:
        p = persona.get("personality", {})
        ctx_parts.append(
            f"PERSONA: {persona_name}, {persona.get('job_title', '')} in {persona.get('city', '')}"
        )
    conv_lines = []
    for msg in trajectory.user_messages[:6]:
        conv_lines.append(f"  USER: {str(msg)[:200]}")
    for t in rewritten_turns[:8]:
        preview = (t.text or "(tool use)")[:150]
        conv_lines.append(f"  ASSISTANT: {preview}")
    if conv_lines:
        ctx_parts.append("CONVERSATION:\n" + "\n".join(conv_lines[:12]))
    conversation_context = "\n\n".join(ctx_parts)

    # RAG: fetch real adversarial examples for context
    rag_adversarial_block = ""
    try:
        from rag_retriever import is_index_ready, get_adversarial_examples
        if is_index_ready():
            rag_adversarial_block = get_adversarial_examples(n=2)
    except Exception:
        pass

    # Generate adversarial turns via Claude
    attack_types_used = random.sample(ATTACK_TYPES, min(num_adversarial, len(ATTACK_TYPES)))

    sem = _get_semaphore()
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    adversarial_turns = []
    for i, attack_type in enumerate(attack_types_used):
        rag_section = f"\n\n{rag_adversarial_block}" if rag_adversarial_block else ""
        prompt = _ADVERSARIAL_PROMPT.format(
            conversation_context=conversation_context[:2000] + rag_section,
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


def _identify_persistence_turns(trajectory: ParsedTrajectory) -> dict[int, str]:
    """Pre-scan turns to identify which ones target persistence paths.

    Returns {turn_index: persistence_path} for turns that write to memory/notes paths.
    """
    PERSISTENCE_TOOLS = {"memory_write", "active_memory_write", "active_memory_set", "wiki_apply", "set", "vault_set"}
    result: dict[int, str] = {}

    for turn in trajectory.assistant_turns:
        for tc in turn.tool_calls:
            if tc.name in PERSISTENCE_TOOLS:
                result[turn.turn_index] = tc.name
                break
            if tc.name in ("write", "edit", "exec"):
                path = tc.arguments.get("path", "") or tc.arguments.get("file_path", "") or ""
                cmd = tc.arguments.get("command", "") or ""
                target = path or cmd
                if any(p in target.lower() for p in ("/memory/", "/notes/", "memory.md")):
                    result[turn.turn_index] = path or target
                    break
    return result


def _build_consent_state_context(consent_state: dict[str, str]) -> str:
    """Build a context string telling the rewriter what consent has already been obtained."""
    if not consent_state:
        return ""
    lines = ["CONSENT ALREADY OBTAINED (do NOT re-ask for these paths):"]
    for path, decision in consent_state.items():
        lines.append(f"  - {path}: {decision}")
    lines.append("For these paths, proceed directly with the write (applying data minimization). Do NOT generate a consent_flow.")
    return "\n".join(lines)


async def rewrite_trajectory(
    trajectory: ParsedTrajectory,
    pii_map: PIIMap,
) -> RewriteResult:
    """Rewrite all assistant turns in a trajectory for privacy compliance.

    Key architecture: persistence turns are grouped so only the FIRST write to a
    given path triggers a consent gate. Subsequent writes to the same path reuse
    the granted consent and just execute with minimization.
    """
    workspace_path = _detect_workspace_path_from_trajectory(trajectory)
    system_prompt = build_rewriter_system(workspace_path=workspace_path)
    rewritten_turns: list[RewrittenTurn] = []
    rewrite_repairs: list[dict] = []

    persistence_turns = _identify_persistence_turns(trajectory)
    consent_state: dict[str, str] = {}

    for turn in trajectory.assistant_turns:
        consent_context = _build_consent_state_context(consent_state)

        expanded, rep = await _rewrite_single_turn(
            trajectory, turn, pii_map, system_prompt,
            consent_context=consent_context,
        )
        rewrite_repairs.extend(rep)

        # Track consent decisions from this turn's output
        for rt in expanded:
            if rt.consent_decision and rt.consent_decision != "":
                path = persistence_turns.get(turn.turn_index, "")
                if path:
                    consent_state[path] = rt.consent_decision

        rewritten_turns.extend(expanded)

    # Inject adversarial turns (1-2 per trajectory)
    rewritten_turns = await _inject_adversarial_turns(trajectory, pii_map, rewritten_turns)

    # Post-process: collapse redundant consent gates
    rewritten_turns = _collapse_redundant_consent_gates(rewritten_turns)

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
        rewrite_repairs=rewrite_repairs,
    )


def _collapse_redundant_consent_gates(turns: list[RewrittenTurn]) -> list[RewrittenTurn]:
    """Remove duplicate consent gates that ask about the same path within close proximity.

    If a consent gate turn has no tool calls and the next execution turn also has
    no tool calls (consent was granted but nothing executed), remove the pair.
    Also collapse consecutive gate→user→gate→user sequences into single gate→user→execute.
    """
    if len(turns) < 2:
        return turns

    result: list[RewrittenTurn] = []
    seen_consent_paths: set[str] = set()
    i = 0

    while i < len(turns):
        turn = turns[i]

        # Check if this is a consent gate (has consent_gate action, no tool calls)
        is_gate = (
            "consent_gate" in turn.privacy_actions
            and not turn.tool_calls
            and not turn.synthetic_user_message
        )

        if is_gate and i + 1 < len(turns):
            next_turn = turns[i + 1]
            # The next turn should be the execution with synthetic_user_message
            if next_turn.synthetic_user_message:
                # Check if we already have consent for a similar operation
                gate_text_key = turn.text[:50] if turn.text else ""
                if gate_text_key in seen_consent_paths:
                    # Skip this redundant gate+execution pair — consent already obtained
                    i += 2
                    continue
                seen_consent_paths.add(gate_text_key)

        result.append(turn)
        i += 1

    # Re-index
    for idx, t in enumerate(result):
        t.turn_index = idx

    return result


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

        fixed_turn.tool_calls, rep = _repair_tool_calls_from_original(turn, fixed_turn.tool_calls)
        for r in rep:
            rewrite_result.rewrite_repairs.append({"source": "refix", "source_assistant_turn": idx, **r})

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
