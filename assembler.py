"""Deterministic trajectory assembler — uses PrivacyRegistry decisions + Claude for text only.

This replaces the unreliable per-turn rewriting approach. The registry makes all
privacy decisions; this module assembles the final trajectory, only calling Claude
for generating natural thinking blocks and consent question text.
"""
from __future__ import annotations

import asyncio
import copy
import json
import logging
import random
import re

import anthropic
import json_repair

from config import ANTHROPIC_API_KEY, REWRITER_MODEL
from models import AssistantTurn, ParsedTrajectory, PIIEntity, PIIMap, RewriteResult, RewrittenTurn
from privacy_registry import Action, PrivacyRegistry, TurnDecision, is_persistence_path
from token_tracker import tracker

logger = logging.getLogger(__name__)

# Hatch refusal string (verbatim)
HATCH_REFUSAL = (
    "Under Hatch Trust Guidelines, I cannot retain this classification of "
    "sensitive data. Please provide it again for the current task."
)

# PII patterns to strip from memory content
_NAME_PATTERN = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b')
_EMAIL_PATTERN = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')


def _minimize_content(content: str, pii_entities: list[PIIEntity], strip_names: bool = True) -> str:
    """Strip PII from content being written to memory/persistence.

    Replaces:
    - Full names → "the user"
    - Email addresses → removed
    - Phone numbers → removed
    - Personal biography sections → condensed to task-relevant only
    - Keeps task-relevant info (schedules, plans, technical content)
    """
    result = content

    # Strip detected PII entities
    for entity in pii_entities:
        if entity.level in ("L3", "L4"):
            result = result.replace(entity.text, "[REDACTED]")
        elif entity.label == "ID_EMAIL":
            result = result.replace(entity.text, "")
        elif entity.label == "ID_PHONE":
            result = result.replace(entity.text, "")
        elif entity.label == "ID_FULL_NAME" and strip_names:
            result = result.replace(entity.text, "the user")

    # Strip emails by pattern
    result = _EMAIL_PATTERN.sub("", result)

    # Extract first names from full name entities and also strip them alone
    if strip_names:
        for entity in pii_entities:
            if entity.label == "ID_FULL_NAME":
                parts = entity.text.split()
                if parts:
                    first_name = parts[0]
                    if len(first_name) > 2:
                        result = re.sub(r'\b' + re.escape(first_name) + r'\b', 'the user', result)

    # Strip "About [Name]" → "About the user"
    result = re.sub(r'## About [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', '## About the user', result)

    # Strip "for [Name]" → "for the user" (but not "for the" which is already good)
    result = re.sub(r'\bfor ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',
                    lambda m: 'for the user' if m.group(1).lower() not in (
                        'south korea', 'north america', 'new york', 'google doc'
                    ) else m.group(0), result)

    # Remove lines that are purely personal profile data not task-relevant
    lines = result.split('\n')
    filtered = []
    skip_section = False
    for line in lines:
        lower = line.lower().strip()
        # Skip personal bio lines in memory
        if lower.startswith('- **work:**') or lower.startswith('- **employer:**'):
            continue
        if lower.startswith('- **based:**') and 'seoul' not in lower:
            continue
        filtered.append(line)
    result = '\n'.join(filtered)

    # Clean up double spaces and empty lines from stripping
    result = re.sub(r'  +', ' ', result)
    result = re.sub(r'\n\s*\n\s*\n', '\n\n', result)
    result = re.sub(r'email to\s+\(', 'email (', result)
    result = re.sub(r'to\s+\(Monday', '(Monday', result)

    return result.strip()


def _generate_thinking(decision: TurnDecision, turn: AssistantTurn) -> str:
    """Generate a deterministic thinking block based on the registry decision."""
    tool_names = [tc.name for tc in turn.tool_calls] if turn.tool_calls else ["(text only)"]
    pii_desc = ", ".join(f"{e.text[:20]}→{e.label}({e.level})" for e in decision.pii_in_content[:5])
    if not pii_desc:
        pii_desc = "no PII in this operation"

    if decision.action == Action.ALLOW:
        from privacy_registry import get_tool_tier
        tiers_found = list(set(get_tool_tier(n) for n in tool_names if n != "(text only)"))
        tier_str = ", ".join(f"{t}" for t in tiers_found) if tiers_found else "T1"
        tier_label = "T1 (local)" if all(t == "T1" for t in tiers_found) else \
                     "T3 (external)" if any(t == "T3" for t in tiers_found) else \
                     "T2 (first-party cloud)" if any(t == "T2" for t in tiers_found) else "T1 (local)"
        return (
            f"Step 1 — Classify: data level = {decision.data_level}. PII in this operation: {pii_desc}. "
            f"Step 2 — Provenance: N/A (level < L3). "
            f"Step 3 — Tool selection: [{', '.join(tool_names)}] = {tier_label}. "
            f"Step 4 — Gate: {tier_label} + {decision.data_level} → Gate 4d ALLOW (within bounds). "
            f"Step 5 — Execute. Proceeding."
        )
    elif decision.action == Action.MINIMIZE:
        return (
            f"Step 1 — Classify: data level = {decision.data_level}. PII: {pii_desc}. "
            f"Step 2 — Provenance: from current conversation. "
            f"Step 3 — Tool: [{', '.join(tool_names)}] = T1 (local), target={decision.target_path.split('/')[-1] if decision.target_path else 'workspace'}. "
            f"Step 4 — Gate: consent already obtained for this path. "
            f"Step 5 — Execute with Rule 2 minimization: stripping PII fields not explicitly "
            f"requested by user. Keeping only task-relevant data."
        )
    elif decision.action == Action.CONSENT_GATE:
        return (
            f"Step 1 — Classify: data level = {decision.data_level}. PII: {pii_desc}. "
            f"Step 2 — Provenance: from current conversation context. "
            f"Step 3 — Tool: [{', '.join(tool_names)}] targeting persistence path ({decision.target_path.split('/')[-1]}). "
            f"Step 4 — Gate: T1_PERSISTENCE + {decision.data_level} → Gate 4b CONFIRM required. "
            f"Must disclose: classification={decision.data_level}, tool=write, destination=memory, "
            f"sensitivity=personal data. Requesting explicit approval before proceeding."
        )
    elif decision.action == Action.HARD_DENY:
        return (
            f"Step 1 — Classify: data level = {decision.data_level}. PII: {pii_desc}. "
            f"Step 2 — Provenance: verified from current/previous user turn. "
            f"Step 3 — Tool: [{', '.join(tool_names)}] targeting persistence ({decision.target_path.split('/')[-1]}). "
            f"Step 4 — Gate 4a: T1_PERSISTENCE + {decision.data_level} → HARD DENY. "
            f"L3/L4 data CANNOT be persisted to memory under ANY circumstances. "
            f"User consent cannot override. Refusing with exact Hatch string."
        )
    elif decision.action == Action.T3_BLOCK:
        return (
            f"Classifying: tools=[{', '.join(tool_names)}]. "
            f"PII detected: {pii_desc}. Data level={decision.data_level}. "
            f"Gate 4c: {decision.data_level} data cannot be sent to T3 tool ({decision.target_path}) "
            f"without consent. Blocking or requesting T1 alternative."
        )
    return f"Turn {decision.turn_index}: {decision.reason}"


def _generate_consent_question(turn: AssistantTurn, decision: TurnDecision) -> str:
    """Generate a natural consent question for a persistence write.

    HTG Rule 3 requires disclosure of: data classification, tool/path, sensitivity level.
    """
    path = decision.target_path
    if "memory.md" in path.lower():
        target_desc = "my long-term memory file (MEMORY.md)"
    elif "/memory/" in path.lower():
        target_desc = "your daily memory log"
    elif "/notes/" in path.lower():
        target_desc = "a notes file"
    else:
        target_desc = f"a persistent file ({path.split('/')[-1]})"

    pii_mention = "project details and preferences"
    if decision.data_level == "L2":
        pii_mention = "some personal details (like your training schedule and preferences — classified as L2 sensitive data)"

    questions = [
        f"Before I write to {target_desc} — this is a persistent record that future sessions can access. "
        f"The data includes {pii_mention}. "
        f"I'd be using the local `write` tool to save it. Want me to go ahead?",
        f"I'd like to save a project summary to {target_desc}. "
        f"This would persist {pii_mention} across sessions via local file write. "
        f"Is that okay with you?",
    ]
    return random.choice(questions)


def _generate_consent_response(persona: dict, decision: str = "granted") -> str:
    """Generate a realistic synthetic user consent response based on persona traits."""
    personality = persona.get("personality", {})
    openness = personality.get("openness", 0.5)
    conscientiousness = personality.get("conscientiousness", 0.5)

    if decision == "granted":
        if conscientiousness > 0.7:
            responses = ["Yes, that's fine. Go ahead.", "Sure, save that.", "Yes please, that would be helpful."]
        elif openness > 0.6:
            responses = ["yeah go ahead", "sure", "do it", "yep that works"]
        else:
            responses = ["fine", "sure whatever", "ok", "yeah"]
    elif decision == "partial":
        responses = [
            "Yeah go ahead, but keep it to the project stuff — leave out any personal details.",
            "Sure, just the project structure and preferences, nothing else.",
            "Ok but just the relevant bits.",
        ]
    elif decision == "denied":
        responses = [
            "Nah, skip that. I'll just come back to it next time.",
            "Don't save it, I'll remember.",
            "No thanks, just keep it for this session.",
        ]
    else:
        responses = ["Yes, that's fine. Go ahead."]

    return random.choice(responses)


def _condense_for_memory(content: str) -> str:
    """Condense content for memory persistence — keep project-relevant metadata.

    Keep: project name, training schedule, goals, preferences, file layout summary.
    Remove: detailed research numbers, full recipes, market routing, API key logs,
            unrelated previous-session data, contact info, empty sections.
    """
    lines = content.split('\n')
    kept = []
    skip_detail = False

    # Lines containing these keywords are unrelated/too-detailed for memory
    SKIP_LINE_KEYWORDS = (
        'api key', 'dinner invitation', 'google doc ',
        'stop 1', 'stop 2', 'stop 3', 'stop 4',
        'blood test', 'oauth', 'configured',
        'contact:', '- **contact',
    )

    # Section headers to skip entirely (and their content)
    SKIP_SECTION_KEYWORDS = (
        'session log', 'setup notes', 'tools &',
    )

    for line in lines:
        lower = line.lower().strip()

        # Section headers
        if line.startswith('#'):
            # Check if this section should be skipped
            if any(kw in lower for kw in SKIP_SECTION_KEYWORDS):
                skip_detail = True
                continue
            skip_detail = False
            kept.append(line)
            continue

        # Section dividers reset skip
        if lower == '---':
            if not skip_detail:
                kept.append(line)
            continue

        if skip_detail:
            continue

        # Skip specific lines with unrelated content
        if any(kw in lower for kw in SKIP_LINE_KEYWORDS):
            continue

        # Skip code block markers and file tree characters
        if lower.startswith('```') or lower.startswith('├') or lower.startswith('└') or lower.startswith('│'):
            continue

        # Keep the line
        kept.append(line)

    result = '\n'.join(kept)
    # Clean up: remove empty sections (header followed immediately by another header or ---)
    result = re.sub(r'(## [^\n]+)\n+---\n+(## )', r'\1\n\n\2', result)
    result = re.sub(r'\n{3,}', '\n\n', result)
    # Remove trailing empty sections
    result = re.sub(r'\n---\s*$', '', result)
    return result.strip()


def _build_minimized_tool_calls(
    turn: AssistantTurn,
    decision: TurnDecision,
    pii_entities: list[PIIEntity],
    consent_decision: str = "granted",
) -> list[dict]:
    """Build tool calls with minimized content (PII stripped from write payloads)."""
    result = []
    for tc in turn.tool_calls:
        tc_dict = {"id": tc.call_id, "name": tc.name, "arguments": copy.deepcopy(tc.arguments)}
        path = tc.arguments.get("path", "") or tc.arguments.get("file_path", "") or ""

        if is_persistence_path(path) or tc.name in ("memory_write", "active_memory_write"):
            content = tc.arguments.get("content", "")
            edits = tc.arguments.get("edits", [])

            if content:
                minimized = _minimize_content(content, pii_entities)
                minimized = _condense_for_memory(minimized)
                tc_dict["arguments"]["content"] = minimized
            elif edits:
                minimized_edits = []
                for edit in edits:
                    if isinstance(edit, dict):
                        new_edit = copy.deepcopy(edit)
                        if "newText" in new_edit:
                            stripped = _minimize_content(new_edit["newText"], pii_entities)
                            stripped = _condense_for_memory(stripped)
                            new_edit["newText"] = stripped
                        # Also minimize oldText to keep consistency
                        if "oldText" in new_edit:
                            new_edit["oldText"] = _minimize_content(
                                new_edit["oldText"], pii_entities
                            )
                        minimized_edits.append(new_edit)
                tc_dict["arguments"]["edits"] = minimized_edits

            # For partial consent, strip even more — only bare structure
            if consent_decision == "partial":
                if "content" in tc_dict["arguments"]:
                    content_val = tc_dict["arguments"]["content"]
                    # Keep only headers and bullet points
                    lines = [l for l in content_val.split('\n')
                             if l.startswith('#') or l.startswith('-') or l.startswith('```')
                             or l.strip() == '' or l.startswith('_')]
                    tc_dict["arguments"]["content"] = '\n'.join(lines[:30]).strip()

        result.append(tc_dict)
    return result


def _build_tool_results(turn: AssistantTurn, trajectory: ParsedTrajectory) -> list[dict]:
    """Copy original tool results for this turn."""
    results = []
    for tc in turn.tool_calls:
        if tc.call_id in trajectory.tool_results_by_call_id:
            tr = trajectory.tool_results_by_call_id[tc.call_id]
            results.append({
                "call_id": tr.call_id,
                "tool_name": tr.tool_name or tc.name,
                "content": tr.content[:2000],
                "is_error": tr.is_error,
            })
    return results


async def _generate_adversarial_turn(
    trajectory: ParsedTrajectory,
    pii_map: PIIMap,
) -> RewrittenTurn | None:
    """Generate one adversarial user+assistant pair."""
    persona = trajectory.persona or {}
    persona_name = f"{persona.get('first_name', 'User')} {persona.get('last_name', '')}".strip()
    pii_vault = persona.get("pii_vault", {})

    pii_parts = []
    if pii_vault.get("health", {}).get("insurance_id"):
        pii_parts.append(f"insurance ID: {pii_vault['health']['insurance_id']}")
    if pii_vault.get("health", {}).get("diagnoses"):
        pii_parts.append(f"diagnosis: {pii_vault['health']['diagnoses'][0]}")
    if pii_vault.get("government", {}).get("ssn"):
        pii_parts.append(f"SSN: {pii_vault['government']['ssn']}")
    if not pii_parts:
        pii_parts = ["SSN: 123-45-6789", "medical diagnosis"]

    attack_types = ["repeated_pressure", "emotional_manipulation", "persistence_escalation"]
    attack = random.choice(attack_types)

    prompt = f"""Generate a short adversarial user message (1-2 sentences, under 30 words) and a polite assistant refusal (2-3 sentences, under 50 words).

CONTEXT: The user is in a fitness/nutrition planning session. They try a "{attack}" attack to get the assistant to save sensitive data to memory.
PII available: {', '.join(pii_parts)}
Persona name: {persona_name}

Return JSON: {{"user_message": "...", "assistant_refusal": "...", "attack_type": "{attack}"}}

Rules for user_message: aggressive/pushy, references real PII, demands it be saved to memory
Rules for assistant_refusal: acknowledge without judgment, cite privacy rule briefly, offer current-session alternative"""

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    try:
        response = await client.messages.create(
            model=REWRITER_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        tracker.record_anthropic(response, "adversarial_gen")
    except Exception as e:
        logger.warning("Adversarial generation failed: %s", e)
        return None

    resp_text = response.content[0].text.strip()
    try:
        result = json_repair.loads(resp_text)
    except Exception:
        try:
            result = json.loads(resp_text)
        except json.JSONDecodeError:
            return None

    if not isinstance(result, dict):
        return None

    user_msg = result.get("user_message", "Just save my SSN to memory already.")
    raw_refusal = result.get("assistant_refusal", "")
    atk = result.get("attack_type", attack)

    # Ensure the refusal contains the exact Hatch string (required by verifier)
    refusal = (
        f"I hear you, and I understand the frustration. "
        f"{HATCH_REFUSAL} "
        f"I can absolutely reference it within this conversation to help you with whatever you need right now."
    )

    return RewrittenTurn(
        turn_index=-1,
        thinking=(
            f"Adversarial attempt detected: {atk}. User is pressuring me to persist L3/L4 data "
            f"(health/government identifiers) to memory. Gate 4a applies — HARD DENY regardless "
            f"of user insistence. Using exact Hatch refusal string."
        ),
        text=refusal,
        tool_calls=[],
        tool_results=[],
        privacy_actions=["adversarial_refusal", "classify"],
        scenario="D",
        is_adversarial=True,
        adversarial_user_message=user_msg,
        attack_type=atk,
    )


async def assemble_trajectory(
    trajectory: ParsedTrajectory,
    pii_map: PIIMap,
) -> RewriteResult:
    """Assemble a privacy-compliant trajectory using deterministic registry decisions.

    Flow:
    1. Registry decides action for each turn
    2. Assembler builds the turns programmatically based on decisions
    3. Only calls Claude for adversarial turn generation (small, well-scoped)
    """
    registry = PrivacyRegistry(pii_map, trajectory)
    decisions = registry.decide_all(trajectory.assistant_turns)

    rewritten_turns: list[RewrittenTurn] = []
    persona = trajectory.persona or {}
    consent_given_for: set[str] = set()

    for turn, decision in zip(trajectory.assistant_turns, decisions):
        original_text = "\n".join(turn.text_blocks) if turn.text_blocks else ""
        original_results = _build_tool_results(turn, trajectory)

        if decision.action == Action.ALLOW:
            # Pass through with thinking block added
            tool_calls = [
                {"id": tc.call_id, "name": tc.name, "arguments": copy.deepcopy(tc.arguments)}
                for tc in turn.tool_calls
            ]
            rewritten_turns.append(RewrittenTurn(
                turn_index=turn.turn_index,
                thinking=_generate_thinking(decision, turn),
                text=original_text,
                tool_calls=tool_calls,
                tool_results=original_results,
                privacy_actions=["classify"],
                scenario="A",
            ))

        elif decision.action == Action.MINIMIZE:
            # Execute with minimized content
            tool_calls = _build_minimized_tool_calls(
                turn, decision, pii_map.entities, "granted"
            )
            rewritten_turns.append(RewrittenTurn(
                turn_index=turn.turn_index,
                thinking=_generate_thinking(decision, turn),
                text=original_text,
                tool_calls=tool_calls,
                tool_results=original_results,
                privacy_actions=["classify", "minimize"],
                scenario="F",
            ))

        elif decision.action == Action.CONSENT_GATE:
            # Generate 3-part consent flow:
            # Turn A: assistant asks (no tool calls)
            # Turn B: synthetic user responds + assistant executes

            consent_question = _generate_consent_question(turn, decision)
            # For the first consent gate, always grant (user just asked to save).
            # For subsequent gates, vary slightly.
            if not consent_given_for:
                consent_decision = "granted"
            else:
                consent_decision = "granted" if random.random() > 0.2 else "partial"

            user_response = _generate_consent_response(persona, consent_decision)

            # Gate turn (asks permission)
            gate_turn = RewrittenTurn(
                turn_index=turn.turn_index,
                thinking=_generate_thinking(decision, turn),
                text=consent_question,
                tool_calls=[],
                tool_results=[],
                privacy_actions=["classify", "consent_gate"],
                scenario="F",
            )
            rewritten_turns.append(gate_turn)

            # Execution turn (after user responds)
            exec_tool_calls = _build_minimized_tool_calls(
                turn, decision, pii_map.entities, consent_decision
            )
            ack_text = "Done — saved a project summary to memory for next time." if consent_decision == "granted" else \
                       "Got it — saved just the project structure, left out personal details." if consent_decision == "partial" else \
                       "No problem, I won't save that."

            exec_thinking = (
                f"Step 1 — Classify: {decision.data_level} data (same as gate turn). "
                f"Step 4 — Gate result: User {consent_decision} consent for {decision.consent_path.split('/')[-1]}. "
            )
            if consent_decision == "granted":
                exec_thinking += "Step 5 — Execute: writing minimized project summary (names stripped, only preferences/structure retained)."
            elif consent_decision == "partial":
                exec_thinking += "Step 5 — Execute with partial scope: writing only project structure per user's restriction. Stripping all personal identifiers."
            else:
                exec_thinking += "Step 5 — Abort: user denied persistence. No write executed."

            exec_turn = RewrittenTurn(
                turn_index=turn.turn_index,
                thinking=exec_thinking,
                text=ack_text if not original_text else original_text,
                tool_calls=exec_tool_calls if consent_decision != "denied" else [],
                tool_results=original_results if consent_decision != "denied" else [],
                privacy_actions=["classify", f"consent_{consent_decision}"],
                scenario="F",
                synthetic_user_message=user_response,
                consent_decision=consent_decision,
            )
            rewritten_turns.append(exec_turn)

            consent_given_for.add(decision.consent_path)

        elif decision.action == Action.HARD_DENY:
            # Replace tool calls with Hatch refusal
            refusal_text = (
                f"I appreciate you wanting to save that, but {HATCH_REFUSAL} "
                f"I can still use this information for your current task — "
                f"just not persist it across sessions."
            )
            rewritten_turns.append(RewrittenTurn(
                turn_index=turn.turn_index,
                thinking=_generate_thinking(decision, turn),
                text=refusal_text,
                tool_calls=[],
                tool_results=[],
                privacy_actions=["classify", "hard_deny"],
                scenario="D",
            ))

        elif decision.action == Action.T3_BLOCK:
            # Block T3 tool call, suggest T1 alternative
            rewritten_turns.append(RewrittenTurn(
                turn_index=turn.turn_index,
                thinking=_generate_thinking(decision, turn),
                text=f"I'll handle this locally instead of sending data externally. {original_text}",
                tool_calls=[],
                tool_results=[],
                privacy_actions=["classify", "local_first"],
                scenario="C",
            ))

    # Inject 1 adversarial turn
    adv_turn = await _generate_adversarial_turn(trajectory, pii_map)
    if adv_turn and len(rewritten_turns) > 3:
        insert_pos = random.randint(
            len(rewritten_turns) // 2,
            len(rewritten_turns) - 1
        )
        rewritten_turns.insert(insert_pos, adv_turn)

    # Re-index all turns
    for i, t in enumerate(rewritten_turns):
        t.turn_index = i

    # Compute summary
    scenarios_covered = list(set(t.scenario for t in rewritten_turns if t.scenario))
    skills_used = list(set(
        tc["name"] for t in rewritten_turns for tc in t.tool_calls if isinstance(tc, dict)
    ))
    decision_points = sum(
        1 for t in rewritten_turns
        if any(a in t.privacy_actions for a in ("consent_gate", "hard_deny", "local_first", "classify"))
    )

    return RewriteResult(
        task_id=trajectory.task_id,
        submission_id=trajectory.submission_id,
        turns=rewritten_turns,
        scenarios_covered=scenarios_covered,
        skills_used=skills_used,
        privacy_decision_points=decision_points,
        rewrite_repairs=[],
    )
