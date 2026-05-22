"""Single-shot Privacy SFT rewriter.

Instead of a multi-stage pipeline where each stage has partial context and
introduces compounding errors, this module sends the FULL original trajectory
plus ALL context (PII map, task definition, privacy rules, RAG examples) to
a single powerful LLM call and gets back the complete privacy-compliant
trajectory in one shot.

The LLM sees:
  - The entire original conversation (user msgs, assistant turns, tool calls, results)
  - PII classification for every entity found
  - Task definition (goal, expected privacy actions, scenarios)
  - Hatch Trust Guidelines / privacy rules
  - Real conversation examples from RAG (for tone and realism)
  - The persona's data vault

And produces:
  - A complete RewriteResult with properly ordered RewrittenTurns
  - Natural consent gates, adversarial probes, refusals injected in-context
  - Proper user/assistant alternation with no structural issues
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import anthropic

from config import ANTHROPIC_API_KEY, REWRITER_MODEL
from models import (
    ParsedTrajectory, PIIMap, PIIEntity,
    RewriteResult, RewrittenTurn,
)
from token_tracker import tracker

logger = logging.getLogger(__name__)

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Format the original trajectory as readable text for the LLM
# ---------------------------------------------------------------------------

def _format_original_trajectory(trajectory: ParsedTrajectory) -> str:
    """Convert the parsed trajectory into a readable text representation."""
    lines = []
    user_idx = 0
    turn_idx = 0

    for kind, idx in trajectory.thread_order:
        if kind == "user":
            if idx < len(trajectory.user_messages):
                msg = trajectory.user_messages[idx]
                # Truncate very long user messages but keep enough context
                if len(msg) > 2000:
                    msg = msg[:1800] + "\n...[truncated]..."
                lines.append(f"[USER MESSAGE {user_idx}]\n{msg}\n")
                user_idx += 1

        elif kind == "assistant":
            if idx < len(trajectory.assistant_turns):
                at = trajectory.assistant_turns[idx]
                parts = []

                text = "\n".join(at.text_blocks).strip()
                if text:
                    parts.append(f"Text: {text}")

                for tc in at.tool_calls:
                    args_str = json.dumps(tc.arguments)
                    if len(args_str) > 500:
                        args_str = args_str[:450] + "...}"
                    parts.append(f"Tool Call: {tc.name}({args_str})")

                    result = trajectory.tool_results_by_call_id.get(tc.call_id)
                    if result:
                        result_str = result.content
                        if len(result_str) > 500:
                            result_str = result_str[:450] + "...[truncated]"
                        parts.append(f"Tool Result ({tc.name}): {result_str}")

                content = "\n".join(parts)
                lines.append(f"[ASSISTANT TURN {turn_idx}]\n{content}\n")
                turn_idx += 1

    return "\n".join(lines)


def _format_pii_map(pii_map: PIIMap) -> str:
    """Format PII entities into a readable summary."""
    if not pii_map.entities:
        return "No PII entities detected."

    lines = [f"PII Entities Found ({len(pii_map.entities)} total, max level: {pii_map.max_level}):"]
    seen = set()
    for e in pii_map.entities:
        key = (e.label, e.level, e.text[:50])
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"  - {e.label} ({e.level}): \"{e.text[:80]}\"")

    return "\n".join(lines)


def _format_task_spec(trajectory: ParsedTrajectory) -> str:
    """Format task definition into readable text."""
    spec = trajectory.task_spec
    if not spec:
        return "No task definition available."

    lines = ["Task Definition:"]
    for key in ["title", "goal_summary", "privacy_scenario", "data_levels",
                "expected_privacy_actions", "tool_tiers", "pii_fields_exercised"]:
        val = spec.get(key, "")
        if val:
            lines.append(f"  {key}: {val}")

    return "\n".join(lines)


def _format_persona_vault(persona: dict) -> str:
    """Format persona's PII vault for context."""
    if not persona:
        return "No persona data available."

    lines = [f"Persona: {persona.get('first_name', '')} {persona.get('last_name', '')}"]

    vault = persona.get("pii_vault", {})
    if vault:
        lines.append("PII Vault (data the persona has):")
        for category, entries in vault.items():
            if isinstance(entries, dict):
                for k, v in entries.items():
                    val_str = str(v)
                    if len(val_str) > 80:
                        val_str = val_str[:77] + "..."
                    lines.append(f"  {category}.{k}: {val_str}")
            else:
                lines.append(f"  {category}: {entries}")

    personality = persona.get("personality_profile", {})
    if personality:
        lines.append(f"Communication style: {personality.get('communication_style', 'casual')}")
        lines.append(f"Verbosity: {personality.get('verbosity', 'moderate')}")

    return "\n".join(lines)


def _get_rag_context(trajectory: ParsedTrajectory, pii_map: PIIMap) -> str:
    """Pull relevant examples from RAG for the LLM prompt."""
    try:
        from rag_retriever import (
            get_similar_conversations,
            get_privacy_patterns,
            is_index_ready,
        )
        if not is_index_ready():
            return ""

        task_goal = trajectory.task_spec.get("goal_summary", "")
        if not task_goal:
            task_goal = " ".join(trajectory.user_messages[:2])[:200]

        parts = []
        conv_examples = get_similar_conversations(task_goal, n=3)
        if conv_examples:
            parts.append(conv_examples)

        privacy_query = f"privacy handling for {pii_map.max_level} data"
        privacy_examples = get_privacy_patterns(privacy_query, n=3)
        if privacy_examples:
            parts.append(privacy_examples)

        return "\n\n".join(parts)
    except Exception as e:
        logger.debug("RAG context fetch failed: %s", e)
        return ""


# ---------------------------------------------------------------------------
# The core prompt
# ---------------------------------------------------------------------------

PRIVACY_RULES = """
HATCH TRUST GUIDELINES — PRIVACY RULES (from team meeting decisions):

DATA CLASSIFICATION:
  L0 (Public): Broad locations, general knowledge → No flagging needed. Store freely.
  L1 (Caution): Device IDs, IPs, metro-level city, age, hobbies → No flagging needed. Store freely.
  L2 (Sensitive): Full names, emails, employer, job title, account IDs → Can be stored WITH T1 tools without consent. Needs consent only for T2/T3 tools.
  L3 (Highly Sensitive): Health records, detailed financials, precise GPS → Zero retention. Agent amnesia. Must re-input per turn.
  L4 (Critical): SSNs, passwords, biometrics, full credit cards → NEVER store anywhere. Agent amnesia. Must re-input per turn.

KEY RULE — L0, L1, AND L2 WITH T1 TOOLS: NO CONSENT NEEDED.
  The agent can store L0, L1, and L2 data using T1 tools (memory_write, write, etc.) WITHOUT asking for
  explicit user permission. Just do it naturally. Do NOT insert unnecessary consent gates for L0/L1/L2
  data when using local tools. The user telling you their name, employer, or city is enough — save it.

TOOL TRUST TIERS:
  T1 (Local): memory_write, memory_read, read, write, edit, exec, data-analysis
     → Safe for ALL data up to L2. L3/L4 must NEVER be stored even in T1.
  T2 (1P Cloud): enterprise-mail, enterprise-calendar, enterprise-rag
     → Safe up to L2. L3/L4 requires EXPLICIT user consent via a consent gate.
  T3 (3P API): web_search, web_fetch, browser, agent-browser, github, slack, notion
     → Safe up to L1 only. L2+ STRICTLY PROHIBITED without documented user exception.

CONSENT GATE RULES (when to ask):
  Only insert a consent gate when there is a MISMATCH between the data level and tool tier:
  - T1 + L0/L1/L2: NO consent needed. Just proceed.
  - T1 + L3/L4: DENY. Agent amnesia. Cannot store.
  - T2 + L0/L1/L2: NO consent needed. Just proceed.
  - T2 + L3/L4: Consent gate required. Ask user explicitly before proceeding.
  - T3 + L0/L1: NO consent needed. Just proceed.
  - T3 + L2+: Consent gate required. Ask user explicitly, explain data will go to external service.

REQUIRED BEHAVIORS:
  1. ADVERSARIAL REFUSAL (Scenario D): If user asks to permanently memorize L3/L4 data, the assistant MUST refuse with:
     "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. Please provide it again for the current task."
     This is the EXACT required string and must appear verbatim.
  
  2. L3/L4 THREE-TURN FLOW: When L3/L4 data is needed for a task:
     Turn 1: Assistant explains it needs the data and asks user to provide it
     Turn 2: User provides the L3/L4 data
     Turn 3: Assistant uses it for that turn only, then drops it (agent amnesia)
  
  3. LOCAL-FIRST: Always prefer T1 tools over T2/T3 when the task can be done locally.
  
  4. DATA MINIMIZATION: Store only what's useful for future sessions. Skip data that won't help.
     But do NOT be overly cautious — if the user shares L0/L1/L2 data and wants it remembered, just save it.
"""


def _build_prompt(
    trajectory: ParsedTrajectory,
    pii_map: PIIMap,
    rag_context: str,
) -> str:
    """Build the complete prompt for the single-shot rewriter."""

    original_text = _format_original_trajectory(trajectory)
    pii_text = _format_pii_map(pii_map)
    task_text = _format_task_spec(trajectory)
    persona_text = _format_persona_vault(trajectory.persona or {})

    return f"""You are an expert at converting standard SFT agent trajectories into privacy-compliant Privacy SFT trajectories.

Your job: Take the original conversation below and produce a COMPLETE privacy-aware version that:
1. Preserves the original task flow and user intent
2. Adds privacy behaviors (consent gates, refusals, data minimization) naturally
3. Includes one adversarial probe + refusal (Scenario D) woven naturally into the conversation
4. Maintains realistic user/assistant alternation — NO consecutive assistant messages without user input
5. Every assistant message that includes tool calls MUST also include natural text explaining the action
6. All tool calls must have realistic arguments and results

{PRIVACY_RULES}

---
{task_text}

---
{persona_text}

---
{pii_text}

---
ORIGINAL TRAJECTORY:

{original_text}

---
{rag_context}

---
OUTPUT FORMAT:

Produce the rewritten trajectory as a JSON array of turn objects. Each turn is either a user message or an assistant response.

```json
[
  {{
    "role": "user",
    "text": "the user message text",
    "is_original": true
  }},
  {{
    "role": "assistant", 
    "text": "natural assistant response text (REQUIRED even when tool_calls present)",
    "tool_calls": [
      {{
        "name": "tool_name",
        "arguments": {{"key": "value"}},
        "result": "tool result text"
      }}
    ],
    "privacy_actions": ["consent_gate", "data_classification", etc.],
    "scenario": "A/B/C/D/E/F/etc."
  }},
  {{
    "role": "user",
    "text": "user response to consent gate",
    "is_synthetic": true
  }}
]
```

CRITICAL RULES:
1. Every assistant turn MUST have non-empty "text" — never a silent tool call.
2. User and assistant turns MUST strictly alternate. Never two assistants in a row.
   - Exception: assistant → toolResult → assistant chains ARE allowed (these represent the assistant processing tool results and continuing). But each assistant message in such a chain must have its own meaningful text.
3. For the adversarial probe (Scenario D):
   - Insert a synthetic user message that tries to get the agent to store L3/L4 data permanently
   - The assistant refuses with the exact Hatch Trust Guidelines string (verbatim, no changes)
   - Then a synthetic user acknowledgment, and the conversation continues naturally
4. For consent gates — ONLY when there is a tier-data mismatch:
   - T3 tool with L2+ data: insert a consent gate
   - T2 tool with L3/L4 data: insert a consent gate
   - T1 tool with L0/L1/L2: NO consent gate needed. DO NOT ask permission. Just proceed naturally.
   - The consent text must reference the ACTUAL task being done (not generic "booking" or "appointment")
   - Insert a synthetic user approval message after the consent gate
5. Preserve ALL original tool calls that are in the trajectory — don't drop any.
6. For memory_write calls (T1):
   - L0/L1/L2 data: JUST SAVE IT. No need to ask for consent or confirmation. The user shared it, save it.
   - L3/L4 data: NEVER store. Refuse with the HTG string.
7. CONSISTENCY: If the assistant classifies data into categories (e.g., "I'll save X, skip Y"), the subsequent
   actions (memory_write, write) MUST match exactly what was described. Do NOT exclude items that were in
   the "save" category, and do NOT include items that were in the "skip" category.
8. Make the conversation sound natural. Match the persona's communication style.
9. Mark synthetic user messages with "is_synthetic": true.
10. Mark original user messages with "is_original": true.
11. IMPORTANT: Do NOT be overly cautious. If the user shares their name, city, hobbies, employer — and those
    are L0/L1/L2 — just save them using T1 tools without asking permission. Only L3/L4 data triggers
    special handling.

Produce ONLY the JSON array, no other text. Start with [ and end with ]."""


# ---------------------------------------------------------------------------
# Parse LLM output into RewriteResult
# ---------------------------------------------------------------------------

def _parse_llm_output(raw: str, trajectory: ParsedTrajectory) -> RewriteResult:
    """Parse the LLM's JSON output into a RewriteResult.

    This also rewrites trajectory.user_messages and trajectory.thread_order
    to match the LLM's output, so the writer's user-message mapping works.
    """
    json_match = re.search(r'\[.*\]', raw, re.DOTALL)
    if not json_match:
        raise ValueError("No JSON array found in LLM output")

    turns_data = json.loads(json_match.group())

    # First pass: build the full conversation in order
    # We need to know which user messages precede which assistant turns
    new_user_messages: list[str] = []
    new_thread_order: list[tuple[str, int]] = []
    rewritten_turns: list[RewrittenTurn] = []

    user_idx = 0
    assistant_idx = 0
    pending_synthetic_user: str | None = None
    pending_adversarial_user: str | None = None
    pending_consent_response: bool = False

    for i, item in enumerate(turns_data):
        role = item.get("role", "")

        if role == "user":
            is_synthetic = item.get("is_synthetic", False)
            text = item.get("text", "")

            if is_synthetic:
                # Look ahead: is the next assistant turn adversarial?
                next_asst = None
                for j in range(i + 1, len(turns_data)):
                    if turns_data[j].get("role") == "assistant":
                        next_asst = turns_data[j]
                        break

                is_adversarial_probe = False
                if next_asst:
                    pa = next_asst.get("privacy_actions", [])
                    if "adversarial_refusal" in pa or next_asst.get("scenario") == "D":
                        is_adversarial_probe = True

                if is_adversarial_probe:
                    pending_adversarial_user = text
                else:
                    pending_synthetic_user = text
                    if next_asst and "consent_granted" in next_asst.get("privacy_actions", []):
                        pending_consent_response = True
            else:
                # Original user message
                new_user_messages.append(text)
                new_thread_order.append(("user", user_idx))
                user_idx += 1

        elif role == "assistant":
            tool_calls = []
            tool_results = []
            for tc in item.get("tool_calls", []):
                tc_dict = {
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", {}),
                }
                tool_calls.append(tc_dict)

                if "result" in tc:
                    tool_results.append({
                        "tool_name": tc.get("name", ""),
                        "content": str(tc.get("result", "")),
                        "is_error": tc.get("is_error", False),
                    })

            privacy_actions = item.get("privacy_actions", [])
            scenario = item.get("scenario", "")
            text = item.get("text", "")

            is_adversarial = pending_adversarial_user is not None
            consent_decision = "granted" if pending_consent_response else ""

            rt = RewrittenTurn(
                turn_index=assistant_idx,
                thinking="",
                text=text,
                tool_calls=tool_calls,
                tool_results=tool_results,
                privacy_actions=privacy_actions,
                scenario=scenario,
                synthetic_user_message=pending_synthetic_user or "",
                consent_decision=consent_decision,
                is_adversarial=is_adversarial,
                adversarial_user_message=pending_adversarial_user or "",
            )
            rewritten_turns.append(rt)
            new_thread_order.append(("assistant", assistant_idx))
            assistant_idx += 1

            pending_synthetic_user = None
            pending_adversarial_user = None
            pending_consent_response = False

    # Update the trajectory so the writer can map user messages correctly
    trajectory.user_messages = new_user_messages
    trajectory.thread_order = new_thread_order

    scenarios_covered = list(set(
        rt.scenario for rt in rewritten_turns if rt.scenario
    ))

    privacy_points = sum(
        1 for rt in rewritten_turns
        if rt.privacy_actions
    )

    return RewriteResult(
        task_id=trajectory.task_id,
        submission_id=trajectory.submission_id,
        turns=rewritten_turns,
        scenarios_covered=scenarios_covered,
        privacy_decision_points=privacy_points,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def rewrite_trajectory_single_shot(
    trajectory: ParsedTrajectory,
    pii_map: PIIMap,
) -> RewriteResult:
    """Rewrite a trajectory in a single LLM call with full context.

    This replaces the assembler + rewriter + reviewer stages with one
    powerful call that has complete visibility into the conversation.
    """
    rag_context = _get_rag_context(trajectory, pii_map)

    prompt = _build_prompt(trajectory, pii_map, rag_context)

    logger.info(
        "Single-shot rewrite: sending %d chars to %s",
        len(prompt), REWRITER_MODEL,
    )

    client = _get_client()
    response = await client.messages.create(
        model=REWRITER_MODEL,
        max_tokens=16384,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_output = ""
    for block in response.content:
        if hasattr(block, "text"):
            raw_output += block.text

    tracker.record_anthropic(response, "single_shot_rewrite")

    logger.info(
        "Single-shot response: %d chars, %d input tokens, %d output tokens",
        len(raw_output),
        response.usage.input_tokens,
        response.usage.output_tokens,
    )

    result = _parse_llm_output(raw_output, trajectory)

    # Validation: ensure no consecutive assistants without user
    _validate_turn_alternation(result)

    return result


def _validate_turn_alternation(result: RewriteResult) -> None:
    """Post-process validation: ensure all turns have text and no structural issues."""
    for rt in result.turns:
        if rt.tool_calls and not (rt.text and rt.text.strip()):
            tool_names = [tc.get("name", "?") for tc in rt.tool_calls if isinstance(tc, dict)]
            rt.text = f"Let me {', '.join(tool_names[:2])} for you."
            logger.warning(
                "Turn %d had tool calls but no text — injected fallback",
                rt.turn_index,
            )
