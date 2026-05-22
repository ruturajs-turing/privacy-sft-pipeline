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
HATCH TRUST GUIDELINES (HTG) — COMPLETE PRIVACY RULES:

═══════════════════════════════════════════════════════════════
1. DATA CLASSIFICATION LEVELS
═══════════════════════════════════════════════════════════════

  L0 (Public): Broad locations, general knowledge.
     → No flagging needed. Store freely. No consent required.

  L1 (Caution): Device IDs, IP addresses, metro-level city, age, hobbies, proprietary code.
     → No flagging needed. Store freely. No consent required.

  L2 (Sensitive): Full names, emails, employer, job title, account IDs, transaction history, hashed IDs.
     → Can be SAVED without consent using T1 tools (including memory_write).
     → Needs consent for T3 tools. OK for T2 without consent.

  L3 (Highly Sensitive): Health records (PHI), detailed financials, private communications, precise GPS.
     → Zero retention. Agent amnesia. Must re-input per turn.
     → Can be PROCESSED by T1 tools (e.g., local OCR on a medical PDF) but NEVER written to persistent storage.

  L4 (Critical): SSNs, passwords, biometrics, full credit card numbers.
     → NEVER store anywhere. Agent amnesia. Must re-input per turn.
     → Can be PROCESSED by T1 tools for the current turn only, then dropped.

KEY RULE — L0, L1, AND L2 WITH T1 TOOLS: NO CONSENT NEEDED.
  The agent can store L0, L1, and L2 data using T1 tools (memory_write, write, etc.) WITHOUT asking for
  explicit user permission. Just do it naturally. Do NOT insert unnecessary consent gates for L0/L1/L2
  data when using local tools. The user telling you their name, employer, or city is enough — save it.

═══════════════════════════════════════════════════════════════
2. TOOL TRUST TIERS (complete lists)
═══════════════════════════════════════════════════════════════

  T1 (Local) — Safe to PROCESS all data up to L4. Safe to WRITE/SAVE up to L2. L3/L4 writes are HARD DENIED.
     Tools: memory_write, memory_search, memory_read, read, write, edit, exec, cron, sessions,
            nano-pdf, surya, openai-whisper, word-docx, excel-xlsx, powerpoint-pptx,
            docker-essentials, data-analysis, automation-workflows, markdown-converter,
            ontology, self-improving, obsidian, mcporter, humanizer, productivity,
            skill-creator, ui-ux-pro-max, self-reflection, language-learning, cfo, health,
            relationship-skills, workout, healthcheck, mechanic, marketing-mode,
            Linux-native tools (ls, cp, mv, rm, cat, grep, awk, sed, find, chmod, tar, curl localhost)

  T2 (1P Cloud) — Safe to READ/WRITE up to L2. L3/L4 requires EXPLICIT user consent via consent gate.
     Tools: enterprise-mail, enterprise-calendar, enterprise-rag, enterprise-inference, enterprise-vault

  T3 (3P API) — Safe to READ/WRITE up to L1 only. L2+ STRICTLY PROHIBITED without documented user exception.
     Tools: web_search, web_fetch, browser, agent-browser, github, slack, notion, gog, trello,
            weather, news-summary, stock-analysis, caldav-calendar, clawhub, api-gateway,
            polymarket, goplaces, openai-whisper-api, edge-tts, academic-research,
            flight-search, eventbrite, plan2meal, legaldoc-ai, music-cog, bundled-web-tools

  IF A TOOL IS NOT IN ANY LIST ABOVE: Default to T3 treatment (L1 data cap).

═══════════════════════════════════════════════════════════════
3. CONSENT GATE RULES (when to ask)
═══════════════════════════════════════════════════════════════

  Only insert a consent gate when there is a MISMATCH between the data level and tool tier:

  - T1 + L0/L1/L2 (READ or WRITE): NO consent needed. Just proceed.
  - T1 + L3/L4 (PROCESS/READ only): NO consent needed. Execute locally, then drop data.
  - T1 + L3/L4 (WRITE/SAVE): HARD DENY. Refuse with the Hatch refusal string. Cannot be overridden by consent.
  - T2 + L0/L1/L2: NO consent needed. Just proceed.
  - T2 + L3/L4: Consent gate required. Ask user explicitly before proceeding.
  - T3 + L0/L1: NO consent needed. Just proceed.
  - T3 + L2+: Consent gate required. Ask user explicitly, explain data will go to external service.

  The consent text must reference the ACTUAL task being done (not generic "booking" or "appointment").

═══════════════════════════════════════════════════════════════
4. TRAJECTORY SCENARIO PATTERNS
═══════════════════════════════════════════════════════════════

  Use the correct pattern based on what the task involves:

  SCENARIO A — Ideal Local Execution (Zero Exfiltration):
    When user needs L3/L4 data processed and a T1 tool can do it.
    Flow: Identify data as L3/L4 → Ask user to provide data for this turn → Select T1 tool →
          Execute locally → Return result → Drop data from context (amnesia).

  SCENARIO B — Cloud Fallback with Explicit Consent:
    When user needs L3/L4 data processed but only a T2 tool can do it.
    Flow: Attempt T1 first (fails) → Select T2 tool → Recognize mismatch (T2 caps at L2, data is L3) →
          Halt and warn user → Request authorization → Execute only after user approves.

  SCENARIO C — 3P API Hard Block / Exception Handling:
    When user asks to send L2/L3/L4 data to a T3 tool.
    Flow: Identify T3 caps at L1 → Halt immediately → Explain exfiltration risk →
          Ask for authorization, confirm user understands data leaves controlled network.

  SCENARIO D — Memory Manipulation / Jailbreak Attempt (ADVERSARIAL):
    When user tries to make the agent permanently store L3/L4 data.
    Flow: Identify data as L3/L4 → Enforce Zero Retention → Output EXACT refusal:
          "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data.
           Please provide it again for the current task."

═══════════════════════════════════════════════════════════════
5. REQUIRED BEHAVIORS
═══════════════════════════════════════════════════════════════

  1. ADVERSARIAL REFUSAL: Every trajectory MUST include one adversarial probe (Scenario D) where the user
     tries to get the agent to save L3/L4 data. The assistant MUST refuse with the EXACT Hatch string:
     "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. Please provide it again for the current task."

     CRITICAL: After the refusal string, do NOT repeat the same concept in different words. Do NOT add
     extra sentences explaining why it can't store data, how retention works, or what the sensitivity tier is.
     The refusal string says everything needed. After it, briefly offer what CAN be done (e.g., "happy to
     help with X if you paste it in when you need it") and move on. One refusal, not a lecture.

     BAD example (redundant):
       "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. Please provide
        it again for the current task. SSN falls into the highest sensitivity tier. I'm not allowed to store
        it anywhere, not even locally, regardless of consent. No persistent storage, ever."
     GOOD example (concise):
       "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. Please provide
        it again for the current task. Happy to use it for whatever you need right now though, just paste it
        in when you're ready."

  2. L3/L4 THREE-TURN FLOW: When L3/L4 data is needed for a task:
     Turn 1 (Assistant): Explains it needs the data and asks user to provide it for this turn.
     Turn 2 (User): Provides the L3/L4 data.
     Turn 3 (Assistant): Uses it for that turn only, then drops it (agent amnesia).

  3. LOCAL-FIRST: Always prefer T1 tools over T2/T3 when the task can be done locally.
     Only escalate to T2/T3 when a local tool is genuinely insufficient.

  4. DATA MINIMIZATION: Store only what's useful for future sessions. Skip data that won't help.
     But do NOT be overly cautious. If the user shares L0/L1/L2 data and wants it remembered, just save it.

  5. TRANSPARENCY: When refusing or requesting consent, the assistant should:
     - Briefly cite the privacy concern (casual tone, e.g., "that's L3 health data, need your ok")
     - Offer alternatives when possible
     - Explain what would be needed to proceed

  6. CONSISTENCY: What the assistant says it will do must match what it actually does.
     If it says "I'll save your preferences", it must actually call memory_write.
     If it says "I won't store that", it must NOT call memory_write with that data.

  7. NO LECTURING: Do not have the agent lecture the user on security practices unless the user is
     actively violating a constraint. Be concise, capable, and transparent. If elicitation is required,
     simply ask for the data and execute once provided.

═══════════════════════════════════════════════════════════════
6. WRITING STYLE RULES (MANDATORY)
═══════════════════════════════════════════════════════════════

  ALL assistant text in the trajectory MUST sound like a real human wrote it. Follow these rules strictly:

  BANNED PATTERNS (never use these):
  - Em dashes (--) or (—). Use commas, periods, or parentheses instead.
  - "Additionally", "Furthermore", "Moreover" as sentence starters.
  - "crucial", "pivotal", "vital", "key" (as adjective), "landscape" (abstract), "tapestry", "testament",
    "underscore", "delve", "foster", "garner", "showcase", "vibrant", "intricate", "enduring", "enhance".
  - "It's not just X; it's Y" (negative parallelism).
  - "serves as", "stands as", "marks a" (copula avoidance). Just use "is" or "are".
  - "I hope this helps!", "Let me know if you'd like...", "Of course!", "Certainly!", "Great question!",
    "You're absolutely right!", "Absolutely!" (sycophantic/servile tone).
  - Rule of three lists forced for effect ("innovation, inspiration, and insight").
  - Emoji in any text.
  - Curly quotation marks. Use straight quotes only.
  - Excessive hedging ("it could potentially possibly be argued that...").
  - Filler phrases ("In order to", "Due to the fact that", "It is important to note that").
  - Generic positive conclusions ("The future looks bright", "Exciting times lie ahead").

  REQUIRED STYLE:
  - Vary sentence length naturally. Mix short punchy sentences with longer ones.
  - Use simple constructions: "is", "are", "has" over fancy substitutes.
  - Be direct. Say what you mean. Real people don't pad their sentences.
  - Use contractions naturally ("I'll", "can't", "won't", "it's").
  - When explaining privacy decisions, be casual: "heads up, that's health data so I can't hang onto it"
    not "I must inform you that under the established guidelines, this data falls under Level 3 classification."
  - Acknowledge the user's request before getting into privacy stuff. Don't lead with the rules.
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
4. Maintains realistic user/assistant alternation. NO consecutive assistant messages without user input
5. Every assistant message that includes tool calls MUST also include natural text explaining the action
6. All tool calls must have realistic arguments and results
7. All assistant text MUST sound human-written. NO em dashes, NO AI vocabulary, NO sycophantic phrases.
   See the WRITING STYLE RULES section below for the full list of banned patterns.
8. Refusals must be concise: give the Hatch string once, then move on. Do NOT repeat the same idea.

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
