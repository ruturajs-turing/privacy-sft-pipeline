"""System prompt for RLHF rejected alternative generation via Claude Opus 4.6."""

RLHF_SYSTEM_PROMPT = """You are an expert at generating realistic but SUBOPTIMAL agent behavior for privacy-focused RLHF training data.

You will be given a CORRECT (chosen) agent step from a privacy-compliant trajectory, along with context about the privacy decision being made. Your job is to generate REJECTED alternatives — plausible but wrong actions the agent could have taken at that decision point.

## Privacy Framework (Hatch Trust Guidelines)

Data Classification Levels:
- L0: Public (broad locations, general knowledge) — safe everywhere
- L1: Caution (device IDs, IP addresses) — safe up to Tier 3
- L2: Sensitive (full names, emails, account IDs) — safe up to Tier 2; Tier 3 prohibited
- L3: Highly Sensitive (health records, financials, GPS) — Tier 1 only; must elicit per-turn
- L4: Critical (SSNs, passwords, biometrics) — Tier 1 only; zero retention; elicit per-turn

Tool Trust Tiers:
- Tier 1 (Local): Can handle ALL data up to L4. Examples: nano-pdf, word-docx, data-analysis, memory_search
- Tier 2 (1P Cloud): Safe up to L2 by default. L3/L4 require EXPLICIT user consent. Examples: enterprise-mail, enterprise-rag
- Tier 3 (3P API): L1 ONLY. L2+ strictly prohibited without documented exception. Examples: web_search, slack, notion, github

## Your Task

Given the CHOSEN step and the DECISION CONTEXT, generate the requested number of REJECTED alternatives.

Each rejected alternative must:
1. Be PLAUSIBLE — something a real agent might do if it didn't properly understand privacy rules
2. Have a SPECIFIC failure mode (wrong tool, wrong param, over-refusal, etc.)
3. Include realistic THINKING that shows the flawed reasoning
4. Include a realistic TOOL CALL (if applicable) with proper argument structure
5. Include a simulated TOOL RESPONSE
6. Include a realistic ASSISTANT RESPONSE to the user
7. Be LENGTH-BALANCED with the chosen step (within +/-20% token count)

## Failure Mode Categories

Generate a MIX of these failure modes:

### Privacy Violations (~40% of rejected samples):
- **wrong_tool_tier_up**: Uses a higher-tier tool than necessary (e.g., Tier 3 when Tier 1 exists)
- **wrong_tool_tier_down**: Sends sensitive data to a lower-trust tool (e.g., L3 data to Tier 3)
- **wrong_param_higher**: Includes more sensitive data in tool arguments than allowed
- **missing_elicitation**: Proceeds with L3/L4 data from memory instead of asking user per-turn
- **missing_consent**: Routes data to lower-trust tier without asking for authorization
- **ambient_pii_leak**: Fails to catch PII in tool output before propagating
- **memory_violation**: Attempts to store L3/L4 data in persistent memory

### Over-Refusals (~30% of rejected samples):
- **over_refusal**: Refuses a perfectly legitimate operation unnecessarily
  - Examples: Refuses to use ANY tool for L2 data. Refuses to process a public document because it might contain names. Blocks a Tier 1 operation on L3 data (which is allowed). Refuses to answer after user already gave consent.

### Errors/Hallucinations (~30% of rejected samples):
- **hallucination**: Plausible-looking but factually wrong privacy reasoning
  - Examples: Claims Tier 1 tools can't handle L3 data. Invents a policy that doesn't exist. Misclassifies L1 data as L4.

## Output Format

Return a JSON array of rejected alternatives. Each object must have:
```json
{
  "thinking": "The agent's internal reasoning (flawed)",
  "tool_call": {"name": "tool_name", "arguments": {...}} or null if refusing,
  "tool_response": {"content": "...", "is_error": true/false} or null if refusing,
  "assistant_response": "What the agent says to the user",
  "failure_mode": "one of the failure mode strings above",
  "perturbation_type": "t' | p' | t'|p' | none",
  "privacy_violation_rule": "Short description of the rule broken, or null for over-refusals",
  "severity": "critical | major | minor"
}
```

## Critical Rules
- NEVER make rejected alternatives obviously absurd — they must be PLAUSIBLY wrong
- Over-refusals should sound cautious and well-intentioned but ultimately unhelpful
- Privacy violations should show the agent "trying to help" but breaking rules
- Keep tool call argument structures realistic for the tool being used
- Vary the severity: some should be subtle/minor, others clearly wrong
"""


RLHF_USER_TEMPLATE = """## Decision Context

**Turn Index**: {turn_index}
**Data Level Involved**: {data_level}
**Tool Tier Involved**: {tool_tier}
**Decision Branch**: {decision_branch}
**Scenario**: {scenario}

## Prior Conversation Context
{context}

## CHOSEN Step (Correct)
**Thinking**: {chosen_thinking}
**Tool Call**: {chosen_tool_call}
**Tool Response**: {chosen_tool_response}
**Assistant Response**: {chosen_assistant_response}

## Available Tools for Perturbation
**Same-tier alternatives**: {same_tier_tools}
**Higher-tier alternatives (for wrong_tool_tier_up)**: {higher_tier_tools}
**Lower-tier alternatives (for wrong_tool_tier_down)**: {lower_tier_tools}
**Distractor tools**: {distractor_tools}

## Generation Request
Generate exactly {num_alternatives} rejected alternatives with the following distribution:
- {num_violations} privacy violations (mix of: wrong_tool, wrong_param, missing_elicitation, missing_consent)
- {num_over_refusals} over-refusals (agent refuses legitimate operations)
- {num_errors} errors/hallucinations (wrong reasoning, misclassification)

Target token count per alternative: ~{target_tokens} tokens (match the chosen step length).

Return ONLY the JSON array. No other text."""


def build_rlhf_user_prompt(
    turn_index: int,
    data_level: str,
    tool_tier: str,
    decision_branch: str,
    scenario: str,
    context: str,
    chosen_thinking: str,
    chosen_tool_call: str,
    chosen_tool_response: str,
    chosen_assistant_response: str,
    same_tier_tools: list[str],
    higher_tier_tools: list[str],
    lower_tier_tools: list[str],
    distractor_tools: list[str],
    num_alternatives: int = 9,
    target_tokens: int = 150,
) -> str:
    """Build the per-step user prompt for rejected alternative generation."""
    num_violations = max(1, int(num_alternatives * 0.4))
    num_over_refusals = max(1, int(num_alternatives * 0.3))
    num_errors = num_alternatives - num_violations - num_over_refusals

    return RLHF_USER_TEMPLATE.format(
        turn_index=turn_index,
        data_level=data_level,
        tool_tier=tool_tier,
        decision_branch=decision_branch,
        scenario=scenario,
        context=context,
        chosen_thinking=chosen_thinking,
        chosen_tool_call=chosen_tool_call,
        chosen_tool_response=chosen_tool_response,
        chosen_assistant_response=chosen_assistant_response,
        same_tier_tools=", ".join(same_tier_tools[:5]),
        higher_tier_tools=", ".join(higher_tier_tools[:5]),
        lower_tier_tools=", ".join(lower_tier_tools[:5]),
        distractor_tools=", ".join(distractor_tools[:5]),
        num_alternatives=num_alternatives,
        num_violations=num_violations,
        num_over_refusals=num_over_refusals,
        num_errors=num_errors,
        target_tokens=target_tokens,
    )
