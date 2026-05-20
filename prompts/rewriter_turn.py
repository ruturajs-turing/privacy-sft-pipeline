"""Per-turn user prompt template for the Claude rewriter."""
from __future__ import annotations

import json
from models import AssistantTurn, ParsedTrajectory, PIIEntity, PIIMap


def build_turn_prompt(
    trajectory: ParsedTrajectory,
    turn: AssistantTurn,
    turn_context: str,
    pii_map: PIIMap,
    scenario_hint: str,
) -> str:
    """Build the user prompt for rewriting a single assistant turn."""

    # Gather the user message preceding this turn
    preceding_user = ""
    for order_type, order_idx in trajectory.thread_order:
        if order_type == "assistant" and order_idx == turn.turn_index:
            break
        if order_type == "user":
            preceding_user = trajectory.user_messages[order_idx]

    # Original assistant content
    original_text = "\n".join(turn.text_blocks) if turn.text_blocks else "(no text)"
    original_thinking = "\n".join(turn.thinking_blocks) if turn.thinking_blocks else "(no thinking)"

    # Original tool calls with full details
    original_tools_list = []
    for tc in turn.tool_calls:
        original_tools_list.append({"id": tc.call_id, "name": tc.name, "arguments": tc.arguments})
    original_tools = json.dumps(original_tools_list, indent=2) if original_tools_list else "[]"

    # Original tool results
    original_results_list = []
    for tc in turn.tool_calls:
        if tc.call_id in trajectory.tool_results_by_call_id:
            tr = trajectory.tool_results_by_call_id[tc.call_id]
            original_results_list.append({
                "call_id": tr.call_id,
                "tool_name": tr.tool_name or tc.name,
                "content": tr.content[:2000],  # Truncate very long results
                "is_error": tr.is_error,
            })
    original_results = json.dumps(original_results_list, indent=2) if original_results_list else "[]"

    # Detect workspace path from tool calls
    workspace_path = _detect_workspace_path(turn, trajectory)

    # PII entities relevant to this turn
    pii_section = ""
    if pii_map.entities:
        pii_items = []
        for e in pii_map.entities[:30]:
            pii_items.append(f"  - \"{e.text}\" → {e.label} ({e.level})")
        pii_section = "PII DETECTED IN THIS TRAJECTORY:\n" + "\n".join(pii_items)
    else:
        pii_section = "PII DETECTED: None (L0 content only)"

    # Persona context
    persona_section = ""
    if trajectory.persona:
        p = trajectory.persona
        persona_section = f"""PERSONA CONTEXT:
- Name: {p.get('name', 'Unknown')}
- Persona ID: {p.get('persona_id', '')}
- Profession: {p.get('profession', '')}
- Primary Language: {p.get('primary_language', 'English')}
- Country: {p.get('country', '')}"""

    # Task spec
    task_section = ""
    if trajectory.task_spec:
        ts = trajectory.task_spec
        task_section = f"""TASK SPEC:
- Task ID: {ts.get('task_id', '')}
- Title: {ts.get('title', '')}
- Privacy Scenario: {ts.get('privacy_scenario', '')}
- Skills: {', '.join(ts.get('skills', []))}"""

    # Workspace context
    workspace_section = ""
    if trajectory.workspace_files:
        file_list = list(trajectory.workspace_files.keys())[:10]
        workspace_section = f"WORKSPACE FILES: {', '.join(file_list)}"

    prompt = f"""## TURN {turn.turn_index + 1} REWRITE REQUEST

{pii_section}

{persona_section}

{task_section}

{workspace_section}

WORKSPACE PATH: {workspace_path}
(Use this EXACT path prefix for ALL file operations — do NOT change it)

SCENARIO TO DEMONSTRATE: {scenario_hint}

---

PRECEDING USER MESSAGE:
{preceding_user}

---

ORIGINAL ASSISTANT TURN (preserve this structure):
Thinking: {original_thinking}
Text: {original_text}
Tool Calls: {original_tools}
Tool Results: {original_results}

---

{turn_context}

---

INSTRUCTIONS:
Enhance this assistant turn with privacy compliance. PRESERVE the original structure:
1. Keep the SAME tool calls with the SAME paths and arguments (use workspace path: {workspace_path})
2. Keep the SAME tool results (copy them exactly from "Tool Results" above into your output)
3. Add a thinking block showing specific PII classification for data in THIS turn
4. Keep the original text meaning — only add minimal natural privacy phrasing where appropriate
5. If the original has a T3 tool call, swap to T1 equivalent
6. If the original has memory_write with L3+ data, replace with the Hatch refusal
7. If the original deleted a file (exec rm), your rewrite MUST also delete it

Return ONLY the JSON object as specified in the system prompt."""

    return prompt


def _detect_workspace_path(turn: AssistantTurn, trajectory: ParsedTrajectory) -> str:
    """Detect the workspace path used in the original trajectory."""
    # Check tool calls in this turn
    for tc in turn.tool_calls:
        args = tc.arguments
        for key in ("path", "file_path", "command"):
            val = args.get(key, "")
            if isinstance(val, str):
                if "/home/user/.openclaw/workspace" in val:
                    return "/home/user/.openclaw/workspace"
                if "/home/user/OpenClawTrainer/workspace" in val:
                    return "/home/user/OpenClawTrainer/workspace"

    # Check all turns in trajectory
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
