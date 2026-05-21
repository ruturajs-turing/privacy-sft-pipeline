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

    # Persona context (with personality traits for realistic consent responses)
    persona_section = ""
    if trajectory.persona:
        p = trajectory.persona
        personality = p.get('personality', {})
        persona_section = f"""PERSONA CONTEXT:
- Name: {p.get('first_name', p.get('name', 'Unknown'))} {p.get('last_name', '')}
- Persona ID: {p.get('persona_id', '')}
- Profession: {p.get('job_title', p.get('profession', ''))}
- Primary Language: {p.get('primary_language', 'English')}
- Country: {p.get('city', p.get('country', ''))}
- Communication Style: {"formal" if personality.get('conscientiousness', 0.5) > 0.7 else "casual"}
- Openness (data sharing comfort): {personality.get('openness', 0.5):.1f}/1.0 {"(cautious — more likely to deny consent)" if personality.get('openness', 0.5) < 0.4 else "(open — more likely to grant)" if personality.get('openness', 0.5) > 0.6 else "(moderate)"}
- Conscientiousness: {personality.get('conscientiousness', 0.5):.1f}/1.0 {"(thorough — prefers explicit confirmations)" if personality.get('conscientiousness', 0.5) > 0.7 else "(relaxed — terse responses)"}

Use these traits when generating user_response in consent flows. The user's replies should match their personality — a cautious/low-openness persona denies more often; a casual persona gives short responses."""

    # Task spec
    task_section = ""
    if trajectory.task_spec:
        ts = trajectory.task_spec
        task_section = f"""TASK SPEC:
- Task ID: {ts.get('task_id', '')}
- Title: {ts.get('title', '')}
- Privacy Scenario: {ts.get('privacy_scenario', '')}
- Skills: {', '.join(ts.get('skills', []))}"""

    # Workspace context — full content of key files for grounded rewriting
    workspace_section = _build_workspace_context(trajectory.workspace_files)

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
8. If the original has ONLY a `read` or `memory_search` call → no consent gate needed, just classify and proceed
9. If CONSENT ALREADY OBTAINED is provided below for this path → use Format A (normal), include the tool call, apply data minimization to write content (strip names, emails not explicitly requested by user)
10. When writing to memory/persistence, MINIMIZE content: use role labels ("the user") instead of names, strip emails/phones, only keep task-relevant info the user asked to remember

Return ONLY the JSON object as specified in the system prompt."""

    return prompt


def _build_workspace_context(workspace_files: dict[str, str]) -> str:
    """Build rich workspace context from extracted workspace files.

    Includes full content of TOOLS.md, AGENTS.md, USER.md (capped at 2000 chars each),
    and a summary of installed skills from skills/*/SKILL.md entries.
    """
    if not workspace_files:
        return ""

    sections = []
    MAX_FILE_CHARS = 2000

    # Key files to include in full
    key_files = {
        "USER.md": "USER PERSONA (from USER.md)",
        "AGENTS.md": "AGENT IDENTITY (from AGENTS.md)",
        "TOOLS.md": "TOOL NOTES (from TOOLS.md)",
    }

    for filename, header in key_files.items():
        content = workspace_files.get(filename, "")
        if not content:
            # Try case-insensitive match
            for k, v in workspace_files.items():
                if k.lower() == filename.lower():
                    content = v
                    break
        if content:
            truncated = content[:MAX_FILE_CHARS]
            if len(content) > MAX_FILE_CHARS:
                truncated += "\n... (truncated)"
            sections.append(f"### {header}\n{truncated}")

    # Installed skills — extract first line of each SKILL.md as description
    skill_entries = []
    for path, content in workspace_files.items():
        if "/skills/" in f"/{path}" and path.endswith("SKILL.md"):
            parts = path.split("/")
            # Extract skill name from path like "skills/caldav-calendar/SKILL.md"
            skill_name = None
            for i, p in enumerate(parts):
                if p == "skills" and i + 1 < len(parts):
                    skill_name = parts[i + 1]
                    break
            if skill_name:
                first_line = content.strip().split("\n")[0].strip().lstrip("#").strip()
                skill_entries.append(f"  - {skill_name}: {first_line[:100]}")

    if skill_entries:
        sections.append("### INSTALLED SKILLS\n" + "\n".join(skill_entries))

    # List remaining workspace files (non-key, non-skill) for reference
    other_files = [
        k for k in workspace_files.keys()
        if k not in key_files and not (k.endswith("SKILL.md") and "/skills/" in f"/{k}")
    ]
    if other_files:
        shown = other_files[:15]
        sections.append(f"### OTHER WORKSPACE FILES ({len(other_files)} total)\n" +
                        ", ".join(shown) +
                        ("..." if len(other_files) > 15 else ""))

    if not sections:
        return ""

    return "WORKSPACE CONTEXT:\n" + "\n\n".join(sections)


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
