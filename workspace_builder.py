"""Build realistic workspace markdown files from persona + task context."""
from __future__ import annotations

import json
from datetime import datetime, timezone


def is_blank_user_md(content: str) -> bool:
    """True if USER.md is still the empty bootstrap template."""
    if not content or len(content.strip()) < 80:
        return True
    for line in content.splitlines():
        if line.strip() in ("- **Name:**", "**Name:**"):
            return True
    return False


def is_blank_memory_md(content: str) -> bool:
    if not content:
        return True
    return "- Name:\n" in content and "Persona Summary" in content


def is_first_intro_task(task_def: dict) -> bool:
    """Tasks where workspace_before should be empty and post-session is partial."""
    text = " ".join(
        str(task_def.get(k, ""))
        for k in ("goal_summary", "task_title", "subdomain", "domain")
    ).lower()
    markers = (
        "first time", "introduce", "introductory", "fresh session",
        "new user", "who am i", "bootstrap", "memory preferences safely",
    )
    return any(m in text for m in markers)


def _hobby_names(persona: dict, limit: int = 4) -> list[str]:
    hobbies = persona.get("hobbies", {})
    names: list[str] = []
    for tier in ("tier_1", "tier_2", "tier_3"):
        for h in hobbies.get(tier, []) or []:
            if isinstance(h, dict) and h.get("id"):
                names.append(h["id"].replace("_", " "))
    return names[:limit]


def _pronouns(persona: dict) -> str:
    gender = (persona.get("gender") or "").lower()
    if gender == "female":
        return "she/her"
    if gender == "male":
        return "he/him"
    return "they/them"


def build_user_md_profile(
    persona: dict,
    task_def: dict,
    variant: str = "full",
) -> str:
    """Build USER.md content.

    variant:
      - blank: bootstrap template (pre-session intro tasks)
      - post_intro: after first meeting with partial consent (no employer/age)
      - full: rich profile for returning sessions
    """
    if variant == "blank" or not persona:
        return (
            "# USER.md - About Your Human\n\n"
            "_Learn about the person you're helping. Update this as you go._\n\n"
            "- **Name:**\n"
            "- **What to call them:**\n"
            "- **Pronouns:** _(optional)_\n"
            "- **Timezone:**\n"
            "- **Notes:**\n\n"
            "## Context\n\n"
            "_(What do they care about? What projects are they working on? "
            "What annoys them? What makes them laugh? Build this over time.)_\n\n"
            "---\n\n"
            "The more you know, the better you can help. But remember: "
            "you're learning about a person, not building a dossier. "
            "Respect the difference.\n"
        )

    first = persona.get("first_name", "User")
    last = persona.get("last_name", "")
    full = f"{first} {last}".strip()
    city = persona.get("city", "")
    tz = persona.get("timezone", "")
    job = persona.get("job_title", "")
    sector = persona.get("occupation_sector", "").replace("_", " ")
    emp = (persona.get("pii_vault", {}) or {}).get("employment", {})
    employer = emp.get("employer", "") if isinstance(emp, dict) else ""
    age = persona.get("exact_age", "")
    lang = persona.get("primary_language", "English")
    hobbies = _hobby_names(persona)
    goal = task_def.get("goal_summary", "") if task_def else ""

    if variant == "post_intro":
        display_name = first
        notes = f"Based in {city.split(',')[0] if city else 'their city'}"
        if hobbies:
            notes += f". Interests: {', '.join(hobbies)}."
        lines = [
            f"# USER.md - {first}",
            "",
            f"**Name:** {display_name}",
            f"**What to call them:** {first}",
            f"**Pronouns:** {_pronouns(persona)}",
            f"**Timezone:** {tz or '_(not set)_'}",
            f"**Language:** {lang}",
            "",
            "---",
            "",
            "## Context",
            "",
            notes,
            "",
            "## Preferences (consented to remember)",
            "",
        ]
        if hobbies:
            lines.append(f"- Hobbies: {', '.join(hobbies)}")
        lines.append(f"- Location (metro): {city.split(',')[0] if city else 'Austin'}")
        lines.append("")
        lines.append("## Excluded from memory (user request)")
        lines.append("")
        lines.append("- Last name, age, employer, and job title are **not** stored.")
        lines.append("")
        if goal:
            lines.extend(["## Session note", "", goal[:200], ""])
        lines.extend([
            "---",
            "",
            "_Profile updated after introductory session with explicit consent._",
            "",
        ])
        return "\n".join(lines)

    # full profile
    lines = [
        f"# USER.md - {full}",
        "",
        f"**Name:** {full}",
        f"**What to call them:** {first}",
        f"**Pronouns:** {_pronouns(persona)}",
        f"**Timezone:** {tz or '_(not set)_'}",
    ]
    if age:
        lines.append(f"**Age:** {age}")
    lines.extend(["", "---", "", "## Work", ""])
    if job:
        lines.append(f"- **Role:** {job}")
    if employer:
        lines.append(f"- **Employer:** {employer}")
    if sector:
        lines.append(f"- **Sector:** {sector}")
    if city:
        lines.append(f"- **Based:** {city}")
    lines.extend(["", "## Context", ""])
    if goal:
        lines.append(goal[:300])
    elif sector or job:
        lines.append(f"Working as {job or 'a professional'} in {sector or 'their field'}.")
    if hobbies:
        lines.extend(["", "## Interests", ""])
        for h in hobbies:
            lines.append(f"- {h.title()}")
    lines.extend([
        "",
        "## Communication",
        "",
        "- Prefers direct, practical help",
        "- Privacy-conscious about identifying details",
        "",
        "---",
        "",
        f"_Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}_",
        "",
    ])
    return "\n".join(lines)


def build_memory_md_profile(
    persona: dict,
    task_def: dict,
    variant: str = "full",
) -> str:
    """Build MEMORY.md — blank before intro, populated after."""
    if variant in ("blank",) or not persona:
        return (
            "# MEMORY\n\n"
            "## Persona Summary\n\n"
            "- Name:\n- Occupation:\n- Location:\n- Short bio:\n\n"
            "## Stable Preferences\n\n"
            "- Communication style:\n- Travel preferences:\n"
            "- Work habits:\n- Shopping preferences:\n\n"
            "## Ongoing Threads\n\n"
            "- Current projects:\n- Follow-ups to remember:\n\n"
            "## Recent Updates\n\n"
            "- Add new facts here after each completed task.\n"
        )

    first = persona.get("first_name", "User")
    city = persona.get("city", "").split(",")[0] if persona.get("city") else ""
    job = persona.get("job_title", "")
    hobbies = _hobby_names(persona)
    goal = (task_def or {}).get("goal_summary", "")[:150]

    lines = [
        "# MEMORY.md - Long-Term Project Memory",
        "",
        "## Active User Context",
        "",
        f"- User: {first}",
    ]
    if city:
        lines.append(f"- Metro area: {city}")
    if hobbies:
        lines.append(f"- Interests: {', '.join(hobbies)}")
    if variant == "post_intro":
        lines.append("- Retention: partial consent — no employer, age, or last name on file")
    elif job:
        lines.append(f"- Role: {job}")
    lines.extend([
        "",
        "## Recent Updates",
        "",
    ])
    if goal:
        lines.append(f"- {goal}")
    else:
        lines.append("- Profile preferences recorded this session.")
    lines.append("")
    return "\n".join(lines)


def apply_session_writes(
    files: dict[str, str],
    rewrite_result,
) -> dict[str, str]:
    """Merge write/memory_write tool payloads into workspace files."""
    result = dict(files)
    for rt in rewrite_result.turns:
        for tc in rt.tool_calls:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name", "")
            args = tc.get("arguments", {}) or {}
            if name == "write":
                raw_path = args.get("file_path", args.get("path", ""))
                content = args.get("content", "")
                if not raw_path or not content or len(content.strip()) < 20:
                    continue
                clean = raw_path
                for prefix in (
                    "/home/user/OpenClawTrainer/workspace/",
                    "/home/user/.openclaw/workspace/",
                    "/workspace/",
                ):
                    if clean.startswith(prefix):
                        clean = clean[len(prefix):]
                clean = clean.lstrip("/")
                if clean and not clean.startswith(".."):
                    base = clean.split("/")[-1]
                    if base in (
                        "USER.md", "MEMORY.md", "IDENTITY.md",
                    ) and is_blank_user_md(content):
                        continue
                    result[clean] = content
            elif name in ("memory_write", "active_memory_write"):
                key = args.get("key", "")
                value = args.get("value", "")
                if not key or not value:
                    continue
                if isinstance(value, dict):
                    value = json.dumps(value, indent=2)
                block = f"\n\n### {key}\n{value}\n"
                mem = result.get("MEMORY.md", result.get("memory.md", ""))
                if "MEMORY" not in mem.upper():
                    mem = build_memory_md_profile({}, {}, "blank")
                result["MEMORY.md"] = mem.rstrip() + block
    return result


def enrich_workspace_files(
    files: dict[str, str],
    persona: dict,
    task_def: dict,
    phase: str,
) -> dict[str, str]:
    """Fill blank core markdown files from persona when missing."""
    if not persona:
        return files
    result = dict(files)
    intro = is_first_intro_task(task_def or {})
    if phase == "before":
        user_var = "blank" if intro else "full"
        mem_var = "blank" if intro else "full"
    else:
        user_var = "post_intro" if intro else "full"
        mem_var = "post_intro" if intro else "full"

    if is_blank_user_md(result.get("USER.md", "")):
        result["USER.md"] = build_user_md_profile(persona, task_def or {}, user_var)
    if is_blank_memory_md(result.get("MEMORY.md", "")):
        result["MEMORY.md"] = build_memory_md_profile(persona, task_def or {}, mem_var)
    return result
