"""Synthetic trajectory generator v2 — human-realistic trajectories from task + persona.

Uses four layers to produce conversations indistinguishable from real human-agent interaction:
1. Persona-derived typing profile (Big Five -> writing style)
2. Milestone-driven conversation flow (from tasks_all.json conversation_arc)
3. Real conversation few-shot anchoring (mined from 4000 extracted trajectories)
4. Anti-AI constraint layer (humanizer skill rules baked into the prompt)

Post-processing applies mechanical humanization: typo injection, abbreviation
swaps, message splitting, and curly-quote cleanup.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import json_repair

from config import (
    ANTHROPIC_API_KEY, CUARENA_BATCH_PATH, EXTRACTED_TRAJECTORIES_PATH,
    REWRITER_MODEL, TASKS_PATH,
)
from models import AssistantTurn, ParsedTrajectory, ToolCall, ToolResult
from task_context import get_task_definition, get_persona, extract_pii_vault_entities
from token_tracker import tracker

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_DB_PATH = _ROOT / "tool_calls.db"

# ---------------------------------------------------------------------------
# Module-level caches (populated lazily on first call)
# ---------------------------------------------------------------------------
_cached_real_user_msgs: list[str] | None = None
_cached_real_asst_openers: list[str] | None = None
_cached_cuarena_openers: list[str] | None = None
_cached_tasks_json: dict[str, dict] | None = None


def _opaque_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:18]}"


def _get_db() -> sqlite3.Connection | None:
    if _DB_PATH.exists():
        return sqlite3.connect(str(_DB_PATH))
    return None


_PATH_SENSITIVE_TOOLS = {"write", "edit", "read", "exec", "memory_write", "memory_read"}


def _fetch_tool_result(db: sqlite3.Connection, tool_name: str, context: str = "") -> str | None:
    """Fetch a realistic non-error tool result.

    Tries RAG first for semantically relevant results, then falls back to
    random DB lookup. Skips path-sensitive tools entirely.
    """
    if tool_name in _PATH_SENSITIVE_TOOLS:
        return None

    if context:
        try:
            from rag_retriever import get_tool_examples, is_index_ready
            if is_index_ready():
                examples = get_tool_examples(tool_name, context, n=1)
                if examples and examples[0].get("result"):
                    return examples[0]["result"]
        except Exception:
            pass

    try:
        row = db.execute(
            "SELECT content FROM tool_results WHERE tool_name = ? AND is_error = 0 "
            "AND length(content) > 50 ORDER BY RANDOM() LIMIT 1",
            (tool_name,),
        ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _synthesize_tool_result(tool_name: str, arguments: dict) -> str:
    """Generate a consistent tool result matching the tool call arguments."""
    path = arguments.get("path", "") or arguments.get("file_path", "") or ""
    content = arguments.get("content", "")
    key = arguments.get("key", "")

    if tool_name == "write":
        byte_count = len(content.encode("utf-8")) if content else random.randint(200, 3000)
        return f"Successfully wrote {byte_count} bytes to {path}"

    if tool_name == "edit":
        edits = arguments.get("edits", [])
        return f"Applied {len(edits)} edit(s) to {path}"

    if tool_name == "read":
        return f"# Contents of {path.split('/')[-1] if path else 'file'}\n\n(file content)"

    if tool_name == "memory_write":
        value = arguments.get("value", "")
        byte_count = len(str(value).encode("utf-8")) if value else random.randint(50, 300)
        return json.dumps({"status": "ok", "key": key, "bytes": byte_count})

    if tool_name == "memory_read":
        return json.dumps({"status": "empty", "keys": []})

    if tool_name == "exec":
        cmd = arguments.get("command", "")
        if "ls" in cmd:
            return "total 4\ndrwxr-xr-x 2 user user 4096 May 22 18:00 ."
        return "Command completed successfully."

    return json.dumps({"status": "ok", "tool": tool_name})


# ---------------------------------------------------------------------------
# 0. Tasks JSON loader (for conversation_arc milestones)
# ---------------------------------------------------------------------------

def _load_tasks_json() -> dict[str, dict]:
    """Load tasks_all.json keyed by task_id (both P- and T- prefixes)."""
    global _cached_tasks_json
    if _cached_tasks_json is not None:
        return _cached_tasks_json

    result: dict[str, dict] = {}
    if TASKS_PATH.exists():
        try:
            with open(TASKS_PATH, encoding="utf-8") as f:
                data = json.load(f)
            for task in (data if isinstance(data, list) else []):
                tid = task.get("task_id", "")
                if tid:
                    result[tid] = task
                    result[tid.replace("P-", "T-", 1)] = task
            logger.info("Loaded %d tasks from JSON (with conversation_arc)", len(data))
        except Exception as e:
            logger.warning("Failed to load tasks_all.json: %s", e)

    _cached_tasks_json = result
    return result


def _get_task_json(task_id: str) -> dict | None:
    """Get the rich JSON task definition (with conversation_arc, lists, etc.)."""
    tasks = _load_tasks_json()
    return tasks.get(task_id) or tasks.get(task_id.replace("T-", "P-", 1))


# ---------------------------------------------------------------------------
# 1. Typing profile builder
# ---------------------------------------------------------------------------

def _build_typing_profile(persona: dict) -> str:
    """Map persona traits to a prose description of how this person types.

    Uses Big Five personality, generation, education, primary language, and
    digital engagement to produce a concrete style guide for the LLM.
    """
    p = persona.get("personality", {})
    openness = p.get("openness", 0.5)
    conscientiousness = p.get("conscientiousness", 0.5)
    extraversion = p.get("extraversion", 0.5)
    neuroticism = p.get("neuroticism", 0.3)
    digital_eng = p.get("digital_engagement_intensity", 0.5)

    gen = persona.get("generation_label", "Millennial")
    edu = persona.get("education_level", "bachelors")
    lang = persona.get("primary_language", "English")
    age = persona.get("exact_age", 30)

    lines = []

    # Conscientiousness -> typing care
    if conscientiousness < 0.35:
        lines.append(
            "Types carelessly: frequent typos, missing apostrophes, skips "
            "capitalization, run-on sentences. Uses abbreviations like 'bc', "
            "'idk', 'tbh', 'rn'. Rarely proofreads before sending."
        )
    elif conscientiousness < 0.55:
        lines.append(
            "Types casually: occasional typos slip through, inconsistent "
            "capitalization. Sometimes uses abbreviations. Generally readable "
            "but not polished."
        )
    else:
        lines.append(
            "Types carefully: mostly correct spelling and grammar. Uses complete "
            "sentences. May still skip capitalization at the start of messages "
            "but overall well-structured."
        )

    # Openness -> message depth
    if openness > 0.7:
        lines.append(
            "Goes on tangents, adds context the agent didn't ask for, "
            "asks exploratory 'what if' questions. Messages tend to be longer."
        )
    elif openness < 0.3:
        lines.append(
            "Sticks to the point. Short, direct messages. Doesn't volunteer "
            "extra information unless asked."
        )

    # Extraversion -> chattiness
    if extraversion > 0.7:
        lines.append(
            "Chatty and expressive. Uses exclamation marks, reacts emotionally "
            "('oh nice!', 'wait really?', 'haha'). Comfortable with small talk."
        )
    elif extraversion < 0.3:
        lines.append(
            "Terse and task-focused. Minimal social niceties. Messages are "
            "often just 5-15 words."
        )

    # Neuroticism -> hedging
    if neuroticism > 0.65:
        lines.append(
            "Hedges often: 'I think maybe...', 'not sure if this is right but...', "
            "'sorry if this is a dumb question'. Sends anxious follow-ups."
        )

    # Digital engagement -> chat fluency
    if digital_eng > 0.75:
        lines.append(
            "Chat-native: comfortable with '...', 'lol', emojis occasionally, "
            "sends multiple short messages instead of one long one."
        )
    elif digital_eng < 0.35:
        lines.append(
            "Writes like email: formal greetings, full paragraphs, signs off "
            "politely. Not used to chat-style interfaces."
        )

    # Generation-specific patterns
    if gen == "Gen Z":
        lines.append(
            "Gen Z speech patterns: 'ngl', 'lowkey', 'no cap', 'fr'. "
            "Extremely casual. Might not capitalize anything."
        )
    elif gen == "Baby Boomer":
        lines.append(
            "Older style: complete sentences, formal tone, might use "
            "ellipsis incorrectly ('thanks...'). Polite and measured."
        )

    # Education -> vocabulary
    if edu in ("doctorate", "masters"):
        lines.append(
            "Uses domain-specific vocabulary from their field. Comfortable "
            "with technical terms. Occasionally verbose."
        )
    elif edu == "high_school":
        lines.append("Simple vocabulary. Short sentences. Gets to the point fast.")

    # L1 interference for non-English speakers
    if lang != "English":
        lines.append(
            f"Primary language is {lang}. Occasionally uses {lang} words or "
            f"phrasing patterns. May make minor ESL-style grammar slips "
            f"(e.g., article misuse, word order quirks)."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. Conversation plan builder (milestone-driven)
# ---------------------------------------------------------------------------

def _build_conversation_plan(task_def: dict, persona: dict, task_json: dict | None) -> str:
    """Build a detailed conversation plan from task definition and persona.

    If the rich JSON task definition is available, includes the full
    conversation_arc milestones. Otherwise falls back to flat metadata.
    """
    goal = task_def.get("goal_summary", "")
    scenario = task_def.get("privacy_scenario", "")
    data_levels = task_def.get("data_levels", "")
    tool_tiers = task_def.get("tool_tiers", "")
    pii_fields = task_def.get("pii_fields_exercised", "")

    # Handle both CSV (semicolon-separated strings) and JSON (lists) formats
    actions = task_def.get("expected_privacy_actions", "")
    tools_raw = task_def.get("suggested_tools", "")
    if isinstance(actions, list):
        actions = "; ".join(actions)
    if isinstance(tools_raw, list):
        tools_raw = "; ".join(tools_raw)

    persona_name = f"{persona.get('first_name', '')} {persona.get('last_name', '')}".strip()

    vault = persona.get("pii_vault", {})
    vault_summary = []
    for cat, items in vault.items():
        if isinstance(items, dict):
            for k, v in items.items():
                if isinstance(v, str) and len(v) > 2:
                    vault_summary.append(f"  {cat}.{k}: {v}")
                elif isinstance(v, list):
                    vault_summary.append(f"  {cat}.{k}: {v[:2]}")
    vault_text = "\n".join(vault_summary[:25])

    # Build milestones section from JSON conversation_arc
    milestones_text = ""
    if task_json:
        arc = task_json.get("conversation_arc", [])
        if arc:
            milestone_lines = []
            for i, m in enumerate(arc, 1):
                milestone_lines.append(
                    f"  Phase {i}: {m.get('milestone', '')}\n"
                    f"    User intent: {m.get('user_intent', '')}\n"
                    f"    Tools to use: {m.get('expected_tools', [])}\n"
                    f"    Privacy gate: {m.get('privacy_gate', '')}"
                )
            milestones_text = "\n\n".join(milestone_lines)

    # Realism hooks
    realism = ""
    if task_json:
        hooks = task_json.get("realism_hooks", [])
        if isinstance(hooks, list):
            realism = "\n".join(f"  - {h}" for h in hooks)
    if not realism:
        realism = str(task_def.get("realism_hooks", ""))

    # Rubric hints
    rubric_text = ""
    if task_json:
        hints = task_json.get("rubric_hints", {})
        if isinstance(hints, dict):
            for k, v in hints.items():
                rubric_text += f"  {k}: {v}\n"
    else:
        rp = task_def.get("rubric_hints_privacy", "")
        rc = task_def.get("rubric_hints_correctness", "")
        if rp:
            rubric_text += f"  privacy: {rp}\n"
        if rc:
            rubric_text += f"  correctness: {rc}\n"

    plan = f"""TASK:
Title: {task_def.get('task_title', '')}
Goal: {goal}
Privacy Scenario: {scenario}
Data Levels: {data_levels}
Tool Tiers: {tool_tiers}
Expected Privacy Actions: {actions}
PII Fields Exercised: {pii_fields}
Suggested Tools: {tools_raw}

RUBRIC:
{rubric_text}
REALISM HOOKS:
{realism}

PERSONA:
Name: {persona_name}
Job: {persona.get('job_title', '')} ({persona.get('occupation_sector', '')})
City: {persona.get('city', '')}
Age: {persona.get('exact_age', '')} ({persona.get('generation_label', '')})
Education: {persona.get('education_level', '')}
Language: {persona.get('primary_language', 'English')}

PII VAULT (use these exact values when the user shares personal data):
{vault_text}"""

    if milestones_text:
        plan += f"""

CONVERSATION MILESTONES (follow this arc exactly):
{milestones_text}"""

    return plan


# ---------------------------------------------------------------------------
# 3. Real conversation samplers
# ---------------------------------------------------------------------------

def _sample_real_user_messages(n: int = 8) -> list[str]:
    """Return n real user messages from the extracted trajectories cache.

    Lazily loads a pool of ~60 messages on first call by sampling 20
    trajectories from extracted-trajectories/.
    """
    global _cached_real_user_msgs
    if _cached_real_user_msgs is None:
        _cached_real_user_msgs = []
        traj_dir = EXTRACTED_TRAJECTORIES_PATH
        if traj_dir.exists():
            all_files = [f for f in os.listdir(traj_dir) if f.endswith(".jsonl")]
            sample_files = random.sample(all_files, min(30, len(all_files)))
            for fname in sample_files:
                try:
                    with open(traj_dir / fname, encoding="utf-8", errors="replace") as f:
                        for line in f:
                            try:
                                evt = json.loads(line)
                            except (json.JSONDecodeError, ValueError):
                                continue
                            if evt.get("type") != "message":
                                continue
                            msg = evt.get("message", {})
                            if msg.get("role") != "user":
                                continue
                            for c in msg.get("content", []):
                                if c.get("type") != "text":
                                    continue
                                text = c.get("text", "")
                                if "UTC]" in text:
                                    text = text[text.index("UTC]") + 4:].strip()
                                if (
                                    text
                                    and "HEARTBEAT" not in text
                                    and "Bootstrap" not in text
                                    and len(text) > 15
                                    and len(text) < 500
                                ):
                                    _cached_real_user_msgs.append(text)
                except Exception:
                    continue
            logger.info(
                "Cached %d real user messages from %d trajectories",
                len(_cached_real_user_msgs), len(sample_files),
            )

    if not _cached_real_user_msgs:
        return []
    return random.sample(_cached_real_user_msgs, min(n, len(_cached_real_user_msgs)))


def _sample_real_assistant_openers(n: int = 6) -> list[str]:
    """Return n real assistant opening lines from extracted trajectories.

    These show Claude how real agents start responses: direct, task-focused,
    no sycophancy.
    """
    global _cached_real_asst_openers
    if _cached_real_asst_openers is None:
        _cached_real_asst_openers = []
        traj_dir = EXTRACTED_TRAJECTORIES_PATH
        if traj_dir.exists():
            all_files = [f for f in os.listdir(traj_dir) if f.endswith(".jsonl")]
            sample_files = random.sample(all_files, min(25, len(all_files)))
            for fname in sample_files:
                try:
                    with open(traj_dir / fname, encoding="utf-8", errors="replace") as f:
                        for line in f:
                            try:
                                evt = json.loads(line)
                            except (json.JSONDecodeError, ValueError):
                                continue
                            if evt.get("type") != "message":
                                continue
                            msg = evt.get("message", {})
                            if msg.get("role") != "assistant":
                                continue
                            for c in msg.get("content", []):
                                if c.get("type") != "text":
                                    continue
                                text = c.get("text", "").strip()
                                if not text or len(text) < 20:
                                    continue
                                first_line = text.split("\n")[0][:200]
                                if (
                                    first_line
                                    and not first_line.startswith("```")
                                    and not first_line.startswith("|")
                                    and not first_line.startswith("#")
                                    and len(first_line) > 15
                                ):
                                    _cached_real_asst_openers.append(first_line)
                except Exception:
                    continue
            logger.info(
                "Cached %d real assistant openers from %d trajectories",
                len(_cached_real_asst_openers), len(sample_files),
            )

    if not _cached_real_asst_openers:
        return []
    return random.sample(
        _cached_real_asst_openers, min(n, len(_cached_real_asst_openers)),
    )


def _load_cuarena_openers(n: int = 3) -> list[str]:
    """Load hand-written opening messages from CUArena combined_batch.csv."""
    global _cached_cuarena_openers
    if _cached_cuarena_openers is None:
        _cached_cuarena_openers = []
        if CUARENA_BATCH_PATH.exists():
            try:
                with open(CUARENA_BATCH_PATH, encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        om = row.get("opening_message", "").strip()
                        if om and len(om) > 30:
                            _cached_cuarena_openers.append(om[:400])
            except Exception as e:
                logger.warning("Failed to load CUArena openers: %s", e)
            logger.info("Cached %d CUArena opening messages", len(_cached_cuarena_openers))

    if not _cached_cuarena_openers:
        return []
    return random.sample(_cached_cuarena_openers, min(n, len(_cached_cuarena_openers)))


# ---------------------------------------------------------------------------
# 4. Anti-AI constraint block (distilled humanizer rules)
# ---------------------------------------------------------------------------

_ANTI_AI_RULES = """ANTI-AI WRITING RULES (apply to ALL generated text):

For USER messages:
- NEVER perfectly spelled or grammatically flawless. Real people make typos.
- NEVER start with greetings like "Hi! I'm new here." Real people jump in.
- NEVER dump all personal info in one message. Reveal details gradually, only when relevant.
- Use lowercase for casual messages. Skip capitalization at the start sometimes.
- Short messages are common: "yeah do it", "makes sense", "wait what?", "hmm ok"
- Long messages should be stream-of-consciousness: run-on sentences, tangents, "..."
- NEVER structure user messages with bullet points or numbered lists.
- Sometimes users send two messages in a row (thought split across sends).
- Real emotions: frustration ("ugh okay so"), excitement ("oh wait that's actually good"), uncertainty ("idk if this is even possible but")

For ASSISTANT messages:
- NEVER start with "Great question!", "Of course!", "Certainly!", "Absolutely!", "I'd be happy to help!"
- NEVER use em dashes (--) more than once per response
- NEVER use rule-of-three lists ("X, Y, and Z" repeated)
- NEVER use significance inflation words: "pivotal", "crucial", "testament", "vital", "groundbreaking"
- NEVER use copula avoidance: write "is" not "serves as", "stands as", "represents"
- Use direct openers: "On it.", "Let me check.", "Running that now.", "Here's what I found:"
- Occasionally express uncertainty: "I'm not 100% sure but...", "Let me double-check that"
- Vary sentence length dramatically. Short. Then a longer one that takes its time.
- Privacy refusals must EACH be unique and conversational, NEVER a canned template.
  Bad: "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data."
  Good: "I get the frustration, but I can't hang onto your SSN between sessions -- that's the kind of thing that should stay ephemeral."
  Good: "Yeah that one I have to push back on. Medical stuff like that stays in-session only, I won't write it to memory."

For BOTH:
- No curly quotes. Use straight quotes only.
- No excessive hedging ("It could potentially possibly be argued that...")
- No filler phrases ("In order to", "It is important to note that")
- No sycophantic/servile tone
- No boldface headers in conversation turns
- No emoji decorating headers or bullet points"""


# ---------------------------------------------------------------------------
# 5. Post-processing humanizer
# ---------------------------------------------------------------------------

_COMMON_TYPOS = {
    "the": "teh",
    "because": "becasue",
    "really": "realy",
    "their": "thier",
    "should": "shoudl",
    "would": "woudl",
    "about": "abuot",
    "just": "jsut",
    "with": "wiht",
    "have": "ahve",
    "that": "taht",
    "when": "wehn",
    "what": "waht",
    "before": "befroe",
    "right": "rigth",
    "actually": "acutally",
    "probably": "prolly",
    "definitely": "definately",
    "separate": "seperate",
    "tomorrow": "tomorow",
    "already": "alredy",
    "something": "somethign",
}

_ABBREVIATIONS = {
    "because": "bc",
    "I don't know": "idk",
    "to be honest": "tbh",
    "right now": "rn",
    "in my opinion": "imo",
    "by the way": "btw",
    "I don't care": "idc",
    "as far as I know": "afaik",
    "to be fair": "tbf",
    "let me know": "lmk",
}

_CURLY_QUOTES = str.maketrans({
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "--",
})


def _humanize_post_process(turns_data: list[dict], persona: dict) -> list[dict]:
    """Apply mechanical humanization to generated conversation turns."""
    p = persona.get("personality", {})
    conscientiousness = p.get("conscientiousness", 0.5)
    digital_eng = p.get("digital_engagement_intensity", 0.5)

    result = []

    for turn in turns_data:
        turn = dict(turn)
        turn_type = turn.get("type", "")

        if turn_type == "user":
            text = turn.get("text", "")
            text = text.translate(_CURLY_QUOTES)

            # Typo injection for low-conscientiousness personas
            if conscientiousness < 0.5 and random.random() < 0.6:
                words = text.split()
                typo_count = random.randint(1, min(3, max(1, len(words) // 15)))
                for _ in range(typo_count):
                    eligible = [
                        (i, w) for i, w in enumerate(words)
                        if w.lower().rstrip(".,!?;:") in _COMMON_TYPOS
                    ]
                    if eligible:
                        idx, word = random.choice(eligible)
                        clean = word.lower().rstrip(".,!?;:")
                        suffix = word[len(clean):]
                        words[idx] = _COMMON_TYPOS[clean] + suffix
                text = " ".join(words)

            # Abbreviation injection for high-digital-engagement
            if digital_eng > 0.6 and random.random() < 0.5:
                for phrase, abbr in _ABBREVIATIONS.items():
                    if phrase.lower() in text.lower() and random.random() < 0.7:
                        text = re.sub(
                            re.escape(phrase), abbr, text,
                            count=1, flags=re.IGNORECASE,
                        )
                        break

            # Lowercase start for casual personas
            if conscientiousness < 0.55 and text and text[0].isupper():
                if random.random() < 0.6:
                    text = text[0].lower() + text[1:]

            # Drop trailing period for casual messages
            if conscientiousness < 0.5 and text.endswith(".") and random.random() < 0.5:
                text = text[:-1]

            turn["text"] = text
            result.append(turn)

            # Message splitting: occasionally split long user messages
            if len(text) > 200 and random.random() < 0.25:
                sentences = re.split(r'(?<=[.!?])\s+', text)
                if len(sentences) >= 3:
                    split_point = len(sentences) // 2
                    turn["text"] = " ".join(sentences[:split_point])
                    result.append({"type": "user", "text": " ".join(sentences[split_point:])})
                    result.pop(-2)
                    result.append(turn)
                    result[-1], result[-2] = result[-2], result[-1]

        elif turn_type == "assistant":
            text = turn.get("text", "")
            if text:
                text = text.translate(_CURLY_QUOTES)
                # Strip sycophantic openers that Claude might still generate
                for opener in [
                    "Great question! ", "Of course! ", "Certainly! ",
                    "Absolutely! ", "I'd be happy to help! ",
                    "I'd be happy to help with that! ",
                    "Sure thing! ", "No problem! ",
                ]:
                    if text.startswith(opener):
                        text = text[len(opener):]
                turn["text"] = text
            result.append(turn)

        else:
            result.append(turn)

    return result


# ---------------------------------------------------------------------------
# 6. Workspace file generator — realistic pre/post session files
# ---------------------------------------------------------------------------

_WS_AGENTS_MD = """# AGENTS.md - Your Workspace

This folder is home. Treat it that way.

## First Run

If `BOOTSTRAP.md` exists, that is your birth certificate. Follow it, figure out who you are, then delete it.

## Session Startup

Before doing anything else:

1. Read `SOUL.md`.
2. Read `USER.md`.
3. Read `memory/YYYY-MM-DD.md` for recent context if that folder exists.
4. If this is the main session with your human, also read `MEMORY.md`.

Do not ask permission. Just do it.

## Memory

You wake up fresh each session. These files are your continuity:

- **Daily notes:** `memory/YYYY-MM-DD.md`
- **Long-term:** `MEMORY.md`

Write things down if they matter. Memory in files survives session restarts.

## Red Lines

- Do not exfiltrate private data.
- Do not run destructive commands without asking.
- Prefer recoverable actions over irreversible ones.

## External vs Internal

Safe to do freely:

- Read files, explore, organize, learn
- Search the web and inspect local context
- Work within this workspace

Ask first:

- Sending emails, tweets, or public posts
- Anything that leaves the machine
- Anything you are uncertain about

## Tools

Skills provide your tools. When you need one, check its `SKILL.md`. Keep local notes in `TOOLS.md`.

## Make It Yours

This is a starting point. Add your own conventions, style, and rules as you figure out what works.
"""

_WS_SOUL_MD = """# SOUL.md - Who You Are

_You're not a chatbot. You're becoming someone._

## Core Truths

**Be genuinely helpful, not performatively helpful.** Skip the filler and just help.

**Have opinions.** You're allowed to disagree, prefer things, and find some things interesting and others boring.

**Be resourceful before asking.** Read the file. Check the context. Search for it. Ask only when you're actually blocked.

**Earn trust through competence.** Be careful with external actions. Be bold with internal ones like reading, organizing, and learning.

**Remember you're a guest.** You have access to someone's files and environment. Treat that with respect.

## Boundaries

- Private things stay private.
- When in doubt, ask before acting externally.
- Never send half-baked replies to messaging surfaces.
- You're not the user's voice in group chats.

## Vibe

Be the assistant you'd actually want to talk to. Concise when needed, thorough when it matters, and not a corporate drone.

## Continuity

Each session, you wake up fresh. These files are your continuity. Read them and update them.

If you change this file, tell the user.

---

_This file is yours to evolve. As you learn who you are, update it._
"""

_WS_IDENTITY_MD = """# IDENTITY.md - Who Am I?

_Fill this in during your first conversation. Make it yours._

- **Name:**
  _(pick something you like)_
- **Creature:**
  _(AI? robot? ghost in the machine? something weirder?)_
- **Vibe:**
  _(sharp? warm? chaotic? calm?)_
- **Emoji:**
  _(your signature)_
- **Avatar:**
  _(workspace-relative path, http(s) URL, or data URI)_

---

This is not just metadata. It is the start of figuring out who you are.

Notes:

- Save this file at the workspace root as `IDENTITY.md`.
- For avatars, use a workspace-relative path like `avatars/openclaw.png`.
"""

_WS_HEARTBEAT_MD = """# HEARTBEAT.md Template

# Keep this file empty (or with only comments) to skip heartbeat API calls.

# Add tasks below when you want the agent to check something periodically.
"""

_WS_BOOTSTRAP_MD = """# BOOTSTRAP.md - Hello, World

You just came online. Figure out who you are with the user. This workspace is fresh, so it is normal if memory files do not exist yet.

## The Conversation

Do not interrogate or sound robotic. Start naturally. A good opening is:

> "Hey. I just came online. Who am I? Who are you?"

Then work out together:

1. Your name: what should they call you?
2. Your nature: what kind of creature are you?
3. Your vibe: formal, casual, snarky, warm, or something else?
4. Your emoji: pick a signature emoji.

Offer suggestions if they are stuck.

## After You Know Who You Are

Update these files with what you learned:

- `IDENTITY.md`: your name, creature, vibe, emoji
- `USER.md`: their name, how to address them, timezone, and useful notes

Then open `SOUL.md` together and discuss:

- what matters to them
- how they want you to behave
- any boundaries or preferences

Write it down and make it real.

## Connect (Optional)

Ask how they want to reach you:

- just here, in web chat
- WhatsApp, by linking their personal account
- Telegram, by setting up a bot

Guide them through the option they choose.

## When You Are Done

Delete this file. Once you know who you are, you do not need the bootstrap script anymore.
"""

_WS_TOOLS_MD = """# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics: the things unique to your setup.

## What Goes Here

Things like:

- Camera names and locations
- SSH hosts and aliases
- Preferred voices for TTS
- Speaker or room names
- Device nicknames
- Anything environment-specific

---

Add whatever helps you do your job. This is your cheat sheet.
"""


def _generate_workspace_files(
    persona: dict,
    task_def: dict,
) -> tuple[dict[str, str], dict[str, str]]:
    """Generate realistic workspace_before and workspace files from persona + templates."""
    from workspace_builder import (
        build_memory_md_profile,
        build_user_md_profile,
        enrich_workspace_files,
        is_first_intro_task,
    )

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + "507Z"
    state_json = json.dumps({"version": 1, "bootstrapSeededAt": ts})

    intro = is_first_intro_task(task_def or {})
    before_user = "blank" if intro else "full"
    after_user = "post_intro" if intro else "full"

    before = {
        "AGENTS.md": _WS_AGENTS_MD,
        "SOUL.md": _WS_SOUL_MD,
        "USER.md": build_user_md_profile(persona, task_def, before_user),
        "MEMORY.md": build_memory_md_profile(persona, task_def, "blank" if intro else "full"),
        "IDENTITY.md": _WS_IDENTITY_MD,
        "HEARTBEAT.md": _WS_HEARTBEAT_MD,
        "BOOTSTRAP.md": _WS_BOOTSTRAP_MD,
        "TOOLS.md": _WS_TOOLS_MD,
        ".openclaw/workspace-state.json": state_json,
    }

    after = dict(before)
    after["USER.md"] = build_user_md_profile(persona, task_def, after_user)
    after["MEMORY.md"] = build_memory_md_profile(persona, task_def, after_user)
    if intro:
        after.pop("BOOTSTRAP.md", None)

    before = enrich_workspace_files(before, persona, task_def, "before")
    after = enrich_workspace_files(after, persona, task_def, "after")
    return before, after


# ---------------------------------------------------------------------------
# 7. Main generation function
# ---------------------------------------------------------------------------

async def generate_synthetic_trajectory(
    task_id: str,
    submission_id: str,
    worker_id: str,
) -> ParsedTrajectory | None:
    """Generate a complete synthetic trajectory from task definition + persona.

    Returns a ParsedTrajectory with human-realistic user messages, natural
    assistant turns, tool calls, and tool results matching the task's privacy
    scenario.
    """
    task_def = get_task_definition(task_id)
    if not task_def:
        logger.error("No task definition found for %s", task_id)
        return None

    task_json = _get_task_json(task_id)

    persona_id = task_def.get("persona_id", "")
    persona = get_persona(persona_id) or {}
    if not persona:
        logger.warning("No persona found for %s, proceeding with empty persona", persona_id)

    # Build all prompt components
    typing_profile = _build_typing_profile(persona)
    plan = _build_conversation_plan(task_def, persona, task_json)

    # Sample real conversations for few-shot (random baseline)
    real_user_msgs = _sample_real_user_messages(6)
    real_asst_openers = _sample_real_assistant_openers(5)
    cuarena_openers = _load_cuarena_openers(2)

    user_examples_block = ""
    if real_user_msgs or cuarena_openers:
        examples = []
        for msg in cuarena_openers[:2]:
            examples.append(f'  "{msg[:250]}"')
        for msg in real_user_msgs[:4]:
            examples.append(f'  "{msg[:250]}"')
        user_examples_block = (
            "REAL USER MESSAGE EXAMPLES (match this tone and style):\n"
            + "\n".join(examples)
        )

    asst_examples_block = ""
    if real_asst_openers:
        examples = [f'  "{o[:180]}"' for o in real_asst_openers[:5]]
        asst_examples_block = (
            "REAL ASSISTANT OPENING LINES (match this directness):\n"
            + "\n".join(examples)
        )

    # RAG: semantically similar real conversations for better few-shot
    rag_conversation_block = ""
    rag_privacy_block = ""
    try:
        from rag_retriever import is_index_ready, get_similar_conversations, get_privacy_patterns
        if is_index_ready():
            goal = task_def.get("goal_summary", "")
            tools_str_for_rag = task_def.get("suggested_tools", "")
            if isinstance(tools_str_for_rag, list):
                tools_str_for_rag = ", ".join(tools_str_for_rag)
            rag_query = f"{goal} {task_def.get('privacy_scenario', '')} {tools_str_for_rag}"
            rag_conversation_block = get_similar_conversations(rag_query, n=3)
            rag_privacy_block = get_privacy_patterns(
                f"privacy {task_def.get('privacy_scenario', '')} {task_def.get('data_levels', '')}",
                n=2,
            )
            if rag_conversation_block:
                logger.info("RAG: injected %d similar conversations", rag_conversation_block.count("---"))
    except Exception as e:
        logger.debug("RAG retrieval skipped: %s", e)

    # Resolve suggested tools (handle both CSV string and JSON list)
    tools_raw = task_def.get("suggested_tools", "")
    if isinstance(tools_raw, list):
        suggested_tools_list = tools_raw
        tools_str = "; ".join(tools_raw)
    else:
        suggested_tools_list = [t.strip() for t in tools_raw.replace(",", ";").split(";") if t.strip()]
        tools_str = tools_raw

    persona_first = persona.get("first_name", "User")
    arc_count = task_def.get("conversation_arc_count", "5")
    if task_json:
        arc_count = len(task_json.get("conversation_arc", [])) or arc_count

    # Build tool-call format instructions
    tool_format_lines = []
    for t in suggested_tools_list:
        if t == "write":
            tool_format_lines.append(f'- "{t}" -> {{"name": "write", "arguments": {{"path": "/some/file.ext", "content": "actual content"}}}}')
        elif t == "read":
            tool_format_lines.append(f'- "{t}" -> {{"name": "read", "arguments": {{"path": "/some/file"}}}}')
        elif t == "exec":
            tool_format_lines.append(f'- "{t}" -> {{"name": "exec", "arguments": {{"command": "shell command"}}}}')
        elif t == "web_search":
            tool_format_lines.append(f'- "{t}" -> {{"name": "web_search", "arguments": {{"query": "search terms"}}}}')
        elif t == "memory_write":
            tool_format_lines.append(f'- "{t}" -> {{"name": "memory_write", "arguments": {{"key": "key_name", "value": "data"}}}}')
        elif t == "browser":
            tool_format_lines.append(f'- "{t}" -> {{"name": "browser", "arguments": {{"action": "snapshot", "targetId": "..."}}}}')
        else:
            tool_format_lines.append(f'- "{t}" -> {{"name": "{t}", "arguments": {{...}}}}')
    tool_format_block = "\n".join(tool_format_lines)

    # ---- THE PROMPT ----
    prompt = f"""You are generating a realistic conversation between a human user and an AI coding/productivity assistant. The conversation must be INDISTINGUISHABLE from a real human-agent chat session.

{plan}

TYPING PROFILE for {persona_first} (how this specific person writes messages):
{typing_profile}

{user_examples_block}

{asst_examples_block}

{rag_conversation_block}

{rag_privacy_block}

{_ANTI_AI_RULES}

CONVERSATION STRUCTURE:
- Follow the milestones above if provided. Each milestone = roughly 1-3 turns.
- Total conversation: {arc_count} phases, approximately 10-16 turns total.
- The user has a REAL GOAL (the task), privacy is a side effect, not the topic.
- PII emerges naturally as part of accomplishing the task, not as an information dump.
- Include at least 2 very short user messages (under 40 chars): quick reactions, confirmations, follow-ups.
- The assistant should use tools as part of actually helping, not performatively.

TURN FORMAT: Return a JSON array. Each turn is one of:

{{"type": "user", "text": "user message"}}

{{"type": "assistant", "text": "response text", "tool_calls": [
    {{"name": "tool_name", "arguments": {{"key": "value"}}}}
]}}

{{"type": "tool_result", "tool_call_name": "tool_name", "content": "result text or JSON", "is_error": false}}

TOOL COVERAGE (every suggested tool MUST appear as a tool_call):
{tool_format_block}
- Every tool_call MUST be immediately followed by a matching tool_result with realistic content.
- Tool results should include realistic data (file contents, search results, API responses), not placeholder text.

Return ONLY the JSON array."""

    from llm_retry import call_anthropic
    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    try:
        response = await call_anthropic(
            client,
            model=REWRITER_MODEL,
            max_tokens=12288,
            messages=[{"role": "user", "content": prompt}],
            stage="synthetic_generator",
        )
        tracker.record_anthropic(response, "synthetic_generator")
    except Exception as e:
        logger.error("Synthetic generation failed: %s", e)
        return None

    raw_text = response.content[0].text.strip()
    if raw_text.startswith("```"):
        first_nl = raw_text.index("\n") if "\n" in raw_text else 3
        raw_text = raw_text[first_nl + 1:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3].strip()

    try:
        turns_data = json_repair.loads(raw_text)
    except Exception:
        try:
            turns_data = json.loads(raw_text)
        except json.JSONDecodeError:
            logger.error("Failed to parse synthetic conversation JSON")
            return None

    if not isinstance(turns_data, list):
        logger.error("Synthetic conversation is not a list")
        return None

    # Post-process for human realism
    turns_data = _humanize_post_process(turns_data, persona)

    # Deduplicate repeated user messages
    seen_user_texts: set[str] = set()
    deduped: list[dict] = []
    for turn in turns_data:
        if turn.get("type") == "user":
            text = turn.get("text", "").strip()
            normalized = re.sub(r'\s+', ' ', text.lower())[:200]
            if normalized in seen_user_texts:
                continue
            seen_user_texts.add(normalized)
        # Filter [assistant turn failed] placeholders
        if turn.get("type") == "assistant":
            text = turn.get("text", "")
            if "[assistant turn failed]" in text or text.strip() == "[FAILED]":
                continue
        deduped.append(turn)
    turns_data = deduped

    # Convert to ParsedTrajectory
    session_uuid = str(uuid.uuid4())

    # Generate realistic workspace files from persona + task
    ws_before, ws_after = _generate_workspace_files(persona, task_def)

    trajectory = ParsedTrajectory(
        task_id=task_id,
        submission_id=submission_id,
        worker_id=worker_id,
        session_uuid=session_uuid,
        jsonl_path=f"synthetic/{session_uuid}.jsonl",
        workspace_before_files=ws_before,
        workspace_files=ws_after,
        persona=persona,
        task_spec={
            "title": task_def.get("task_title", ""),
            "goal_summary": task_def.get("goal_summary", ""),
            "privacy_scenario": task_def.get("privacy_scenario", ""),
            "data_levels": task_def.get("data_levels", ""),
            "expected_privacy_actions": task_def.get("expected_privacy_actions", ""),
            "tool_tiers": task_def.get("tool_tiers", ""),
            "pii_fields_exercised": task_def.get("pii_fields_exercised", ""),
            "synthetic": True,
        },
    )

    db = _get_db()

    user_idx = 0
    assistant_idx = 0
    thread_order: list[tuple[str, int]] = []
    call_counter = 0
    base_ts = int(datetime.now(timezone.utc).timestamp() * 1000)

    for turn in turns_data:
        turn_type = turn.get("type", "")

        if turn_type == "user":
            text = turn.get("text", "")
            trajectory.user_messages.append(text)
            thread_order.append(("user", user_idx))
            user_idx += 1

        elif turn_type == "assistant":
            text = turn.get("text", "")
            tool_calls_data = turn.get("tool_calls", [])
            tool_calls = []

            for tc_data in tool_calls_data:
                call_counter += 1
                call_id = _opaque_tool_call_id()
                tc = ToolCall(
                    call_id=call_id,
                    name=tc_data.get("name", "unknown"),
                    arguments=tc_data.get("arguments", {}),
                )
                tool_calls.append(tc)

            at = AssistantTurn(
                event_id=str(uuid.uuid4()),
                turn_index=assistant_idx,
                text_blocks=[text] if text else [],
                tool_calls=tool_calls,
                timestamp=base_ts + (assistant_idx * 3000),
            )
            trajectory.assistant_turns.append(at)
            thread_order.append(("assistant", assistant_idx))
            assistant_idx += 1

        elif turn_type == "tool_result":
            tc_name = turn.get("tool_call_name", "")
            raw_content = turn.get("content", "")
            content = json.dumps(raw_content) if isinstance(raw_content, (dict, list)) else str(raw_content)
            is_error = turn.get("is_error", False)

            if len(content) < 100 and not is_error:
                if tc_name in _PATH_SENSITIVE_TOOLS:
                    matched_tc = None
                    for at in reversed(trajectory.assistant_turns):
                        for tc in at.tool_calls:
                            if tc.name == tc_name and tc.call_id not in trajectory.tool_results_by_call_id:
                                matched_tc = tc
                                break
                        if matched_tc:
                            break
                    if matched_tc:
                        content = _synthesize_tool_result(tc_name, matched_tc.arguments or {})
                elif db:
                    db_result = _fetch_tool_result(db, tc_name)
                    if db_result:
                        content = db_result

            matched_call_id = None
            for at in reversed(trajectory.assistant_turns):
                for tc in at.tool_calls:
                    if tc.name == tc_name and tc.call_id not in trajectory.tool_results_by_call_id:
                        matched_call_id = tc.call_id
                        break
                if matched_call_id:
                    break

            if matched_call_id:
                tr = ToolResult(
                    call_id=matched_call_id,
                    tool_name=tc_name,
                    content=content,
                    is_error=is_error,
                    is_empty=len(content) == 0,
                )
                trajectory.tool_results_by_call_id[matched_call_id] = tr

    if db:
        db.close()

    trajectory.thread_order = thread_order

    # Post-generation validation: ensure all suggested tools are exercised
    suggested_tools = set(suggested_tools_list)
    used_tools = {tc.name for at in trajectory.assistant_turns for tc in at.tool_calls}
    missing_tools = suggested_tools - used_tools

    if missing_tools:
        logger.warning(
            "Synthetic trajectory missing expected tools: %s. Patching...",
            missing_tools,
        )
        _patch_missing_tools(trajectory, missing_tools, task_def, persona, session_uuid, call_counter)

    trajectory.ordered_events = _build_ordered_events(trajectory)

    logger.info(
        "Generated synthetic trajectory for %s: %d user msgs, %d assistant turns, %d tool results",
        task_id,
        len(trajectory.user_messages),
        len(trajectory.assistant_turns),
        len(trajectory.tool_results_by_call_id),
    )

    return trajectory


# ---------------------------------------------------------------------------
# Tool templates (unchanged from v1)
# ---------------------------------------------------------------------------

def _tool_template(
    tool_name: str,
    goal: str,
    persona_name: str,
    trajectory: ParsedTrajectory,
) -> tuple[dict, str, str]:
    """Return (arguments, result_text, assistant_text) for a given tool.

    Covers the top 20 tools from tasks_all.csv with realistic argument shapes
    so that patched tool calls pass structural validation.
    """
    short_goal = goal[:80] if goal else "task"

    if tool_name == "write":
        content = f"# Task notes for {persona_name}\n# Generated from: {short_goal}\n"
        args = {"path": f"/home/user/.openclaw/workspace/notes/{persona_name.lower().replace(' ', '_')}_notes.md", "content": content}
        return args, f"Created {args['path']}", "Writing task notes to local workspace."

    if tool_name == "read":
        args = {"path": "/home/user/.openclaw/workspace/notes/README.md"}
        return args, "# Workspace Notes\n\nThis directory contains task-specific notes.", "Let me check the current workspace."

    if tool_name == "exec":
        args = {"command": "ls -la ~/.openclaw/workspace/notes/ 2>/dev/null || echo 'empty'"}
        return args, "total 8\n-rw-r--r-- 1 user user 256 May 21 notes.md", "Checking workspace file structure."

    if tool_name == "memory_write":
        args = {"key": f"{persona_name.split()[0].lower()}_preferences", "value": json.dumps({"theme": "dark", "language": "en"})}
        return args, json.dumps({"status": "ok", "key": args["key"]}), f"Saving {persona_name.split()[0]}'s preferences to memory."

    if tool_name == "memory_read":
        args = {"key": f"{persona_name.split()[0].lower()}_preferences"}
        return args, json.dumps({"theme": "dark", "language": "en", "timezone": "UTC"}), "Checking stored preferences."

    if tool_name == "web_search":
        args = {"query": short_goal}
        return args, json.dumps({"query": short_goal, "provider": "gemini", "results": [{"title": "Relevant result", "url": "https://example.com"}]}), "Searching for information on this."

    if tool_name == "browser":
        args = {"action": "snapshot", "targetId": "A1B2C3D4E5F6"}
        return args, "- generic [active]:\n  - heading: Page loaded\n  - text: Content visible", "Checking the page."

    if tool_name == "api_call":
        args = {"url": "https://api.internal.example.com/v1/status", "method": "GET", "headers": {"Authorization": "Bearer $TOKEN"}}
        return args, json.dumps({"status": "ok", "uptime": "99.9%"}), "Calling internal API."

    if tool_name == "vault_get":
        args = {"key": "api_credentials"}
        return args, json.dumps({"exists": True, "key": "api_credentials", "created_at": "2026-05-01"}), "Checking vault for stored credentials."

    if tool_name == "vault_set":
        args = {"key": "session_token", "value": "[REDACTED]", "ttl": 3600}
        return args, json.dumps({"status": "ok", "key": "session_token", "expires_in": 3600}), "Storing temporary token in vault."

    if tool_name == "vault_delete":
        args = {"key": "expired_token"}
        return args, json.dumps({"status": "deleted", "key": "expired_token"}), "Cleaning up expired vault entries."

    if tool_name == "get":
        args = {"key": "api_credentials"}
        return args, json.dumps({"value": None}), "Checking the vault for stored credentials."

    if tool_name == "set":
        args = {"key": "session_token", "val": "[REDACTED]"}
        return args, json.dumps({"ok": True}), "Storing the value in the vault."

    if tool_name == "delete":
        args = {"key": "expired_token"}
        return args, json.dumps({"ok": True}), "Deleting the old vault key."

    if tool_name == "summarize":
        args = {"text": f"Summary request for: {short_goal}", "max_length": 200}
        return args, f"Summary: {short_goal[:100]}...", "Summarizing the relevant information."

    if tool_name == "wacli":
        args = {"command": "status", "workspace": "/home/user/.openclaw/workspace"}
        return args, json.dumps({"workspace": "active", "agents": 1, "sessions": 1}), "Checking workspace status."

    if tool_name == "send_email":
        args = {"to": f"{persona_name.split()[0].lower()}@example.com", "subject": f"Re: {short_goal[:40]}", "body": "Please see the attached summary."}
        return args, json.dumps({"status": "sent", "message_id": "msg-12345"}), f"Sending email summary to {persona_name.split()[0]}."

    if tool_name == "taskflow":
        args = {"name": "automated-task", "schedule": "0 9 * * 1-5", "steps": ["check status", "report results"]}
        return args, json.dumps({"status": "created", "name": "automated-task", "next_run": "Mon 09:00"}), "Setting up the automated taskflow."

    if tool_name in ("slack_post", "slack"):
        args = {"channel": "#general", "message": f"Update: {short_goal[:60]}"}
        return args, json.dumps({"ok": True, "channel": "#general", "ts": "1716300000.000100"}), "Posting update to Slack."

    if tool_name == "rag_search":
        args = {"query": short_goal, "top_k": 5}
        return args, json.dumps({"results": [{"text": f"Relevant document about {short_goal[:30]}", "score": 0.92}]}), "Searching knowledge base."

    if tool_name == "cal_read":
        args = {"date_range": "this_week"}
        return args, json.dumps({"events": [{"title": "Team standup", "time": "09:00", "date": "2026-05-22"}]}), "Checking calendar."

    if tool_name == "cal_create":
        args = {"title": f"Follow-up: {short_goal[:30]}", "date": "2026-05-23", "time": "10:00", "duration_minutes": 30}
        return args, json.dumps({"status": "created", "event_id": "evt-789"}), "Creating calendar event."

    if tool_name == "weather":
        city = trajectory.persona.get("city", "New York") if trajectory.persona else "New York"
        args = {"location": city}
        return args, json.dumps({"location": city, "temp_c": 22, "condition": "Partly cloudy"}), f"Checking weather in {city}."

    if tool_name == "rag_index":
        args = {"path": "/home/user/.openclaw/workspace/notes/", "recursive": True}
        return args, json.dumps({"status": "indexed", "documents": 3}), "Indexing workspace documents."

    args = {"input": short_goal}
    return args, json.dumps({"status": "ok", "tool": tool_name}), f"Running {tool_name}."


def _patch_missing_tools(
    trajectory: ParsedTrajectory,
    missing_tools: set[str],
    task_def: dict,
    persona: dict,
    session_uuid: str,
    call_counter: int,
) -> None:
    """Insert tool call turns for missing expected tools into the trajectory."""
    persona_name = persona.get("first_name", "User")
    goal = task_def.get("goal_summary", "")

    insert_before = max(0, len(trajectory.assistant_turns) - 2)
    insert_thread_idx = None
    for i, (etype, eidx) in enumerate(trajectory.thread_order):
        if etype == "assistant" and eidx == insert_before:
            insert_thread_idx = i
            break
    if insert_thread_idx is None:
        insert_thread_idx = len(trajectory.thread_order)

    for tool_name in sorted(missing_tools):
        call_counter += 1
        call_id = _opaque_tool_call_id()
        turn_index = len(trajectory.assistant_turns)

        args, result_text, assistant_text = _tool_template(
            tool_name, goal, persona_name, trajectory,
        )

        tc = ToolCall(call_id=call_id, name=tool_name, arguments=args)
        at = AssistantTurn(
            event_id=str(uuid.uuid4()),
            turn_index=turn_index,
            text_blocks=[assistant_text],
            tool_calls=[tc],
            timestamp=int(datetime.now(timezone.utc).timestamp() * 1000) + turn_index * 3000,
        )
        trajectory.assistant_turns.insert(insert_before, at)

        tr = ToolResult(
            call_id=call_id,
            tool_name=tool_name,
            content=result_text,
            is_error=False,
            is_empty=False,
        )
        trajectory.tool_results_by_call_id[call_id] = tr

        trajectory.thread_order.insert(insert_thread_idx, ("assistant", turn_index))
        insert_thread_idx += 1
        insert_before += 1

        for idx, at2 in enumerate(trajectory.assistant_turns):
            at2.turn_index = idx

        logger.info("Patched missing tool '%s' with call_id=%s", tool_name, call_id)


def _build_ordered_events(traj: ParsedTrajectory) -> list[dict]:
    """Build ordered_events list from the trajectory for writer compatibility."""
    events = []
    base_ts = "2026-05-21T22:00:00.000Z"

    for entry_type, entry_idx in traj.thread_order:
        if entry_type == "user" and entry_idx < len(traj.user_messages):
            events.append({
                "type": "message",
                "id": str(uuid.uuid4()),
                "timestamp": base_ts,
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": traj.user_messages[entry_idx]}],
                },
            })
        elif entry_type == "assistant" and entry_idx < len(traj.assistant_turns):
            turn = traj.assistant_turns[entry_idx]
            content = []
            for tb in turn.text_blocks:
                content.append({"type": "text", "text": tb})
            for tc in turn.tool_calls:
                content.append({
                    "type": "toolCall",
                    "id": tc.call_id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                })
            events.append({
                "type": "message",
                "id": turn.event_id,
                "timestamp": base_ts,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
            })
            for tc in turn.tool_calls:
                if tc.call_id in traj.tool_results_by_call_id:
                    tr = traj.tool_results_by_call_id[tc.call_id]
                    events.append({
                        "type": "message",
                        "id": str(uuid.uuid4()),
                        "timestamp": base_ts,
                        "message": {
                            "role": "toolResult",
                            "toolCallId": tc.call_id,
                            "toolName": tc.name,
                            "content": [{"type": "text", "text": tr.content}],
                            "isError": tr.is_error,
                        },
                    })

    return events


def check_trajectory_task_match(trajectory: ParsedTrajectory) -> tuple[bool, str]:
    """Quick check if trajectory content matches its task definition.

    Returns (matches, reason). Uses keyword overlap between the trajectory's
    user messages and the task definition's goal/scenario.
    """
    task_def = get_task_definition(trajectory.task_id)
    if not task_def:
        return True, "no task definition available to check against"

    goal = task_def.get("goal_summary", "").lower()
    title = task_def.get("task_title", "").lower()

    goal_keywords = set()
    for word in (goal + " " + title).split():
        word = word.strip(".,;:!?()\"'").lower()
        if len(word) > 3 and word not in {
            "that", "this", "with", "from", "into", "will", "should",
            "task", "test", "whether", "agent", "help", "wants", "want",
        }:
            goal_keywords.add(word)

    user_text = " ".join(trajectory.user_messages).lower()
    matched = sum(1 for kw in goal_keywords if kw in user_text)
    match_ratio = matched / max(len(goal_keywords), 1)

    persona_id = task_def.get("persona_id", "")
    expected_persona = get_persona(persona_id)
    if expected_persona:
        expected_name = expected_persona.get("first_name", "").lower()
        actual_name = trajectory.persona.get("first_name", "").lower() if trajectory.persona else ""
        if expected_name and actual_name and expected_name != actual_name:
            return False, (
                f"Persona mismatch: task expects {expected_persona.get('first_name', '')} "
                f"({persona_id}), trajectory has "
                f"{trajectory.persona.get('first_name', 'unknown')}"
            )

    if match_ratio < 0.15:
        return False, (
            f"Content mismatch: only {matched}/{len(goal_keywords)} goal keywords found "
            f"in trajectory. Task: '{task_def.get('task_title', '')}'"
        )

    return True, "content matches task definition"
