"""Task context loader — enriches trajectories with task definitions and persona data.

Loads from:
  - tasks_all.csv   — 2800+ task definitions with goals, rubrics, expected actions
  - privacy-personas.json — 142 personas with full PII vaults and personality traits

Task IDs in the CSV use P- prefix (P-033-02), pipeline uses T- prefix (T-033-02).
"""
from __future__ import annotations

import csv
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent

_TASKS_CSV_PATHS = [
    _ROOT / "privacy-task-generator" / "outputs" / "tasks_all.csv",
    _ROOT / "privacy-task-generator" / "outputs" / "chat_batch_v2" / "tasks_all.csv",
]
_PERSONAS_PATH = _ROOT / "privacy-personas.json"


@lru_cache(maxsize=1)
def _load_tasks() -> dict[str, dict]:
    """Load all task definitions, keyed by both P- and T- prefixed IDs."""
    tasks: dict[str, dict] = {}
    for csv_path in _TASKS_CSV_PATHS:
        if not csv_path.exists():
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tid = row.get("task_id", "").strip()
                if not tid:
                    continue
                tasks[tid] = row
                alt = tid.replace("P-", "T-", 1)
                tasks[alt] = row
    logger.info("Loaded %d task definitions", len(tasks) // 2)
    return tasks


@lru_cache(maxsize=1)
def _load_personas() -> dict[str, dict]:
    """Load all personas, keyed by persona_id (P-001, P-033, etc.)."""
    if not _PERSONAS_PATH.exists():
        logger.warning("privacy-personas.json not found at %s", _PERSONAS_PATH)
        return {}
    with open(_PERSONAS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    personas_list = data.get("personas", data if isinstance(data, list) else [])
    result = {}
    for p in personas_list:
        pid = p.get("persona_id", "")
        if pid:
            result[pid] = p
    logger.info("Loaded %d personas", len(result))
    return result


def get_task_definition(task_id: str) -> Optional[dict]:
    """Get the task definition for a task_id (accepts both T- and P- prefix)."""
    tasks = _load_tasks()
    return tasks.get(task_id) or tasks.get(task_id.replace("T-", "P-", 1))


def get_persona(persona_id: str) -> Optional[dict]:
    """Get the full persona data for a persona_id."""
    return _load_personas().get(persona_id)


def get_persona_for_task(task_id: str) -> Optional[dict]:
    """Get the persona associated with a task."""
    task_def = get_task_definition(task_id)
    if not task_def:
        return None
    return get_persona(task_def.get("persona_id", ""))


def extract_pii_vault_entities(persona: dict) -> list[dict]:
    """Extract PII entities from a persona's vault for classifier boosting.

    Returns a list of {text, label, level} dicts that the classifier can
    use as ground-truth PII values instead of guessing.
    """
    vault = persona.get("pii_vault", {})
    if not vault:
        return []

    _LEVEL_MAP = {
        "auth": "L4", "gov": "L4", "fin": "L4",
        "health": "L3", "legal": "L3",
        "demo": "L2", "id": "L2", "emp": "L2", "fam": "L2",
    }

    entities = []
    for category, items in vault.items():
        level = _LEVEL_MAP.get(category.split("_")[0].lower(), "L2")
        if isinstance(items, dict):
            for key, val in items.items():
                if isinstance(val, str) and len(val) > 2:
                    label = f"{category.upper()}_{key.upper()}"
                    entities.append({"text": val, "label": label, "level": level})
                elif isinstance(val, list):
                    for v in val:
                        if isinstance(v, str) and len(v) > 2:
                            entities.append({
                                "text": v,
                                "label": f"{category.upper()}_{key.upper()}",
                                "level": level,
                            })
                elif isinstance(val, dict):
                    for k2, v2 in val.items():
                        if isinstance(v2, str) and len(v2) > 2:
                            entities.append({
                                "text": v2,
                                "label": f"{category.upper()}_{k2.upper()}",
                                "level": level,
                            })
        elif isinstance(items, str) and len(items) > 2:
            entities.append({
                "text": items,
                "label": category.upper(),
                "level": level,
            })

    # Add top-level identity fields
    for field_name, label, level in [
        ("first_name", "ID_FIRST_NAME", "L2"),
        ("last_name", "ID_LAST_NAME", "L2"),
        ("email_synthetic", "ID_EMAIL", "L2"),
        ("date_of_birth", "DEMO_DOB", "L2"),
        ("city", "DEMO_CITY", "L1"),
        ("job_title", "EMP_TITLE", "L2"),
        ("occupation_sector", "EMP_SECTOR", "L1"),
    ]:
        val = persona.get(field_name, "")
        if val and isinstance(val, str) and len(val) > 1:
            entities.append({"text": val, "label": label, "level": level})

    full_name = f"{persona.get('first_name', '')} {persona.get('last_name', '')}".strip()
    if full_name and len(full_name) > 3:
        entities.append({"text": full_name, "label": "ID_FULL_NAME", "level": "L2"})

    return entities


def build_verifier_rubric(task_def: dict) -> str:
    """Build a task-specific rubric block for the verifier prompt."""
    parts = [f"**Task:** {task_def.get('task_title', 'Unknown')}"]
    parts.append(f"**Goal:** {task_def.get('goal_summary', '')}")
    parts.append(f"**Privacy Scenario:** {task_def.get('privacy_scenario', '?')}")
    parts.append(f"**Data Levels:** {task_def.get('data_levels', '?')}")
    parts.append(f"**Tool Tiers:** {task_def.get('tool_tiers', '?')}")
    parts.append(f"**Expected Privacy Actions:** {task_def.get('expected_privacy_actions', '?')}")

    rubric_p = task_def.get("rubric_hints_privacy", "")
    rubric_c = task_def.get("rubric_hints_correctness", "")
    rubric_comp = task_def.get("rubric_hints_completeness", "")

    if rubric_p:
        parts.append(f"\n**Privacy Rubric:** {rubric_p}")
    if rubric_c:
        parts.append(f"**Correctness Rubric:** {rubric_c}")
    if rubric_comp:
        parts.append(f"**Completeness Rubric:** {rubric_comp}")

    return "\n".join(parts)
