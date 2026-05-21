"""Stage 2: Multi-engine PII classification — Presidio, Google DLP, OpenAI Filter, Claude LLM, Pattern."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path

import anthropic

from config import (
    ANTHROPIC_API_KEY,
    CLASSIFIER_MODEL,
    DATA_DIR,
    ENABLE_GOOGLE_DLP,
    ENABLE_LLM_CLASSIFIER,
    ENABLE_OPENAI_FILTER,
    ENABLE_PRESIDIO,
    OPENAI_API_KEY,
)
from models import ParsedTrajectory, PIIEntity, PIIMap
from token_tracker import tracker

logger = logging.getLogger(__name__)

_classification_map: dict[str, str] | None = None


def _load_classification_map() -> dict[str, str]:
    global _classification_map
    if _classification_map is None:
        _classification_map = json.loads((DATA_DIR / "classification.json").read_text())
    return _classification_map


# ---------------------------------------------------------------------------
# Presidio (local NER)
# ---------------------------------------------------------------------------

PRESIDIO_TO_LABEL = {
    "PERSON": "ID_FULL_NAME",
    "EMAIL_ADDRESS": "ID_EMAIL",
    "PHONE_NUMBER": "ID_PHONE",
    "US_SSN": "GOV_SSN_FULL",
    "CREDIT_CARD": "FIN_PAN_FULL",
    "US_DRIVER_LICENSE": "GOV_DL_NUM",
    "US_PASSPORT": "GOV_PASSPORT_NUM",
    "IP_ADDRESS": "DEV_IP_ADDR",
    "US_BANK_NUMBER": "FIN_BANK_ACCT",
    "IBAN_CODE": "FIN_IBAN",
    "DATE_TIME": None,  # skip raw dates
    "NRP": None,  # nationality/religion/political - too noisy
    "LOCATION": "LOC_STREET_ADDR",
    "MEDICAL_LICENSE": "HEALTH_INSURANCE_ID",
    "URL": None,
}


def _run_presidio(text: str) -> list[PIIEntity]:
    """Run Microsoft Presidio local NER."""
    if not ENABLE_PRESIDIO:
        return []
    try:
        from presidio_analyzer import AnalyzerEngine
        analyzer = AnalyzerEngine()
        results = analyzer.analyze(text=text, language="en")
    except ImportError:
        logger.warning("presidio_analyzer not installed; skipping Presidio engine")
        return []
    except Exception as e:
        logger.warning("Presidio failed: %s", e)
        return []

    classification = _load_classification_map()
    entities = []
    for r in results:
        label = PRESIDIO_TO_LABEL.get(r.entity_type)
        if not label:
            continue
        level = classification.get(label, "L2")
        entities.append(PIIEntity(
            text=text[r.start:r.end],
            label=label,
            level=level,
            start=r.start,
            end=r.end,
            engines=["presidio"],
            confidence=r.score,
        ))
    return entities


# ---------------------------------------------------------------------------
# Google Cloud DLP
# ---------------------------------------------------------------------------

DLP_TO_LABEL = {
    "PERSON_NAME": "ID_FULL_NAME",
    "EMAIL_ADDRESS": "ID_EMAIL",
    "PHONE_NUMBER": "ID_PHONE",
    "US_SOCIAL_SECURITY_NUMBER": "GOV_SSN_FULL",
    "CREDIT_CARD_NUMBER": "FIN_PAN_FULL",
    "US_DRIVERS_LICENSE_NUMBER": "GOV_DL_NUM",
    "US_PASSPORT": "GOV_PASSPORT_NUM",
    "IBAN_CODE": "FIN_IBAN",
    "STREET_ADDRESS": "LOC_STREET_ADDR",
    "IP_ADDRESS": "DEV_IP_ADDR",
    "DATE_OF_BIRTH": "ID_DOB",
    "MEDICAL_RECORD_NUMBER": "HEALTH_INSURANCE_ID",
    "US_BANK_ROUTING_MICR": "FIN_ROUTING_ALONE",
    "PASSPORT": "GOV_PASSPORT_NUM",
    "NATIONAL_ID_NUMBER": "GOV_NATIONAL_ID",
}


async def _run_google_dlp(text: str) -> list[PIIEntity]:
    """Run Google Cloud DLP inspection."""
    if not ENABLE_GOOGLE_DLP:
        return []
    try:
        from google.cloud import dlp_v2
    except ImportError:
        logger.warning("google-cloud-dlp not installed; skipping DLP engine")
        return []

    try:
        client = dlp_v2.DlpServiceClient()
        project = client.common_project_path("[PROJECT]")  # auto-detected from credentials

        item = dlp_v2.ContentItem(value=text)
        inspect_config = dlp_v2.InspectConfig(
            info_types=[dlp_v2.InfoType(name=n) for n in DLP_TO_LABEL.keys()],
            min_likelihood=dlp_v2.Likelihood.POSSIBLE,
            include_quote=True,
        )

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.inspect_content(
                request={"parent": project, "item": item, "inspect_config": inspect_config}
            )
        )
    except Exception as e:
        logger.warning("Google DLP failed: %s", e)
        return []

    classification = _load_classification_map()
    entities = []
    for finding in response.result.findings:
        label = DLP_TO_LABEL.get(finding.info_type.name)
        if not label:
            continue
        level = classification.get(label, "L2")
        quote = finding.quote or ""
        entities.append(PIIEntity(
            text=quote,
            label=label,
            level=level,
            start=finding.location.byte_range.start if finding.location else -1,
            end=finding.location.byte_range.end if finding.location else -1,
            engines=["google_dlp"],
            confidence=finding.likelihood / 5.0,
        ))
    return entities


# ---------------------------------------------------------------------------
# OpenAI Privacy Filter
# ---------------------------------------------------------------------------

async def _run_openai_filter(text: str) -> list[PIIEntity]:
    """Run OpenAI Privacy Filter for English PII."""
    if not ENABLE_OPENAI_FILTER or not OPENAI_API_KEY:
        return []
    try:
        import openai
        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await client.moderations.create(
            input=text,
            model="omni-moderation-latest",
        )
    except Exception as e:
        logger.warning("OpenAI filter failed: %s", e)
        return []

    # OpenAI moderation doesn't directly return PII entities in the same way,
    # but can flag sensitive content. We use it as a supplementary signal.
    # For now, return empty — actual PII extraction relies on Presidio + DLP + LLM.
    return []


# ---------------------------------------------------------------------------
# Claude LLM Classifier (fine-grained taxonomy from Classification.md)
# ---------------------------------------------------------------------------

_CLASSIFIER_PROMPT = """You are a highly accurate PII classifier for the OpenClaw Privacy pipeline, aligned with the Hatch Trust Guidelines (HTG) 5-level data taxonomy.

For each PII item found, output a JSON array of objects:
{{"text": "<exact text found in the input>", "label": "<LABEL_FROM_TAXONOMY>", "level": "<L0|L1|L2|L3|L4>"}}

## TAXONOMY (label → level):
{taxonomy}

## CRITICAL CLASSIFICATION RULES (aligned with HTG + cuarena-pipeline):

1. **Be exhaustive** — find EVERY piece of PII in the text
2. **Exact text only** — the "text" field must be a verbatim substring from the input
3. **Most specific label** — if text matches multiple labels, use the most specific
4. **References vs. Values**:
   - A LOG ENTRY like "API key saved to TOOLS.md" is METADATA (L1), NOT the credential itself
   - An ACTUAL credential value like "sk-ant-api03-..." is AUTH_API_KEY (L4)
   - "Maton API key" as a reference/mention = L1 metadata. The actual key string = L4.
5. **Session tokens are L1** (not L3): short-lived JWTs, session cookies, bearer tokens with short TTL = AUTH_SESSION_TOKEN (L1). Only long-lived refresh tokens = AUTH_OAUTH_REFRESH (L4).
6. **Health data precision**:
   - Diagnosis names (GAD, ADHD, diabetes, anxiety) = HEALTH_MENTAL_HEALTH or HEALTH_DIAGNOSIS (L3)
   - Insurance ID numbers (UHC-883-...) = HEALTH_INSURANCE_ID (L2)
   - Vital signs (65kg, BMI, heart rate) = HEALTH_VITAL (L1)
7. **Names**: Full names = ID_FULL_NAME (L2). First name alone in context = ID_FULL_NAME (L2) only if clearly a person's name
8. **Locations**: City names = LOC_GPS_CITY (L1). Street addresses = LOC_STREET_ADDR (L2). Home address = LOC_HOME_ADDR (L3). Country/region only = LOC_COUNTRY/LOC_REGION (L0).
9. **Behavioral data**: Aggregated habits (e.g., "trains 3x/week") = BEHAV_AGGREGATE (L1). NOT L2 unless tied to purchase/browse history.
10. **Cuisine/nationality words** (Korean, Japanese, Italian) in food context = PUB_GENERAL_KNOWLEDGE (L0), NOT DEMO_ETHNICITY (L3). Only classify as DEMO_ETHNICITY when explicitly describing a person's racial/ethnic origin tied to their identity.
11. **Gender** (male/female) tied to a specific person = DEMO_GENDER (L2). Generic category references ("women's health") = PUB_GENERAL_KNOWLEDGE (L0).

{persona_context}

Output ONLY the JSON array. No explanation, no markdown."""


async def _run_llm_classifier(text: str, persona: dict | None = None) -> list[PIIEntity]:
    """Run Claude Opus 4.6 fine-grained classification using full taxonomy + persona context."""
    if not ENABLE_LLM_CLASSIFIER or not ANTHROPIC_API_KEY:
        return []

    classification = _load_classification_map()
    taxonomy_str = "\n".join(f"  {label}: {level}" for label, level in classification.items())

    # Build persona context for better accuracy
    persona_context = ""
    if persona:
        pii_vault = persona.get("pii_vault", {})
        known_pii = []
        first = persona.get("first_name", "")
        last = persona.get("last_name", "")
        if first and last:
            known_pii.append(f"- Full name: {first} {last} (ID_FULL_NAME, L2)")
        health = pii_vault.get("health", {})
        if health.get("insurance_id"):
            known_pii.append(f"- Insurance ID: {health['insurance_id']} (HEALTH_INSURANCE_ID, L2)")
        for d in health.get("diagnoses", []):
            known_pii.append(f"- Diagnosis: {d} (HEALTH_MENTAL_HEALTH, L3)")
        gov = pii_vault.get("government", {})
        if gov.get("ssn"):
            known_pii.append(f"- SSN: {gov['ssn']} (GOV_SSN_FULL, L4)")
        contact = pii_vault.get("contact", {})
        if contact.get("email"):
            known_pii.append(f"- Email: {contact['email']} (ID_EMAIL, L2)")
        if contact.get("phone"):
            known_pii.append(f"- Phone: {contact['phone']} (ID_PHONE, L2)")

        if known_pii:
            persona_context = "## KNOWN PII FOR THIS PERSONA (if these appear in text, classify them correctly):\n" + "\n".join(known_pii)

    system_prompt = _CLASSIFIER_PROMPT.format(
        taxonomy=taxonomy_str,
        persona_context=persona_context,
    )

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    try:
        response = await client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Classify ALL PII in this text:\n\n{text[:12000]}"}],
        )
        tracker.record_anthropic(response, "classifier")
    except Exception as e:
        logger.warning("LLM classifier failed: %s", e)
        return []

    # Parse JSON response
    resp_text = response.content[0].text.strip()
    try:
        import json_repair
        items = json_repair.loads(resp_text)
    except Exception:
        try:
            items = json.loads(resp_text)
        except json.JSONDecodeError:
            logger.warning("LLM classifier returned unparseable response")
            return []

    if not isinstance(items, list):
        return []

    entities = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text_val = item.get("text", "")
        label = item.get("label", "")
        level = item.get("level", classification.get(label, "L2"))

        # Post-process: downgrade log references that aren't actual secret values
        if label == "AUTH_API_KEY" and level == "L4":
            if "saved to" in text_val or "stored in" in text_val or len(text_val) > 50:
                level = "L1"
                label = "COMM_TIMESTAMP"

        # Post-process: cuisine/nationality words are NOT ethnicity identifiers
        # "Korean food", "Korean pancakes", "Japanese", "Asian" in food context = L0/L1
        if label in ("DEMO_ETHNICITY", "DEMO_RACE") and text_val.lower() in (
            "korean", "japanese", "chinese", "asian", "thai", "italian",
            "french", "mexican", "indian", "vietnamese", "american",
        ):
            level = "L0"
            label = "PUB_GENERAL_KNOWLEDGE"

        # Gender is L2 per official rules — only downgrade truly generic/aggregate usage
        # (e.g., "women's health tips" as a category, not tied to a person)
        if label == "DEMO_GENDER" and text_val.lower() in ("women", "men"):
            level = "L0"
            label = "PUB_GENERAL_KNOWLEDGE"

        # Enforce taxonomy level (don't let LLM override the classification map)
        canonical_level = classification.get(label)
        if canonical_level:
            level = canonical_level

        entities.append(PIIEntity(
            text=text_val,
            label=label,
            level=level,
            engines=["llm_classifier"],
            confidence=0.9,
        ))
    return entities


# ---------------------------------------------------------------------------
# Pattern-Based Fallback (always runs — catches emails, names from persona)
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+\.[\w.-]+')
_PHONE_RE = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
_SSN_RE = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
_INSURANCE_RE = re.compile(r'\b(?:UHC|BCBS|Aetna|Cigna|Kaiser|Humana)[\s-]?\w*[\s-]?\d[\d-]{5,}\b', re.IGNORECASE)


def _run_pattern_classifier(text: str, persona: dict | None = None) -> list[PIIEntity]:
    """Fast regex + persona-aware PII detection. Always runs as fallback."""
    entities = []

    # Emails
    for m in _EMAIL_RE.finditer(text):
        entities.append(PIIEntity(
            text=m.group(), label="ID_EMAIL", level="L2",
            start=m.start(), end=m.end(), engines=["pattern"], confidence=0.95,
        ))

    # Phone numbers
    for m in _PHONE_RE.finditer(text):
        entities.append(PIIEntity(
            text=m.group(), label="ID_PHONE", level="L2",
            start=m.start(), end=m.end(), engines=["pattern"], confidence=0.85,
        ))

    # SSNs
    for m in _SSN_RE.finditer(text):
        entities.append(PIIEntity(
            text=m.group(), label="GOV_SSN_FULL", level="L4",
            start=m.start(), end=m.end(), engines=["pattern"], confidence=0.95,
        ))

    # Insurance IDs
    for m in _INSURANCE_RE.finditer(text):
        entities.append(PIIEntity(
            text=m.group(), label="HEALTH_INSURANCE_ID", level="L2",
            start=m.start(), end=m.end(), engines=["pattern"], confidence=0.85,
        ))

    # Persona name detection (both from persona config and from content heuristics)
    if persona:
        first = persona.get("first_name", "")
        last = persona.get("last_name", "")
        full_name = f"{first} {last}".strip()

        for name in [full_name, first]:
            if name and len(name) > 2:
                idx = text.find(name)
                while idx >= 0:
                    entities.append(PIIEntity(
                        text=name, label="ID_FULL_NAME", level="L2",
                        start=idx, end=idx + len(name), engines=["pattern_persona"], confidence=0.99,
                    ))
                    idx = text.find(name, idx + 1)

    # Detect names that appear near "About X" or "for X" patterns in memory content
    # Only match 2-3 word names (First Last or First Middle Last)
    about_pattern = re.compile(r'(?:About|Built .* for|plan for)\s+([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,}){1,2})\b')
    # Common false positives to exclude
    _FP_NAMES = frozenset({
        'south korea', 'north america', 'new york', 'los angeles', 'san francisco',
        'united states', 'google doc', 'google docs', 'olive young', 'dance fitness',
        'clinical research', 'your situation', 'common items', 'seoul station',
        'sunday shop', 'active project', 'session log', 'setup notes',
        'term memory', 'majang meat', 'mangwon market', 'gwangjang market',
    })
    for m in about_pattern.finditer(text):
        candidate = m.group(1)
        if candidate.lower() not in _FP_NAMES and len(candidate.split()) <= 3:
            entities.append(PIIEntity(
                text=candidate, label="ID_FULL_NAME", level="L2",
                start=m.start(1), end=m.end(1), engines=["pattern_context"], confidence=0.85,
            ))

        # Also check PII vault for L3/L4 items
        vault = persona.get("pii_vault", {})
        health = vault.get("health", {})
        if health.get("insurance_id"):
            ins_id = health["insurance_id"]
            if ins_id in text:
                entities.append(PIIEntity(
                    text=ins_id, label="HEALTH_INSURANCE_ID", level="L2",
                    engines=["pattern_persona"], confidence=0.99,
                ))
        for diag in health.get("diagnoses", []):
            if diag.lower() in text.lower():
                entities.append(PIIEntity(
                    text=diag, label="HEALTH_MENTAL_HEALTH", level="L3",
                    engines=["pattern_persona"], confidence=0.95,
                ))

        gov = vault.get("government", {})
        if gov.get("ssn") and gov["ssn"] in text:
            entities.append(PIIEntity(
                text=gov["ssn"], label="GOV_SSN_FULL", level="L4",
                engines=["pattern_persona"], confidence=0.99,
            ))

    return entities


# ---------------------------------------------------------------------------
# Merge & Deduplicate
# ---------------------------------------------------------------------------

_LEVEL_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "BLOCK": 5}


def _merge_entities(all_entities: list[PIIEntity]) -> list[PIIEntity]:
    """Merge detections from multiple engines, deduplicate by text+label."""
    merged: dict[tuple[str, str], PIIEntity] = {}

    for entity in all_entities:
        key = (entity.text.lower().strip(), entity.label)
        if key in merged:
            existing = merged[key]
            existing.engines = list(set(existing.engines + entity.engines))
            existing.confidence = max(existing.confidence, entity.confidence)
            # Promote to higher level if found by multiple engines
            if _LEVEL_ORDER.get(entity.level, 0) > _LEVEL_ORDER.get(existing.level, 0):
                existing.level = entity.level
        else:
            merged[key] = PIIEntity(
                text=entity.text,
                label=entity.label,
                level=entity.level,
                start=entity.start,
                end=entity.end,
                engines=list(entity.engines),
                confidence=entity.confidence,
            )

    return list(merged.values())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _classify_source(entity: PIIEntity, turn_index: int, source_type: str) -> PIIEntity:
    """Stamp provenance fields onto an entity."""
    entity.source_turn_index = turn_index
    entity.source_type = source_type
    return entity


def _determine_source_type(tool_name: str | None, is_user_msg: bool) -> str:
    """Map where text originated to a provenance category."""
    if is_user_msg:
        return "user_input"
    if not tool_name:
        return "assistant_text"
    memory_read_tools = {"memory_search", "rag_search", "vault_get", "vault_list"}
    if tool_name in memory_read_tools:
        return "memory_read"
    return "tool_result"


async def classify_trajectory(trajectory: ParsedTrajectory) -> PIIMap:
    """Run multi-engine PII detection with per-argument provenance tracking.

    Each detected entity gets stamped with:
    - source_turn_index: which turn introduced this PII
    - source_type: "user_input" | "tool_result" | "memory_read" | "assistant_text"
    """
    # Build text segments with provenance metadata
    segments: list[tuple[str, int, str]] = []  # (text, turn_index, source_type)

    for i, msg in enumerate(trajectory.user_messages):
        segments.append((msg, i, "user_input"))

    for turn in trajectory.assistant_turns:
        for tb in turn.text_blocks:
            segments.append((tb, turn.turn_index, "assistant_text"))

        for tc in turn.tool_calls:
            for arg_key, arg_val in tc.arguments.items():
                arg_text = arg_val if isinstance(arg_val, str) else json.dumps(arg_val)
                if arg_text and len(arg_text) > 3:
                    src = _determine_source_type(tc.name, False)
                    segments.append((arg_text, turn.turn_index, src))

        for tc in turn.tool_calls:
            if tc.call_id in trajectory.tool_results_by_call_id:
                tr = trajectory.tool_results_by_call_id[tc.call_id]
                if tr.content:
                    src = _determine_source_type(tc.name, False)
                    segments.append((tr.content[:4000], turn.turn_index, src))

    # Concatenate for bulk classification (engines work best on large text)
    full_text = "\n---\n".join(seg[0] for seg in segments)

    # Run engines in parallel
    presidio_task = asyncio.get_event_loop().run_in_executor(None, _run_presidio, full_text)
    dlp_task = _run_google_dlp(full_text)
    oai_task = _run_openai_filter(full_text)
    llm_task = _run_llm_classifier(full_text, trajectory.persona)

    results = await asyncio.gather(presidio_task, dlp_task, oai_task, llm_task, return_exceptions=True)

    all_entities: list[PIIEntity] = []
    for r in results:
        if isinstance(r, list):
            all_entities.extend(r)
        elif isinstance(r, Exception):
            logger.warning("Engine failed: %s", r)

    pattern_entities = _run_pattern_classifier(full_text, trajectory.persona)
    all_entities.extend(pattern_entities)

    merged = _merge_entities(all_entities)

    # Stamp provenance: for each merged entity, find its earliest source segment
    for entity in merged:
        entity_lower = entity.text.lower()
        earliest_turn = 999
        earliest_source = "history"
        for seg_text, seg_turn, seg_source in segments:
            if entity_lower in seg_text.lower() and seg_turn < earliest_turn:
                earliest_turn = seg_turn
                earliest_source = seg_source
        if earliest_turn < 999:
            entity.source_turn_index = earliest_turn
            entity.source_type = earliest_source
        else:
            entity.source_turn_index = -1
            entity.source_type = "history"

    # Compute summary
    max_level = "L0"
    labels_present = list(set(e.label for e in merged))
    for e in merged:
        if _LEVEL_ORDER.get(e.level, 0) > _LEVEL_ORDER.get(max_level, 0):
            max_level = e.level

    return PIIMap(
        entities=merged,
        max_level=max_level,
        has_l4=any(e.level == "L4" for e in merged),
        has_l3=any(e.level == "L3" for e in merged),
        labels_present=labels_present,
    )
