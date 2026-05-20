"""Stage 2: Multi-engine PII classification — Presidio, Google DLP, OpenAI Filter, Claude LLM."""
from __future__ import annotations

import asyncio
import json
import logging
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

_CLASSIFIER_PROMPT = """You are a PII classifier. Analyze the text and identify ALL personally identifiable information.

For each PII item found, output a JSON array of objects:
{{"text": "<exact text>", "label": "<LABEL_FROM_TAXONOMY>", "level": "<L0|L1|L2|L3|L4>"}}

Use ONLY labels from this taxonomy:
{taxonomy}

Rules:
- Be exhaustive — find EVERY piece of PII
- If the same text matches multiple labels, use the most specific one
- Include names, emails, phone numbers, addresses, financial data, health info, credentials
- Only output the JSON array, nothing else
- If no PII found, output []"""


async def _run_llm_classifier(text: str) -> list[PIIEntity]:
    """Run Claude Opus 4.6 fine-grained classification using full taxonomy."""
    if not ENABLE_LLM_CLASSIFIER or not ANTHROPIC_API_KEY:
        return []

    classification = _load_classification_map()
    taxonomy_str = "\n".join(f"  {label}: {level}" for label, level in classification.items())

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    try:
        response = await client.messages.create(
            model=CLASSIFIER_MODEL,
            max_tokens=4096,
            system=_CLASSIFIER_PROMPT.format(taxonomy=taxonomy_str),
            messages=[{"role": "user", "content": f"Classify PII in this text:\n\n{text[:8000]}"}],
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
        label = item.get("label", "")
        level = item.get("level", classification.get(label, "L2"))
        entities.append(PIIEntity(
            text=item.get("text", ""),
            label=label,
            level=level,
            engines=["llm_classifier"],
            confidence=0.9,
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

async def classify_trajectory(trajectory: ParsedTrajectory) -> PIIMap:
    """Run multi-engine PII detection on a full trajectory."""
    all_text_blocks = []

    # Gather all text from user messages and assistant turns
    for msg in trajectory.user_messages:
        all_text_blocks.append(msg)
    for turn in trajectory.assistant_turns:
        for tb in turn.text_blocks:
            all_text_blocks.append(tb)
        for tc in turn.tool_calls:
            # Check tool arguments too
            args_str = json.dumps(tc.arguments) if tc.arguments else ""
            if args_str:
                all_text_blocks.append(args_str)

    full_text = "\n---\n".join(all_text_blocks)

    # Run engines in parallel
    presidio_task = asyncio.get_event_loop().run_in_executor(None, _run_presidio, full_text)
    dlp_task = _run_google_dlp(full_text)
    oai_task = _run_openai_filter(full_text)
    llm_task = _run_llm_classifier(full_text)

    results = await asyncio.gather(presidio_task, dlp_task, oai_task, llm_task, return_exceptions=True)

    all_entities: list[PIIEntity] = []
    for r in results:
        if isinstance(r, list):
            all_entities.extend(r)
        elif isinstance(r, Exception):
            logger.warning("Engine failed: %s", r)

    merged = _merge_entities(all_entities)

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
