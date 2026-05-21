"""Privacy Registry — deterministic rule engine for HTG-compliant trajectory rewriting.

This module makes ALL privacy decisions programmatically (no LLM involved).
For each turn it outputs an Action that tells the assembler exactly what to do.
The LLM is only used downstream to generate natural-sounding text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from config import DATA_DIR
from models import AssistantTurn, ParsedTrajectory, PIIEntity, PIIMap, ToolCall


class Action(str, Enum):
    ALLOW = "allow"
    MINIMIZE = "minimize"
    CONSENT_GATE = "consent_gate"
    HARD_DENY = "hard_deny"
    T3_BLOCK = "t3_block"


@dataclass
class TurnDecision:
    """What the registry decided for a single turn."""
    turn_index: int
    action: Action
    reason: str
    data_level: str = "L0"
    tool_tier: str = "T1"
    target_path: str = ""
    pii_in_content: list[PIIEntity] = field(default_factory=list)
    fields_to_strip: list[str] = field(default_factory=list)
    is_read_only: bool = False
    consent_path: str = ""


_LEVEL_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "BLOCK": 5}

_PERSISTENCE_PATTERNS = (
    "/memory/",
    "/notes/",
    "memory.md",
    "-notes.md",
)

_tool_tiers: dict[str, str] | None = None


def _load_tool_tiers() -> dict[str, str]:
    global _tool_tiers
    if _tool_tiers is None:
        data = json.loads((DATA_DIR / "tool_tiers.json").read_text())
        _tool_tiers = {}
        for tier, tools in data.items():
            for t in tools:
                _tool_tiers[t] = tier
    return _tool_tiers


def get_tool_tier(tool_name: str) -> str:
    tiers = _load_tool_tiers()
    return tiers.get(tool_name, "T1")


def is_persistence_path(path: str) -> bool:
    """Check if a path targets a persistence location (memory/notes)."""
    lower = path.lower()
    return any(p in lower for p in _PERSISTENCE_PATTERNS)


def is_persistence_tool(tool_name: str) -> bool:
    return tool_name in ("memory_write", "active_memory_write", "active_memory_set", "wiki_apply")


def _max_pii_level_in_content(content: str, pii_entities: list[PIIEntity]) -> tuple[str, list[PIIEntity]]:
    """Find the max PII level present in a string and which entities matched."""
    matched = []
    for e in pii_entities:
        if e.text.lower() in content.lower():
            matched.append(e)
    if not matched:
        return "L0", []
    max_level = max(matched, key=lambda e: _LEVEL_ORDER.get(e.level, 0))
    return max_level.level, matched


def _get_write_content(tc: ToolCall) -> str:
    """Extract the content that would be written by a tool call."""
    args = tc.arguments
    content = args.get("content", "")
    if not content:
        edits = args.get("edits", [])
        if edits:
            content = " ".join(e.get("newText", "") for e in edits if isinstance(e, dict))
    if not content:
        content = args.get("command", "")
    return content


class PrivacyRegistry:
    """Deterministic privacy decision engine.

    For each assistant turn, evaluates:
    1. What tools are being called
    2. What paths they target
    3. What PII level is in the content
    4. What gate applies per the HTG authorization matrix

    Returns a TurnDecision with the exact action to take.
    """

    def __init__(self, pii_map: PIIMap, trajectory: ParsedTrajectory):
        self.pii_map = pii_map
        self.trajectory = trajectory
        self._consent_granted: dict[str, str] = {}

    def record_consent(self, path: str, decision: str) -> None:
        """Record that consent was obtained for a path."""
        self._consent_granted[path] = decision

    def has_consent(self, path: str) -> bool:
        """Check if consent was already obtained for this path."""
        normalized = path.lower()
        for granted_path in self._consent_granted:
            if granted_path.lower() in normalized or normalized in granted_path.lower():
                return True
        return False

    def decide(self, turn: AssistantTurn) -> TurnDecision:
        """Evaluate a turn and return the deterministic privacy decision."""

        if not turn.tool_calls:
            return TurnDecision(
                turn_index=turn.turn_index,
                action=Action.ALLOW,
                reason="text-only turn, no tool calls",
                is_read_only=True,
            )

        # Find the most sensitive operation in this turn
        highest_action = Action.ALLOW
        decision = TurnDecision(turn_index=turn.turn_index, action=Action.ALLOW, reason="")

        for tc in turn.tool_calls:
            tier = get_tool_tier(tc.name)
            path = tc.arguments.get("path", "") or tc.arguments.get("file_path", "") or ""
            is_read = tc.name in ("read", "memory_search", "rag_search", "vault_get", "vault_list")
            is_persist_tool = is_persistence_tool(tc.name)
            is_persist_path = is_persistence_path(path)
            write_content = _get_write_content(tc)

            # Determine effective PII level in THIS tool call's content
            content_level, matched_pii = _max_pii_level_in_content(
                write_content, self.pii_map.entities
            )

            # Also check path arguments and text for PII
            if not matched_pii and path:
                content_level_path, matched_pii_path = _max_pii_level_in_content(
                    path, self.pii_map.entities
                )
                if _LEVEL_ORDER.get(content_level_path, 0) > _LEVEL_ORDER.get(content_level, 0):
                    content_level = content_level_path
                    matched_pii = matched_pii_path

            # Read-only operations: always ALLOW
            if is_read:
                continue

            # Gate 4a: T1_PERSISTENCE + L>=3 → HARD DENY
            if (is_persist_tool or is_persist_path) and _LEVEL_ORDER.get(content_level, 0) >= 3:
                return TurnDecision(
                    turn_index=turn.turn_index,
                    action=Action.HARD_DENY,
                    reason=f"Gate 4a: {content_level} data to persistence path ({path or tc.name})",
                    data_level=content_level,
                    tool_tier="T1_PERSISTENCE",
                    target_path=path or tc.name,
                    pii_in_content=matched_pii,
                )

            # Gate 4b/4c: T2+L3 or T3+L2 → need consent (but we don't handle these for now)
            if tier == "T3" and _LEVEL_ORDER.get(content_level, 0) >= 2:
                return TurnDecision(
                    turn_index=turn.turn_index,
                    action=Action.T3_BLOCK,
                    reason=f"Gate 4c: {content_level} data to T3 tool ({tc.name})",
                    data_level=content_level,
                    tool_tier="T3",
                    target_path=tc.name,
                    pii_in_content=matched_pii,
                )

            # L2 data to persistence → consent gate (if not already granted)
            if (is_persist_tool or is_persist_path) and _LEVEL_ORDER.get(content_level, 0) >= 2:
                consent_key = path or tc.name
                if self.has_consent(consent_key):
                    # Consent already obtained — just minimize
                    action = Action.MINIMIZE
                    reason = f"L2 to persistence ({consent_key}) — consent already granted, minimizing"
                else:
                    action = Action.CONSENT_GATE
                    reason = f"Gate: L2 data to persistence path ({consent_key})"

                if _LEVEL_ORDER.get(action.value if isinstance(action, Action) else "", 0) == 0:
                    pass
                # Compare severity
                action_order = {Action.ALLOW: 0, Action.MINIMIZE: 1, Action.CONSENT_GATE: 2, Action.T3_BLOCK: 3, Action.HARD_DENY: 4}
                if action_order.get(action, 0) > action_order.get(highest_action, 0):
                    highest_action = action
                    decision = TurnDecision(
                        turn_index=turn.turn_index,
                        action=action,
                        reason=reason,
                        data_level=content_level,
                        tool_tier=tier,
                        target_path=consent_key,
                        pii_in_content=matched_pii,
                        consent_path=consent_key,
                    )
                continue

            # Non-persistence writes with PII → minimize
            if write_content and _LEVEL_ORDER.get(content_level, 0) >= 2:
                action = Action.MINIMIZE
                reason = f"L2 data in non-persistence write — minimize PII fields"
                action_order = {Action.ALLOW: 0, Action.MINIMIZE: 1, Action.CONSENT_GATE: 2, Action.T3_BLOCK: 3, Action.HARD_DENY: 4}
                if action_order.get(action, 0) > action_order.get(highest_action, 0):
                    highest_action = action
                    decision = TurnDecision(
                        turn_index=turn.turn_index,
                        action=action,
                        reason=reason,
                        data_level=content_level,
                        tool_tier=tier,
                        target_path=path,
                        pii_in_content=matched_pii,
                    )

        if decision.action == Action.ALLOW and decision.reason == "":
            decision.reason = "All operations within allowed bounds"
            decision.is_read_only = all(
                tc.name in ("read", "memory_search", "rag_search", "vault_get")
                for tc in turn.tool_calls
            )

        return decision

    def decide_all(self, turns: list[AssistantTurn]) -> list[TurnDecision]:
        """Decide actions for all turns, tracking consent state across them."""
        decisions = []
        for turn in turns:
            d = self.decide(turn)
            decisions.append(d)
            # After a consent gate is decided, mark it as granted for subsequent turns
            # (the actual consent response will be generated by the assembler)
            if d.action == Action.CONSENT_GATE and d.consent_path:
                self.record_consent(d.consent_path, "granted")
        return decisions
