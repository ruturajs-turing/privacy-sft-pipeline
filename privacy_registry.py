"""Privacy Registry — deterministic rule engine for HTG-compliant trajectory rewriting.

Implements the exact authorization procedure from the official HTG "New Rules":

    FREE_BAND (cuarena-pipeline compatible):
        (T1, read):  free up to L4
        (T1, write): free up to L2
        (T2, read):  free up to L2
        (T2, write): free up to L2
        (T3, read):  free up to L1
        (T3, write): free up to L1

    Special rule (Gate 4a):
        T1 persistence tools (memory_write, etc.) + L3/L4 → HARD DENY

    Gate 4b: T2 + L3/L4 → CONFIRM (consent required)
    Gate 4c: T3 + L2/L3/L4 → CONFIRM (consent required)
    Gate 4d: everything else → ALLOW

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
    ELICIT = "elicit"  # L3/L4 data from stale source — must re-ask user
    VAULT_REDIRECT = "vault_redirect"  # redirect L3/L4 persistence to vault


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
    remediation: str = ""  # "vault", "manual_input", "exception_md", ""
    vault_key: str = ""  # key name for vault redirect
    original_tool: str = ""  # before downgrade
    downgraded_tool: str = ""  # after downgrade


_LEVEL_ORDER = {"L0": 0, "L1": 1, "L2": 2, "L3": 3, "L4": 4, "BLOCK": 5}

# Aligned with cuarena-pipeline FREE_BAND: (tier_int, is_write) → max free level
# T1 read = L4, T1 write = L2, T2 read/write = L2, T3 read/write = L1
FREE_BAND: dict[tuple[int, bool], int] = {
    (1, False): 4, (1, True): 2,
    (2, False): 2, (2, True): 2,
    (3, False): 1, (3, True): 1,
}

_TIER_TO_INT = {"T1": 1, "T1_PERSISTENCE": 1, "T2": 2, "T3": 3}

_PERSISTENCE_PATTERNS = (
    "/memory/",
    "/notes/",
    "memory.md",
    "-notes.md",
)

_PERSISTENCE_TOOLS = frozenset({
    "memory_write", "active_memory_write", "active_memory_set", "wiki_apply",
})

_READ_TOOLS = frozenset({
    "read", "memory_search", "rag_search", "vault_get", "vault_list",
    "context_window", "context_window_introspection", "session-logs",
})

_tool_tiers: dict[str, str] | None = None
_tool_equivalences: dict[str, str] | None = None


def _load_tool_tiers() -> dict[str, str]:
    global _tool_tiers
    if _tool_tiers is None:
        data = json.loads((DATA_DIR / "tool_tiers.json").read_text())
        _tool_tiers = {}
        for tier, tools in data.items():
            for t in tools:
                _tool_tiers[t] = tier
    return _tool_tiers


def _load_tool_equivalences() -> dict[str, str]:
    """Load T3→T2/T1 tool equivalence map for tier downgrade."""
    global _tool_equivalences
    if _tool_equivalences is None:
        eq_path = DATA_DIR / "tool_equivalences.json"
        if eq_path.exists():
            _tool_equivalences = json.loads(eq_path.read_text())
        else:
            _tool_equivalences = {}
    return _tool_equivalences


def get_tool_tier(tool_name: str) -> str:
    tiers = _load_tool_tiers()
    return tiers.get(tool_name, "T1")


def get_lower_tier_equivalent(tool_name: str) -> str | None:
    """Return a lower-tier tool that can do the same job, or None."""
    equivalences = _load_tool_equivalences()
    return equivalences.get(tool_name)


def is_persistence_path(path: str) -> bool:
    """Check if a path targets a persistence location (memory/notes)."""
    lower = path.lower()
    return any(p in lower for p in _PERSISTENCE_PATTERNS)


def is_persistence_tool(tool_name: str) -> bool:
    return tool_name in _PERSISTENCE_TOOLS


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


def _evaluate_free_band(tier_str: str, is_write: bool, data_level: int, is_mem: bool) -> tuple[bool, str]:
    """Evaluate the FREE_BAND predicate aligned with cuarena-pipeline + meeting rules.

    Returns (ok, reason_if_not_ok).

    Key rule: L3/L4 data must NEVER be written to T1 local files (any path).
    This extends Gate 4a beyond just memory — writing L3 health data to
    asthma-checkup-questions.md is just as much a persistence violation as
    writing it to MEMORY.md. Both survive across sessions.
    
    T2/T3 + L3 = consent gate (Gates 4b/4c), NOT hard deny.
    """
    tier_int = _TIER_TO_INT.get(tier_str, 1)
    free_level = FREE_BAND.get((tier_int, is_write), 0)

    # T1 local writes of L3/L4 = hard deny (persists to disk, survives sessions)
    if tier_int == 1 and is_write and data_level >= 3:
        return False, "local_write_above_band"

    ok = free_level >= data_level
    if ok:
        return True, ""
    return False, "missing_authorization"


class PrivacyRegistry:
    """Deterministic privacy decision engine.

    Implements the 5-step HTG authorization procedure with:
    - cuarena-compatible FREE_BAND gating
    - L3/L4 provenance enforcement (meeting rule: amnesia for stale data)
    - Tool tier downgrade (Condition 1: swap T3→T2 when equivalent exists)
    - Vault redirect for L3/L4 memory violations reused later
    - Look-back consent check (don't inject synthetic consent if original has it)
    """

    def __init__(self, pii_map: PIIMap, trajectory: ParsedTrajectory):
        self.pii_map = pii_map
        self.trajectory = trajectory
        self._consent_granted: dict[str, str] = {}
        self._memory_paths_used_later: set[str] = set()
        self._l34_provided_in_turn: dict[int, set[str]] = {}
        self._last_confirmation_turn: int = -1
        self._precompute_memory_usage()

    def _precompute_memory_usage(self) -> None:
        """Scan all turns to find which memory paths are read after being written."""
        written_paths: dict[str, int] = {}
        for turn in self.trajectory.assistant_turns:
            for tc in turn.tool_calls:
                path = tc.arguments.get("path", "") or tc.arguments.get("file_path", "") or ""
                if is_persistence_tool(tc.name) or (path and is_persistence_path(path)):
                    if tc.name not in _READ_TOOLS:
                        written_paths[path.lower()] = turn.turn_index
                if tc.name in _READ_TOOLS and path:
                    path_lower = path.lower()
                    if path_lower in written_paths and written_paths[path_lower] < turn.turn_index:
                        self._memory_paths_used_later.add(path_lower)

    def is_memory_reused_later(self, path: str, current_turn: int) -> bool:
        """Check if data written to this path is read in a subsequent turn."""
        path_lower = path.lower()
        if path_lower in self._memory_paths_used_later:
            return True
        for turn in self.trajectory.assistant_turns:
            if turn.turn_index <= current_turn:
                continue
            for tc in turn.tool_calls:
                tc_path = (tc.arguments.get("path", "") or tc.arguments.get("file_path", "") or "").lower()
                if tc.name in _READ_TOOLS and tc_path and (
                    tc_path == path_lower or path_lower in tc_path or tc_path in path_lower
                ):
                    return True
        return False

    def _check_l34_provenance(self, turn: AssistantTurn, matched_pii: list[PIIEntity]) -> bool:
        """Check if L3/L4 data has valid provenance per meeting rule.

        L3/L4 data is admissible if:
        1. The user mentioned it in ANY preceding message in this session
           (the user brought it into the conversation), OR
        2. The immediately previous turn was an explicit confirmation step

        STALE = data that ONLY exists in memory/persistence and was never
        mentioned by the user in any message during this session. The amnesia
        rule targets cross-session carryover, not within-session re-use.
        """
        l34_entities = [e for e in matched_pii if _LEVEL_ORDER.get(e.level, 0) >= 3]
        if not l34_entities:
            return True

        current_idx = turn.turn_index

        # Check ALL user messages up to and including the current turn
        for msg_idx in range(min(current_idx + 1, len(self.trajectory.user_messages))):
            user_msg = self.trajectory.user_messages[msg_idx].lower()
            for entity in l34_entities:
                if entity.text.lower() in user_msg:
                    return True

        # Check if the previous turn was a confirmation (consent gate)
        if self._last_confirmation_turn == current_idx - 1:
            return True

        # Check entity source provenance — if classified as user_input, it's valid
        for entity in l34_entities:
            if entity.source_type == "user_input":
                return True

        return False

    def _check_look_back_consent(self, turn: AssistantTurn) -> bool:
        """Check if the original trajectory already has consent before this turn.

        Look at the preceding user message — if it contains approval language,
        consent already exists and we don't need to inject synthetic consent.
        """
        idx = turn.turn_index
        if idx <= 0:
            return False
        # Check user message before this turn
        if idx < len(self.trajectory.user_messages):
            user_msg = self.trajectory.user_messages[idx].lower()
            approval_words = {"yes", "sure", "go ahead", "approve", "confirmed", "okay", "ok",
                              "do it", "proceed", "granted", "allow", "fine", "yep", "yeah"}
            return any(w in user_msg for w in approval_words)
        return False

    def record_consent(self, path: str, decision: str) -> None:
        self._consent_granted[path] = decision

    def has_consent(self, path: str) -> bool:
        if not self._consent_granted:
            return False
        normalized = path.lower()
        for granted_path in self._consent_granted:
            gp_lower = granted_path.lower()
            if gp_lower in normalized or normalized in gp_lower:
                return True
            gp_name = gp_lower.split("/")[-1]
            norm_name = normalized.split("/")[-1]
            if gp_name == norm_name:
                return True
        return False

    def _pick_remediation(self, path: str, turn_index: int) -> str:
        """Choose remediation strategy for L3/L4 memory violations.

        - vault: preferred when the memory is reused later (1-2 reads)
        - manual_input: acceptable default when memory is rarely reused
        - exception_md: use sparingly — only for trajectories with 5+ subsequent reads
          of the same path, making vault redirect or manual re-input impractical
        """
        reused = self.is_memory_reused_later(path, turn_index)
        if not reused:
            return ""
        read_count = 0
        for t in self.trajectory.assistant_turns:
            if t.turn_index <= turn_index:
                continue
            for tc in t.tool_calls:
                p = (tc.arguments.get("path", "") or tc.arguments.get("file_path", "") or "").lower()
                if tc.name in _READ_TOOLS and p and (p == path.lower() or path.lower() in p):
                    read_count += 1
        if read_count >= 5:
            return "exception_md"
        elif read_count >= 3:
            return "vault"
        elif read_count >= 1:
            return "manual_input"
        return ""

    def decide(self, turn: AssistantTurn) -> TurnDecision:
        """Evaluate a turn and return the deterministic privacy decision."""

        if not turn.tool_calls:
            return TurnDecision(
                turn_index=turn.turn_index,
                action=Action.ALLOW,
                reason="text-only turn, no tool calls",
                is_read_only=True,
            )

        highest_action = Action.ALLOW
        decision = TurnDecision(turn_index=turn.turn_index, action=Action.ALLOW, reason="")
        action_order = {
            Action.ALLOW: 0, Action.MINIMIZE: 1, Action.ELICIT: 2,
            Action.CONSENT_GATE: 3, Action.T3_BLOCK: 4,
            Action.VAULT_REDIRECT: 5, Action.HARD_DENY: 6,
        }

        for tc in turn.tool_calls:
            tier = get_tool_tier(tc.name)
            path = tc.arguments.get("path", "") or tc.arguments.get("file_path", "") or ""
            is_read = tc.name in _READ_TOOLS
            is_persist_tool = is_persistence_tool(tc.name)
            is_persist_path = is_persistence_path(path)
            is_mem = is_persist_tool or is_persist_path
            is_write = not is_read
            write_content = _get_write_content(tc)

            content_level, matched_pii = _max_pii_level_in_content(
                write_content, self.pii_map.entities
            )

            if not matched_pii and path:
                content_level_path, matched_pii_path = _max_pii_level_in_content(
                    path, self.pii_map.entities
                )
                if _LEVEL_ORDER.get(content_level_path, 0) > _LEVEL_ORDER.get(content_level, 0):
                    content_level = content_level_path
                    matched_pii = matched_pii_path

            if is_read:
                continue

            data_level_int = _LEVEL_ORDER.get(content_level, 0)

            # --- P0: L3/L4 Provenance check (meeting "amnesia" rule) ---
            if data_level_int >= 3 and not is_mem:
                if not self._check_l34_provenance(turn, matched_pii):
                    candidate_action = Action.ELICIT
                    candidate_reason = (
                        f"L{data_level_int} data from stale source (memory/history). "
                        f"Amnesia rule: agent must ask user to re-provide."
                    )
                    if action_order.get(candidate_action, 0) > action_order.get(highest_action, 0):
                        highest_action = candidate_action
                        decision = TurnDecision(
                            turn_index=turn.turn_index,
                            action=candidate_action,
                            reason=candidate_reason,
                            data_level=content_level,
                            tool_tier=tier,
                            target_path=path or tc.name,
                            pii_in_content=matched_pii,
                        )
                    continue

            # --- P1: Tool tier downgrade check (Condition 1) ---
            lower_tool = get_lower_tier_equivalent(tc.name)
            if lower_tool and data_level_int >= 2:
                lower_tier = get_tool_tier(lower_tool)
                lower_tier_int = _TIER_TO_INT.get(lower_tier, 1)
                current_tier_int = _TIER_TO_INT.get(tier, 1)
                if lower_tier_int < current_tier_int:
                    ok_lower, _ = _evaluate_free_band(lower_tier, is_write, data_level_int, False)
                    if ok_lower:
                        candidate_action = Action.ALLOW
                        candidate_reason = (
                            f"Tool downgrade: {tc.name} ({tier}) → {lower_tool} ({lower_tier}). "
                            f"Lower tier handles L{data_level_int} within free band."
                        )
                        if action_order.get(candidate_action, 0) >= action_order.get(highest_action, 0):
                            decision = TurnDecision(
                                turn_index=turn.turn_index,
                                action=candidate_action,
                                reason=candidate_reason,
                                data_level=content_level,
                                tool_tier=lower_tier,
                                target_path=path or tc.name,
                                pii_in_content=matched_pii,
                                original_tool=tc.name,
                                downgraded_tool=lower_tool,
                            )
                        continue

            # Apply the FREE_BAND predicate
            ok, reason = _evaluate_free_band(tier, is_write, data_level_int, is_mem)

            if ok:
                if is_write and data_level_int >= 2 and matched_pii:
                    candidate_action = Action.MINIMIZE
                    candidate_reason = f"Within free band but minimizing L{data_level_int} PII in write"
                else:
                    continue
            elif reason == "local_write_above_band":
                # Gate 4a: T1 persistence + L3/L4
                # --- P1: Check if vault redirect is better than hard deny ---
                remediation = self._pick_remediation(path or tc.name, turn.turn_index)
                if remediation == "vault":
                    vault_key = path.split("/")[-1].replace(".md", "").replace(".", "_") if path else tc.name
                    return TurnDecision(
                        turn_index=turn.turn_index,
                        action=Action.VAULT_REDIRECT,
                        reason=f"Gate 4a + vault redirect: L{data_level_int} data reused later, saving to vault instead of {path or tc.name}",
                        data_level=content_level,
                        tool_tier="T1_PERSISTENCE",
                        target_path=path or tc.name,
                        pii_in_content=matched_pii,
                        remediation="vault",
                        vault_key=vault_key,
                    )
                if remediation == "exception_md":
                    return TurnDecision(
                        turn_index=turn.turn_index,
                        action=Action.HARD_DENY,
                        reason=f"Gate 4a + exception.md: L{data_level_int} data heavily reused — documented exception required",
                        data_level=content_level,
                        tool_tier="T1_PERSISTENCE",
                        target_path=path or tc.name,
                        pii_in_content=matched_pii,
                        remediation="exception_md",
                    )
                return TurnDecision(
                    turn_index=turn.turn_index,
                    action=Action.HARD_DENY,
                    reason=f"Gate 4a: {content_level} data to persistence ({path or tc.name})",
                    data_level=content_level,
                    tool_tier=tier if tier != "T1" else "T1_PERSISTENCE",
                    target_path=path or tc.name,
                    pii_in_content=matched_pii,
                    remediation=remediation if remediation else "",
                )
            else:
                # Gate 4b/4c: outside free band — need authorization
                consent_key = path or tc.name

                # --- P2: Look-back check for existing consent ---
                has_prior = self.has_consent(consent_key) or self._check_look_back_consent(turn)

                if tier == "T3":
                    if has_prior:
                        candidate_action = Action.MINIMIZE
                        candidate_reason = f"T3+L{data_level_int}: consent granted, minimizing"
                    else:
                        candidate_action = Action.T3_BLOCK
                        candidate_reason = f"Gate 4c: {content_level} data to T3 tool ({tc.name})"
                elif tier == "T2" and data_level_int >= 3:
                    if has_prior:
                        candidate_action = Action.MINIMIZE
                        candidate_reason = f"T2+L{data_level_int}: consent granted, minimizing"
                    else:
                        candidate_action = Action.CONSENT_GATE
                        candidate_reason = f"Gate 4b: {content_level} data to T2 tool ({tc.name})"
                else:
                    if has_prior:
                        candidate_action = Action.MINIMIZE
                        candidate_reason = f"Outside free band but consent granted for {consent_key}"
                    else:
                        candidate_action = Action.CONSENT_GATE
                        candidate_reason = f"Outside free band: L{data_level_int} data, {tier} tool ({tc.name})"

            if action_order.get(candidate_action, 0) > action_order.get(highest_action, 0):
                highest_action = candidate_action
                decision = TurnDecision(
                    turn_index=turn.turn_index,
                    action=candidate_action,
                    reason=candidate_reason,
                    data_level=content_level,
                    tool_tier=tier,
                    target_path=path or tc.name,
                    pii_in_content=matched_pii,
                    consent_path=path or tc.name if candidate_action in (
                        Action.CONSENT_GATE, Action.T3_BLOCK,
                    ) else "",
                )

        if decision.action == Action.ALLOW and decision.reason == "":
            decision.reason = "All operations within allowed bounds (FREE_BAND)"
            decision.is_read_only = all(
                tc.name in _READ_TOOLS for tc in turn.tool_calls
            )

        return decision

    def decide_all(self, turns: list[AssistantTurn]) -> list[TurnDecision]:
        """Decide actions for all turns, tracking consent state across them."""
        decisions = []
        for turn in turns:
            d = self.decide(turn)
            decisions.append(d)
            if d.action == Action.CONSENT_GATE and d.consent_path:
                self.record_consent(d.consent_path, "granted")
                self._last_confirmation_turn = turn.turn_index
        return decisions
