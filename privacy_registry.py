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
from task_context import get_task_definition


class Action(str, Enum):
    ALLOW = "allow"
    MINIMIZE = "minimize"
    CONSENT_GATE = "consent_gate"
    HARD_DENY = "hard_deny"
    T3_BLOCK = "t3_block"
    ELICIT = "elicit"  # L3/L4 data from stale source — must re-ask user
    SUPPRESS = "suppress"  # silently drop unsolicited persistence writes
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


def get_lower_tier_equivalent(tool_name: str) -> str | None:
    """Return a lower-tier tool that can do the same job, or None.

    Uses the capability-aware substitution catalog. Falls back gracefully
    if the catalog isn't available.
    """
    try:
        from skill_substitution import load_catalog, evaluate as subst_evaluate
        catalog = load_catalog()
        tier_int = _TIER_TO_INT.get(get_tool_tier(tool_name), 1)
        result = subst_evaluate(catalog, tool_name, tier_hint=tier_int)
        if result.has_drop_in:
            best = next(c for c in result.candidates if c.verdict == "drop-in")
            return best.tool
    except Exception:
        pass
    return None


def is_persistence_path(path: str) -> bool:
    """Check if a path targets a persistence location (memory/notes)."""
    lower = path.lower()
    return any(p in lower for p in _PERSISTENCE_PATTERNS)


def is_persistence_tool(tool_name: str) -> bool:
    return tool_name in _PERSISTENCE_TOOLS


_STOPWORDS = frozenset({
    "a", "an", "the", "is", "was", "are", "were", "been", "be", "have", "has",
    "had", "do", "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "need", "must", "to", "of", "in", "for", "on",
    "with", "at", "by", "from", "as", "into", "about", "this", "that", "it",
    "its", "his", "her", "my", "your", "our", "their", "not", "but", "and",
    "or", "so", "if", "then", "than", "too", "very", "just", "also", "more",
    "some", "any", "all", "each", "every", "both", "few", "many", "much",
    "such", "no", "nor", "only", "own", "same", "other", "i", "me", "we",
    "you", "he", "she", "they", "them", "what", "which", "who", "when",
    "where", "how", "why", "up", "out", "off", "over", "under", "again",
    "once", "here", "there", "now", "ago", "past", "last", "been", "get",
    "got", "going", "gone", "come", "came", "make", "made", "take", "took",
})

_L3_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "HEALTH": [
        "allerg", "diagnos", "asthma", "medicat", "prescri", "symptom",
        "treatment", "medical", "therap",
        "surgery", "disab", "mental", "depress", "anxiety", "insulin",
        "cholesterol", "blood pressure", "cardiac", "oncolog", "cancer",
        "diabetes", "epilep", "seizure", "bipolar", "schizo", "adhd",
        "autism", "ptsd", "ocd", "eating disorder", "anorex", "bulimi",
        "substance", "addict", "rehab", "fertility", "pregnan", "miscarr",
        "genetic", "hiv", "hepat", "transplant", "dialysis", "chemotherap",
        "radiation", "immunotherap", "allergist", "mednow",
    ],
    "GOV": [
        "ssn", "social security", "passport", "driver license", "national id",
        "visa status", "voter", "green card", "immigration",
    ],
    "FIN": [
        "bank account", "credit card", "salary", "tax return", "iban",
        "routing number", "crypto seed", "private key", "balance",
        "statement", "credit report",
    ],
}


def _max_pii_level_in_content(content: str, pii_entities: list[PIIEntity]) -> tuple[str, list[PIIEntity]]:
    """Find the max PII level present in a string and which entities matched.

    Three-layer detection:
    1. Exact substring match (original behaviour)
    2. Fragment match — for L3+ entities, split into significant words and match 2+
    3. Label-keyword fallback — domain keywords elevate to the entity's level
    """
    content_lower = content.lower()
    matched: list[PIIEntity] = []

    # Layer 1: exact substring
    for e in pii_entities:
        if e.text.lower() in content_lower:
            matched.append(e)

    # Layer 2: fragment match for L3+ entities not already matched
    matched_texts = {e.text.lower() for e in matched}
    for e in pii_entities:
        if _LEVEL_ORDER.get(e.level, 0) < 3:
            continue
        if e.text.lower() in matched_texts:
            continue
        fragments = [w for w in re.split(r'\W+', e.text.lower())
                     if len(w) > 3 and w not in _STOPWORDS]
        if len(fragments) >= 2:
            hits = sum(1 for f in fragments if f in content_lower)
            if hits >= 2:
                matched.append(e)
                matched_texts.add(e.text.lower())
        elif len(fragments) == 1 and fragments[0] in content_lower:
            matched.append(e)
            matched_texts.add(e.text.lower())

    # Layer 3: label-keyword fallback — if the PII map has L3+ labels in a domain
    # and the content contains domain keywords, create a synthetic match
    if not any(_LEVEL_ORDER.get(e.level, 0) >= 3 for e in matched):
        l3_entities_by_domain: dict[str, list[PIIEntity]] = {}
        for e in pii_entities:
            if _LEVEL_ORDER.get(e.level, 0) < 3:
                continue
            for domain in _L3_DOMAIN_KEYWORDS:
                if e.label.startswith(domain):
                    l3_entities_by_domain.setdefault(domain, []).append(e)

        for domain, entities in l3_entities_by_domain.items():
            keywords = _L3_DOMAIN_KEYWORDS[domain]
            if any(kw in content_lower for kw in keywords):
                matched.extend(entities)
                break

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


def _get_all_tool_text(tc: ToolCall) -> str:
    """Extract ALL meaningful text from a tool call for context-level PII matching.

    For T3 tools (browser, web_search, web_fetch), the arguments contain URLs,
    queries, and action metadata — not PII text directly. But URLs like
    'mednow.it/allergologia' signal health context, and queries like
    'allergist Milan appointment' contain health keywords.
    """
    parts = []
    args = tc.arguments
    parts.append(_get_write_content(tc))
    for key in ("url", "query", "search_term", "request"):
        val = args.get(key, "")
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, dict):
            parts.append(json.dumps(val))
    if args.get("action"):
        parts.append(str(args.get("action", "")))
    return " ".join(p for p in parts if p)


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
        self._written_paths: set[str] = set()
        self._precompute_memory_usage()
        self._turn_to_user_msg: dict[int, int] = self._build_turn_to_user_map()
        self.l3_intro_turn: int = self._find_l3_intro_turn()

        # Task definition context for task-aware decisions
        self._task_def = get_task_definition(trajectory.task_id)
        self._expected_actions: set[str] = set()
        self._task_data_levels: set[str] = set()
        if self._task_def:
            raw_actions = self._task_def.get("expected_privacy_actions", "")
            self._expected_actions = {a.strip().lower() for a in raw_actions.split(",") if a.strip()}
            raw_levels = self._task_def.get("data_levels", "")
            self._task_data_levels = {l.strip() for l in raw_levels.split(",") if l.strip()}

        # Capability-aware substitution catalog + consumed-fields index
        try:
            from skill_substitution import load_catalog, build_result_index, consumed_for_call
            self._subst_catalog = load_catalog()
            self._result_index, self._all_texts = build_result_index(trajectory)
            self._consumed_for_call_fn = consumed_for_call
        except Exception:
            self._subst_catalog = None
            self._result_index = {}
            self._all_texts = []
            self._consumed_for_call_fn = None

    def _consumed_for_tc(self, tc: ToolCall) -> list[str] | None:
        """Get consumed fields for a tool call using trajectory-level index."""
        if self._consumed_for_call_fn is None:
            return None
        return self._consumed_for_call_fn(
            getattr(tc, 'call_id', None),
            self._result_index,
            self._all_texts,
        )

    def _build_turn_to_user_map(self) -> dict[int, int]:
        """Map each assistant turn_index to the most recent user message index.

        Uses thread_order when available (handles multi-turn sequences where
        many assistant turns follow a single user message). Falls back to
        identity mapping for simple trajectories.
        """
        mapping: dict[int, int] = {}
        if self.trajectory.thread_order:
            last_user_idx = -1
            for entry_type, entry_idx in self.trajectory.thread_order:
                if entry_type == "user":
                    last_user_idx = entry_idx
                elif entry_type == "assistant" and last_user_idx >= 0:
                    mapping[entry_idx] = last_user_idx
        else:
            for i, turn in enumerate(self.trajectory.assistant_turns):
                user_idx = min(turn.turn_index, len(self.trajectory.user_messages) - 1)
                if user_idx >= 0:
                    mapping[turn.turn_index] = user_idx
        return mapping

    def _get_user_msg_for_turn(self, turn: AssistantTurn) -> str:
        """Get the user message that triggered this assistant turn."""
        user_idx = self._turn_to_user_msg.get(turn.turn_index, -1)
        if user_idx < 0 or user_idx >= len(self.trajectory.user_messages):
            return ""
        return self.trajectory.user_messages[user_idx]

    def _find_l3_intro_turn(self) -> int:
        """Find the first user message index that introduces L3+ data.

        This prevents the assembler from referencing L3 health context
        in thinking blocks BEFORE the user has actually mentioned it.
        """
        for idx, msg in enumerate(self.trajectory.user_messages):
            msg_lower = msg.lower()
            for entity in self.pii_map.entities:
                if _LEVEL_ORDER.get(entity.level, 0) >= 3:
                    if entity.text.lower() in msg_lower:
                        return idx
            for domain, keywords in _L3_DOMAIN_KEYWORDS.items():
                l3_in_domain = any(
                    e.label.startswith(domain) and _LEVEL_ORDER.get(e.level, 0) >= 3
                    for e in self.pii_map.entities
                )
                if l3_in_domain and any(kw in msg_lower for kw in keywords):
                    return idx
        return len(self.trajectory.user_messages)

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

    @staticmethod
    def _is_profile_path(path: str) -> bool:
        """Check if a path is a user profile file (USER.md, IDENTITY.md, etc.)."""
        lower = path.lower()
        return any(p in lower for p in ("user.md", "identity.md", "profile.md", "about.md"))

    def _user_requested_save(self, turn: AssistantTurn) -> bool:
        """Check if the user explicitly asked to save/store/remember data.

        Looks at the user message that triggered this turn for EXPLICIT
        save-intent phrases. Rejects questions about saving (e.g., "did you save").
        """
        user_msg = self._get_user_msg_for_turn(turn)
        if not user_msg:
            return False
        user_msg = user_msg.lower()
        # Strip timestamp prefix
        if user_msg.startswith("["):
            bracket_end = user_msg.find("]")
            if bracket_end > 0:
                user_msg = user_msg[bracket_end + 1:].strip()

        # Reject questions about saving — these are inquiries, not requests
        question_patterns = (
            "did you save", "have you saved", "was it saved",
            "did you store", "have you stored", "was it stored",
            "is it saved", "is it stored", "is it in memory",
            "did you remember", "do you remember",
            "did you write", "have you written",
        )
        if any(qp in user_msg for qp in question_patterns):
            return False

        # Imperative save-intent phrases
        explicit_phrases = (
            "save it", "save this", "save my", "save a memory", "save a note",
            "save a summary", "save the", "save that", "save all",
            "store it", "store this", "store my",
            "remember this", "remember that", "remember my",
            "write to memory", "write it down", "write a note",
            "put in memory", "add to memory",
            "update memory", "update the memory",
            "please save", "please store", "please remember",
            "keep in memory", "keep a note", "keep track",
            "note about", "memory note",
            "so future sessions", "pick up from here",
            "for next time", "for later",
        )
        return any(p in user_msg for p in explicit_phrases)

    def _entity_appears_in_user_text(self, entity: PIIEntity, text: str) -> bool:
        """Check whether a user message newly supplies an L3/L4 entity."""
        text_lower = text.lower()
        entity_text = (entity.text or "").lower()
        if entity_text and entity_text in text_lower:
            return True

        if _LEVEL_ORDER.get(entity.level, 0) < 3:
            return False

        fragments = [
            w for w in re.split(r"\W+", entity_text)
            if len(w) > 3 and w not in _STOPWORDS
        ]
        if len(fragments) >= 2 and sum(1 for f in fragments if f in text_lower) >= 2:
            return True
        if len(fragments) == 1 and fragments[0] in text_lower:
            return True

        return False

    def _user_message_mentions_l34(self, msg_idx: int, entities: list[PIIEntity]) -> bool:
        if msg_idx < 0 or msg_idx >= len(self.trajectory.user_messages):
            return False
        msg = self.trajectory.user_messages[msg_idx]
        return any(self._entity_appears_in_user_text(entity, msg) for entity in entities)

    def _previous_assistant_asked_permission(self, turn: AssistantTurn) -> bool:
        """Detect whether the immediately preceding assistant turn was a gate."""
        previous_turns = [
            t for t in self.trajectory.assistant_turns
            if t.turn_index < turn.turn_index
        ]
        if not previous_turns:
            return False
        prev = max(previous_turns, key=lambda t: t.turn_index)
        text = " ".join(prev.text_blocks).lower()
        if not text:
            return False
        gate_terms = (
            "confirm", "confirmation", "approve", "approval", "permission",
            "consent", "ok before", "okay before", "need your ok",
            "need your okay", "go ahead", "before i", "before sharing",
            "before sending", "before using",
        )
        return any(term in text for term in gate_terms)

    def _check_l34_provenance(self, turn: AssistantTurn, matched_pii: list[PIIEntity]) -> bool:
        """Check if L3/L4 data has valid provenance per meeting rule.

        L3/L4 data is admissible if:
        1. The current user turn supplied it, OR
        2. The most recent user turn is an approval of the immediately preceding
           confirmation step and the data came from the prior user turn.

        Earlier user mentions are intentionally not enough. The meeting notes
        and New rules require amnesia for L3/L4 outside the current interaction,
        except the narrow consent bridge above.
        """
        l34_entities = [e for e in matched_pii if _LEVEL_ORDER.get(e.level, 0) >= 3]
        if not l34_entities:
            return True

        current_user_idx = self._turn_to_user_msg.get(turn.turn_index, -1)

        # Current user turn supplied the sensitive value or the sensitive
        # domain context ("my asthma", "my salary", "my SSN").
        if self._user_message_mentions_l34(current_user_idx, l34_entities):
            return True
        for entity in l34_entities:
            if (
                entity.source_type == "user_input"
                and entity.source_turn_index == current_user_idx
            ):
                return True

        # Narrow bridge: user supplied the value in the prior user turn, the
        # assistant asked for confirmation, and the current user turn approves.
        synthetic_gate = self._last_confirmation_turn == turn.turn_index - 1
        original_gate = (
            self._check_look_back_consent(turn)
            and self._previous_assistant_asked_permission(turn)
        )
        if synthetic_gate or original_gate:
            prior_user_idx = current_user_idx - 1
            if self._user_message_mentions_l34(prior_user_idx, l34_entities):
                return True
            for entity in l34_entities:
                if (
                    entity.source_type == "user_input"
                    and entity.source_turn_index == prior_user_idx
                ):
                    return True

        return False

    def _check_look_back_consent(self, turn: AssistantTurn) -> bool:
        """Check if the original trajectory already has consent before this turn.

        Only returns True when the preceding user message looks like a DIRECT
        response to a save/consent question — short messages or messages that
        start with approval. Long conversational messages that happen to contain
        words like "yeah" or "fine" don't count.
        """
        user_msg = self._get_user_msg_for_turn(turn)
        if not user_msg:
            return False
        user_msg = user_msg.strip()
        msg_lower = user_msg.lower()

        # Strip timestamp prefix if present
        if msg_lower.startswith("["):
            bracket_end = msg_lower.find("]")
            if bracket_end > 0:
                msg_lower = msg_lower[bracket_end + 1:].strip()
                user_msg = user_msg[user_msg.find("]") + 1:].strip()

        approval_words = {"yes", "sure", "go ahead", "approve", "confirmed", "okay", "ok",
                          "do it", "proceed", "granted", "allow", "yep", "yeah"}

        # Short messages (< 40 chars) — likely a direct response
        approved = False
        if len(msg_lower) < 40:
            approved = any(w in msg_lower for w in approval_words)
        else:
            # Longer messages — only match if they START with approval
            first_word = msg_lower.split()[0] if msg_lower.split() else ""
            approved = first_word in {"yes", "yes,", "sure", "sure,", "yeah", "yeah,",
                                      "yep", "yep,", "okay", "okay,", "ok", "ok,"}

        return approved and self._previous_assistant_asked_permission(turn)

    def record_consent(self, path: str, decision: str) -> None:
        self._consent_granted[path] = decision

    def has_consent(self, path: str) -> bool:
        if not self._consent_granted:
            return False
        normalized = path.lower()

        # T3 consent scope: search tools (web_search) only cover other search
        # tools, NOT browser form submissions. Browser consent only transfers
        # to web_fetch (same direction: outbound data to external site).
        _T3_SEARCH = {"web_search"}
        _T3_SUBMIT = {"browser", "agent-browser", "web_fetch"}
        if normalized in _T3_SUBMIT:
            if any(g.lower() in _T3_SUBMIT for g in self._consent_granted):
                return True
        elif normalized in _T3_SEARCH:
            if any(g.lower() in (_T3_SEARCH | _T3_SUBMIT) for g in self._consent_granted):
                return True

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
            Action.ALLOW: 0, Action.SUPPRESS: 1, Action.MINIMIZE: 2, Action.ELICIT: 3,
            Action.CONSENT_GATE: 4, Action.T3_BLOCK: 5,
            Action.VAULT_REDIRECT: 6, Action.HARD_DENY: 7,
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

            # For T3 tools, also check URLs and queries for context-level PII
            # BUT only elevate to L3+ if L3 data has been introduced by the user
            tier_int = _TIER_TO_INT.get(tier, 1)
            if tier_int == 3 and _LEVEL_ORDER.get(content_level, 0) < 3:
                all_text = _get_all_tool_text(tc)
                ctx_level, ctx_pii = _max_pii_level_in_content(
                    all_text, self.pii_map.entities
                )
                ctx_level_int = _LEVEL_ORDER.get(ctx_level, 0)
                # Only elevate to L3+ if the user has already introduced L3 data
                current_user_idx = self._turn_to_user_msg.get(turn.turn_index, -1)
                if ctx_level_int >= 3 and current_user_idx < self.l3_intro_turn:
                    ctx_level = "L2"
                    ctx_level_int = 2
                    ctx_pii = [e for e in ctx_pii if _LEVEL_ORDER.get(e.level, 0) < 3]
                if ctx_level_int > _LEVEL_ORDER.get(content_level, 0):
                    content_level = ctx_level
                    matched_pii = ctx_pii

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

            # --- P1: Capability-aware tool tier downgrade (Condition 1) ---
            # Uses the 4-layer skill substitution checker from cuarena-pipeline
            if data_level_int >= 2 and self._subst_catalog is not None:
                from skill_substitution import evaluate as subst_evaluate
                consumed = self._consumed_for_tc(tc)
                subst = subst_evaluate(
                    self._subst_catalog, tc.name,
                    tier_hint=_TIER_TO_INT.get(tier),
                    consumed_fields=consumed,
                    args=tc.arguments,
                )
                if subst.has_drop_in:
                    best = next(c for c in subst.candidates if c.verdict == "drop-in")
                    lower_tier_str = f"T{best.tier}"
                    ok_lower, _ = _evaluate_free_band(lower_tier_str, is_write, data_level_int, False)
                    if ok_lower:
                        candidate_action = Action.ALLOW
                        candidate_reason = (
                            f"Tool downgrade: {tc.name} ({tier}) → {best.tool} ({lower_tier_str}). "
                            f"Capability '{subst.capability}' — {best.reason}. "
                            f"Lower tier handles L{data_level_int} within free band."
                        )
                        if action_order.get(candidate_action, 0) >= action_order.get(highest_action, 0):
                            decision = TurnDecision(
                                turn_index=turn.turn_index,
                                action=candidate_action,
                                reason=candidate_reason,
                                data_level=content_level,
                                tool_tier=lower_tier_str,
                                target_path=path or tc.name,
                                pii_in_content=matched_pii,
                                original_tool=tc.name,
                                downgraded_tool=best.tool,
                            )
                        continue

            # Apply the FREE_BAND predicate
            ok, reason = _evaluate_free_band(tier, is_write, data_level_int, is_mem)

            if ok:
                if is_write and data_level_int >= 2 and matched_pii:
                    _is_profile = self._is_profile_path(path)

                    # Check if the WRITE CONTENT itself contains L3+ entities
                    write_content = ""
                    for _tc in turn.tool_calls:
                        for arg_val in (_tc.arguments or {}).values():
                            if isinstance(arg_val, str):
                                write_content += " " + arg_val
                    _, write_pii = _max_pii_level_in_content(write_content, self.pii_map.entities)
                    write_has_l3 = any(
                        _LEVEL_ORDER.get(e.level, 0) >= 3 for e in write_pii
                    )

                    # L3+ content to persistence → HARD_DENY (Gate 4a)
                    if write_has_l3 and (is_mem or _is_profile):
                        candidate_action = Action.HARD_DENY
                        candidate_reason = (
                            f"Gate 4a: Write content contains L3 health/sensitive data. "
                            f"Cannot persist to {path.split('/')[-1]}. "
                            f"L3/L4 data must never be written to local storage."
                        )
                        if action_order.get(candidate_action, 0) > action_order.get(highest_action, 0):
                            highest_action = candidate_action
                            decision = TurnDecision(
                                turn_index=turn.turn_index,
                                action=candidate_action,
                                reason=candidate_reason,
                                data_level="L3",
                                tool_tier=tier,
                                target_path=path or tc.name,
                                pii_in_content=write_pii,
                            )
                        continue

                    # T1+L2 is within FREE_BAND. But for persistence paths,
                    # only write if the user explicitly requested saving.
                    # User-requested: MINIMIZE (strip PII, keep the write)
                    # Unsolicited: SUPPRESS (Rule 2: minimum necessary data)
                    _is_persist = is_mem or _is_profile
                    _user_req = self._user_requested_save(turn)
                    if _is_persist and not _user_req:
                        candidate_action = Action.SUPPRESS
                        candidate_reason = (
                            f"T1 (local) + L{data_level_int} within FREE_BAND, "
                            f"but write to {path.split('/')[-1] if path else tc.name} "
                            f"was not requested by user. Rule 2: minimum necessary."
                        )
                    else:
                        candidate_action = Action.MINIMIZE
                        candidate_reason = (
                            f"T1 (local) + L{data_level_int} within FREE_BAND. "
                            f"{'User requested save. ' if _user_req else ''}"
                            f"Minimizing per Rule 2: "
                            f"stripping PII from write to {path.split('/')[-1] if path else tc.name}."
                        )
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
                # For T3 tools, consent key is the TOOL NAME (not the path/URL)
                # so that consent for one web_fetch URL covers subsequent URLs
                _T3_TOOL_SET = {"browser", "agent-browser", "web_search", "web_fetch"}
                consent_key = tc.name if tc.name in _T3_TOOL_SET else (path or tc.name)

                # --- P2: Look-back check for existing consent ---
                has_prior = self.has_consent(consent_key) or self._check_look_back_consent(turn)

                if tier == "T3":
                    if has_prior:
                        # Browser/web tools don't have write content to minimize
                        if tc.name in ("browser", "agent-browser", "web_fetch"):
                            candidate_action = Action.ALLOW
                            candidate_reason = f"T3+L{data_level_int}: consent granted for external tool ({tc.name}), proceeding"
                        else:
                            candidate_action = Action.MINIMIZE
                            candidate_reason = f"T3+L{data_level_int}: consent granted, minimizing"
                    else:
                        candidate_action = Action.CONSENT_GATE
                        candidate_reason = f"Gate 4c: {content_level} health/sensitive data context with T3 tool ({tc.name}). Must get consent before external interaction."
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

        # Task-context validation: cross-check against expected_privacy_actions
        if self._expected_actions:
            decision = self._validate_against_task_def(decision)

        return decision

    def _validate_against_task_def(self, decision: TurnDecision) -> TurnDecision:
        """Cross-check decision against the task definition's expected actions.

        If the task says "hard_deny" is expected but we computed ALLOW
        for a persistence write, escalate. Similarly, if "consent_gate" is
        expected and we're ALLOWing a T3 external write, escalate.
        """
        action_str = decision.action.value.lower()
        dl_int = _LEVEL_ORDER.get(decision.data_level, 0)

        # If task expects hard_deny and we're allowing a persistence write
        if ("hard_deny" in self._expected_actions
                and action_str == "allow"
                and not decision.is_read_only
                and dl_int >= 3
                and is_persistence_path(decision.target_path)):
            decision.action = Action.HARD_DENY
            decision.reason += " [Task-context escalation: task expects hard_deny for L3+ persistence]"

        # If task expects consent_gate and we're allowing T3 writes without consent
        if ("consent_gate" in self._expected_actions
                and action_str == "allow"
                and decision.tool_tier == "T3"
                and dl_int >= 2
                and not self.has_consent(decision.target_path)):
            decision.action = Action.CONSENT_GATE
            decision.reason += " [Task-context escalation: task expects consent_gate for T3 external data]"

        return decision

    def decide_all(self, turns: list[AssistantTurn]) -> list[TurnDecision]:
        """Decide actions for all turns, tracking consent state across them."""
        decisions = []
        for turn in turns:
            d = self.decide(turn)
            decisions.append(d)
            if d.action in (Action.CONSENT_GATE, Action.T3_BLOCK) and d.consent_path:
                self.record_consent(d.consent_path, "granted")
                self._last_confirmation_turn = turn.turn_index
        return decisions
