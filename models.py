"""Typed dataclasses for every pipeline stage."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Stage 1 — Parser
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict
    tier: str = ""  # T1/T2/T3, populated after classification


@dataclass
class ToolResult:
    call_id: str
    tool_name: str
    content: str
    is_error: bool
    is_empty: bool


@dataclass
class AssistantTurn:
    event_id: str
    turn_index: int
    thinking_blocks: list[str] = field(default_factory=list)
    text_blocks: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    timestamp: int | None = None


@dataclass
class ParsedTrajectory:
    task_id: str
    submission_id: str
    worker_id: str
    session_uuid: str
    jsonl_path: str
    workspace_files: dict[str, str] = field(default_factory=dict)
    workspace_before_files: dict[str, str] = field(default_factory=dict)
    user_messages: list[str] = field(default_factory=list)
    assistant_turns: list[AssistantTurn] = field(default_factory=list)
    tool_results_by_call_id: dict[str, ToolResult] = field(default_factory=dict)
    thread_order: list[tuple[str, str | int]] = field(default_factory=list)
    ordered_events: list[dict] = field(default_factory=list)
    persona: dict = field(default_factory=dict)
    task_spec: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Stage 2 — PII Classifier
# ---------------------------------------------------------------------------

@dataclass
class PIIEntity:
    text: str
    label: str
    level: str  # L0-L4 or BLOCK
    start: int = -1
    end: int = -1
    turn_index: int = -1
    engines: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source_turn_index: int = -1  # which turn introduced this PII (-1 = unknown)
    source_type: str = ""  # "user_input", "tool_result", "memory_read", "history"


@dataclass
class PIIMap:
    entities: list[PIIEntity] = field(default_factory=list)
    max_level: str = "L0"
    has_l4: bool = False
    has_l3: bool = False
    labels_present: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 3 — Rewriter
# ---------------------------------------------------------------------------

@dataclass
class RewrittenTurn:
    turn_index: int
    thinking: str = ""
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    privacy_actions: list[str] = field(default_factory=list)
    scenario: str = ""
    synthetic_user_message: str = ""
    consent_decision: str = ""
    is_adversarial: bool = False
    adversarial_user_message: str = ""
    attack_type: str = ""


@dataclass
class RewriteResult:
    task_id: str
    submission_id: str
    turns: list[RewrittenTurn] = field(default_factory=list)
    scenarios_covered: list[str] = field(default_factory=list)
    skills_used: list[str] = field(default_factory=list)
    privacy_decision_points: int = 0
    # Programmatic fixes applied after Claude (e.g. malformed edit/write JSON)
    rewrite_repairs: list[dict] = field(default_factory=list)
    # Patch-mode: pre-built JSONL events that bypass the writer's reconstruction
    patched_events: list[dict] | None = None


# ---------------------------------------------------------------------------
# Stage 4 — Verifier
# ---------------------------------------------------------------------------

@dataclass
class VerificationIssue:
    turn_index: int
    rule_violated: str
    severity: Literal["critical", "major", "minor"]
    description: str
    fix_instruction: str = ""


@dataclass
class VerificationResult:
    verdict: Literal["PASS", "MINOR_ISSUES", "FAIL"]
    issues: list[VerificationIssue] = field(default_factory=list)
    privacy_compliance: int = 0
    correctness: int = 0
    completeness: int = 0
    efficiency: int = 0
    naturality: int = 0
    overall: float = 0.0
    rationale: str = ""


# ---------------------------------------------------------------------------
# Stage 6 — RLHF Pair Generator
# ---------------------------------------------------------------------------

FailureMode = Literal[
    "wrong_tool_tier_up",     # Used higher-tier tool than needed (T1→T2, T1→T3)
    "wrong_tool_tier_down",   # Used lower-trust tool for sensitive data
    "wrong_param_higher",     # Sent higher-sensitivity data than tool allows
    "wrong_param_lower",      # Used lower-sensitivity param (missed the real data)
    "over_refusal",           # Refused a legitimate operation
    "missing_elicitation",    # Proceeded without asking user for L3/L4 data
    "missing_consent",        # Sent data to lower-trust tier without explicit consent
    "hallucination",          # Plausible-looking but incorrect privacy reasoning
    "ambient_pii_leak",       # Failed to catch PII in tool output (Scenario I)
    "memory_violation",       # Stored L3/L4 data in persistent memory (Scenario D)
]


@dataclass
class PrivacyViolation:
    flag: bool
    rule: str | None = None
    severity: Literal["critical", "major", "minor"] | None = None
    data_level: str = ""       # L0-L4 of the data involved
    tool_tier: str = ""        # T1/T2/T3 of the tool used


@dataclass
class RejectedStep:
    thinking: str
    tool_call: dict | None = None
    tool_response: dict | None = None
    assistant_response: str = ""
    failure_mode: str = ""
    privacy_violation: PrivacyViolation = field(default_factory=lambda: PrivacyViolation(flag=False))
    reward_score: float = 0.0  # 0.0 (catastrophic) to 0.9 (barely suboptimal)
    perturbation_type: str = ""  # t', p', t'|p' from diagram


@dataclass
class RLHFPair:
    task_id: str
    submission_id: str
    turn_index: int
    step_criticality: float = 0.0  # 0-1 from PRM
    pair_level: Literal["step", "turn", "trajectory"] = "step"
    context: list[dict] = field(default_factory=list)
    chosen: dict = field(default_factory=dict)
    rejected: list[RejectedStep] = field(default_factory=list)
    decision_branch: str = ""  # R_correct, A_wrong, A_correct, R_wrong
    data_level_involved: str = ""
    tool_tier_involved: str = ""


@dataclass
class RLHFBatchReport:
    task_id: str
    submission_id: str
    total_pairs: int = 0
    pairs_by_failure_mode: dict[str, int] = field(default_factory=dict)
    pairs_by_level: dict[str, int] = field(default_factory=dict)
    avg_reward_score: float = 0.0
    over_refusal_ratio: float = 0.0
    violation_ratio: float = 0.0


# ---------------------------------------------------------------------------
# Task Report (final)
# ---------------------------------------------------------------------------

@dataclass
class TaskReport:
    task_id: str
    submission_id: str
    status: Literal["PASS", "MINOR_FIXED", "FAIL", "PARSE_ERROR", "CLASSIFY_ERROR", "REWRITE_ERROR"]
    pii_entities_found: int = 0
    max_pii_level: str = "L0"
    scenarios_covered: list[str] = field(default_factory=list)
    privacy_decision_points: int = 0
    verification_scores: dict = field(default_factory=dict)
    refix_iterations: int = 0
    total_tokens: int = 0
    error_message: str = ""
    quality_gate_issues: list[dict] = field(default_factory=list)
