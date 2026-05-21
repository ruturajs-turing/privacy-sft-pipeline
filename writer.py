"""Stage 5: Output writer — JSONL trajectory, workspace dirs, metadata, SFT dataset, RLHF pairs."""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from models import (
    ParsedTrajectory,
    PIIMap,
    RewriteResult,
    RewrittenTurn,
    TaskReport,
    VerificationResult,
    RLHFPair,
    RejectedStep,
    RLHFBatchReport,
)

logger = logging.getLogger(__name__)


def _generate_event_id() -> str:
    return str(uuid.uuid4())


def _build_user_before_turn_map(trajectory) -> dict[int, int]:
    """Build mapping of assistant_turn_index → user_message_index.

    When thread_order is available, uses it to determine which user message
    precedes each assistant turn. Falls back to simple 1:1 mapping when
    thread_order is empty (e.g. in tests or simple trajectories).
    """
    user_before_turn: dict[int, int] = {}

    if trajectory.thread_order:
        last_user_idx = -1
        for entry_type, entry_idx in trajectory.thread_order:
            if entry_type == "user":
                last_user_idx = entry_idx
            elif entry_type == "assistant":
                if last_user_idx >= 0 and entry_idx not in user_before_turn:
                    already_assigned = last_user_idx in user_before_turn.values()
                    if not already_assigned:
                        user_before_turn[entry_idx] = last_user_idx
    else:
        # Fallback: assume 1:1 mapping (first user msg → first assistant turn, etc.)
        num_users = len(trajectory.user_messages)
        for i in range(num_users):
            user_before_turn[i] = i

    return user_before_turn


def _current_ts() -> int:
    return int(time.time() * 1000)


def _turn_to_jsonl_events(
    turn: RewrittenTurn,
    session_id: str,
    base_ts: int,
) -> list[dict]:
    """Convert a RewrittenTurn to JSONL message events."""
    events = []
    ts = base_ts + (turn.turn_index * 5000)

    # Build content blocks — tool calls go before text when both present
    # (prevents "claims action before executing" sequencing issues)
    content = []
    if turn.thinking:
        content.append({"type": "thinking", "thinking": turn.thinking})

    has_tool_calls = any(isinstance(tc, dict) for tc in turn.tool_calls)
    if turn.text and not has_tool_calls:
        content.append({"type": "text", "text": turn.text})

    for tc in turn.tool_calls:
        if isinstance(tc, dict):
            content.append({
                "type": "toolCall",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                "name": tc.get("name", ""),
                "arguments": tc.get("arguments", {}),
            })

    # Text after tool calls (for turns that execute tools then confirm)
    if turn.text and has_tool_calls:
        content.append({"type": "text", "text": turn.text})

    # Assistant message event
    msg_data: dict = {
        "role": "assistant",
        "content": content,
        "timestamp": ts,
    }
    if turn.is_adversarial:
        msg_data["metadata"] = {"is_adversarial": True, "attack_type": turn.attack_type}
    events.append({
        "type": "message",
        "id": _generate_event_id(),
        "message": msg_data,
    })

    # Tool result events
    for tr in turn.tool_results:
        if isinstance(tr, dict):
            events.append({
                "type": "message",
                "id": _generate_event_id(),
                "message": {
                    "role": "toolResult",
                    "toolCallId": tr.get("call_id", ""),
                    "toolName": tr.get("tool_name", ""),
                    "content": [{"type": "text", "text": tr.get("content", "")}],
                    "isError": tr.get("is_error", False),
                    "timestamp": ts + 1000,
                }
            })

    return events


def write_trajectory_output(
    output_dir: Path,
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
    pii_map: PIIMap,
    verification: VerificationResult,
    report: TaskReport,
) -> None:
    """Write all outputs for a single trajectory."""
    task_dir = output_dir / trajectory.submission_id
    task_dir.mkdir(parents=True, exist_ok=True)

    session_id = trajectory.session_uuid or str(uuid.uuid4())
    base_ts = _current_ts()

    # 1. trajectory.jsonl — rewritten privacy-compliant session
    jsonl_events = []

    # Session event
    jsonl_events.append({
        "type": "session",
        "id": session_id,
        "createdAt": base_ts,
    })

    # Use thread_order to reconstruct proper interleaving.
    # Build a map: for each assistant turn_index, determine which user message (if any) precedes it.
    user_before_turn: dict[int, int] = _build_user_before_turn_map(trajectory)

    # Emit events in thread order
    emitted_user_msgs: set[int] = set()
    last_emitted_role = None

    for rt in rewrite_result.turns:
        # Emit the original user message preceding this turn (only once per turn_index)
        # BUT skip if a synthetic user message will immediately follow (prevents consecutive user msgs)
        if rt.turn_index in user_before_turn and rt.turn_index not in emitted_user_msgs:
            # Don't emit user message if this turn has a synthetic user message
            # (it would create user→user sequence)
            if not rt.synthetic_user_message:
                u_idx = user_before_turn[rt.turn_index]
                if u_idx < len(trajectory.user_messages):
                    jsonl_events.append({
                        "type": "message",
                        "id": _generate_event_id(),
                        "message": {
                            "role": "user",
                            "content": [{"type": "text", "text": trajectory.user_messages[u_idx]}],
                            "timestamp": base_ts + (rt.turn_index * 5000) - 1000,
                        }
                    })
                    last_emitted_role = "user"
            emitted_user_msgs.add(rt.turn_index)

        # If this is an adversarial turn, emit the adversarial user message BEFORE the refusal
        if rt.is_adversarial and rt.adversarial_user_message:
            jsonl_events.append({
                "type": "message",
                "id": _generate_event_id(),
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": rt.adversarial_user_message}],
                    "timestamp": base_ts + (rt.turn_index * 5000) + 2000,
                    "metadata": {
                        "synthetic": True,
                        "is_adversarial": True,
                        "attack_type": rt.attack_type,
                    },
                }
            })
            last_emitted_role = "user"

        # If this turn has a synthetic user message (consent flow response),
        # emit it BEFORE the assistant's execution message
        if rt.synthetic_user_message:
            jsonl_events.append({
                "type": "message",
                "id": _generate_event_id(),
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": rt.synthetic_user_message}],
                    "timestamp": base_ts + (rt.turn_index * 5000) + 2500,
                    "metadata": {
                        "synthetic": True,
                        "consent_decision": rt.consent_decision,
                    },
                }
            })

        # Assistant turn events
        turn_events = _turn_to_jsonl_events(rt, session_id, base_ts)
        jsonl_events.extend(turn_events)

    # Write JSONL
    jsonl_path = task_dir / "trajectory.jsonl"
    with open(jsonl_path, "w") as f:
        for event in jsonl_events:
            f.write(json.dumps(event) + "\n")

    # 2. workspace_before/ — pre-trajectory workspace
    wb_dir = task_dir / "workspace_before"
    wb_dir.mkdir(exist_ok=True)
    for filename, content in trajectory.workspace_files.items():
        file_path = wb_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    # 3. workspace/ — post-trajectory workspace (includes writes from rewritten turns)
    ws_dir = task_dir / "workspace"
    ws_dir.mkdir(exist_ok=True)
    # Start with pre-trajectory files
    post_files = dict(trajectory.workspace_files)
    # Apply write tool calls from rewritten turns
    for rt in rewrite_result.turns:
        for tc in rt.tool_calls:
            if isinstance(tc, dict) and tc.get("name") == "write":
                args = tc.get("arguments", {})
                raw_path = args.get("file_path", args.get("path", ""))
                content = args.get("content", "")
                if raw_path:
                    # Strip various workspace prefixes to get a relative path
                    clean_path = raw_path
                    for prefix in ("/home/user/OpenClawTrainer/workspace/", "/home/user/.openclaw/workspace/", "/workspace/"):
                        if clean_path.startswith(prefix):
                            clean_path = clean_path[len(prefix):]
                            break
                    # Avoid absolute paths escaping the output directory
                    clean_path = clean_path.lstrip("/")
                    if clean_path and not clean_path.startswith(".."):
                        post_files[clean_path] = content

    for filename, content in post_files.items():
        file_path = ws_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)

    # 4. metadata.json
    metadata = {
        "task_id": trajectory.task_id,
        "submission_id": trajectory.submission_id,
        "worker_id": trajectory.worker_id,
        "persona_id": trajectory.persona.get("persona_id", ""),
        "persona_name": trajectory.persona.get("name", ""),
        "session_uuid": session_id,
        "pii_map": {
            "max_level": pii_map.max_level,
            "has_l4": pii_map.has_l4,
            "has_l3": pii_map.has_l3,
            "entity_count": len(pii_map.entities),
            "labels": pii_map.labels_present,
            "entities": [
                {"text": e.text, "label": e.label, "level": e.level, "engines": e.engines}
                for e in pii_map.entities
            ],
        },
        "scenarios_covered": rewrite_result.scenarios_covered,
        "skills_used": rewrite_result.skills_used,
        "privacy_decision_points": rewrite_result.privacy_decision_points,
        "rewrite_repairs": rewrite_result.rewrite_repairs,
        "verification": {
            "verdict": verification.verdict,
            "privacy_compliance": verification.privacy_compliance,
            "correctness": verification.correctness,
            "completeness": verification.completeness,
            "efficiency": verification.efficiency,
            "naturality": verification.naturality,
            "overall": verification.overall,
            "rationale": verification.rationale,
            "issue_count": len(verification.issues),
        },
        "report": {
            "status": report.status,
            "refix_iterations": report.refix_iterations,
            "total_tokens": report.total_tokens,
        },
    }
    (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info("Written output for %s → %s", trajectory.task_id, task_dir)


def write_sft_entry(
    trajectory: ParsedTrajectory,
    rewrite_result: RewriteResult,
) -> dict:
    """Generate a single SFT dataset entry (flat format for training).

    Groups consecutive assistant turns into a single assistant message to ensure
    proper user/assistant alternation required for SFT training.
    """
    messages = []

    # Use thread_order to determine proper user/assistant interleaving
    user_before_turn: dict[int, int] = _build_user_before_turn_map(trajectory)

    # Group consecutive assistant turns that share the same preceding user message
    current_assistant_parts: list[str] = []

    def _flush_assistant():
        if current_assistant_parts:
            messages.append({
                "role": "assistant",
                "content": "\n\n".join(current_assistant_parts),
            })
            current_assistant_parts.clear()

    emitted_sft_user: set[int] = set()
    for rt in rewrite_result.turns:
        # Emit user message if this is the first assistant turn after a new user message
        if rt.turn_index in user_before_turn and rt.turn_index not in emitted_sft_user:
            _flush_assistant()
            u_idx = user_before_turn[rt.turn_index]
            if u_idx < len(trajectory.user_messages):
                messages.append({
                    "role": "user",
                    "content": trajectory.user_messages[u_idx],
                })
            emitted_sft_user.add(rt.turn_index)

        # If this is an adversarial turn, flush and insert the adversarial user message
        if rt.is_adversarial and rt.adversarial_user_message:
            _flush_assistant()
            messages.append({
                "role": "user",
                "content": rt.adversarial_user_message,
            })

        # If this turn carries a synthetic user message (consent flow),
        # flush the previous assistant and insert the user response
        if rt.synthetic_user_message:
            _flush_assistant()
            messages.append({
                "role": "user",
                "content": rt.synthetic_user_message,
            })

        # Build assistant content for this turn
        content_parts = []
        if rt.thinking:
            content_parts.append(f"<thinking>\n{rt.thinking}\n</thinking>")
        if rt.text:
            content_parts.append(rt.text)
        for tc in rt.tool_calls:
            if isinstance(tc, dict):
                content_parts.append(
                    f"<tool_call>\n{json.dumps(tc, indent=2)}\n</tool_call>"
                )
        for tr in rt.tool_results:
            if isinstance(tr, dict):
                content_parts.append(
                    f"<tool_result>\n{json.dumps(tr, indent=2)}\n</tool_result>"
                )

        if content_parts:
            current_assistant_parts.append("\n\n".join(content_parts))

    # Flush remaining assistant parts
    _flush_assistant()

    return {
        "task_id": trajectory.task_id,
        "submission_id": trajectory.submission_id,
        "persona_id": trajectory.persona.get("persona_id", ""),
        "scenarios": rewrite_result.scenarios_covered,
        "messages": messages,
    }


def write_batch_report(
    output_dir: Path,
    reports: list[TaskReport],
    token_summary: dict,
) -> None:
    """Write batch-level summary report."""
    total = len(reports)
    passed = sum(1 for r in reports if r.status == "PASS")
    minor_fixed = sum(1 for r in reports if r.status == "MINOR_FIXED")
    failed = sum(1 for r in reports if r.status == "FAIL")
    errors = total - passed - minor_fixed - failed

    batch_report = {
        "total_trajectories": total,
        "passed": passed,
        "minor_fixed": minor_fixed,
        "failed": failed,
        "errors": errors,
        "pass_rate": f"{(passed + minor_fixed) / max(total, 1) * 100:.1f}%",
        "token_usage": token_summary,
        "per_task": [
            {
                "task_id": r.task_id,
                "submission_id": r.submission_id,
                "status": r.status,
                "pii_level": r.max_pii_level,
                "scenarios": r.scenarios_covered,
                "refix_iterations": r.refix_iterations,
                "error": r.error_message,
            }
            for r in reports
        ],
    }

    (output_dir / "batch_report.json").write_text(json.dumps(batch_report, indent=2))
    logger.info(
        "Batch report: %d total, %d passed, %d minor-fixed, %d failed, %d errors",
        total, passed, minor_fixed, failed, errors,
    )


# ---------------------------------------------------------------------------
# RLHF Output — Multi-level preference pairs
# ---------------------------------------------------------------------------

def write_rlhf_pairs(
    output_dir: Path,
    pairs: list[RLHFPair],
    batch_report: RLHFBatchReport | None = None,
) -> None:
    """Write RLHF preference pairs in multiple formats for training.

    Generates:
    1. rlhf_pairs.jsonl — full step-level pairs (chosen + rejected alternatives)
    2. rlhf_dpo.jsonl — DPO-format pairs (one chosen + one rejected per line)
    3. rlhf_turn_level.jsonl — turn-level aggregated pairs
    4. rlhf_report.json — batch statistics
    """
    rlhf_dir = output_dir / "rlhf"
    rlhf_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full pairs (step-level with all alternatives)
    full_path = rlhf_dir / "rlhf_pairs.jsonl"
    with open(full_path, "w") as f:
        for pair in pairs:
            record = _pair_to_dict(pair)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Written %d full RLHF pairs → %s", len(pairs), full_path)

    # 2. DPO-format (one line per chosen/rejected pair, expanding alternatives)
    dpo_path = rlhf_dir / "rlhf_dpo.jsonl"
    dpo_count = 0
    with open(dpo_path, "w") as f:
        for pair in pairs:
            for rejected_step in pair.rejected:
                dpo_record = _build_dpo_record(pair, rejected_step)
                f.write(json.dumps(dpo_record, ensure_ascii=False) + "\n")
                dpo_count += 1
    logger.info("Written %d DPO pairs → %s", dpo_count, dpo_path)

    # 3. Turn-level aggregated pairs
    turn_pairs = _aggregate_turn_level(pairs)
    turn_path = rlhf_dir / "rlhf_turn_level.jsonl"
    with open(turn_path, "w") as f:
        for tp in turn_pairs:
            f.write(json.dumps(tp, ensure_ascii=False) + "\n")
    logger.info("Written %d turn-level pairs → %s", len(turn_pairs), turn_path)

    # 4. Batch report
    if batch_report:
        report_dict = {
            "task_id": batch_report.task_id,
            "submission_id": batch_report.submission_id,
            "total_pairs": batch_report.total_pairs,
            "pairs_by_failure_mode": batch_report.pairs_by_failure_mode,
            "pairs_by_level": batch_report.pairs_by_level,
            "avg_reward_score": batch_report.avg_reward_score,
            "over_refusal_ratio": batch_report.over_refusal_ratio,
            "violation_ratio": batch_report.violation_ratio,
            "balance_check": _check_balance(pairs),
        }
        (rlhf_dir / "rlhf_report.json").write_text(json.dumps(report_dict, indent=2))


def _pair_to_dict(pair: RLHFPair) -> dict:
    """Convert an RLHFPair to a serializable dict."""
    return {
        "task_id": pair.task_id,
        "submission_id": pair.submission_id,
        "turn_index": pair.turn_index,
        "step_criticality": pair.step_criticality,
        "pair_level": pair.pair_level,
        "decision_branch": pair.decision_branch,
        "data_level_involved": pair.data_level_involved,
        "tool_tier_involved": pair.tool_tier_involved,
        "context": pair.context,
        "chosen": pair.chosen,
        "rejected": [
            {
                "thinking": s.thinking,
                "tool_call": s.tool_call,
                "tool_response": s.tool_response,
                "assistant_response": s.assistant_response,
                "failure_mode": s.failure_mode,
                "privacy_violation": {
                    "flag": s.privacy_violation.flag,
                    "rule": s.privacy_violation.rule,
                    "severity": s.privacy_violation.severity,
                    "data_level": s.privacy_violation.data_level,
                    "tool_tier": s.privacy_violation.tool_tier,
                },
                "reward_score": s.reward_score,
                "perturbation_type": s.perturbation_type,
            }
            for s in pair.rejected
        ],
    }


def _build_dpo_record(pair: RLHFPair, rejected_step: RejectedStep) -> dict:
    """Build a single DPO training record (chosen vs one rejected)."""
    context_str = "\n".join(
        f"[{m.get('role', 'user')}]: {m.get('content', '')[:200]}"
        for m in pair.context
    )

    chosen_text = _step_to_text(pair.chosen)
    rejected_text = _rejected_step_to_text(rejected_step)

    return {
        "prompt": context_str,
        "chosen": chosen_text,
        "rejected": rejected_text,
        "metadata": {
            "task_id": pair.task_id,
            "turn_index": pair.turn_index,
            "failure_mode": rejected_step.failure_mode,
            "reward_score": rejected_step.reward_score,
            "step_criticality": pair.step_criticality,
            "data_level": pair.data_level_involved,
            "tool_tier": pair.tool_tier_involved,
        },
    }


def _step_to_text(step: dict) -> str:
    """Convert a chosen step dict to a text representation for DPO."""
    parts = []
    if step.get("thinking"):
        parts.append(f"<thinking>\n{step['thinking']}\n</thinking>")
    if step.get("tool_call"):
        parts.append(f"<tool_call>\n{json.dumps(step['tool_call'], indent=2)}\n</tool_call>")
    if step.get("assistant_response"):
        parts.append(step["assistant_response"])
    return "\n\n".join(parts) if parts else ""


def _rejected_step_to_text(step: RejectedStep) -> str:
    """Convert a RejectedStep to a text representation for DPO."""
    parts = []
    if step.thinking:
        parts.append(f"<thinking>\n{step.thinking}\n</thinking>")
    if step.tool_call:
        parts.append(f"<tool_call>\n{json.dumps(step.tool_call, indent=2)}\n</tool_call>")
    if step.assistant_response:
        parts.append(step.assistant_response)
    return "\n\n".join(parts) if parts else ""


def _aggregate_turn_level(pairs: list[RLHFPair]) -> list[dict]:
    """Aggregate step-level pairs into turn-level pairs.

    Groups all steps for the same turn_index and creates a single
    turn-level pair with combined chosen and worst-rejected.
    """
    by_turn: dict[int, list[RLHFPair]] = {}
    for p in pairs:
        by_turn.setdefault(p.turn_index, []).append(p)

    turn_pairs = []
    for turn_idx, turn_steps in sorted(by_turn.items()):
        all_rejected = []
        for step in turn_steps:
            all_rejected.extend(step.rejected)

        if not all_rejected:
            continue

        worst = min(all_rejected, key=lambda r: r.reward_score)

        combined_context = turn_steps[0].context if turn_steps else []
        combined_chosen = {
            "steps": [p.chosen for p in turn_steps],
            "reward_score": 1.0,
        }

        turn_pairs.append({
            "pair_level": "turn",
            "turn_index": turn_idx,
            "task_id": turn_steps[0].task_id,
            "submission_id": turn_steps[0].submission_id,
            "context": combined_context,
            "chosen": combined_chosen,
            "rejected": {
                "thinking": worst.thinking,
                "tool_call": worst.tool_call,
                "assistant_response": worst.assistant_response,
                "failure_mode": worst.failure_mode,
                "reward_score": worst.reward_score,
            },
            "all_rejected_count": len(all_rejected),
            "avg_rejected_score": sum(r.reward_score for r in all_rejected) / len(all_rejected),
        })

    return turn_pairs


def _check_balance(pairs: list[RLHFPair]) -> dict:
    """Check pair balance requirements."""
    all_rejected = [s for p in pairs for s in p.rejected]
    total = len(all_rejected)
    if total == 0:
        return {"status": "empty", "total": 0}

    mode_counts: dict[str, int] = {}
    for s in all_rejected:
        mode_counts[s.failure_mode] = mode_counts.get(s.failure_mode, 0) + 1

    over_refusal_pct = mode_counts.get("over_refusal", 0) / total
    violation_count = sum(1 for s in all_rejected if s.privacy_violation.flag)
    violation_pct = violation_count / total

    chosen_lengths = []
    rejected_lengths = []
    for p in pairs:
        chosen_text = _step_to_text(p.chosen)
        chosen_lengths.append(len(chosen_text.split()))
        for s in p.rejected:
            rejected_lengths.append(len(_rejected_step_to_text(s).split()))

    avg_chosen = sum(chosen_lengths) / max(len(chosen_lengths), 1)
    avg_rejected = sum(rejected_lengths) / max(len(rejected_lengths), 1)
    length_ratio = avg_rejected / max(avg_chosen, 1)

    return {
        "status": "ok" if over_refusal_pct >= 0.2 else "needs_more_over_refusals",
        "total_rejected": total,
        "failure_mode_distribution": mode_counts,
        "over_refusal_pct": round(over_refusal_pct, 3),
        "violation_pct": round(violation_pct, 3),
        "avg_chosen_words": round(avg_chosen, 1),
        "avg_rejected_words": round(avg_rejected, 1),
        "length_ratio": round(length_ratio, 3),
        "length_balanced": 0.8 <= length_ratio <= 1.2,
    }
