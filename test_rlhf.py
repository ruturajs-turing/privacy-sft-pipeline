#!/usr/bin/env python3
"""Standalone test for the RLHF pair generator — runs on T-033-02 test trajectory.

Usage:
    python3 test_rlhf.py [--dry-run]

With --dry-run, identifies decision points and simulates pair generation without calling LLMs.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
logger = logging.getLogger("test_rlhf")

TEST_SUBMISSION = "347d195e-031d-4ad4-966b-bf85c050e604"
TEST_TASK = "T-033-02"

# This file is a standalone smoke script. Pytest should not collect its async
# LLM-facing helpers as unit tests.
__test__ = False


def _build_mock_trajectory():
    """Build a minimal ParsedTrajectory from the existing test output."""
    from models import ParsedTrajectory, AssistantTurn, ToolCall, ToolResult

    test_dir = Path(__file__).parent / "test_output" / TEST_SUBMISSION
    jsonl_path = test_dir / "trajectory.jsonl"

    if not jsonl_path.exists():
        logger.error("Test trajectory not found: %s", jsonl_path)
        sys.exit(1)

    events = []
    with open(jsonl_path) as f:
        for line in f:
            events.append(json.loads(line.strip()))

    user_messages = []
    assistant_turns = []
    tool_results_by_call_id = {}
    thread_order = []
    turn_idx = 0

    for event in events:
        if event.get("type") != "message":
            continue
        msg = event.get("message", {})
        role = msg.get("role")

        if role == "user":
            content = msg.get("content", [])
            text = ""
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text += block.get("text", "")
            user_messages.append(text)
            thread_order.append(("user", len(user_messages) - 1))

        elif role == "assistant":
            content = msg.get("content", [])
            thinking_blocks = []
            text_blocks = []
            tool_calls = []

            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "thinking":
                        thinking_blocks.append(block.get("thinking", ""))
                    elif block.get("type") == "text":
                        text_blocks.append(block.get("text", ""))
                    elif block.get("type") == "toolCall":
                        tool_calls.append(ToolCall(
                            call_id=block.get("id", ""),
                            name=block.get("name", ""),
                            arguments=block.get("arguments", {}),
                        ))

            turn = AssistantTurn(
                event_id=event.get("id", ""),
                turn_index=turn_idx,
                thinking_blocks=thinking_blocks,
                text_blocks=text_blocks,
                tool_calls=tool_calls,
            )
            assistant_turns.append(turn)
            thread_order.append(("assistant", turn_idx))
            turn_idx += 1

        elif role == "toolResult":
            call_id = msg.get("toolCallId", "")
            content_text = ""
            raw_content = msg.get("content", [])
            for block in raw_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    content_text += block.get("text", "")
            tool_results_by_call_id[call_id] = ToolResult(
                call_id=call_id,
                tool_name=msg.get("toolName", ""),
                content=content_text,
                is_error=msg.get("isError", False),
                is_empty=not content_text,
            )

    traj = ParsedTrajectory(
        task_id=TEST_TASK,
        submission_id=TEST_SUBMISSION,
        worker_id="test-worker",
        session_uuid="test-session",
        jsonl_path=str(jsonl_path),
        user_messages=user_messages,
        assistant_turns=assistant_turns,
        tool_results_by_call_id=tool_results_by_call_id,
        thread_order=thread_order,
    )
    return traj


def _build_mock_rewrite_result(trajectory):
    """Build a mock RewriteResult from the trajectory's assistant turns."""
    from models import RewriteResult, RewrittenTurn

    turns = []
    for at in trajectory.assistant_turns:
        tool_calls_dicts = [
            {"name": tc.name, "arguments": tc.arguments, "call_id": tc.call_id}
            for tc in at.tool_calls
        ]
        tool_results_dicts = []
        for tc in at.tool_calls:
            if tc.call_id in trajectory.tool_results_by_call_id:
                tr = trajectory.tool_results_by_call_id[tc.call_id]
                tool_results_dicts.append({
                    "call_id": tr.call_id,
                    "tool_name": tr.tool_name,
                    "content": tr.content[:200],
                    "is_error": tr.is_error,
                })

        privacy_actions = []
        thinking = "\n".join(at.thinking_blocks)
        if "L3" in thinking or "HEALTH" in thinking:
            privacy_actions.append("privacy_classification")
        if "Gate" in thinking:
            privacy_actions.append("gate_evaluation")
        if "consent" in thinking.lower():
            privacy_actions.append("consent_check")

        turns.append(RewrittenTurn(
            turn_index=at.turn_index,
            thinking=thinking,
            text="\n".join(at.text_blocks),
            tool_calls=tool_calls_dicts,
            tool_results=tool_results_dicts,
            privacy_actions=privacy_actions,
            scenario="A",
        ))

    return RewriteResult(
        task_id=TEST_TASK,
        submission_id=TEST_SUBMISSION,
        turns=turns,
        scenarios_covered=["A"],
        skills_used=["memory_search", "exec", "read", "write", "health"],
        privacy_decision_points=len([t for t in turns if t.privacy_actions]),
    )


def _build_mock_pii_map():
    """Build a mock PIIMap for the asthma checkup trajectory."""
    from models import PIIMap, PIIEntity

    entities = [
        PIIEntity(text="asthma", label="HEALTH_DIAGNOSIS", level="L3", turn_index=-1, engines=["llm"]),
        PIIEntity(text="shortness of breath", label="HEALTH_DIAGNOSIS", level="L3", turn_index=6, engines=["llm"]),
        PIIEntity(text="Albuterol", label="HEALTH_MEDICATION", level="L3", turn_index=6, engines=["llm"]),
        PIIEntity(text="Dr. Lokajc", label="HEALTH_PROVIDER_APPT", level="L2", turn_index=12, engines=["llm"]),
        PIIEntity(text="AET-879-376-2674", label="HEALTH_INSURANCE_ID", level="L2", turn_index=12, engines=["llm"]),
    ]

    return PIIMap(
        entities=entities,
        max_level="L3",
        has_l4=False,
        has_l3=True,
        labels_present=["HEALTH_DIAGNOSIS", "HEALTH_MEDICATION", "HEALTH_PROVIDER_APPT", "HEALTH_INSURANCE_ID"],
    )


async def test_decision_points():
    """Test decision point identification without LLM calls."""
    from rlhf_generator import identify_decision_points

    trajectory = _build_mock_trajectory()
    rewrite_result = _build_mock_rewrite_result(trajectory)
    pii_map = _build_mock_pii_map()

    logger.info("Trajectory: %d user messages, %d assistant turns", len(trajectory.user_messages), len(trajectory.assistant_turns))
    logger.info("Rewrite: %d turns, %d privacy decision points", len(rewrite_result.turns), rewrite_result.privacy_decision_points)
    logger.info("PII: %d entities, max level %s", len(pii_map.entities), pii_map.max_level)

    points = identify_decision_points(trajectory, rewrite_result, pii_map)

    logger.info("\n=== Decision Points Found: %d ===", len(points))
    for i, dp in enumerate(points):
        logger.info(
            "  [%d] Turn %d | Tool: %s (T%s) | Data: %s | Branch: %s | Criticality: %.2f | Scenario: %s",
            i, dp.turn_index, dp.tool_name, dp.tool_tier[1:] if dp.tool_tier else "?",
            dp.data_level, dp.decision_branch, dp.criticality, dp.scenario,
        )

    return trajectory, rewrite_result, pii_map, points


async def test_full_generation():
    """Test full RLHF pair generation with LLM calls."""
    from rlhf_generator import generate_rlhf_pairs, compute_batch_report
    from reward_scorer import score_rejected_alternatives
    from writer import write_rlhf_pairs

    trajectory = _build_mock_trajectory()
    rewrite_result = _build_mock_rewrite_result(trajectory)
    pii_map = _build_mock_pii_map()

    logger.info("=== Running full RLHF generation ===")
    pairs = await generate_rlhf_pairs(trajectory, rewrite_result, pii_map, max_pairs_per_trajectory=20)

    if not pairs:
        logger.warning("No pairs generated (check API keys)")
        return

    logger.info("Generated %d pairs, scoring with PRM...", len(pairs))
    pairs = await score_rejected_alternatives(pairs, task_description="Asthma checkup questions for doctor")

    report = compute_batch_report(TEST_TASK, TEST_SUBMISSION, pairs)
    logger.info("\n=== RLHF Report ===")
    logger.info("  Total pairs: %d", report.total_pairs)
    logger.info("  By failure mode: %s", json.dumps(report.pairs_by_failure_mode, indent=4))
    logger.info("  Avg reward score: %.3f", report.avg_reward_score)
    logger.info("  Over-refusal ratio: %.3f", report.over_refusal_ratio)
    logger.info("  Violation ratio: %.3f", report.violation_ratio)

    output_dir = Path(__file__).parent / "test_output" / TEST_SUBMISSION
    write_rlhf_pairs(output_dir, pairs, report)
    logger.info("\nOutput written to %s/rlhf/", output_dir)


async def main():
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        logger.info("=== DRY RUN: Decision point identification only ===\n")
        trajectory, rewrite_result, pii_map, points = await test_decision_points()

        if points:
            logger.info("\n=== Would generate ~10 rejected alternatives across the top decision points ===")
            logger.info("  Covering failure modes: wrong_tool_tier_up/down, wrong_param, over_refusal, hallucination, etc.")
        else:
            logger.warning("No decision points found — RLHF generation would produce no pairs")
    else:
        await test_full_generation()


if __name__ == "__main__":
    asyncio.run(main())
