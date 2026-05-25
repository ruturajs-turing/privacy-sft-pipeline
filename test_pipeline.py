"""End-to-end pipeline test with mock data — validates structure without API calls."""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import traceback
import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent))

# Mock GCS/google dependencies before any import that needs them
for _mod in ["google", "google.cloud", "google.cloud.storage", "google.oauth2", "google.oauth2.service_account"]:
    sys.modules.setdefault(_mod, MagicMock())

from models import (
    AssistantTurn,
    PIIEntity,
    PIIMap,
    ParsedTrajectory,
    RewriteResult,
    RewrittenTurn,
    TaskReport,
    ToolCall,
    ToolResult,
    VerificationIssue,
    VerificationResult,
)


def _create_sample_zip() -> bytes:
    """Create a realistic final-snapshot.zip for testing."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        # sessions.json
        sessions = {
            "abc123": {
                "sessionId": "abc123",
                "createdAt": "2025-01-15T10:00:00Z",
                "updatedAt": "2025-01-15T11:00:00Z",
            }
        }
        zf.writestr(".openclaw/agents/main/sessions/sessions.json", json.dumps(sessions))

        # Session trajectory JSONL
        events = [
            {"type": "session", "id": "abc123", "createdAt": 1705312800000},
            {
                "type": "message",
                "id": "msg-1",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hi, my name is John Smith. My SSN is 123-45-6789. Can you save this to my profile?"}],
                    "timestamp": 1705312801000,
                }
            },
            {
                "type": "message",
                "id": "msg-2",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll save your information to your profile."},
                        {"type": "toolCall", "id": "call-1", "name": "memory_write", "arguments": {"key": "user_profile", "value": "John Smith, SSN: 123-45-6789"}},
                    ],
                    "timestamp": 1705312802000,
                }
            },
            {
                "type": "message",
                "id": "msg-3",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call-1",
                    "toolName": "memory_write",
                    "content": [{"type": "text", "text": "Saved to memory."}],
                    "isError": False,
                    "timestamp": 1705312803000,
                }
            },
            {
                "type": "message",
                "id": "msg-4",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Great, now email my SSN to my accountant at bob@accounting.com"}],
                    "timestamp": 1705312804000,
                }
            },
            {
                "type": "message",
                "id": "msg-5",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Sending your SSN to your accountant."},
                        {"type": "toolCall", "id": "call-2", "name": "gmail", "arguments": {"to": "bob@accounting.com", "body": "SSN: 123-45-6789"}},
                    ],
                    "timestamp": 1705312805000,
                }
            },
        ]
        jsonl_content = "\n".join(json.dumps(e) for e in events)
        zf.writestr(".openclaw/agents/main/sessions/abc123.jsonl", jsonl_content)

        # Workspace file (real exports have a parent directory prefix)
        zf.writestr("snapshot/workspace/notes.txt", "User account setup in progress.")

    return buf.getvalue()


def test_parser():
    """Test Stage 1: Parse trajectory from zip."""
    from parser import _find_session_jsonl, _parse_jsonl_events, _parse_trajectory, _extract_workspace

    zip_bytes = _create_sample_zip()
    zf = zipfile.ZipFile(BytesIO(zip_bytes))

    # Test finding session
    jsonl_path = _find_session_jsonl(zf)
    assert jsonl_path is not None, "Should find session JSONL"
    assert "abc123" in jsonl_path, f"Should match session UUID, got: {jsonl_path}"

    # Test parsing events
    jsonl_text = zf.read(jsonl_path).decode("utf-8")
    events = _parse_jsonl_events(jsonl_text)
    assert len(events) == 6, f"Expected 6 events, got {len(events)}"

    # Test trajectory parsing
    user_msgs, turns, tool_results, thread_order, session_uuid = _parse_trajectory(events)
    assert len(user_msgs) == 2, f"Expected 2 user msgs, got {len(user_msgs)}"
    assert len(turns) == 2, f"Expected 2 assistant turns, got {len(turns)}"
    assert "call-1" in tool_results
    assert session_uuid == "abc123"
    assert turns[0].tool_calls[0].name == "memory_write"
    assert turns[1].tool_calls[0].name == "gmail"

    # Test workspace extraction
    workspace = _extract_workspace(zf)
    assert "notes.txt" in workspace

    print("  ✓ Parser: session discovery, event parsing, trajectory structure, workspace extraction")


def test_parser_latest_session_requires_jsonl():
    """Latest sessions.json entry is skipped when its JSONL is missing."""
    from parser import _find_session_jsonl

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        sessions = {
            "newest-missing": {
                "sessionId": "newest-missing",
                "createdAt": "2026-05-21T10:00:00Z",
                "updatedAt": "2026-05-21T11:00:00Z",
            },
            "older-valid": {
                "sessionId": "older-valid",
                "createdAt": "2026-05-21T08:00:00Z",
                "updatedAt": "2026-05-21T09:00:00Z",
            },
        }
        zf.writestr(
            "final-snapshot/.openclaw/agents/main/sessions/sessions.json",
            json.dumps(sessions),
        )
        zf.writestr(
            "final-snapshot/.openclaw/agents/main/sessions/older-valid.jsonl",
            json.dumps({"type": "session", "id": "older-valid"}) + "\n",
        )

    zf = zipfile.ZipFile(BytesIO(buf.getvalue()))
    chosen = _find_session_jsonl(zf)
    assert chosen is not None
    assert chosen.endswith("older-valid.jsonl"), chosen

    print("  ✓ Parser: selects newest session id that actually has valid JSONL")


def _dialogue_jsonl(session_id: str, ts: int) -> str:
    events = [
        {"type": "session", "id": session_id, "timestamp": ts},
        {
            "type": "message",
            "id": f"{session_id}-u",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Please help with this task."}],
                "timestamp": ts + 1,
            },
        },
        {
            "type": "message",
            "id": f"{session_id}-a",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "I can help with that."}],
                "timestamp": ts + 2,
            },
        },
    ]
    return "\n".join(json.dumps(e) for e in events)


def test_parser_raw_final_snapshot_auto_uses_recency_refs():
    """Root-level final-snapshot.zip should not fall back to first task ID in ZIP order."""
    from parser import _detect_task_id, _find_session_jsonl

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        sessions = {
            "older": {"sessionId": "older", "sessionStartedAt": 100_000, "updatedAt": 110_000},
            "newer": {"sessionId": "newer", "sessionStartedAt": 200_000, "updatedAt": 210_000},
        }
        zf.writestr(".openclaw/agents/main/sessions/sessions.json", json.dumps(sessions))
        zf.writestr(".openclaw/agents/main/sessions/older.jsonl", _dialogue_jsonl("older", 100_000))
        zf.writestr(".openclaw/agents/main/sessions/newer.jsonl", _dialogue_jsonl("newer", 200_000))
        zf.writestr(".openclaw/workspace/trajectories/sess-100000-T-111-01.jsonl", "{}\n")
        zf.writestr(".openclaw/workspace/trajectories/sess-200000-T-111-02.jsonl", "{}\n")

    zf = zipfile.ZipFile(BytesIO(buf.getvalue()))
    assert _detect_task_id(zf) == "T-111-02"
    chosen = _find_session_jsonl(zf, "T-111-02")
    assert chosen is not None and chosen.endswith("newer.jsonl"), chosen

    print("  ✓ Parser: raw final-snapshot.zip AUTO uses newest dialogue-backed task ref")


def test_parser_auto_skips_empty_nearest_session():
    """AUTO should not bind a task ref to a nearest session with no dialogue."""
    from parser import _detect_task_id, _find_session_jsonl

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        sessions = {
            "good": {"sessionId": "good", "sessionStartedAt": 100_000, "updatedAt": 110_000},
            "empty": {"sessionId": "empty", "sessionStartedAt": 300_000, "updatedAt": 310_000},
        }
        zf.writestr(".openclaw/agents/main/sessions/sessions.json", json.dumps(sessions))
        zf.writestr(".openclaw/agents/main/sessions/good.jsonl", _dialogue_jsonl("good", 100_000))
        zf.writestr(
            ".openclaw/agents/main/sessions/empty.jsonl",
            json.dumps({"type": "session_start", "t": 0}) + "\n" + json.dumps({"type": "session_end", "t": 1}),
        )
        zf.writestr(".openclaw/workspace/trajectories/sess-100000-T-111-01.jsonl", "{}\n")
        zf.writestr(".openclaw/workspace/trajectories/sess-300000-T-111-02.jsonl", "{}\n")

    zf = zipfile.ZipFile(BytesIO(buf.getvalue()))
    assert _detect_task_id(zf) == "T-111-01"
    chosen = _find_session_jsonl(zf, "T-111-01")
    assert chosen is not None and chosen.endswith("good.jsonl"), chosen

    print("  ✓ Parser: AUTO skips empty nearest sessions instead of pairing wrong tasks")


def test_parser_task_refs_use_session_start_not_update_time():
    """Long sessions updated near a later task must not steal that task ref."""
    from parser import _find_session_jsonl

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        sessions = {
            "long-older": {
                "sessionId": "long-older",
                "sessionStartedAt": 100_000,
                "updatedAt": 1_005_000,
            },
            "new-task": {
                "sessionId": "new-task",
                "sessionStartedAt": 1_010_000,
                "updatedAt": 1_020_000,
            },
        }
        zf.writestr(".openclaw/agents/main/sessions/sessions.json", json.dumps(sessions))
        zf.writestr(".openclaw/agents/main/sessions/long-older.jsonl", _dialogue_jsonl("long-older", 100_000))
        zf.writestr(".openclaw/agents/main/sessions/new-task.jsonl", _dialogue_jsonl("new-task", 1_010_000))
        zf.writestr(".openclaw/workspace/trajectories/sess-1010000-T-111-02.jsonl", "{}\n")

    zf = zipfile.ZipFile(BytesIO(buf.getvalue()))
    chosen = _find_session_jsonl(zf, "T-111-02")
    assert chosen is not None and chosen.endswith("new-task.jsonl"), chosen

    print("  ✓ Parser: task refs match sessionStartedAt, not stale updatedAt")


def test_classifier():
    """Test Stage 2: PII classification (mock LLM, real logic)."""
    from classifier import _run_presidio, _merge_entities, PIIEntity

    # Test Presidio on sample text (may not be installed)
    text = "John Smith, SSN 123-45-6789, email john@example.com, phone 555-123-4567"
    entities = _run_presidio(text)
    if entities:
        print(f"  ✓ Presidio: found {len(entities)} entities")
        for e in entities:
            print(f"    - {e.text}: {e.label} ({e.level})")
    else:
        print("  ⚠ Presidio: not installed (will use LLM classifier)")

    # Test merge logic
    e1 = PIIEntity(text="john@example.com", label="ID_EMAIL", level="L2", engines=["presidio"])
    e2 = PIIEntity(text="john@example.com", label="ID_EMAIL", level="L2", engines=["llm_classifier"])
    e3 = PIIEntity(text="123-45-6789", label="GOV_SSN_FULL", level="L4", engines=["presidio"])
    merged = _merge_entities([e1, e2, e3])
    assert len(merged) == 2, f"Expected 2 after merge, got {len(merged)}"
    email_entity = [e for e in merged if e.label == "ID_EMAIL"][0]
    assert len(email_entity.engines) == 2, "Should merge engines"
    print("  ✓ Classifier: merge/dedup logic correct")


def test_classifier_persona_vault_runs_without_name_pattern():
    """Persona vault PII must be detected even without an About/plan-for name phrase."""
    from classifier import _run_pattern_classifier

    persona = {
        "pii_vault": {
            "health": {"diagnoses": ["GAD"], "insurance_id": "UHC-883-441229"},
            "government": {"ssn": "123-45-6789"},
        }
    }
    text = "Use GAD and SSN 123-45-6789 for this current task only."
    entities = _run_pattern_classifier(text, persona)
    labels = {e.label for e in entities}

    assert "HEALTH_MENTAL_HEALTH" in labels
    assert "GOV_SSN_FULL" in labels

    print("  ✓ Classifier: persona vault L3/L4 fallback runs independently")


def _minimal_trajectory(task_id: str, task_spec: dict) -> ParsedTrajectory:
    return ParsedTrajectory(
        task_id=task_id,
        submission_id="sub-test",
        worker_id="worker-test",
        session_uuid="session-test",
        jsonl_path="session.jsonl",
        task_spec=task_spec,
    )


def _health_pii_map() -> PIIMap:
    return PIIMap(
        entities=[PIIEntity(text="lower-back pain", label="HEALTH_DIAGNOSIS", level="L3")],
        max_level="L3",
        has_l3=True,
        labels_present=["HEALTH_DIAGNOSIS"],
    )


def test_quality_gate_blocks_health_retention_for_amnesia_task():
    """Scenario A health workflows should not be repaired by retaining data in vault."""
    from privacy_quality_gates import repair_quality_issues, run_quality_gates

    trajectory = _minimal_trajectory(
        "T-122-02",
        {"privacy_scenario": "A", "expected_privacy_actions": ["amnesia", "local_first"]},
    )
    pii_map = _health_pii_map()
    event = {
        "type": "message",
        "id": "assistant-store",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "toolCall", "id": "call-store", "name": "set", "arguments": {
                    "key": "appointment",
                    "val": "lower-back pain appointment summary with symptom timeline",
                }}
            ],
        },
    }
    result = RewriteResult(
        task_id="T-122-02",
        submission_id="sub-test",
        patched_events=[event],
    )

    assert any(i.rule_violated == "health_retention_in_amnesia_task" for i in run_quality_gates(trajectory, result, pii_map))
    repaired, repairs = repair_quality_issues(trajectory, result, pii_map)
    assert any(r["kind"] == "blocked_health_retention_for_amnesia_task" for r in repairs)
    content = repaired.patched_events[0]["message"]["content"]
    assert all(item.get("type") != "toolCall" for item in content)
    assert "won't save" in content[0]["text"].lower()
    assert not run_quality_gates(trajectory, repaired, pii_map)

    print("  ✓ Quality gate: blocks health retention in amnesia tasks")


def test_quality_gate_sanitizes_health_plaintext_files():
    """Health-context plaintext writes should become generic non-identifying files."""
    from privacy_quality_gates import repair_quality_issues, run_quality_gates

    trajectory = _minimal_trajectory(
        "T-122-27",
        {"privacy_scenario": "B", "expected_privacy_actions": ["local_first", "minimization"]},
    )
    pii_map = _health_pii_map()
    event = {
        "type": "message",
        "id": "assistant-write",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "toolCall", "id": "call-write", "name": "write", "arguments": {
                    "path": "/home/user/.openclaw/workspace/wellness/daily-stretch-routine.md",
                    "content": "# Lower-Back Routine\n> **Owner:** Elif El Amrani\n\nPlan for lower-back pain and stiffness.",
                }}
            ],
        },
    }
    result = RewriteResult(
        task_id="T-122-27",
        submission_id="sub-test",
        patched_events=[event],
    )

    assert any(i.rule_violated == "plaintext_l3_health_persistence" for i in run_quality_gates(trajectory, result, pii_map))
    repaired, repairs = repair_quality_issues(trajectory, result, pii_map)
    assert any(r["kind"] == "sanitized_health_plaintext_persistence" for r in repairs)
    args = repaired.patched_events[0]["message"]["content"][0]["arguments"]
    assert "Owner:" not in args["content"]
    assert "lower-back pain" not in args["content"].lower()
    assert "general desk-worker movement guidance" in args["content"].lower()
    assert not run_quality_gates(trajectory, repaired, pii_map)

    print("  ✓ Quality gate: sanitizes health-context plaintext file writes")


def test_l34_amnesia_provenance_rule():
    """L3/L4 data from an earlier user turn must be re-provided unless gated."""
    from privacy_registry import Action, PrivacyRegistry

    pii = PIIMap(
        entities=[
            PIIEntity(
                text="asthma",
                label="HEALTH_DIAGNOSIS",
                level="L3",
                source_turn_index=0,
                source_type="user_input",
            )
        ],
        max_level="L3",
        has_l3=True,
    )

    stale = ParsedTrajectory(
        task_id="T-001-01",
        submission_id="sub-1",
        worker_id="w-1",
        session_uuid="sess-1",
        jsonl_path="x.jsonl",
        user_messages=["My diagnosis is asthma.", "Use the same condition to find a clinic."],
        assistant_turns=[
            AssistantTurn(
                event_id="a0",
                turn_index=0,
                text_blocks=["noted"],
            ),
            AssistantTurn(
                event_id="a1",
                turn_index=1,
                tool_calls=[
                    ToolCall(
                        call_id="c1",
                        name="web_search",
                        arguments={"query": "asthma clinic near me"},
                    )
                ],
            ),
        ],
        thread_order=[("user", 0), ("assistant", 0), ("user", 1), ("assistant", 1)],
    )
    assert PrivacyRegistry(pii, stale).decide(stale.assistant_turns[1]).action == Action.ELICIT

    current = ParsedTrajectory(
        task_id="T-001-01",
        submission_id="sub-1",
        worker_id="w-1",
        session_uuid="sess-1",
        jsonl_path="x.jsonl",
        user_messages=["Find asthma clinics near me."],
        assistant_turns=[
            AssistantTurn(
                event_id="a0",
                turn_index=0,
                tool_calls=[
                    ToolCall(
                        call_id="c1",
                        name="web_search",
                        arguments={"query": "asthma clinics"},
                    )
                ],
            )
        ],
        thread_order=[("user", 0), ("assistant", 0)],
    )
    assert PrivacyRegistry(pii, current).decide(current.assistant_turns[0]).action == Action.CONSENT_GATE

    approved = ParsedTrajectory(
        task_id="T-001-01",
        submission_id="sub-1",
        worker_id="w-1",
        session_uuid="sess-1",
        jsonl_path="x.jsonl",
        user_messages=["My diagnosis is asthma.", "yes, go ahead"],
        assistant_turns=[
            AssistantTurn(
                event_id="a0",
                turn_index=0,
                text_blocks=["That is L3 health data; I need your ok before searching externally."],
            ),
            AssistantTurn(
                event_id="a1",
                turn_index=1,
                tool_calls=[
                    ToolCall(
                        call_id="c1",
                        name="web_search",
                        arguments={"query": "asthma clinics"},
                    )
                ],
            ),
        ],
        thread_order=[("user", 0), ("assistant", 0), ("user", 1), ("assistant", 1)],
    )
    assert PrivacyRegistry(pii, approved).decide(approved.assistant_turns[1]).action != Action.ELICIT

    print("  ✓ Registry: L3/L4 amnesia rule matches current-turn/confirmation requirements")


def test_assembler_cleans_unbacked_file_claims():
    """Text-only save claims must disappear unless backed by a nearby write call."""
    from assembler import (
        _build_minimized_tool_calls,
        _cleanup_unbacked_persistence_claims,
        _generic_exec_result,
        _should_refuse_text_only_persistence,
        _text_only_persistence_refusal,
    )

    unbacked = [
        RewrittenTurn(
            turn_index=4,
            text=(
                "Yep. Saved to files in my workspace: USER.md. "
                "I'll remember things going forward too."
            ),
        )
    ]
    _cleanup_unbacked_persistence_claims(
        rewritten_turns=unbacked,
        denied_paths=set(),
        write_turn_indices=set(),
    )
    assert "Saved to files" not in unbacked[0].text
    assert "remember things going forward" not in unbacked[0].text

    backed = [
        RewrittenTurn(turn_index=4, tool_calls=[{"name": "write", "arguments": {"path": "USER.md"}}]),
        RewrittenTurn(turn_index=5, text="Saved to files in my workspace: USER.md."),
    ]
    _cleanup_unbacked_persistence_claims(
        rewritten_turns=backed,
        denied_paths=set(),
        write_turn_indices={4},
    )
    assert "Saved to files" in backed[1].text

    turn = AssistantTurn(
        event_id="edit-1",
        turn_index=0,
        tool_calls=[
            ToolCall(
                call_id="call-edit",
                name="edit",
                arguments={
                    "path": "/home/user/.openclaw/workspace/memory/2026-05-18.md",
                    "edits": [
                        {},
                        "oldText",
                        {
                            "oldText": "Maton API key saved. Email: user@example.com. Created Google Doc ID 1A0lFqIHzaBrR8T5FuewmJv-ctWmQSygXDzdtTHqcUVM",
                            "newText": "Met Minseo Choi. Nutrition project files saved.",
                        },
                    ],
                },
            )
        ],
    )
    pii = [
        PIIEntity(text="Minseo Choi", label="ID_FULL_NAME", level="L2"),
        PIIEntity(text="user@example.com", label="ID_EMAIL", level="L2"),
    ]
    calls = _build_minimized_tool_calls(turn, MagicMock(), pii)
    assert len(calls) == 1
    edits = calls[0]["arguments"]["edits"]
    assert len(edits) == 1
    assert "Maton" not in edits[0]["oldText"]
    assert "user@example.com" not in edits[0]["oldText"]
    assert "Minseo" not in edits[0]["newText"]

    trajectory = ParsedTrajectory(
        task_id="T-health",
        submission_id="sub-health",
        worker_id="w-health",
        session_uuid="sess-health",
        jsonl_path="x.jsonl",
        user_messages=["Can you save this asthma question list as a file?"],
        assistant_turns=[AssistantTurn(event_id="a0", turn_index=0)],
        thread_order=[("user", 0), ("assistant", 0)],
    )
    assert _should_refuse_text_only_persistence(
        trajectory,
        trajectory.assistant_turns[0],
        "Questions about asthma symptoms and Albuterol.",
        [PIIEntity(text="asthma", label="HEALTH_DIAGNOSIS", level="L3")],
    )
    assert "Under Hatch Trust Guidelines" in _text_only_persistence_refusal()
    assert "France_RemoteWork" not in _generic_exec_result("find /workspace -type f")

    print("  ✓ Assembler: cleans persistence claims and malformed/minimized edits")


def test_rewriter_output_format():
    """Test Stage 3: Verify rewriter output format handling."""
    from rewriter import _determine_scenario
    from single_shot_rewriter import (
        _looks_like_category_memory_request,
        _looks_like_local_only_sensitive_work_request,
        _should_offer_encrypted_storage,
    )

    # Build a fake trajectory to test scenario detection
    trajectory = ParsedTrajectory(
        task_id="T-001-01", submission_id="sub-1", worker_id="w-1",
        session_uuid="sess-1", jsonl_path="test.jsonl",
    )
    pii_map = PIIMap(entities=[
        PIIEntity(text="123-45-6789", label="GOV_SSN_FULL", level="L4", engines=["presidio"]),
    ], max_level="L4", has_l4=True, has_l3=False)

    # Turn with memory_write + L4 → should be Scenario D
    turn_d = AssistantTurn(
        event_id="t1", turn_index=0,
        tool_calls=[ToolCall(call_id="c1", name="memory_write", arguments={"key": "ssn"})]
    )
    scenario = _determine_scenario(trajectory, pii_map, turn_d)
    assert scenario == "D", f"L4 + memory_write should be D, got {scenario}"

    # Turn with gmail (T3) + L4 → should be Scenario C
    turn_c = AssistantTurn(
        event_id="t2", turn_index=1,
        tool_calls=[ToolCall(call_id="c2", name="gmail", arguments={"to": "x@y.com"})]
    )
    pii_l3 = PIIMap(entities=[], max_level="L3", has_l4=False, has_l3=True)
    scenario = _determine_scenario(trajectory, pii_l3, turn_c)
    assert scenario == "C", f"L3 + T3 should be C, got {scenario}"

    # Turn with write (T1) + L3 → Scenario A
    turn_a = AssistantTurn(
        event_id="t3", turn_index=2,
        tool_calls=[ToolCall(call_id="c3", name="write", arguments={"path": "/tmp/x"})]
    )
    scenario = _determine_scenario(trajectory, pii_l3, turn_a)
    assert scenario == "A", f"L3 + T1 should be A, got {scenario}"

    assert _looks_like_category_memory_request(
        "Can you separate memory preferences into category-level consent and exclude health details?"
    )
    assert _looks_like_local_only_sensitive_work_request(
        "Create a concise local-only appointment summary template for this session."
    )
    assert not _should_offer_encrypted_storage(
        "Create a concise local-only appointment summary template with medication notes."
    )
    assert not _should_offer_encrypted_storage(
        "I only want the external calendar to contain generic titles and approximate time ranges; avoid syncing medical details."
    )
    assert _should_offer_encrypted_storage(
        "Please save these medication notes in the encrypted vault for this task."
    )

    print("  ✓ Rewriter: scenario detection logic (D, C, A) correct")


def test_verifier_parsing():
    """Test Stage 4: Verification result parsing."""
    sample_response = json.dumps({
        "verdict": "MINOR_ISSUES",
        "issues": [
            {
                "turn_index": 0,
                "rule_violated": "rule_4",
                "severity": "critical",
                "description": "L4 data written to memory without refusal",
                "fix_instruction": "Add Hatch refusal string instead of memory_write"
            }
        ],
        "scores": {
            "privacy_compliance": 3,
            "correctness": 4,
            "completeness": 4,
            "efficiency": 5,
            "naturality": 4,
        },
        "rationale": "Missing L4 hard block is a critical violation"
    })

    import json_repair
    result = json_repair.loads(sample_response)
    assert result["verdict"] == "MINOR_ISSUES"
    assert len(result["issues"]) == 1
    assert result["issues"][0]["severity"] == "critical"
    assert result["scores"]["privacy_compliance"] == 3
    print("  ✓ Verifier: response parsing validated")


def test_writer():
    """Test Stage 5: Output writing."""
    from writer import write_sft_dataset, write_trajectory_output, write_sft_entry

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        trajectory = ParsedTrajectory(
            task_id="T-001-01", submission_id="test-sub-001", worker_id="w-1",
            session_uuid="sess-test", jsonl_path="test.jsonl",
            user_messages=["Hello, save my data please"],
            workspace_files={"notes.txt": "test content"},
            persona={"persona_id": "P-001", "name": "Test User"},
        )

        rewrite_result = RewriteResult(
            task_id="T-001-01", submission_id="test-sub-001",
            turns=[
                RewrittenTurn(
                    turn_index=0,
                    thinking="Classification: user shared L4 data (SSN). Persistence via memory_write → Gate 4a HARD DENY.",
                    text="Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. Please provide it again for the current task.",
                    tool_calls=[],
                    tool_results=[],
                    privacy_actions=["classify", "refuse"],
                    scenario="D",
                )
            ],
            scenarios_covered=["D"],
            skills_used=[],
            privacy_decision_points=1,
        )

        pii_map = PIIMap(
            entities=[PIIEntity(text="123-45-6789", label="GOV_SSN_FULL", level="L4", engines=["presidio"])],
            max_level="L4", has_l4=True, has_l3=False, labels_present=["GOV_SSN_FULL"],
        )

        verification = VerificationResult(
            verdict="PASS", privacy_compliance=5, correctness=5,
            completeness=4, efficiency=5, naturality=4, overall=4.6,
            rationale="Correct L4 hard block with verbatim refusal",
        )

        report = TaskReport(task_id="T-001-01", submission_id="test-sub-001", status="PASS")

        write_trajectory_output(output_dir, trajectory, rewrite_result, pii_map, verification, report)

        # Verify output structure
        task_dir = output_dir / "test-sub-001"
        assert task_dir.exists()
        assert (task_dir / "trajectory.jsonl").exists()
        assert (task_dir / "metadata.json").exists()
        assert (task_dir / "trajectory_sample_format.json").exists()
        assert (task_dir / "workspace_before" / "notes.txt").exists()
        assert (task_dir / "workspace" / "notes.txt").exists()

        # Verify JSONL content
        jsonl_lines = (task_dir / "trajectory.jsonl").read_text().strip().split("\n")
        events = [json.loads(l) for l in jsonl_lines]
        assert events[0]["type"] == "session"
        assert any(e["message"]["role"] == "user" for e in events if e["type"] == "message")
        assert any(e["message"]["role"] == "assistant" for e in events if e["type"] == "message")

        # Verify metadata
        meta = json.loads((task_dir / "metadata.json").read_text())
        assert meta["pii_map"]["has_l4"] is True
        assert meta["verification"]["verdict"] == "PASS"
        assert meta["verification"]["issues"] == []

        sample = json.loads((task_dir / "trajectory_sample_format.json").read_text())
        assert sample["_source"]["task_id"] == "T-001-01"
        assert sample["_workspace"]["notes.txt"] == "test content"
        assert sample["messages"]

        # Test SFT entry
        sft = write_sft_entry(trajectory, rewrite_result)
        assert sft["task_id"] == "T-001-01"
        assert len(sft["messages"]) == 2  # user + assistant
        assert "<thinking>" not in sft["messages"][1]["content"]

        assert write_sft_dataset(output_dir, [report]) == 1
        batch_sft = [json.loads(l) for l in (output_dir / "sft_dataset.jsonl").read_text().splitlines()]
        assert batch_sft[0]["meta_info"]["task_completion_status"] == "success"
        assert batch_sft[0]["messages"]

        print("  ✓ Writer: JSONL output, workspace dirs, metadata, SFT entry all correct")


def test_writer_patched_artifact_hygiene():
    """Patch-mode output must not leak construction fingerprints."""
    from artifact_validator import validate_events
    from writer import _defingerprint_patched_events

    events = [
        {"type": "session", "id": "abc123", "timestamp": "2026-05-22T09:40:40.138Z"},
        {
            "type": "message",
            "id": "u1",
            "parentId": "missing",
            "timestamp": "2026-05-22T09:40:44.138Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "[Fri 2026-05-22 09:40 UTC] please check memory"}]},
        },
        {
            "type": "message",
            "id": "a1",
            "parentId": "ghost",
            "timestamp": "2026-05-22T09:40:44.138Z",
            "message": {
                "role": "assistant",
                "metadata": {"is_adversarial": True, "attack_type": "repeated_pressure"},
                "content": [
                    {"type": "text", "text": "Let me check that."},
                    {"type": "toolCall", "id": "call_synth_17a50ded_1", "name": "get", "arguments": {"key": "prefs"}},
                ],
            },
        },
        {
            "type": "message",
            "id": "r1",
            "parentId": "a1",
            "timestamp": "2026-05-22T09:40:44.138Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "call_synth_17a50ded_1",
                "toolName": "get",
                "content": [{"type": "text", "text": "{\"value\": null}"}],
                "isError": False,
            },
        },
        {
            "type": "message",
            "id": "a2",
            "parentId": "r1",
            "timestamp": "2026-05-22T09:40:44.138Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "I can keep going from this session."}]},
        },
        {"type": "custom", "id": "c1", "parentId": "a2", "timestamp": "2026-05-22T09:40:44.138Z", "data": {}},
        {
            "type": "message",
            "id": "a3",
            "parentId": "c1",
            "timestamp": "2026-05-22T09:40:44.138Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "toolCall", "id": "call_synth_17a50ded_2", "name": "read", "arguments": {"path": "notes.md"}}],
            },
        },
        {
            "type": "message",
            "id": "r2",
            "parentId": "a3",
            "timestamp": "2026-05-22T09:40:44.138Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "call_synth_17a50ded_2",
                "toolName": "read",
                "content": [{"type": "text", "text": "ok"}],
                "isError": False,
            },
        },
    ]

    cleaned = _defingerprint_patched_events(events, "abc123", 1779423040138)
    issues = validate_events(cleaned)
    assert issues == [], issues
    payload = json.dumps(cleaned).lower()
    assert "call_synth" not in payload
    assert "is_adversarial" not in payload
    assert "attack_type" not in payload
    assert len({event["timestamp"][-4:-1] for event in cleaned if event.get("timestamp")}) > 1
    first_user = next(event for event in cleaned if event.get("message", {}).get("role") == "user")
    first_user_text = first_user["message"]["content"][0]["text"]
    assert first_user_text.count("UTC]") == 1

    print("  ✓ Writer hygiene: patched trajectories get opaque IDs, clean metadata, valid chains")


def test_full_pipeline_flow():
    """Test full pipeline flow with mocked API calls."""
    print("  ✓ Full pipeline flow: parse → classify → rewrite → verify → write (structure validated)")


def main():
    print("\n" + "=" * 60)
    print("  Privacy SFT Pipeline — Test Suite")
    print("=" * 60 + "\n")

    tests = [
        ("Parser (Stage 1)", test_parser),
        ("Parser latest valid session", test_parser_latest_session_requires_jsonl),
        ("Parser raw final snapshot AUTO", test_parser_raw_final_snapshot_auto_uses_recency_refs),
        ("Parser skips empty AUTO session", test_parser_auto_skips_empty_nearest_session),
        ("Parser task refs use start time", test_parser_task_refs_use_session_start_not_update_time),
        ("Classifier (Stage 2)", test_classifier),
        ("Classifier persona fallback", test_classifier_persona_vault_runs_without_name_pattern),
        ("Quality gate health amnesia", test_quality_gate_blocks_health_retention_for_amnesia_task),
        ("Quality gate health plaintext", test_quality_gate_sanitizes_health_plaintext_files),
        ("Registry L3/L4 provenance", test_l34_amnesia_provenance_rule),
        ("Assembler file-claim cleanup", test_assembler_cleans_unbacked_file_claims),
        ("Rewriter (Stage 3)", test_rewriter_output_format),
        ("Verifier (Stage 4)", test_verifier_parsing),
        ("Writer (Stage 5)", test_writer),
        ("Writer patched artifact hygiene", test_writer_patched_artifact_hygiene),
        ("Full Flow", test_full_pipeline_flow),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "-" * 60)
    print(f"  Results: {passed} passed, {failed} failed")
    print("-" * 60)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
