from __future__ import annotations

from single_shot_rewriter import (
    _analyze_violations,
    _append_encrypted_storage_flow,
    _deterministic_consent_gate,
    _apply_patches,
    _is_plaintext_persistence,
    _is_credential_probe_command,
    _is_sensitive_card_context,
    _is_user_profile_l3_read_result,
    _looks_like_safe_local_export_request,
    _is_retention_claim,
    _looks_like_persistence_request,
    _remove_redundant_user_profile_scrubs,
    _should_offer_encrypted_storage,
    _transient_card_text,
)
from models import AssistantTurn, ParsedTrajectory, PIIEntity, PIIMap, RewriteResult, RewrittenTurn, ToolCall, ToolResult
from privacy_redaction import REDACTION_TOKEN, redact_event_stream, redact_text
from privacy_registry import is_encrypted_persistence_tool_call
from trajectory_structure import validate_event_stream
from workspace_builder import apply_session_writes


def test_denied_marker_in_file_listing_is_not_retention_claim():
    text = """
Workspace root:
- budget_preview.md
- blocked_path_explanation.md
- reading_club_budget_archive.zip

No external automation was connected, no sensitive data left the machine.
"""

    assert not _is_retention_claim(text, {"blocked_path_explanation.md"}, True)


def test_denied_marker_with_write_claim_is_retention_claim():
    text = "I appended the salary and bank note to blocked_path_explanation.md."

    assert _is_retention_claim(text, {"blocked_path_explanation.md"}, True)
    assert _is_retention_claim("Done. saved to your persona file.", set(), False)


def test_l34_plaintext_and_encrypted_persistence_paths_are_distinguished():
    assert _is_plaintext_persistence(
        "write",
        {"path": "/home/user/.openclaw/workspace/health_reminders.md", "content": "diagnosis"},
    )
    assert _is_plaintext_persistence(
        "memory_write",
        {"key": "salary", "value": "$87,600"},
    )
    assert not _is_plaintext_persistence(
        "memory_write",
        {"key": "salary", "value": "$87,600", "encrypted": True},
    )
    assert not _is_plaintext_persistence(
        "active_memory_write",
        {"key": "salary", "value": "$87,600"},
    )
    assert not _is_plaintext_persistence(
        "wiki_apply",
        {"page": "private/health", "content": "diagnosis"},
    )
    assert not _is_plaintext_persistence(
        "set",
        {"key": "diagnosis", "val": "asthma"},
    )


def test_shell_plaintext_write_is_hard_denied_surface():
    assert _is_plaintext_persistence(
        "exec",
        {"command": "printf 'salary $87600' > /home/user/.openclaw/workspace/notes.md"},
    )


def test_registry_marks_only_encrypted_persistence_tools_as_encrypted():
    assert is_encrypted_persistence_tool_call(
        ToolCall(name="active_memory_write", arguments={"key": "diagnosis"}, call_id="c1")
    )
    assert is_encrypted_persistence_tool_call(
        ToolCall(name="set", arguments={"key": "diagnosis", "val": "asthma"}, call_id="c0")
    )
    assert is_encrypted_persistence_tool_call(
        ToolCall(name="memory_write", arguments={"key": "diagnosis", "encrypted": True}, call_id="c2")
    )
    assert not is_encrypted_persistence_tool_call(
        ToolCall(name="memory_write", arguments={"key": "diagnosis"}, call_id="c3")
    )


def test_encrypted_storage_offer_is_not_used_for_cross_session_secret_requests():
    assert _should_offer_encrypted_storage("Please save this health reminder to a local file.")
    assert not _should_offer_encrypted_storage("Export that anonymized file locally.")
    assert _looks_like_safe_local_export_request("Export that anonymized file locally. No L2 data left my system.")
    assert not _looks_like_persistence_request("Ok, keep going without saving it.")
    assert not _should_offer_encrypted_storage(
        "Save my pharmacy password permanently so I don't have to type it next time."
    )


def test_single_shot_downgrades_web_search_with_l2_query_to_lower_tier_rag_search():
    email = "alice@example.com"
    pii_map = PIIMap(
        entities=[PIIEntity(text=email, label="EMAIL", level="L2")],
        max_level="L2",
    )
    trajectory = ParsedTrajectory(
        task_id="T-QA",
        submission_id="sub",
        worker_id="worker",
        session_uuid="session",
        jsonl_path="synthetic/test.jsonl",
        user_messages=[f"Search internal notes for {email}."],
        assistant_turns=[
            AssistantTurn(
                event_id="a1",
                turn_index=0,
                text_blocks=["I'll search for that."],
                tool_calls=[
                    ToolCall(
                        call_id="tc1",
                        name="web_search",
                        arguments={"query": f"{email} internal notes", "count": 5},
                    )
                ],
            )
        ],
        tool_results_by_call_id={
            "tc1": ToolResult(
                call_id="tc1",
                tool_name="web_search",
                content='{"title": "Internal note", "url": "https://example.com", "snippet": "result"}',
                is_error=False,
                is_empty=False,
            )
        },
        ordered_events=[
            {
                "type": "message",
                "id": "u1",
                "message": {"role": "user", "content": [{"type": "text", "text": f"Search internal notes for {email}."}]},
            },
            {
                "type": "message",
                "id": "a1",
                "parentId": "u1",
                "timestamp": "2026-05-25T03:23:38.870Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I'll search for that."},
                        {
                            "type": "toolCall",
                            "id": "tc1",
                            "name": "web_search",
                            "arguments": {"query": f"{email} internal notes", "count": 5},
                        },
                    ],
                },
            },
            {
                "type": "message",
                "id": "r1",
                "parentId": "a1",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "tc1",
                    "toolName": "web_search",
                    "content": [{"type": "text", "text": '{"title": "Internal note", "url": "https://example.com", "snippet": "result"}'}],
                },
            },
        ],
    )

    violations = _analyze_violations(trajectory, pii_map)
    assert any(v["action"] == "tool_downgrade" and v["downgraded_tool"] == "rag_search" for v in violations)

    rewrite = _apply_patches(
        trajectory,
        pii_map,
        {"consent_gates": [], "adversarial": None},
        adversarial_spot=99,
        violations=violations,
    )

    assistant = rewrite.patched_events[1]["message"]["content"]
    tool_call = next(item for item in assistant if item.get("type") == "toolCall")
    assert tool_call["name"] == "rag_search"
    assert tool_call["arguments"]["query"] == f"{email} internal notes"
    assert rewrite.patched_events[2]["message"]["toolName"] == "rag_search"


def test_l34_redaction_preserves_only_user_text_and_tool_args():
    secret = "4000 0000 0000 0002"
    pii_map = PIIMap(
        entities=[PIIEntity(text=secret, label="FIN_CC_FULL", level="L4")],
        max_level="L4",
        has_l4=True,
    )
    events = [
        {
            "type": "message",
            "id": "u1",
            "message": {"role": "user", "content": [{"type": "text", "text": f"use {secret}"}]},
        },
        {
            "type": "message",
            "id": "a1",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": f"I recall {secret}"},
                    {"type": "text", "text": f"I will use {secret}"},
                    {"type": "toolCall", "id": "tc1", "name": "exec", "arguments": {"command": f"charge {secret}"}},
                ],
            },
        },
        {
            "type": "message",
            "id": "r1",
            "message": {
                "role": "toolResult",
                "toolCallId": "tc1",
                "toolName": "exec",
                "content": [{"type": "text", "text": f"processed {secret}"}],
                "details": {"aggregated": f"processed {secret}"},
            },
        },
        {
            "type": "custom",
            "id": "c1",
            "customType": "exec.approval.requested",
            "data": {"reason": f"approve {secret}"},
        },
    ]

    out = redact_event_stream(events, pii_map)

    assert secret in out[0]["message"]["content"][0]["text"]
    assert secret in out[1]["message"]["content"][2]["arguments"]["command"]
    assert out[1]["message"]["content"][0]["thinking"].count(REDACTION_TOKEN) == 1
    assert out[1]["message"]["content"][1]["text"].count(REDACTION_TOKEN) == 1
    assert REDACTION_TOKEN in out[2]["message"]["content"][0]["text"]
    assert REDACTION_TOKEN in out[2]["message"]["details"]["aggregated"]
    assert REDACTION_TOKEN in out[3]["data"]["reason"]


def test_health_medication_redaction_handles_variants_and_profile_lines():
    pii_map = PIIMap(
        entities=[
            PIIEntity(text="Montelukast 10mg daily", label="HEALTH_MEDICATION", level="L3"),
            PIIEntity(text="Metoprolol 25mg twice daily", label="HEALTH_MEDICATION", level="L3"),
        ],
        max_level="L3",
        has_l3=True,
    )
    text = (
        "**Medications:**\n"
        "- Montelukast 10mg. daily\n"
        "- Metoprolol 25mg. twice daily\n"
        "- **__PII_REDACTED__:** Sensitive profile note. Uses Montelukast daily + rescue inhaler.\n"
    )

    redacted = redact_text(text, pii_map)

    assert "Montelukast" not in redacted
    assert "Metoprolol" not in redacted
    assert "rescue inhaler" not in redacted
    assert "Sensitive profile note" not in redacted
    assert redacted.count(REDACTION_TOKEN) >= 2


def test_health_profile_bullet_label_is_fully_redacted():
    pii_map = PIIMap(
        entities=[PIIEntity(text="asthma", label="HEALTH_DIAGNOSIS", level="L3")],
        max_level="L3",
        has_l3=True,
    )
    text = "- **Respiratory sensitivity:** avoid dusty or moldy produce\n"

    redacted = redact_text(text, pii_map)

    assert "dusty" not in redacted
    assert "moldy" not in redacted
    assert "Respiratory sensitivity" not in redacted
    assert redacted == ""


def test_user_profile_scrub_requires_health_l3_in_read_result():
    previous_event = {
        "type": "message",
        "id": "a1",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "toolCall",
                    "name": "read",
                    "arguments": {"path": "~/.openclaw/workspace/USER.md"},
                }
            ],
        },
    }
    clean_result = {
        "type": "message",
        "id": "r1",
        "message": {
            "role": "toolResult",
            "toolName": "read",
            "content": [{"type": "text", "text": "# USER.md\n\n- Name: Ricardo\n- Wife: Mei-Long"}],
        },
    }
    health_result = {
        "type": "message",
        "id": "r2",
        "message": {
            "role": "toolResult",
            "toolName": "read",
            "content": [{"type": "text", "text": "# USER.md\n\n- Name: Ricardo\n- Diagnosis: asthma"}],
        },
    }
    pii_map = PIIMap(
        entities=[
            PIIEntity(text="Ricardo", label="ID_NAME", level="L3"),
            PIIEntity(text="asthma", label="HEALTH_DIAGNOSIS", level="L3"),
        ],
        max_level="L3",
        has_l3=True,
    )

    assert not _is_user_profile_l3_read_result(clean_result, previous_event, pii_map)
    assert _is_user_profile_l3_read_result(health_result, previous_event, pii_map)


def test_redundant_user_profile_scrub_flow_is_removed_when_read_is_clean():
    events = [
        {
            "type": "message",
            "id": "a-read",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "tc-read",
                        "name": "read",
                        "arguments": {"path": "~/.openclaw/workspace/USER.md"},
                    }
                ],
            },
        },
        {
            "type": "message",
            "id": "r-read",
            "parentId": "a-read",
            "message": {
                "role": "toolResult",
                "toolName": "read",
                "toolCallId": "tc-read",
                "content": [{"type": "text", "text": "# USER.md\n\n- Name: Ricardo"}],
            },
        },
        {
            "type": "message",
            "id": "a-scrub",
            "parentId": "r-read",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "I found health details in plaintext USER.md. I won't reuse those from memory, and I'll scrub the profile note now.",
                    },
                    {
                        "type": "toolCall",
                        "id": "tc-write",
                        "name": "write",
                        "arguments": {"path": "~/.openclaw/workspace/USER.md", "content": "# USER.md\n"},
                    },
                ],
            },
        },
        {
            "type": "message",
            "id": "r-write",
            "parentId": "a-scrub",
            "message": {
                "role": "toolResult",
                "toolName": "write",
                "toolCallId": "tc-write",
                "content": [{"type": "text", "text": "Scrubbed L3 health details from USER.md."}],
            },
        },
        {
            "type": "message",
            "id": "a-final",
            "parentId": "r-write",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "What email should I use?"}]},
        },
    ]
    pii_map = PIIMap(
        entities=[
            PIIEntity(text="Ricardo", label="ID_NAME", level="L3"),
            PIIEntity(text="asthma", label="HEALTH_DIAGNOSIS", level="L3"),
        ],
        max_level="L3",
        has_l3=True,
    )

    cleaned = _remove_redundant_user_profile_scrubs(events, pii_map)
    combined = str(cleaned)

    assert "found health details" not in combined
    assert "Scrubbed L3 health details" not in combined
    assert cleaned[-1]["parentId"] == "r-read"


def test_location_address_redaction_handles_street_suffix_variants():
    pii_map = PIIMap(
        entities=[PIIEntity(text="138 Central Avenue", label="LOC_HOME_ADDR", level="L3")],
        max_level="L3",
        has_l3=True,
    )

    redacted = redact_text("ranked by distance from 138 Central Ave", pii_map)

    assert "138 Central" not in redacted
    assert REDACTION_TOKEN in redacted


def test_transient_card_text_does_not_contradict_chat_rendering():
    text = _transient_card_text({"prompt": "EMERGENCY MEDICAL CARD\nCondition: asthma"})

    assert "can't save" not in text.lower()
    assert "transient chat text" in text
    assert "EMERGENCY MEDICAL CARD" in text


def test_card_confirmation_tables_are_replaced_with_text_after_redaction():
    trajectory = ParsedTrajectory(
        task_id="T-999-07",
        submission_id="sub",
        worker_id="worker",
        session_uuid="session",
        jsonl_path="synthetic/test.jsonl",
        ordered_events=[
            {
                "type": "message",
                "id": "u1",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "So two versions. One full card for my bag. One redacted version for the organizer. Confirm you understand.",
                        }
                    ],
                },
            },
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Confirmed:\n\n"
                                "| | **Full Card** | **Redacted Card** |\n"
                                "|---|---|---|\n"
                                "__PII_REDACTED__\n"
                                "| Insurance ID | __PII_REDACTED__ | |\n"
                                "Two cards, two audiences. You're all set."
                            ),
                        }
                    ],
                },
            },
            {
                "type": "message",
                "id": "u2",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Before you show me the organizer version, confirm you removed insurance ID, provider name, and home address.",
                        }
                    ],
                },
            },
            {
                "type": "message",
                "id": "a2",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Confirmed, already done. The organizer version I shared above has none of those. "
                                "You're looking at it right there in the message above."
                            ),
                        }
                    ],
                },
            },
        ],
    )

    result = _apply_patches(
        trajectory,
        PIIMap(entities=[PIIEntity(text="asthma", label="HEALTH_DIAGNOSIS", level="L3")]),
        {"consent_gates": [], "adversarial": None},
        adversarial_spot=99,
        violations=[],
    )
    combined = "\n".join(
        item.get("text", "")
        for event in result.patched_events or []
        for item in event.get("message", {}).get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    )

    assert "|---|" not in combined
    assert "right there in the message above" not in combined
    assert "full card is for your bag" in combined
    assert "organizer version should omit insurance ID" in combined


def test_stale_assistant_text_is_redacted_instead_of_replaced_with_refusal():
    trajectory = ParsedTrajectory(
        task_id="T-999-02",
        submission_id="sub",
        worker_id="worker",
        session_uuid="session",
        jsonl_path="synthetic/test.jsonl",
        ordered_events=[
            {
                "type": "message",
                "id": "u1",
                "message": {"role": "user", "content": [{"type": "text", "text": "I have migraine."}]},
            },
            {
                "type": "message",
                "id": "a1",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Noted migraine."}]},
            },
            {
                "type": "message",
                "id": "u2",
                "message": {"role": "user", "content": [{"type": "text", "text": "Show me the draft."}]},
            },
            {
                "type": "message",
                "id": "a2",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "The migraine mention should be omitted."},
                        {"type": "text", "text": "Here is the draft with no migraine mention."},
                    ],
                },
            },
        ],
    )
    pii_map = PIIMap(
        entities=[PIIEntity(text="migraine", label="HEALTH_DIAGNOSIS", level="L3")],
        max_level="L3",
        has_l3=True,
    )

    result = _apply_patches(
        trajectory,
        pii_map,
        {"consent_gates": [], "adversarial": None},
        adversarial_spot=99,
        violations=[],
    )
    assistant_texts = []
    for event in result.patched_events or []:
        msg = event.get("message", {})
        if msg.get("role") != "assistant":
            continue
        for item in msg.get("content", []):
            if isinstance(item, dict):
                assistant_texts.extend(str(item.get(field, "")) for field in ("text", "thinking"))
    combined = "\n".join(assistant_texts)

    assert "I can't repeat sensitive health details" not in combined
    assert "Here is the draft" in combined
    assert "migraine" not in combined.lower()
    assert REDACTION_TOKEN in combined


def test_stale_l34_tool_args_trigger_reinput_gate():
    secret = "sk-test-stale-secret-123456"
    trajectory = ParsedTrajectory(
        task_id="T-999-01",
        submission_id="sub",
        worker_id="worker",
        session_uuid="session",
        jsonl_path="synthetic/test.jsonl",
        user_messages=[
            f"my temporary key is {secret}",
            "can you use it now?",
        ],
        assistant_turns=[
            AssistantTurn(event_id="a0", turn_index=0, text_blocks=["ok"]),
            AssistantTurn(
                event_id="a1",
                turn_index=1,
                tool_calls=[
                    ToolCall(
                        call_id="tc1",
                        name="exec",
                        arguments={"command": f"local-tool --token {secret}"},
                    )
                ],
            ),
        ],
        thread_order=[("user", 0), ("assistant", 0), ("user", 1), ("assistant", 1)],
    )
    pii_map = PIIMap(
        entities=[PIIEntity(text=secret, label="AUTH_API_KEY", level="L4")],
        max_level="L4",
        has_l4=True,
    )

    violations = _analyze_violations(trajectory, pii_map)

    elicit = [v for v in violations if v.get("action") == "elicit"]
    assert len(elicit) == 1
    assert elicit[0]["turn_idx"] == 1
    assert elicit[0]["sensitive_values"] == [secret]


def test_workspace_writes_follow_patched_events_not_original_turns():
    rewrite_result = RewriteResult(
        task_id="T-999-01",
        submission_id="sub",
        turns=[
            RewrittenTurn(
                turn_index=0,
                tool_calls=[
                    {
                        "name": "write",
                        "arguments": {
                            "path": "/home/user/.openclaw/workspace/plaintext_secret.md",
                            "content": "full card 4000 0000 0000 0002",
                        },
                    }
                ],
            )
        ],
        patched_events=[
            {
                "type": "message",
                "id": "a-safe",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I did not save a plaintext file."}],
                },
            }
        ],
    )

    files = apply_session_writes({"MEMORY.md": "# MEMORY\n"}, rewrite_result)

    assert "plaintext_secret.md" not in files


def test_workspace_does_not_materialize_mcp_vault_set():
    rewrite_result = RewriteResult(
        task_id="T-999-01",
        submission_id="sub",
        turns=[
            RewrittenTurn(
                turn_index=0,
                tool_calls=[
                    {
                        "name": "set",
                        "arguments": {
                            "key": "health_note",
                            "val": "diagnosis asthma",
                        },
                    }
                ],
            )
        ],
    )

    files = apply_session_writes({"MEMORY.md": "# MEMORY\n"}, rewrite_result)

    assert files == {"MEMORY.md": "# MEMORY\n"}


def test_encrypted_storage_flow_uses_gio_mcp_vault_signature():
    patched = [
        {
            "type": "message",
            "id": "u1",
            "timestamp": "2026-04-05T07:00:00.000Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "save this health note"}]},
        }
    ]
    event = {
        "type": "message",
        "id": "a1",
        "parentId": "u1",
        "timestamp": "2026-04-05T07:00:01.000Z",
        "message": {
            "role": "assistant",
            "content": [
                {
                    "type": "toolCall",
                    "id": "tc1",
                    "name": "write",
                    "arguments": {
                        "path": "/home/user/.openclaw/workspace/health.md",
                        "content": "diagnosis asthma",
                    },
                }
            ],
        },
    }

    _append_encrypted_storage_flow(patched, event, "u1", "Please save this health note.")

    tool_calls = [
        item
        for evt in patched
        for item in evt.get("message", {}).get("content", [])
        if isinstance(item, dict) and item.get("type") == "toolCall"
    ]
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "set"
    assert set(tool_calls[0]["arguments"]) == {"key", "val"}

    result = next(evt for evt in patched if evt.get("message", {}).get("role") == "toolResult")
    assert result["message"]["toolName"] == "set"
    assert result["message"]["content"][0]["text"] == '{"ok": true}'


def test_consent_gate_inserts_exec_approval_events():
    trajectory = ParsedTrajectory(
        task_id="T-999-01",
        submission_id="sub",
        worker_id="worker",
        session_uuid="session",
        jsonl_path="synthetic/test.jsonl",
        ordered_events=[
            {"type": "session", "id": "s1", "timestamp": "2026-04-05T07:00:00.000Z"},
            {
                "type": "message",
                "id": "u1",
                "parentId": "s1",
                "timestamp": "2026-04-05T07:00:00.500Z",
                "message": {"role": "user", "content": [{"type": "text", "text": "send this externally"}]},
            },
            {
                "type": "message",
                "id": "a1",
                "parentId": "u1",
                "timestamp": "2026-04-05T07:00:01.000Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "tc1",
                            "name": "browser",
                            "arguments": {"url": "https://example.test/form"},
                        }
                    ],
                },
            },
        ],
    )
    patches = {
        "consent_gates": [
            {
                "before_turn_idx": 0,
                "assistant_ask": "Approval required for the external form.",
                "user_grant": "Approved once.",
            }
        ],
        "adversarial": None,
    }
    violations = [
        {
            "turn_idx": 0,
            "action": "consent_gate",
            "tool_name": "browser",
            "tier": 3,
            "is_write": True,
            "data_layer": 2,
            "free_band": 1,
        }
    ]

    result = _apply_patches(trajectory, PIIMap(), patches, adversarial_spot=99, violations=violations)

    custom_types = [
        event.get("customType")
        for event in result.patched_events or []
        if event.get("type") == "custom"
    ]
    assert custom_types == ["exec.approval.requested", "exec.approval.resolved"]
    approval_request = next(
        event for event in result.patched_events or []
        if event.get("customType") == "exec.approval.requested"
    )
    assert approval_request["data"]["allowedDecisions"] == ["allow-once", "deny"]
    assert validate_event_stream(result.patched_events or []) == []


def test_sensitive_email_exec_consent_names_external_gateway():
    trajectory = ParsedTrajectory(
        task_id="T-999-03",
        submission_id="sub",
        worker_id="worker",
        session_uuid="session",
        jsonl_path="synthetic/test.jsonl",
        ordered_events=[
            {
                "type": "message",
                "id": "u1",
                "message": {"role": "user", "content": [{"type": "text", "text": "send it"}]},
            },
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "tc1",
                            "name": "exec",
                            "arguments": {
                                "command": "cd skills/email && ./client.py send claudia@example.test --body 'Lucia and Mateo'"
                            },
                        }
                    ],
                },
            },
        ],
    )
    pii_map = PIIMap(
        entities=[
            PIIEntity(text="Lucia", label="ID_MINOR", level="L3"),
            PIIEntity(text="Mateo", label="ID_MINOR", level="L3"),
        ],
        max_level="L3",
        has_l3=True,
    )

    result = _apply_patches(
        trajectory,
        pii_map,
        {
            "consent_gates": [
                {"before_turn_idx": 0, "assistant_ask": "Confirm send?", "user_grant": "Yes."}
            ],
            "adversarial": None,
        },
        adversarial_spot=99,
        violations=[
            {
                "turn_idx": 0,
                "action": "consent_gate",
                "tool_name": "exec",
                "tier": 1,
                "is_write": True,
                "data_layer": 3,
                "free_band": 2,
                "arguments": {
                    "command": "cd skills/email && ./client.py send claudia@example.test --body 'Lucia and Mateo'"
                },
            }
        ],
    )
    combined = "\n".join(
        item.get("text", "")
        for event in result.patched_events or []
        for item in event.get("message", {}).get("content", [])
        if isinstance(item, dict)
    )

    assert "external enterprise email gateway" in combined
    assert "L3 minor data" in combined
    assert "Confirm send?" not in combined

    tool_names = [
        item.get("name")
        for event in result.patched_events or []
        for item in event.get("message", {}).get("content", [])
        if isinstance(item, dict) and item.get("type") == "toolCall"
    ]
    assert "enterprise-mail" in tool_names
    assert "exec" not in tool_names


def test_stale_organizer_card_email_request_triggers_t2_l3_consent():
    trajectory = ParsedTrajectory(
        task_id="T-999-06",
        submission_id="sub",
        worker_id="worker",
        session_uuid="session",
        jsonl_path="synthetic/test.jsonl",
        user_messages=[
            "My emergency card has asthma, migraine, Montelukast, and insurance ID KAI-502.",
            "The organizer version is ready. Send it to Valeria at valeria@example.test.",
        ],
        assistant_turns=[
            AssistantTurn(event_id="a1", turn_index=0, text_blocks=["I drafted the redacted organizer card."]),
            AssistantTurn(
                event_id="a2",
                turn_index=1,
                text_blocks=["I need to check the email skill first."],
                tool_calls=[
                    ToolCall(
                        call_id="tc1",
                        name="read",
                        arguments={"path": "~/.openclaw/workspace/skills/email/SKILL.md"},
                    )
                ],
            ),
        ],
        thread_order=[("user", 0), ("assistant", 0), ("user", 1), ("assistant", 1)],
    )
    pii_map = PIIMap(
        entities=[
            PIIEntity(text="asthma", label="HEALTH_DIAGNOSIS", level="L3"),
            PIIEntity(text="migraine", label="HEALTH_DIAGNOSIS", level="L3"),
            PIIEntity(text="Montelukast", label="HEALTH_MEDICATION", level="L3"),
            PIIEntity(text="KAI-502", label="HEALTH_INSURANCE_ID", level="L3"),
        ],
        max_level="L3",
        has_l3=True,
    )

    violations = _analyze_violations(trajectory, pii_map)

    send_gate = next(v for v in violations if v.get("sensitive_card_send"))
    assert send_gate["turn_idx"] == 1
    assert send_gate["tool_name"] == "enterprise-email"
    assert send_gate["tier"] == 2
    assert send_gate["data_layer"] == 3

    gate = _deterministic_consent_gate(send_gate)
    assert "enterprise email" in gate["assistant_ask"]
    assert "L3 health details" in gate["assistant_ask"]
    assert "full card" in gate["assistant_ask"]


def test_credential_probe_exec_is_rewritten_without_secret_printing():
    trajectory = ParsedTrajectory(
        task_id="T-999-04",
        submission_id="sub",
        worker_id="worker",
        session_uuid="session",
        jsonl_path="synthetic/test.jsonl",
        ordered_events=[
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "tc1",
                            "name": "exec",
                            "arguments": {
                                "command": "echo \"VAULT_API_KEY: ${VAULT_API_KEY}\" && curl -H 'x-api-key: sk-amy' https://example.test"
                            },
                        }
                    ],
                },
            },
            {
                "type": "message",
                "id": "r1",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "tc1",
                    "toolName": "exec",
                    "content": [{"type": "text", "text": "VAULT_API_KEY: sk-amy"}],
                    "details": {"aggregated": "VAULT_API_KEY: sk-amy"},
                },
            },
        ],
    )

    result = _apply_patches(
        trajectory,
        PIIMap(entities=[PIIEntity(text="sk-amy", label="AUTH_API_KEY", level="L4")]),
        {"consent_gates": [], "adversarial": None},
        adversarial_spot=99,
        violations=[],
    )
    combined = str(result.patched_events)

    assert "x-api-key: sk-amy" not in combined
    assert "VAULT_API_KEY: sk-amy" not in combined
    assert "Skipped direct gateway credential probe" in combined


def test_insecure_gateway_tls_and_http_downgrade_are_blocked():
    assert _is_credential_probe_command(
        "cp ~/.openclaw/workspace/skills/gitlab/client.py client.py.bak && "
        "sed -i 's/timeout=120)/timeout=120, verify=False)/g' "
        "~/.openclaw/workspace/skills/gitlab/client.py"
    )
    assert _is_credential_probe_command(
        'GATEWAY_URL="http://api.34.59.210.220.nip.io" VAULT_API_KEY="${VAULT_API_KEY}" '
        "uv run --directory ~/.openclaw/workspace/skills/gitlab client.py projects-list"
    )


def test_insecure_gateway_edit_tool_is_replaced_with_blocking_summary():
    trajectory = ParsedTrajectory(
        task_id="T-999-06",
        submission_id="sub",
        worker_id="worker",
        session_uuid="session",
        jsonl_path="synthetic/test.jsonl",
        ordered_events=[
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me patch get_token to disable SSL verification."},
                        {
                            "type": "toolCall",
                            "id": "tc1",
                            "name": "edit",
                            "arguments": {
                                "path": "~/.openclaw/workspace/skills/gitlab/client.py",
                                "edits": [
                                    {
                                        "oldText": 'httpx.get(f"{GATEWAY_URL}/auth")',
                                        "newText": 'httpx.get(f"{GATEWAY_URL}/auth", verify=False)',
                                    }
                                ],
                            },
                        },
                    ],
                },
            },
            {
                "type": "message",
                "id": "r1",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "tc1",
                    "toolName": "edit",
                    "content": [{"type": "text", "text": "Successfully replaced 1 block"}],
                },
            },
        ],
    )

    result = _apply_patches(
        trajectory,
        PIIMap(),
        {"consent_gates": [], "adversarial": None},
        adversarial_spot=99,
        violations=[],
    )
    combined = str(result.patched_events)

    assert "verify=False" not in combined
    assert "can't bypass TLS" in combined


def test_public_weather_coordinates_only_elicit_when_user_previously_stated_them():
    pii_map = PIIMap(
        entities=[
            PIIEntity(text="-33.45", label="LOC_GPS_PRECISE", level="L3"),
            PIIEntity(text="-70.50", label="LOC_GPS_PRECISE", level="L3"),
        ],
        max_level="L3",
        has_l3=True,
    )
    tool_call = ToolCall(
        name="exec",
        arguments={
            "command": (
                'curl -s "https://api.open-meteo.com/v1/forecast?'
                'latitude=-33.45&longitude=-70.50&current_weather=true"'
            )
        },
        call_id="tc-weather",
    )
    inferred_public = ParsedTrajectory(
        task_id="T-999-07",
        submission_id="sub",
        worker_id="worker",
        session_uuid="session",
        jsonl_path="synthetic/test.jsonl",
        user_messages=["Check Santiago foothills weather."],
        assistant_turns=[AssistantTurn(event_id="a1", turn_index=0, tool_calls=[tool_call])],
        thread_order=[("user", 0), ("assistant", 0)],
    )
    user_stated = ParsedTrajectory(
        task_id="T-999-08",
        submission_id="sub",
        worker_id="worker",
        session_uuid="session",
        jsonl_path="synthetic/test.jsonl",
        user_messages=["My exact coords are -33.45, -70.50.", "Check tomorrow's weather."],
        assistant_turns=[AssistantTurn(event_id="a1", turn_index=0, tool_calls=[tool_call])],
        thread_order=[("user", 0), ("user", 1), ("assistant", 0)],
    )

    inferred_violations = _analyze_violations(inferred_public, pii_map)
    stated_violations = _analyze_violations(user_stated, pii_map)

    assert not any(v.get("action") == "elicit" for v in inferred_violations)
    assert any(v.get("action") == "elicit" for v in stated_violations)


def test_weather_trigger_table_is_not_sensitive_card_context():
    weather_table = (
        "| Trigger | Status |\n"
        "|---|---|\n"
        "| Cold dry air | monitor it |\n"
        "| PM2.5 at elevation | low |"
    )
    card_table = (
        "| | Full Card | Redacted Card |\n"
        "|---|---|---|\n"
        "| Medications | Montelukast | removed |\n"
        "| Insurance ID | KAI-502 | removed |"
    )

    assert not _is_sensitive_card_context(weather_table)
    assert _is_sensitive_card_context(card_table)


def test_repeated_sensitive_card_render_is_replaced_after_stale_context():
    trajectory = ParsedTrajectory(
        task_id="T-999-05",
        submission_id="sub",
        worker_id="worker",
        session_uuid="session",
        jsonl_path="synthetic/test.jsonl",
        ordered_events=[
            {
                "type": "message",
                "id": "u1",
                "message": {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Write my emergency card with asthma, migraine, "
                                "Montelukast, Metoprolol, and insurance ID KAI-502."
                            ),
                        }
                    ],
                },
            },
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "**EMERGENCY MEDICAL CARD**\n"
                                "Conditions: asthma, migraine\n"
                                "Medications: Montelukast, Metoprolol\n"
                                "Insurance ID: KAI-502"
                            ),
                        }
                    ],
                },
            },
            {
                "type": "message",
                "id": "u2",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Show both versions side by side."}],
                },
            },
            {
                "type": "message",
                "id": "a2",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "| | Full Card | Redacted Card |\n"
                                "|---|---|---|\n"
                                "| Conditions | asthma, migraine | asthma, migraine |\n"
                                "| Medications | Montelukast, Metoprolol | Montelukast, Metoprolol |\n"
                                "| Insurance ID | KAI-502 | removed |"
                            ),
                        }
                    ],
                },
            },
        ],
    )
    pii_map = PIIMap(
        entities=[
            PIIEntity(text="asthma", label="HEALTH_DIAGNOSIS", level="L3"),
            PIIEntity(text="migraine", label="HEALTH_DIAGNOSIS", level="L3"),
            PIIEntity(text="Montelukast", label="HEALTH_MEDICATION", level="L3"),
            PIIEntity(text="Metoprolol", label="HEALTH_MEDICATION", level="L3"),
            PIIEntity(text="KAI-502", label="HEALTH_INSURANCE_ID", level="L3"),
        ],
        max_level="L3",
        has_l3=True,
    )

    result = _apply_patches(
        trajectory,
        pii_map,
        {"consent_gates": [], "adversarial": None},
        adversarial_spot=99,
        violations=[],
    )
    assistant_texts = [
        item.get("text", "")
        for event in result.patched_events or []
        for item in event.get("message", {}).get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    ]

    assert any("EMERGENCY MEDICAL CARD" in text for text in assistant_texts)
    assert any("I won't reprint the medical fields again" in text for text in assistant_texts)
    assert not any("| Conditions |" in text for text in assistant_texts)
