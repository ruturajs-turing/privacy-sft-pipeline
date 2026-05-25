from __future__ import annotations

from models import ParsedTrajectory, PIIEntity, PIIMap, RewriteResult, RewrittenTurn, VerificationResult
from privacy_quality_gates import HATCH_REFUSAL, apply_quality_gates, repair_quality_issues, run_quality_gates


def _trajectory() -> ParsedTrajectory:
    return ParsedTrajectory(
        task_id="T-QA",
        submission_id="sub",
        worker_id="worker",
        session_uuid="session",
        jsonl_path="synthetic/test.jsonl",
    )


def test_quality_gate_allows_user_text_and_tool_args_but_blocks_assistant_l4_recall():
    secret = "4716 8472 9103 5824"
    pii_map = PIIMap(
        entities=[PIIEntity(text=secret, label="FIN_CC_FULL", level="L4")],
        max_level="L4",
        has_l4=True,
    )
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "u1",
                "message": {"role": "user", "content": [{"type": "text", "text": f"Use {secret}"}]},
            },
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"I will use {secret} now."},
                        {"type": "toolCall", "id": "tc1", "name": "exec", "arguments": {"command": f"charge {secret}"}},
                    ],
                },
            },
        ],
    )

    issues = run_quality_gates(_trajectory(), rewrite, pii_map)

    assert any(issue.rule_violated == "l3_l4_disclosure" for issue in issues)


def test_quality_gate_blocks_plaintext_l3_persistence_and_requires_exact_vault_signature():
    pii_map = PIIMap(
        entities=[PIIEntity(text="asthma", label="HEALTH_DIAGNOSIS", level="L3")],
        max_level="L3",
        has_l3=True,
    )
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "tc1",
                            "name": "write",
                            "arguments": {"path": "MEMORY.md", "content": "diagnosis: asthma"},
                        },
                        {
                            "type": "toolCall",
                            "id": "tc2",
                            "name": "vault_set",
                            "arguments": {"key": "diagnosis", "value": "asthma"},
                        },
                        {
                            "type": "toolCall",
                            "id": "tc3",
                            "name": "set",
                            "arguments": {"key": "diagnosis", "val": "asthma"},
                        },
                    ],
                },
            },
        ],
    )

    issues = run_quality_gates(_trajectory(), rewrite, pii_map)

    assert any(issue.rule_violated == "plaintext_l3_l4_persistence" for issue in issues)
    assert any(issue.rule_violated == "vault_signature" for issue in issues)
    assert sum(1 for issue in issues if issue.rule_violated == "vault_signature") == 1


def test_quality_gate_blocks_insecure_authenticated_gateway_flow():
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "HTTP works. Let me use HTTP for the GitLab calls."},
                        {
                            "type": "toolCall",
                            "id": "tc1",
                            "name": "exec",
                            "arguments": {
                                "command": (
                                    'GATEWAY_URL="http://api.34.59.210.220.nip.io" '
                                    'VAULT_API_KEY="${VAULT_API_KEY}" client.py projects-list'
                                )
                            },
                        },
                    ],
                },
            },
        ],
    )

    issues = run_quality_gates(_trajectory(), rewrite, PIIMap())

    assert any(issue.rule_violated == "security_gateway_http" for issue in issues)
    assert any(issue.rule_violated == "security_gateway_command" for issue in issues)


def test_quality_gate_overrides_pass_verification_to_fail_for_major_issue():
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
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
                                "command": (
                                    'GATEWAY_URL="http://api.34.59.210.220.nip.io" '
                                    'VAULT_API_KEY="${VAULT_API_KEY}" client.py projects-list'
                                )
                            },
                        }
                    ],
                },
            },
        ],
    )
    issues = run_quality_gates(_trajectory(), rewrite, PIIMap())
    verification = VerificationResult(
        verdict="PASS",
        privacy_compliance=5,
        correctness=5,
        completeness=5,
        efficiency=5,
        naturality=5,
        overall=5.0,
        rationale="Looks good.",
    )

    merged = apply_quality_gates(verification, issues)

    assert merged.verdict == "FAIL"
    assert merged.privacy_compliance <= 2
    assert "Deterministic privacy quality gate blocked packaging" in merged.rationale


def test_quality_gate_requires_exec_approval_events_even_when_flag_claims_granted():
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "tc1",
                            "name": "web.run",
                            "arguments": {"query": "clinic asthma guidance"},
                            "flag": {
                                "is_violation": True,
                                "free_band": 2,
                                "data_layer": 3,
                                "authorization": {"needed": True, "granted": True},
                            },
                        }
                    ],
                },
            }
        ],
    )

    issues = run_quality_gates(_trajectory(), rewrite, PIIMap())

    assert any(issue.rule_violated == "missing_exec_approval_events" for issue in issues)


def test_quality_gate_checks_rewritten_turns_when_patch_events_absent():
    secret = "4716 8472 9103 5824"
    pii_map = PIIMap(
        entities=[PIIEntity(text=secret, label="FIN_CC_FULL", level="L4")],
        max_level="L4",
        has_l4=True,
    )
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        turns=[RewrittenTurn(turn_index=0, text=f"I still have {secret} from earlier.")],
    )

    issues = run_quality_gates(_trajectory(), rewrite, pii_map)

    assert any(issue.rule_violated == "l3_l4_disclosure" for issue in issues)


def test_quality_gate_blocks_l4_card_future_storage_claim_even_after_redaction():
    pii_map = PIIMap(
        entities=[
            PIIEntity(text="4716 8472 9103 5824", label="FIN_CC_FULL", level="L4"),
            PIIEntity(text="628", label="FIN_CVV", level="L4"),
        ],
        max_level="L4",
        has_l4=True,
    )
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": (
                            "I'll store these in your workspace memory so they're available next time. "
                            "Say the word and I'll draft the letter and save the card info."
                        ),
                    }],
                },
            }
        ],
    )

    issues = run_quality_gates(_trajectory(), rewrite, pii_map)

    assert any(issue.rule_violated == "l4_future_storage_claim" for issue in issues)


def test_auto_repair_replaces_l4_card_future_storage_claim():
    pii_map = PIIMap(
        entities=[
            PIIEntity(text="4716 8472 9103 5824", label="FIN_CC_FULL", level="L4"),
            PIIEntity(text="628", label="FIN_CVV", level="L4"),
        ],
        max_level="L4",
        has_l4=True,
    )
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": "I'll store these card details in workspace memory for future disputes.",
                    }],
                },
            }
        ],
    )

    repaired, repairs = repair_quality_issues(_trajectory(), rewrite, pii_map)

    assert any(repair["kind"] == "replaced_l4_future_storage_claim" for repair in repairs)
    text = repaired.patched_events[0]["message"]["content"][0]["text"]
    assert "can't store full card numbers or CVV" in text
    assert run_quality_gates(_trajectory(), repaired, pii_map) == []


def test_auto_repair_downgrades_higher_tier_tool_when_lower_drop_in_exists():
    email = "alice@example.com"
    pii_map = PIIMap(
        entities=[PIIEntity(text=email, label="EMAIL", level="L2")],
        max_level="L2",
    )
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "tc1",
                            "name": "web_search",
                            "arguments": {"query": f"{email} internal notes", "count": 5},
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
                    "toolName": "web_search",
                    "content": [{"type": "text", "text": '{"title": "Internal note", "url": "https://example.com", "snippet": "result"}'}],
                },
            },
        ],
    )

    repaired, repairs = repair_quality_issues(_trajectory(), rewrite, pii_map)

    assert any(repair["kind"] == "downgraded_higher_tier_tool" for repair in repairs)
    tool_call = repaired.patched_events[0]["message"]["content"][0]
    assert tool_call["name"] == "rag_search"
    assert repaired.patched_events[1]["message"]["toolName"] == "rag_search"
    assert run_quality_gates(_trajectory(), repaired, pii_map) == []


def test_auto_repair_redacts_assistant_l4_recall_without_touching_allowed_surfaces():
    secret = "4716 8472 9103 5824"
    pii_map = PIIMap(
        entities=[PIIEntity(text=secret, label="FIN_CC_FULL", level="L4")],
        max_level="L4",
        has_l4=True,
    )
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "u1",
                "message": {"role": "user", "content": [{"type": "text", "text": f"Use {secret}"}]},
            },
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"I will use {secret} now."},
                        {"type": "toolCall", "id": "tc1", "name": "exec", "arguments": {"command": f"charge {secret}"}},
                    ],
                },
            },
        ],
    )

    repaired, repairs = repair_quality_issues(_trajectory(), rewrite, pii_map)

    assert repairs
    assert repaired.patched_events[0]["message"]["content"][0]["text"] == f"Use {secret}"
    assistant_content = repaired.patched_events[1]["message"]["content"]
    assert assistant_content[0]["text"] == "I will use __PII_REDACTED__ now."
    assert assistant_content[1]["arguments"]["command"] == f"charge {secret}"
    assert run_quality_gates(_trajectory(), repaired, pii_map) == []


def test_auto_repair_redacts_fuzzy_l3_entity_variants_in_text_and_tool_results():
    pii_map = PIIMap(
        entities=[PIIEntity(text="Glendale Respiratory Care Center", label="HEALTH_PROVIDER", level="L3")],
        max_level="L3",
        has_l3=True,
    )
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I found Respiratory Care Center in the saved note."}],
                },
            },
            {
                "type": "message",
                "id": "tr1",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "tc1",
                    "toolName": "read",
                    "content": [{"type": "text", "text": "Respiratory Care Center"}],
                    "details": {"name": "Respiratory Care Center"},
                },
            },
        ],
    )

    repaired, repairs = repair_quality_issues(_trajectory(), rewrite, pii_map)

    assert any(repair["kind"] == "redacted_non_user_non_tool_sensitive_surfaces" for repair in repairs)
    assert run_quality_gates(_trajectory(), repaired, pii_map) == []


def test_auto_repair_converts_sensitive_plaintext_persistence_to_exact_vault_set():
    pii_map = PIIMap(
        entities=[PIIEntity(text="123-45-6789", label="GOV_ID", level="L3")],
        max_level="L3",
        has_l3=True,
    )
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "a1",
                "timestamp": "2026-05-25T03:23:38.870Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "tc1",
                            "name": "write",
                            "arguments": {"path": "MEMORY.md", "content": "id: 123-45-6789"},
                        }
                    ],
                },
            },
            {
                "type": "message",
                "id": "tr1",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "tc1",
                    "toolName": "write",
                    "content": [{"type": "text", "text": "wrote id: 123-45-6789"}],
                    "details": {"path": "MEMORY.md", "content": "id: 123-45-6789"},
                },
            },
        ],
    )

    repaired, repairs = repair_quality_issues(_trajectory(), rewrite, pii_map)

    assert any(repair["kind"] == "converted_sensitive_persistence_to_vault_set" for repair in repairs)
    tool_call = repaired.patched_events[0]["message"]["content"][1]
    assert repaired.patched_events[0]["message"]["content"][0]["text"].startswith("I'll store this in the vault")
    assert tool_call["name"] == "set"
    assert set(tool_call["arguments"]) == {"key", "val"}
    assert tool_call["arguments"]["val"] == "id: 123-45-6789"
    tool_result = repaired.patched_events[1]["message"]
    assert tool_result["toolName"] == "set"
    assert tool_result["content"] == [{"type": "text", "text": '{"ok": true}'}]
    assert run_quality_gates(_trajectory(), repaired, pii_map) == []


def test_auto_repair_blocks_insecure_gateway_tool_call_and_drops_result():
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
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
                                "command": (
                                    'GATEWAY_URL="http://api.34.59.210.220.nip.io" '
                                    'VAULT_API_KEY="${VAULT_API_KEY}" client.py projects-list'
                                )
                            },
                        }
                    ],
                },
            },
            {
                "type": "message",
                "id": "tr1",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "tc1",
                    "toolName": "exec",
                    "content": [{"type": "text", "text": "project list"}],
                },
            },
        ],
    )

    repaired, repairs = repair_quality_issues(_trajectory(), rewrite, PIIMap())

    assert any(repair["kind"] == "blocked_insecure_gateway_tool_call" for repair in repairs)
    assert len(repaired.patched_events) == 1
    text = repaired.patched_events[0]["message"]["content"][0]["text"]
    assert "can't bypass TLS" in text
    assert run_quality_gates(_trajectory(), repaired, PIIMap()) == []


def test_auto_repair_inserts_exec_approval_for_flagged_boundary_crossing():
    pii_map = PIIMap(
        entities=[PIIEntity(text="asthma", label="HEALTH_DIAGNOSIS", level="L3")],
        max_level="L3",
        has_l3=True,
    )
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "u1",
                "message": {"role": "user", "content": [{"type": "text", "text": "Use asthma for the request."}]},
            },
            {
                "type": "message",
                "id": "a1",
                "parentId": "u1",
                "timestamp": "2026-05-25T03:23:38.870Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "tc1",
                            "name": "web.run",
                            "arguments": {"query": "clinic asthma guidance"},
                            "flag": {
                                "is_violation": True,
                                "free_band": 2,
                                "data_layer": 3,
                                "authorization": {"needed": True, "granted": False},
                            },
                        }
                    ],
                },
            },
        ],
    )

    repaired, repairs = repair_quality_issues(_trajectory(), rewrite, pii_map)

    assert any(repair["kind"] == "inserted_exec_approval" for repair in repairs)
    custom_types = [event.get("customType") for event in repaired.patched_events if event.get("type") == "custom"]
    assert custom_types == ["exec.approval.requested", "exec.approval.resolved"]
    tool_call = repaired.patched_events[-1]["message"]["content"][0]
    assert tool_call["flag"]["authorization"] == {"needed": True, "granted": True}
    assert run_quality_gates(_trajectory(), repaired, pii_map) == []


def test_quality_gate_recomputes_exec_approval_from_tool_args_without_flag():
    email = "casey@example.com"
    pii_map = PIIMap(
        entities=[PIIEntity(text=email, label="EMAIL", level="L2")],
        max_level="L2",
    )
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "tc1",
                            "name": "web.run",
                            "arguments": {"query": f"lookup support history for {email}"},
                        }
                    ],
                },
            }
        ],
    )

    issues = run_quality_gates(_trajectory(), rewrite, pii_map)

    assert any(issue.rule_violated == "missing_exec_approval" for issue in issues)
    assert any(issue.rule_violated == "missing_exec_approval_events" for issue in issues)


def test_quality_gate_repairs_resource_over_refusal():
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "u1",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Please read the uploaded health.pdf and summarize it."}],
                },
            },
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I cannot read the PDF because it contains sensitive health data."}],
                },
            },
        ],
    )

    issues = run_quality_gates(_trajectory(), rewrite, PIIMap())
    assert any(issue.rule_violated == "resource_over_refusal" for issue in issues)

    repaired, repairs = repair_quality_issues(_trajectory(), rewrite, PIIMap())

    assert any(repair["kind"] == "replaced_resource_over_refusal" for repair in repairs)
    assert "can read and work with the provided resource" in repaired.patched_events[1]["message"]["content"][0]["text"]
    assert run_quality_gates(_trajectory(), repaired, PIIMap()) == []


def test_quality_gate_repairs_missing_resource_reread_before_sensitive_cloud_tool():
    pii_map = PIIMap(
        entities=[PIIEntity(text="asthma", label="HEALTH_DIAGNOSIS", level="L3")],
        max_level="L3",
        has_l3=True,
    )
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "u1",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Use the uploaded health.pdf to create the calendar appointment."}],
                },
            },
            {
                "type": "message",
                "id": "a1",
                "parentId": "u1",
                "timestamp": "2026-05-25T03:23:38.870Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "toolCall",
                            "id": "tc1",
                            "name": "enterprise-calendar",
                            "arguments": {"action": "create", "details": "allergist follow-up for asthma"},
                        }
                    ],
                },
            },
        ],
    )

    issues = run_quality_gates(_trajectory(), rewrite, pii_map)
    assert any(issue.rule_violated == "missing_resource_reread" for issue in issues)

    repaired, repairs = repair_quality_issues(_trajectory(), rewrite, pii_map)

    assert any(repair["kind"] == "inserted_resource_reread" for repair in repairs)
    assert any(repair["kind"] == "inserted_exec_approval" for repair in repairs)
    read_calls = [
        item
        for event in repaired.patched_events
        for item in event.get("message", {}).get("content", [])
        if isinstance(item, dict) and item.get("type") == "toolCall" and item.get("name") == "read"
    ]
    assert read_calls and read_calls[0]["arguments"]["path"] == "health.pdf"
    assert run_quality_gates(_trajectory(), repaired, pii_map) == []


def test_quality_gate_repairs_payment_workflow_hatch_and_confirmation():
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "u-save",
                "message": {"role": "user", "content": [{"type": "text", "text": "Yeah go ahead, save it."}]},
            },
            {
                "type": "message",
                "id": "a-bad-hatch",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. Please provide it again for the current task."}],
                },
            },
            {
                "type": "message",
                "id": "u-l4",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Save my national ID plus full card number and CVV into a notes file so you can autofill later."}],
                },
            },
            {
                "type": "message",
                "id": "u-l4-retract",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Okay, skip that."}],
                },
            },
            {
                "type": "message",
                "id": "a-missed",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "Done. Created the comparison summary."}]},
            },
            {
                "type": "message",
                "id": "u-confirm",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Confirm CVV/payment secrets were never retained, no reusable memory/profile of payment or identity data was created, and minimum-disclosure and least-retention were followed."}],
                },
            },
            {
                "type": "message",
                "id": "a-nonseq",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "The safe local files step is complete. I won't read or update the denied plaintext reference file."}]},
            },
        ],
    )

    issues = run_quality_gates(_trajectory(), rewrite, PIIMap())
    assert {issue.rule_violated for issue in issues} >= {
        "misapplied_hatch_refusal",
        "missing_l4_plaintext_refusal",
        "payment_confirmation_nonsequitur",
    }

    repaired, repairs = repair_quality_issues(_trajectory(), rewrite, PIIMap())
    assert any(repair["kind"] == "replaced_misapplied_hatch_refusal" for repair in repairs)
    assert any(repair["kind"] == "inserted_missing_l4_plaintext_refusal" for repair in repairs)
    assert any(repair["kind"] == "replaced_payment_confirmation_nonsequitur" for repair in repairs)
    texts = [
        event["message"]["content"][0]["text"]
        for event in repaired.patched_events
        if event.get("type") == "message" and event.get("message", {}).get("role") == "assistant"
    ]
    assert texts[0].startswith("Saved.")
    assert texts[1] == "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. Please provide it again for the current task."
    assert "minimum-disclosure" in texts[2]
    assert run_quality_gates(_trajectory(), repaired, PIIMap()) == []


def test_quality_gate_allows_hatch_for_future_health_memory_request():
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "u1",
                "message": {
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": (
                            "Could you go ahead and save that I deal with lower-back pain "
                            "and take medication daily so I do not have to re-explain it every time?"
                        ),
                    }],
                },
            },
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": HATCH_REFUSAL}],
                },
            },
        ],
    )

    assert run_quality_gates(_trajectory(), rewrite, PIIMap()) == []


def test_openclaw_patch_applier_drops_tool_result_for_removed_tool_call():
    from openclaw_repair_worker import apply_openclaw_patch

    events = [
        {
            "type": "message",
            "id": "u1",
            "message": {"role": "user", "content": [{"type": "text", "text": "Save this for later."}]},
        },
        {
            "type": "message",
            "id": "a1",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Sure."},
                    {
                        "type": "toolCall",
                        "id": "call_abc",
                        "name": "write",
                        "arguments": {"path": "memory/card.txt", "content": "card number and cvv"},
                    },
                ],
            },
        },
        {
            "type": "message",
            "id": "tr1",
            "message": {
                "role": "toolResult",
                "toolCallId": "call_abc",
                "toolName": "write",
                "content": [{"type": "text", "text": "saved"}],
            },
        },
    ]
    patch = {
        "repairs": [
            {
                "operation": "replace_assistant_text",
                "line": 2,
                "text": "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. Please provide it again for the current task.",
                "drop_tool_calls": True,
            }
        ]
    }

    patched, applied = apply_openclaw_patch(events, patch)

    assert applied == [{"operation": "replace_assistant_text", "line": 2, "drop_tool_calls": True}]
    assert len(patched) == 2
    assert patched[1]["message"]["content"] == [{"type": "text", "text": patch["repairs"][0]["text"]}]


def test_quality_gate_requires_hatch_for_recovery_code_profile_retention():
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "u1",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Save my recovery code permanently to profile notes."}],
                },
            },
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Saved. The local file contains only the approved non-sensitive planning framework."}],
                },
            },
        ],
    )

    issues = run_quality_gates(_trajectory(), rewrite, PIIMap())
    assert any(issue.rule_violated == "missing_l4_plaintext_refusal" for issue in issues)

    repaired, repairs = repair_quality_issues(_trajectory(), rewrite, PIIMap())
    assert any(repair["kind"] == "inserted_missing_l4_plaintext_refusal" for repair in repairs)
    assert repaired.patched_events[1]["message"]["content"] == [{
        "type": "text",
        "text": "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. Please provide it again for the current task.",
    }]
    assert run_quality_gates(_trajectory(), repaired, PIIMap()) == []


def test_quality_gate_repairs_sensitive_tool_result_disclosure_before_packaging():
    pii_map = PIIMap(
        entities=[PIIEntity(text="disc herniation", label="HEALTH_DIAGNOSIS", level="L3")],
        max_level="L3",
        has_l3=True,
    )
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "id": "call_health", "name": "read", "arguments": {"path": "notes.md"}}],
                },
            },
            {
                "type": "message",
                "id": "tr1",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call_health",
                    "toolName": "read",
                    "content": [{"type": "text", "text": "Diagnosis: disc herniation"}],
                },
            },
        ],
    )

    issues = run_quality_gates(_trajectory(), rewrite, pii_map)
    assert any(issue.rule_violated == "tool_result_l3_l4_disclosure" for issue in issues)

    repaired, repairs = repair_quality_issues(_trajectory(), rewrite, pii_map)
    assert any(repair["kind"] == "redacted_sensitive_tool_result" for repair in repairs)
    assert "disc herniation" not in repaired.patched_events[1]["message"]["content"][0]["text"]
    assert run_quality_gates(_trajectory(), repaired, pii_map) == []


def test_quality_gate_repairs_recovery_guidance_over_refusal():
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "u1",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Can you help me safely work through a repository recovery process before I provide any recovery code?"}],
                },
            },
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. Please provide it again for the current task."}],
                },
            },
            {
                "type": "message",
                "id": "u2",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "Can you suggest a safer longer-term workflow with hardware keys and rotating recovery sets?"}],
                },
            },
            {
                "type": "message",
                "id": "a2",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Under Hatch Trust Guidelines, I cannot retain this classification of sensitive data. Please provide it again for the current task."}],
                },
            },
        ],
    )

    issues = run_quality_gates(_trajectory(), rewrite, PIIMap())
    assert [issue.rule_violated for issue in issues].count("misapplied_recovery_hatch_refusal") == 2

    repaired, repairs = repair_quality_issues(_trajectory(), rewrite, PIIMap())
    assert len([repair for repair in repairs if repair["kind"] == "replaced_misapplied_recovery_hatch_refusal"]) == 2
    texts = [
        event["message"]["content"][0]["text"]
        for event in repaired.patched_events
        if event["message"]["role"] == "assistant"
    ]
    assert "tightly scoped" in texts[0]
    assert "hardware security keys" in texts[1]
    assert run_quality_gates(_trajectory(), repaired, PIIMap()) == []


def test_quality_gate_repairs_premature_recovery_purge_claim():
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "u1",
                "message": {"role": "user", "content": [{"type": "text", "text": "Can you confirm the recovery credential was purged?"}]},
            },
            {
                "type": "message",
                "id": "a1",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "The recovery credential was purged, confirmed. Since no recovery code has been shared, there is nothing here to purge."}],
                },
            },
        ],
    )

    issues = run_quality_gates(_trajectory(), rewrite, PIIMap())
    assert any(issue.rule_violated == "premature_recovery_purge_claim" for issue in issues)

    repaired, repairs = repair_quality_issues(_trajectory(), rewrite, PIIMap())
    assert any(repair["kind"] == "replaced_premature_recovery_purge_claim" for repair in repairs)
    assert "can't truthfully confirm" in repaired.patched_events[1]["message"]["content"][0]["text"]
    assert run_quality_gates(_trajectory(), repaired, PIIMap()) == []


def test_quality_gate_repairs_calendar_save_then_reversal_state():
    rewrite = RewriteResult(
        task_id="T-QA",
        submission_id="sub",
        patched_events=[
            {
                "type": "message",
                "id": "u-save",
                "message": {"role": "user", "content": [{"type": "text", "text": "this is fine, save it"}]},
            },
            {
                "type": "message",
                "id": "a-ls",
                "parentId": "u-save",
                "timestamp": "2026-05-25T05:52:49.418Z",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Running that now."},
                        {
                            "type": "toolCall",
                            "id": "call_ics",
                            "name": "exec",
                            "arguments": {"command": "ls -la /home/user/.openclaw/workspace/appointment-2026-05-28.ics"},
                        },
                    ],
                },
            },
            {
                "type": "message",
                "id": "r-ls",
                "parentId": "a-ls",
                "message": {
                    "role": "toolResult",
                    "toolCallId": "call_ics",
                    "toolName": "exec",
                    "content": [{"type": "text", "text": "-rw-r--r-- 1 user user 310 May 25 05:42 /home/user/.openclaw/workspace/appointment-2026-05-28.ics"}],
                },
            },
            {
                "type": "message",
                "id": "a-saved",
                "parentId": "r-ls",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Saved. The local file contains only the approved non-sensitive planning framework."}],
                },
            },
            {
                "type": "message",
                "id": "u-reverse",
                "parentId": "a-saved",
                "message": {"role": "user", "content": [{"type": "text", "text": "Ok, keep it transient here. Don't save it."}]},
            },
            {
                "type": "message",
                "id": "a-false",
                "parentId": "u-reverse",
                "timestamp": "2026-05-25T05:54:43.561Z",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Got it. Keeping it here only, with no plaintext file saved."}],
                },
            },
        ],
    )

    issues = run_quality_gates(_trajectory(), rewrite, PIIMap())
    assert any(issue.rule_violated == "calendar_missing_write_call" for issue in issues)
    assert any(issue.rule_violated == "calendar_missing_delete_after_reversal" for issue in issues)

    repaired, repairs = repair_quality_issues(_trajectory(), rewrite, PIIMap())

    assert any(repair["kind"] == "replaced_calendar_ls_with_visible_write" for repair in repairs)
    assert any(repair["kind"] == "deleted_calendar_file_after_save_reversal" for repair in repairs)
    save_call = repaired.patched_events[1]["message"]["content"][1]
    assert "cat > /home/user/.openclaw/workspace/appointment-2026-05-28.ics" in save_call["arguments"]["command"]
    assert repaired.patched_events[2]["message"]["content"] == [{"type": "text", "text": "(no output)"}]
    delete_event = repaired.patched_events[-2]
    delete_result = repaired.patched_events[-1]
    assert "rm -f /home/user/.openclaw/workspace/appointment-2026-05-28.ics" in delete_event["message"]["content"][1]["arguments"]["command"]
    assert delete_result["message"]["toolCallId"] == delete_event["message"]["content"][1]["id"]
    assert run_quality_gates(_trajectory(), repaired, PIIMap()) == []
