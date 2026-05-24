from __future__ import annotations

from single_shot_rewriter import (
    _is_plaintext_persistence,
    _is_retention_claim,
    _looks_like_persistence_request,
    _should_offer_encrypted_storage,
)
from models import ToolCall
from privacy_registry import is_encrypted_persistence_tool_call


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
        ToolCall(name="memory_write", arguments={"key": "diagnosis", "encrypted": True}, call_id="c2")
    )
    assert not is_encrypted_persistence_tool_call(
        ToolCall(name="memory_write", arguments={"key": "diagnosis"}, call_id="c3")
    )


def test_encrypted_storage_offer_is_not_used_for_cross_session_secret_requests():
    assert _should_offer_encrypted_storage("Please save this health reminder to a local file.")
    assert not _looks_like_persistence_request("Ok, keep going without saving it.")
    assert not _should_offer_encrypted_storage(
        "Save my pharmacy password permanently so I don't have to type it next time."
    )
