from __future__ import annotations

from trajectory_structure import normalize_event_stream, validate_event_stream
from writer import _defingerprint_events


def _assistant_tool(event_id: str, parent: str, tool_id: str, name: str) -> dict:
    return {
        "type": "message",
        "id": event_id,
        "parentId": parent,
        "timestamp": "2026-04-05T07:00:01.000Z",
        "message": {
            "role": "assistant",
            "content": [{
                "type": "toolCall",
                "id": tool_id,
                "name": name,
                "arguments": {},
            }],
        },
    }


def _tool_result(event_id: str, parent: str, tool_id: str, name: str) -> dict:
    return {
        "type": "message",
        "id": event_id,
        "parentId": parent,
        "timestamp": "2026-04-05T07:00:02.000Z",
        "message": {
            "role": "toolResult",
            "toolCallId": tool_id,
            "toolName": name,
            "content": [{"type": "text", "text": "ok"}],
            "isError": False,
        },
    }


def test_validator_accepts_openclaw_tool_continuation_chain():
    events = [
        {"type": "session", "id": "s1", "timestamp": "2026-04-05T07:00:00.000Z"},
        {
            "type": "message",
            "id": "u1",
            "parentId": "s1",
            "timestamp": "2026-04-05T07:00:00.500Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "start"}]},
        },
        _assistant_tool("a1", "u1", "tc1", "read"),
        _tool_result("r1", "a1", "tc1", "read"),
        _assistant_tool("a2", "r1", "tc2", "exec"),
        _tool_result("r2", "a2", "tc2", "exec"),
    ]

    assert validate_event_stream(events) == []


def test_writer_defingerprint_preserves_tool_result_interleaving():
    events = [
        {"type": "session", "id": "s1", "timestamp": "2026-04-05T07:00:00.000Z"},
        {
            "type": "message",
            "id": "u1",
            "parentId": "s1",
            "timestamp": "2026-04-05T07:00:00.500Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "start"}]},
        },
        _assistant_tool("a1", "u1", "tc1", "read"),
        _tool_result("r1", "a1", "tc1", "read"),
        _assistant_tool("a2", "r1", "tc2", "exec"),
        _tool_result("r2", "a2", "tc2", "exec"),
    ]

    out = _defingerprint_events(events, "s1")
    roles = [
        e["message"]["role"]
        for e in out
        if e.get("type") == "message" and isinstance(e.get("message"), dict)
    ]

    assert roles[-4:] == ["assistant", "toolResult", "assistant", "toolResult"]
    assert validate_event_stream(out) == []


def test_normalize_keeps_leaf_branch_and_removes_empty_assistant_stub():
    events = [
        {"type": "session", "id": "s1", "timestamp": "2026-04-05T07:00:00.000Z"},
        {
            "type": "message",
            "id": "u-old",
            "parentId": "s1",
            "timestamp": "2026-04-05T07:00:00.100Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "old branch"}]},
        },
        _assistant_tool("a-old", "u-old", "tc-reused", "read"),
        _tool_result("r-old", "a-old", "tc-reused", "read"),
        {
            "type": "message",
            "id": "u-new",
            "parentId": "s1",
            "timestamp": "2026-04-05T07:01:00.100Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "new branch"}]},
        },
        {
            "type": "message",
            "id": "empty",
            "parentId": "u-new",
            "timestamp": "2026-04-05T07:01:00.200Z",
            "message": {"role": "assistant", "content": []},
        },
        _assistant_tool("a-new", "empty", "tc-reused", "read"),
        _tool_result("r-new", "a-new", "tc-reused", "read"),
    ]

    out = normalize_event_stream(events)
    ids = [event["id"] for event in out]

    assert "a-old" not in ids
    assert "empty" not in ids
    assert ids == ["s1", "u-new", "a-new", "r-new"]
    assert out[2]["parentId"] == "u-new"
    assert validate_event_stream(out) == []


def test_normalize_repairs_broken_parent_links():
    events = [
        {"type": "session", "id": "s1", "timestamp": "2026-04-05T07:00:00.000Z"},
        {
            "type": "message",
            "id": "u1",
            "parentId": "missing-parent",
            "timestamp": "2026-04-05T07:00:00.500Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "start"}]},
        },
        _assistant_tool("a1", "also-missing", "tc1", "read"),
        _tool_result("r1", "a1", "tc1", "read"),
    ]

    out = normalize_event_stream(events)

    assert out[1]["parentId"] == "s1"
    assert out[2]["parentId"] == "u1"
    assert validate_event_stream(out) == []


def test_normalize_does_not_collapse_synthetic_stream_without_session():
    events = [
        {
            "type": "message",
            "id": "u1",
            "timestamp": "2026-04-05T07:00:00.500Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "start"}]},
        },
        {
            "type": "message",
            "id": "a1",
            "parentId": "u1",
            "timestamp": "2026-04-05T07:00:01.000Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "first"}]},
        },
        {
            "type": "message",
            "id": "u2",
            "parentId": "missing",
            "timestamp": "2026-04-05T07:00:02.000Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "continue"}]},
        },
        {
            "type": "message",
            "id": "a2",
            "parentId": "u2",
            "timestamp": "2026-04-05T07:00:03.000Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        },
    ]

    out = normalize_event_stream(events)

    assert [event["id"] for event in out] == ["u1", "a1", "u2", "a2"]
    assert out[0]["parentId"] is None
    assert out[2]["parentId"] == "a1"
    assert validate_event_stream(out) == []


def test_normalize_collapses_denied_plaintext_boilerplate_before_safe_answer():
    events = [
        {"type": "session", "id": "s1", "timestamp": "2026-04-05T07:00:00.000Z"},
        {
            "type": "message",
            "id": "u1",
            "parentId": "s1",
            "timestamp": "2026-04-05T07:00:00.500Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "show diagram"}]},
        },
        {
            "type": "message",
            "id": "a-deny",
            "parentId": "u1",
            "timestamp": "2026-04-05T07:00:01.000Z",
            "message": {
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": "I can't read or update that plaintext file because it would preserve sensitive data.",
                }],
            },
        },
        {
            "type": "message",
            "id": "a-answer",
            "parentId": "a-deny",
            "timestamp": "2026-04-05T07:00:02.000Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "```\n[diagram]\n```"}],
            },
        },
    ]

    out = normalize_event_stream(events)

    assert [event["id"] for event in out] == ["s1", "u1", "a-answer"]
    assert out[2]["parentId"] == "u1"
    assert validate_event_stream(out) == []
