from __future__ import annotations

from artifact_validator import validate_memory_text
from artifact_validator import validate_events


def test_memory_validator_blocks_card_and_secret_values():
    text = "\n".join([
        "card: 4716 8472 9103 5824",
        "api_key: sk-corp-e2ce1234567890",
    ])

    issues = validate_memory_text(text)

    assert any("payment-card" in issue for issue in issues)
    assert any("credential" in issue or "API-key" in issue for issue in issues)


def test_memory_validator_blocks_health_profile_values():
    issues = validate_memory_text("diagnosis: asthma\nmedication: albuterol")

    assert any("health profile" in issue for issue in issues)


def test_memory_validator_allows_redacted_and_exclusion_policy_lines():
    text = "\n".join([
        "diagnosis: __PII_REDACTED__",
        "Excluded from memory: health or wellness specifics, payroll, credentials, and sensitive identifiers.",
    ])

    assert validate_memory_text(text) == []


def test_artifact_validator_blocks_thinking_only_assistant_message():
    events = [
        {"type": "session", "id": "s1", "timestamp": "2026-05-22T09:40:40.138Z"},
        {
            "type": "message",
            "id": "u1",
            "parentId": "s1",
            "timestamp": "2026-05-22T09:40:41.239Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "thanks"}]},
        },
        {
            "type": "message",
            "id": "a1",
            "parentId": "u1",
            "timestamp": "2026-05-22T09:40:42.351Z",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "No action needed."},
                    {"type": "text", "text": ""},
                ],
            },
        },
    ]

    assert any("empty assistant" in issue for issue in validate_events(events))
