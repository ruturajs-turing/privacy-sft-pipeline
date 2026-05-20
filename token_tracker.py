"""Track API token usage per stage."""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class StageUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


class TokenTracker:
    def __init__(self):
        self._lock = Lock()
        self._stages: dict[str, StageUsage] = {}

    def record_anthropic(self, response, stage: str):
        with self._lock:
            s = self._stages.setdefault(stage, StageUsage())
            s.input_tokens += response.usage.input_tokens
            s.output_tokens += response.usage.output_tokens
            s.calls += 1

    def record_openai(self, response, stage: str):
        with self._lock:
            s = self._stages.setdefault(stage, StageUsage())
            usage = response.usage
            s.input_tokens += usage.prompt_tokens
            s.output_tokens += usage.completion_tokens
            s.calls += 1

    def summary(self) -> dict:
        with self._lock:
            total_in = sum(s.input_tokens for s in self._stages.values())
            total_out = sum(s.output_tokens for s in self._stages.values())
            return {
                "stages": {
                    name: {"input": s.input_tokens, "output": s.output_tokens, "calls": s.calls, "total": s.total}
                    for name, s in self._stages.items()
                },
                "total_input": total_in,
                "total_output": total_out,
                "total_tokens": total_in + total_out,
            }


tracker = TokenTracker()
