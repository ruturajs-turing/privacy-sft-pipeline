from __future__ import annotations

import json

from models import TaskReport
from writer import write_batch_report


def test_batch_report_treats_minor_fixed_without_gate_issues_as_production_ready(tmp_path):
    report = TaskReport(
        task_id="T-QA",
        submission_id="sub",
        status="MINOR_FIXED",
        quality_gate_issues=[],
    )

    write_batch_report(tmp_path, [report], {"total_tokens": 0})

    batch = json.loads((tmp_path / "batch_report.json").read_text())
    assert batch["pass_rate"] == "100.0%"
    assert batch["delivery_ready"] is True
    assert batch["production_ready"] is True


def test_batch_report_blocks_production_ready_when_quality_gate_issues_remain(tmp_path):
    report = TaskReport(
        task_id="T-QA",
        submission_id="sub",
        status="MINOR_FIXED",
        quality_gate_issues=[{"rule_violated": "l3_l4_disclosure"}],
    )

    write_batch_report(tmp_path, [report], {"total_tokens": 0})

    batch = json.loads((tmp_path / "batch_report.json").read_text())
    assert batch["delivery_ready"] is True
    assert batch["production_ready"] is False
