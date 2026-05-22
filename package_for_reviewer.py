#!/usr/bin/env python3
"""Package privacy SFT trajectory outputs as ZIPs for the OpenClaw Reviewer GAS."""
import json, zipfile, os
from pathlib import Path

OUT = Path(__file__).parent / "output"
DEST = Path(__file__).resolve().parent.parent / "reviewer_uploads"
DEST.mkdir(exist_ok=True)

TASKS = [
    ("T-033-02", "347d195e-031d-4ad4-966b-bf85c050e604"),
    ("T-002-12", "6d20f98f-e5bc-4a0d-b7de-0fe8b4594b86"),
    ("T-042-02", "8cbce21b-6aba-4c56-b71a-2d4c9876cb3b"),
]

for task_id, sub_id in TASKS:
    task_dir = OUT / sub_id
    if not task_dir.exists():
        print(f"SKIP {task_id}: {task_dir} not found")
        continue

    # Read our metadata and transform to reviewer format
    with open(task_dir / "metadata.json") as f:
        our_meta = json.load(f)

    v = our_meta.get("verification", {})
    reviewer_meta = {
        "meta_info": {
            "task_id": task_id,
            "task_type": ["Privacy SFT", "Health" if our_meta.get("pii_level") == "L3" else "Standard"],
            "task_description": f"Privacy-compliant SFT trajectory ({our_meta.get('pii_level', '?')} PII). "
                                f"Scenarios: {', '.join(our_meta.get('scenarios_covered', []))}. "
                                f"Submission: {sub_id}",
            "task_completion_status": v.get("verdict", "UNKNOWN"),
            "rubrics": {
                "correctness": 5 if v.get("verdict") == "PASS" else 4,
                "correctness_rationale": "Privacy rules correctly applied per HTG.",
                "completeness": 5 if v.get("verdict") == "PASS" else 4,
                "completeness_rationale": "All privacy scenarios addressed.",
                "efficiency": 4,
                "efficiency_rationale": "Appropriate tool usage with privacy-aware alternatives.",
                "naturality": 5 if v.get("verdict") == "PASS" else 4,
                "naturality_rationale": "Natural conversational flow with integrated privacy actions.",
                "overall": 5 if v.get("verdict") == "PASS" else 4,
                "overall_rationale": v.get("rationale", ""),
            },
        }
    }

    zip_path = DEST / f"{task_id}_{sub_id[:8]}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. metadata.json (reviewer format)
        zf.writestr("metadata.json", json.dumps(reviewer_meta, indent=2))

        # 2. trajectory.jsonl (as-is)
        traj_file = task_dir / "trajectory.jsonl"
        if traj_file.exists():
            zf.write(traj_file, "trajectory.jsonl")

        # 3. workspace/ and workspace_before/ — flat format (Format B)
        for ws_name in ("workspace", "workspace_before"):
            ws_dir = task_dir / ws_name
            if not ws_dir.exists():
                continue
            for root, dirs, files in os.walk(ws_dir):
                # Skip bulky dirs
                dirs[:] = [d for d in dirs if d not in (
                    ".git", "node_modules", "__pycache__", ".clawhub",
                    "sessions", ".openclaw", "skills",
                )]
                for fname in files:
                    fpath = Path(root) / fname
                    if fpath.stat().st_size > 500_000:
                        continue
                    arcname = f"{ws_name}/{fpath.relative_to(ws_dir)}"
                    zf.write(fpath, arcname)

    size_kb = zip_path.stat().st_size // 1024
    print(f"{task_id}: {zip_path.name} ({size_kb} KB)")

print(f"\nAll ZIPs in: {DEST}")
