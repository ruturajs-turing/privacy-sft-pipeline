"""FastAPI backend for the Privacy SFT/RLHF Comparison UI."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="OpenClaw Privacy Comparison API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = Path(__file__).parent.parent.parent / "test_output"


def _parse_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


@app.get("/api/trajectories")
def list_trajectories():
    """List all processed trajectory submission IDs."""
    if not OUTPUT_DIR.exists():
        return {"trajectories": []}

    trajectories = []
    for d in sorted(OUTPUT_DIR.iterdir()):
        if d.is_dir() and (d / "trajectory.jsonl").exists():
            meta = {}
            meta_path = d / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
            trajectories.append({
                "submission_id": d.name,
                "task_id": meta.get("task_id", ""),
                "has_rlhf": (d / "rlhf" / "rlhf_pairs.jsonl").exists(),
                "status": meta.get("report", {}).get("status", ""),
            })

    return {"trajectories": trajectories}


@app.get("/api/trajectory/{submission_id}/original")
def get_original(submission_id: str):
    """Return original SFT trajectory events (pre-privacy conversion)."""
    # Strategy: look for the original trajectory in multiple locations
    candidates = [
        OUTPUT_DIR / "original-trajectory.jsonl",
        OUTPUT_DIR / f"original-{submission_id[:8]}.jsonl",
        OUTPUT_DIR / submission_id / "original.jsonl",
    ]

    for path in candidates:
        if path.exists():
            events = _parse_jsonl(path)
            message_events = [e for e in events if e.get("type") == "message"]
            if message_events:
                return {"events": events, "source": path.name}

    # Fallback: try extracting from the original zip
    zip_candidates = [
        OUTPUT_DIR / f"original-{submission_id[:8]}.zip",
        OUTPUT_DIR / f"{submission_id}.zip",
    ]
    for zip_path in zip_candidates:
        if zip_path.exists():
            events = _extract_from_zip(zip_path)
            if events:
                return {"events": events, "source": zip_path.name}

    return {"events": [], "source": "not_found"}


def _extract_from_zip(zip_path: Path) -> list[dict]:
    """Extract trajectory events from an export zip."""
    import zipfile
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            jsonl_files = [n for n in zf.namelist() if n.endswith(".jsonl") and "session" not in n.lower()]
            if not jsonl_files:
                jsonl_files = [n for n in zf.namelist() if n.endswith(".jsonl")]
            if jsonl_files:
                # Prefer the largest JSONL file (likely the full session)
                jsonl_files.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
                with zf.open(jsonl_files[0]) as f:
                    events = []
                    for line in f:
                        line = line.decode("utf-8").strip()
                        if line:
                            events.append(json.loads(line))
                    return events
    except Exception:
        pass
    return []


@app.get("/api/trajectory/{submission_id}/privacy")
def get_privacy(submission_id: str):
    """Return rewritten privacy-compliant trajectory."""
    task_dir = OUTPUT_DIR / submission_id
    traj_path = task_dir / "trajectory.jsonl"

    if not traj_path.exists():
        raise HTTPException(404, f"Privacy trajectory not found for {submission_id}")

    events = _parse_jsonl(traj_path)
    return {"events": events}


@app.get("/api/trajectory/{submission_id}/rlhf")
def get_rlhf(submission_id: str):
    """Return RLHF preference pairs."""
    task_dir = OUTPUT_DIR / submission_id
    rlhf_path = task_dir / "rlhf" / "rlhf_pairs.jsonl"

    if not rlhf_path.exists():
        raise HTTPException(404, f"RLHF pairs not found for {submission_id}")

    pairs = _parse_jsonl(rlhf_path)
    return {"pairs": pairs, "total": len(pairs)}


@app.get("/api/trajectory/{submission_id}/metadata")
def get_metadata(submission_id: str):
    """Return pipeline metadata and RLHF report."""
    task_dir = OUTPUT_DIR / submission_id

    metadata = {}
    meta_path = task_dir / "metadata.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())

    rlhf_report = {}
    report_path = task_dir / "rlhf" / "rlhf_report.json"
    if report_path.exists():
        rlhf_report = json.loads(report_path.read_text())

    return {"metadata": metadata, "rlhf_report": rlhf_report}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
