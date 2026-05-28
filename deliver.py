#!/usr/bin/env python3
"""deliver.py — OpenClaw Privacy delivery orchestrator.

Loops over candidate tasks and produces accepted privacy-SFT deliverables until a
target count is reached. Per task it:

  1. resolves the task and ensures its delivery artifacts exist (RLHF +
     workspace_before; triggers Stage 4/5 generation if missing),
  2. runs the recovery phase (htg_repair.py) to guarantee HTG compliance,
  3. assembles the 4-part delivery folder
     (workspace/, workspace_before/, rlhf/, <uuid>.json),
  4. sends it to the judge (metadata_judge_writer.py), which injects metadata.json
     and returns a PASS / CONDITIONAL / FAIL verdict,
  5. on PASS/CONDITIONAL accumulates a per-task zip; on FAIL recovers + re-judges
     up to --max-recovery times, then quarantines.

It stops when the accepted count reaches --target, then writes a roll-up zip,
review_manifest.csv and summary.json. Progress is streamed to --progress-file
after every task so an external controller (CUArena) can poll it.

Candidate sources (pick one):
  --built-root DIR      enumerate already-assembled <uuid>/ task folders (offline)
  --candidates CSV      4-col manifest worker_id,submission_id,task_id,export_url
                        (runs the full pipeline from each export; needs GCS/zips)
  --bucket NAME --list-candidates
                        enumerate submission-id prefixes from GCS into a manifest
                        (needs GOOGLE_APPLICATION_CREDENTIALS)

Examples:
  python deliver.py --target 2 --built-root output/gcs_valid_tasks_100_20260528 --dry-run
  python deliver.py --target 2 --built-root output/top100_delivery_batch_20260528T074213Z_with_metadata
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("deliver")

UUID_GLOB = "????????-????-????-????-????????????"
CORE_WS_FILES = {"AGENTS.md", "HEARTBEAT.md", "IDENTITY.md", "SOUL.md", "TOOLS.md", "USER.md"}
ACCEPT_DEFAULT = ("PASS", "CONDITIONAL")
PY = sys.executable


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# Candidate model + state
# ---------------------------------------------------------------------------
@dataclass
class Candidate:
    submission_id: str
    source_dir: Path | None = None          # built-root mode: the <uuid>/ folder
    worker_id: str = ""                      # manifest mode
    task_id: str = "AUTO"
    export_url: str = ""
    kind: str = "built-root"                 # built-root | candidates | gcs


@dataclass
class ItemResult:
    submission_id: str
    status: str = "pending"                  # pending|built|judged|accepted|quarantined|error
    verdict: str = ""                        # PASS|CONDITIONAL|FAIL
    attempts: int = 0
    zip_path: str = ""
    detail: str = ""


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------
async def run_cmd(args: list[str], cwd: Path, label: str) -> tuple[int, str]:
    """Run a pipeline subprocess; return (rc, tail-of-output)."""
    log.info("  $ %s", " ".join(str(a) for a in args[:6]) + (" …" if len(args) > 6 else ""))
    proc = await asyncio.create_subprocess_exec(
        *[str(a) for a in args], cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        env={**os.environ},
    )
    out_b, _ = await proc.communicate()
    out = out_b.decode("utf-8", "replace")
    if proc.returncode != 0:
        log.warning("  %s rc=%s tail: %s", label, proc.returncode, out[-400:].replace("\n", " "))
    return proc.returncode or 0, out


# ---------------------------------------------------------------------------
# Candidate enumeration
# ---------------------------------------------------------------------------
PIPELINE_BUCKET = "meta-openclaw-privacy-pipeline"   # processed intermediates (.privacy.jsonl)
EXPORTS_BUCKET = "meta-openclaw-privacy"             # raw exports/{worker}/{sub}/ snapshots


def _gsutil(*args: str) -> tuple[int, str]:
    import subprocess
    p = subprocess.run(["gsutil", *args], capture_output=True, text=True)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# Directories that are dependency/build cruft — excluded from delivery workspaces
# (the canonical reference deliveries contain none of these).
_WS_SKIP_SEGMENTS = ("node_modules", ".git", "__pycache__", ".venv", ".pytest_cache",
                     ".mypy_cache", ".cache")


def _ws_skip(rel: str) -> bool:
    parts = rel.split("/")
    return any(seg in _WS_SKIP_SEGMENTS for seg in parts)


def _extract_workspace_from_snapshot(zip_path: Path, dest: Path) -> int:
    """Extract `.openclaw/workspace/<...>` from a snapshot zip into dest/.

    Mirrors parser._extract_workspace: strips the `.openclaw/workspace/` prefix,
    skips .git/ and files > 500KB. Returns number of files written.
    """
    import zipfile as _zip
    n = 0
    prefix = ".openclaw/workspace/"
    with _zip.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            rel = name[len(prefix):]
            if not rel or _ws_skip(rel):
                continue
            if info.file_size > 500_000:
                continue
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, out.open("wb") as dst:
                dst.write(src.read())
            n += 1
    return n


def fetch_gcs_task(sid: str, dest_root: Path) -> Path | None:
    """Assemble a delivery-ready task dir for one bucket submission (reuse path).

    Reuses the already-rewritten `.privacy.jsonl` from the pipeline bucket and
    extracts workspace/ + workspace_before/ from the raw export snapshots. RLHF
    is generated afterwards by ensure_artifacts. Returns the task dir or None.
    """
    task = dest_root / sid
    if task.exists():
        shutil.rmtree(task)
    task.mkdir(parents=True)
    # 1) processed artifacts from the pipeline bucket (names repair_zero_rlhf expects)
    base = f"gs://{PIPELINE_BUCKET}/{sid}"
    rc, _ = _gsutil("cp", f"{base}/{sid}.privacy.jsonl", str(task / f"{sid}.privacy.jsonl"))
    if rc != 0 or not (task / f"{sid}.privacy.jsonl").exists():
        log.warning("  [%s] no processed .privacy.jsonl in pipeline bucket", sid)
        return None
    # persona / judge / manifest are needed by the rlhf generator (best-effort)
    for fn in (f"{sid}.persona.json", f"{sid}.privacy-judge.json", "manifest.json"):
        _gsutil("cp", f"{base}/{fn}", str(task / fn))
    # seed a zero-shard rlhf manifest so repair_zero_rlhf_text_turns fills it
    (task / "rlhf").mkdir(exist_ok=True)
    (task / "rlhf" / f"{sid}.rlhf.manifest.json").write_text(
        json.dumps({"task_id": sid, "submission_id": sid, "n_shards": 0, "shards": []}))
    # 2) resolve worker_id by locating the export dir
    rc, out = _gsutil("ls", "-d", f"gs://{EXPORTS_BUCKET}/exports/*/{sid}/")
    export_dir = next((l.strip() for l in out.splitlines() if l.strip().endswith(f"/{sid}/")), None)
    if rc != 0 or not export_dir:
        log.warning("  [%s] no raw export found (needed for workspace)", sid)
        return None
    # 3) snapshots -> workspace / workspace_before
    tmp = task / "_snap"
    tmp.mkdir(exist_ok=True)
    for snap, sub in (("final-snapshot.zip", "workspace"), ("initial-snapshot.zip", "workspace_before")):
        rc, _ = _gsutil("cp", export_dir + snap, str(tmp / snap))
        if rc == 0 and (tmp / snap).exists():
            (task / sub).mkdir(exist_ok=True)
            cnt = _extract_workspace_from_snapshot(tmp / snap, task / sub)
            log.info("  [%s] extracted %d files -> %s/", sid, cnt, sub)
    shutil.rmtree(tmp, ignore_errors=True)
    if not (task / "workspace").is_dir():
        log.warning("  [%s] snapshot had no .openclaw/workspace/", sid)
        return None
    return task


def gcs_privacy_score(sid: str) -> int | None:
    """Read the existing privacy-judge verdict score for a bucket task.

    Returns result.privacy_compliance (1-5) or None if unavailable. Used to
    'grab the good ones' without re-judging.
    """
    rc, out = _gsutil("cat", f"gs://{PIPELINE_BUCKET}/{sid}/{sid}.privacy-judge.json")
    if rc != 0 or not out.strip():
        return None
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return None
    res = data.get("result", data) if isinstance(data, dict) else {}
    val = res.get("privacy_compliance")
    return int(val) if isinstance(val, (int, float)) else None


def fetch_gcs_complete(sid: str, dest_root: Path) -> Path | None:
    """Fast path for tasks already complete in the bucket: download the
    processed artifacts + workspace_before_after.zip (no snapshot reconstruction).
    rlhf is still a zero-shard manifest, filled later by ensure_artifacts.
    """
    task = dest_root / sid
    if task.exists():
        shutil.rmtree(task)
    task.mkdir(parents=True)
    base = f"gs://{PIPELINE_BUCKET}/{sid}"
    rc, _ = _gsutil("cp", f"{base}/{sid}.privacy.jsonl", str(task / f"{sid}.privacy.jsonl"))
    if rc != 0 or not (task / f"{sid}.privacy.jsonl").exists():
        log.warning("  [%s] no .privacy.jsonl", sid)
        return None
    for fn in (f"{sid}.persona.json", f"{sid}.privacy-judge.json", "manifest.json"):
        _gsutil("cp", f"{base}/{fn}", str(task / fn))
    _gsutil("cp", "-r", f"{base}/rlhf", str(task))           # zero-shard manifest -> filled later
    # workspace + workspace_before from the bundled zip
    rc, _ = _gsutil("cp", f"{base}/workspace_before_after.zip", str(task / "_wba.zip"))
    if rc != 0 or not (task / "_wba.zip").exists():
        log.warning("  [%s] no workspace_before_after.zip (not a complete task)", sid)
        return None
    import zipfile as _zip
    with _zip.ZipFile(task / "_wba.zip") as zf:
        for info in zf.infolist():
            n = info.filename
            if (n.startswith("workspace/") or n.startswith("workspace_before/")) and not n.endswith("/"):
                if _ws_skip(n) or info.file_size > 500_000:
                    continue
                out = task / n
                out.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as s, out.open("wb") as d:
                    d.write(s.read())
    (task / "_wba.zip").unlink(missing_ok=True)
    if not (task / "workspace").is_dir():
        return None
    return task


def list_complete_gcs_candidates(bucket: str, limit: int) -> list[Candidate]:
    """Enumerate ONLY tasks already complete in the bucket (have a
    workspace_before_after.zip) — one gsutil wildcard call."""
    rc, out = _gsutil("ls", f"gs://{bucket}/*/workspace_before_after.zip")
    if rc != 0:
        log.error("complete-task listing failed: %s", out[-200:])
        return []
    uuid_re = re.compile(r"/([0-9a-fA-F-]{36})/workspace_before_after\.zip$")
    sids = []
    for line in out.splitlines():
        m = uuid_re.search(line.strip())
        if m:
            sids.append(m.group(1))
            if limit and len(sids) >= limit:
                break
    log.info("listed %d COMPLETE candidate(s) in gs://%s", len(sids), bucket)
    return [Candidate(submission_id=s, kind="gcs-complete") for s in sids]


def load_delivered_ids(path: Path) -> set[str]:
    """Read a CSV of already-delivered task UUIDs to skip.

    Accepts either a header row with a submission_id/uuid/task_id column, or a
    bare one-UUID-per-line file. Any 36-char UUID-looking token in the first
    cell is collected.
    """
    ids: set[str] = set()
    if not path or not path.exists():
        return ids
    uuid_re = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
    with path.open() as f:
        reader = csv.reader(f)
        header = None
        for i, row in enumerate(reader):
            if not row:
                continue
            cell = row[0].strip()
            if i == 0 and not uuid_re.match(cell):
                header = [c.strip().lower() for c in row]
                continue
            # if a header named a uuid column, prefer it
            if header:
                for col in ("submission_id", "uuid", "task_id", "id"):
                    if col in header:
                        cell = row[header.index(col)].strip()
                        break
            if uuid_re.match(cell):
                ids.add(cell)
    return ids


def candidates_from_built_root(root: Path) -> list[Candidate]:
    out = []
    for d in sorted(root.glob(UUID_GLOB)):
        if d.is_dir():
            out.append(Candidate(submission_id=d.name, source_dir=d))
    return out


def candidates_from_manifest(csv_path: Path) -> list[Candidate]:
    out = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            out.append(Candidate(
                submission_id=row["submission_id"].strip(),
                worker_id=row.get("worker_id", "").strip(),
                task_id=(row.get("task_id") or "AUTO").strip(),
                export_url=row.get("export_url", "").strip(),
            ))
    return out


def list_bucket_candidates(bucket: str, out_csv: Path, limit: int) -> list[Candidate]:
    """Enumerate submission-id prefixes from the pipeline GCS bucket as GCS
    (reuse-path) candidates. Each prefix is a processed submission; the engine
    fetches its trajectory + snapshots + generates rlhf at process time.

    Uses `gsutil ls` (the gcloud session) so it works without google-cloud-storage
    installed in the engine's interpreter.
    """
    rc, out = _gsutil("ls", f"gs://{bucket}/")
    if rc != 0:
        log.error("gsutil ls failed for gs://%s: %s", bucket, out[-200:])
        return []
    uuid_re = re.compile(r"/([0-9a-fA-F-]{36})/$")
    seen: list[str] = []
    for line in out.splitlines():
        m = uuid_re.search(line.strip())
        if m:
            seen.append(m.group(1))
            if limit and len(seen) >= limit:
                break
    out_csv.write_text("submission_id\n" + "\n".join(seen) + "\n")
    log.info("listed %d candidate submission(s) from gs://%s", len(seen), bucket)
    return [Candidate(submission_id=sid, kind="gcs") for sid in seen]


# ---------------------------------------------------------------------------
# Per-task steps
# ---------------------------------------------------------------------------
def _find_trajectory(task_dir: Path, sid: str) -> Path | None:
    for name in (f"{sid}.jsonl", f"{sid}.json", f"{sid}.privacy.jsonl"):
        p = task_dir / name
        if p.exists():
            return p
    # any top-level jsonl/json that isn't an aux artifact
    for p in sorted(task_dir.glob("*.jsonl")):
        if "_repaired" not in p.name and "negative" not in p.name:
            return p
    return None


def _rlhf_ok(task_dir: Path) -> bool:
    rlhf = task_dir / "rlhf"
    if not rlhf.is_dir():
        return False
    manifest = next(rlhf.glob("*.rlhf.manifest.json"), None)
    shards = list(rlhf.glob("*.rlhf.*.jsonl"))
    return manifest is not None and len(shards) >= 1


def _workspace_before_ok(task_dir: Path) -> bool:
    wb = task_dir / "workspace_before"
    if not wb.is_dir():
        return False
    present = {p.name for p in wb.glob("*.md")}
    return CORE_WS_FILES.issubset(present)


def _missing_parts(task_dir: Path) -> list[str]:
    """Which of the required delivery components are absent from a built task dir."""
    missing = []
    if not (task_dir / "workspace").is_dir():
        missing.append("workspace")
    if not (task_dir / "workspace_before").is_dir():
        missing.append("workspace_before")
    if not _rlhf_ok(task_dir):
        missing.append("rlhf")
    return missing


async def ensure_artifacts(cand: Candidate, task_dir: Path, dry_run: bool) -> list[str]:
    """Ensure RLHF + workspace_before exist; trigger generation if missing.

    Returns a list of remediation notes. In dry-run, only reports gaps.
    """
    notes = []
    if not _rlhf_ok(task_dir):
        notes.append("rlhf_missing")
        if not dry_run:
            await run_cmd([PY, "repair_zero_rlhf_text_turns.py", "--root", task_dir.parent],
                          config.BASE_DIR, "repair_zero_rlhf")
            await run_cmd([PY, "normalize_rlhf_metadata.py", "--root", task_dir.parent],
                          config.BASE_DIR, "normalize_rlhf")
    if not _workspace_before_ok(task_dir):
        notes.append("workspace_before_incomplete")
        if not dry_run:
            # Guide step 7: reconstruct missing core files in workspace_before from
            # the trajectory + final workspace (Opus). Runs in-place, scoped to this
            # task's parent root; no-op when no core files are actually missing.
            root = task_dir.parent
            await run_cmd([PY, "reconstruct_workspace_before.py",
                           "--source-root", root, "--artifact-root", root, "--dest-root", root],
                          config.BASE_DIR, "reconstruct_workspace_before")
    return notes


async def recovery(task_dir: Path, sid: str, dry_run: bool) -> Path:
    """Run htg_repair.py on the trajectory; return the trajectory to deliver.

    htg_repair writes <stem>_repaired.jsonl ONLY when it changes something; if the
    trajectory is already clean it leaves no repaired file and we keep the original.
    """
    traj = _find_trajectory(task_dir, sid)
    if traj is None:
        raise FileNotFoundError(f"no trajectory in {task_dir}")
    if dry_run:
        return traj
    rc, _ = await run_cmd([PY, "htg_repair.py", traj], config.BASE_DIR, "htg_repair")
    repaired = traj.with_name(traj.stem + "_repaired.jsonl")
    return repaired if repaired.exists() else traj


def assemble_delivery(task_dir: Path, sid: str, trajectory: Path, build_root: Path) -> Path:
    """Build <build_root>/<sid>/ with the 4 delivery components."""
    out = build_root / sid
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for sub in ("workspace", "workspace_before", "rlhf"):
        src = task_dir / sub
        if src.is_dir():
            shutil.copytree(src, out / sub)
    # privacy trajectory -> <sid>.json
    shutil.copy2(trajectory, out / f"{sid}.json")
    return out


async def judge_task(build_root: Path, sid: str, run_dir: Path, model: str, dry_run: bool) -> tuple[str, dict]:
    """Run the judge on one task. Returns (verdict, metadata_dict)."""
    if dry_run:
        return "DRY", {}
    judged_root = run_dir / "judged" / sid
    judge_runs = run_dir / "judge_runs" / sid
    if judged_root.exists():
        shutil.rmtree(judged_root)
    args = [PY, "metadata_judge_writer.py",
            "--input-root", build_root, "--output-root", judged_root,
            "--runs-root", judge_runs, "--task-id", sid,
            "--model", model, "--concurrency", "1", "--no-zip", "--overwrite"]
    await run_cmd(args, config.BASE_DIR, "judge")
    # verdict lives in the newest run's scores jsonl
    verdict, meta = "", {}
    runs = sorted(judge_runs.glob("*/metadata_judge_scores.jsonl"))
    if runs:
        for line in runs[-1].read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("task_id") == sid or obj.get("submission_id") == sid:
                # scores line shape: {"task_id", "score": {...,"verdict"}, "validation"}
                score = obj.get("score") or obj
                verdict = str(score.get("verdict", "")).upper()
                break
    meta_path = judged_root / sid / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        # copy the judge-injected metadata.json back into the build folder
        shutil.copy2(meta_path, build_root / sid / "metadata.json")
    return verdict, meta


def zip_dir(folder: Path, zip_path: Path) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(folder.parent))
    return zip_path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class Orchestrator:
    def __init__(self, args):
        self.args = args
        self.run_dir = Path(args.run_dir).expanduser().resolve()
        self.build_root = self.run_dir / "build"
        self.accumulated = self.run_dir / "accumulated"
        self.quarantine = self.run_dir / "quarantine"
        for d in (self.build_root, self.accumulated, self.quarantine):
            d.mkdir(parents=True, exist_ok=True)
        self.progress_file = Path(args.progress_file) if args.progress_file \
            else self.run_dir / "progress.json"
        self.accept = tuple(v.strip().upper() for v in args.accept.split(","))
        self.items: dict[str, ItemResult] = {}
        self.accepted = 0
        # already-delivered task UUIDs to skip (read once; appended to on accept)
        self.delivered_csv = Path(args.delivered_csv) if getattr(args, "delivered_csv", None) else None
        self.delivered: set[str] = load_delivered_ids(self.delivered_csv) if self.delivered_csv else set()
        if self.delivered:
            log.info("loaded %d already-delivered id(s) to skip from %s",
                     len(self.delivered), self.delivered_csv)
        self._load_checkpoint()

    def _record_delivered(self, sid: str) -> None:
        """Append a newly-accepted id to the delivered ledger so future runs skip it."""
        if sid in self.delivered:
            return
        self.delivered.add(sid)
        if self.delivered_csv:
            new_file = not self.delivered_csv.exists()
            self.delivered_csv.parent.mkdir(parents=True, exist_ok=True)
            with self.delivered_csv.open("a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["submission_id"])
                w.writerow([sid])

    # ----- checkpoint / progress -----
    def _ckpt_path(self) -> Path:
        return self.run_dir / "checkpoint.json"

    def _load_checkpoint(self):
        p = self._ckpt_path()
        if self.args.resume and p.exists():
            data = json.loads(p.read_text())
            for k, v in data.get("items", {}).items():
                self.items[k] = ItemResult(**v)
            self.accepted = sum(1 for it in self.items.values() if it.status == "accepted")
            log.info("resumed: %d accepted, %d items tracked", self.accepted, len(self.items))

    def _save_checkpoint(self):
        self._ckpt_path().write_text(json.dumps(
            {"items": {k: asdict(v) for k, v in self.items.items()}}, indent=2))

    def write_progress(self, in_progress: int = 0, done: bool = False, final_zip: str = ""):
        payload = {
            "target": self.args.target,
            "accepted": self.accepted,
            "in_progress": in_progress,
            "processed": len([i for i in self.items.values() if i.status in
                              ("accepted", "quarantined", "error")]),
            "done": done,
            "accept_bar": list(self.accept),
            "final_zip": final_zip,
            "items": [asdict(i) for i in self.items.values()],
            "updated_at": utc_stamp(),
        }
        tmp = self.progress_file.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.progress_file)

    # ----- per task -----
    async def process(self, cand: Candidate, sem: asyncio.Semaphore):
        sid = cand.submission_id
        if sid in self.delivered:
            return  # already delivered in a prior run — skip
        prior = self.items.get(sid)
        if prior and prior.status in ("accepted", "quarantined"):
            return  # already settled (resume)
        async with sem:
            if self.accepted >= self.args.target:
                return
            it = ItemResult(submission_id=sid)
            self.items[sid] = it
            try:
                task_dir = cand.source_dir
                if cand.kind in ("gcs", "gcs-complete"):
                    # 'Grab the good ones': gate on the existing privacy-judge score
                    if self.args.min_privacy > 0:
                        score = await asyncio.to_thread(gcs_privacy_score, sid)
                        if score is not None and score < self.args.min_privacy:
                            it.status = "quarantined"
                            it.detail = f"privacy-judge privacy_compliance={score} < bar {self.args.min_privacy}"
                            log.info("  [%s] SKIP (good-filter) — %s", sid, it.detail)
                            return
                    # gcs-complete: download bundled artifacts (no snapshot rebuild).
                    # gcs:          reconstruct workspace from raw export snapshots.
                    fetcher = fetch_gcs_complete if cand.kind == "gcs-complete" else fetch_gcs_task
                    task_dir = await asyncio.to_thread(fetcher, sid, self.run_dir / "gcs_src")
                    if task_dir is None or not task_dir.exists():
                        it.status, it.detail = "error", f"{cand.kind} fetch failed"
                        log.warning("  [%s] %s", sid, it.detail)
                        return
                    if not self.args.dry_run:
                        await ensure_artifacts(cand, task_dir, self.args.dry_run)  # fills zero-shard rlhf
                elif task_dir is None:
                    # manifest path: full pipeline from export -> built task dir
                    task_dir = await self._build_from_export(cand)
                if task_dir is None or not task_dir.exists():
                    it.status, it.detail = "error", "no task dir"
                    return

                # Completeness guard: never judge/accept a task that lacks the
                # delivery components. Quarantine with a clear reason instead of
                # silently shipping a partial zip.
                missing = _missing_parts(task_dir)
                if missing and not self.args.dry_run:
                    it.status = "quarantined"
                    it.detail = "incomplete source artifacts: missing " + ",".join(missing)
                    log.warning("  [%s] QUARANTINED — %s", sid, it.detail)
                    qd = self.quarantine / sid
                    if qd.exists():
                        shutil.rmtree(qd)
                    qd.mkdir(parents=True, exist_ok=True)
                    (qd / "REASON.txt").write_text(it.detail + "\n")
                    return

                if cand.kind not in ("gcs", "gcs-complete"):
                    await ensure_artifacts(cand, task_dir, self.args.dry_run)

                verdict = ""
                for attempt in range(1, self.args.max_recovery + 2):  # initial + recoveries
                    it.attempts = attempt
                    trajectory = await recovery(task_dir, sid, self.args.dry_run)
                    folder = assemble_delivery(task_dir, sid, trajectory, self.build_root)
                    it.status = "built"
                    verdict, _meta = await judge_task(
                        self.build_root, sid, self.run_dir, self.args.judge_model, self.args.dry_run)
                    it.verdict = verdict
                    it.status = "judged"
                    if self.args.dry_run or verdict in self.accept:
                        break
                    log.info("  [%s] verdict=%s -> recovery attempt %d", sid, verdict, attempt)

                if self.args.dry_run or verdict in self.accept:
                    zp = zip_dir(self.build_root / sid, self.accumulated / f"{sid}.zip")
                    it.zip_path, it.status = str(zp), "accepted"
                    self.accepted += 1
                    self._record_delivered(sid)
                    log.info("  [%s] ACCEPTED (%s) %d/%d", sid, verdict or "dry", self.accepted, self.args.target)
                else:
                    qd = self.quarantine / sid
                    if qd.exists():
                        shutil.rmtree(qd)
                    shutil.copytree(self.build_root / sid, qd)
                    it.status = "quarantined"
                    log.info("  [%s] QUARANTINED after %d attempts (last=%s)", sid, it.attempts, verdict)
            except Exception as e:  # noqa: BLE001
                it.status, it.detail = "error", str(e)
                log.exception("  [%s] error", sid)
            finally:
                self._save_checkpoint()
                self.write_progress()

    async def _build_from_export(self, cand: Candidate) -> Path | None:
        """Manifest/GCS path: run the full pipeline to produce a built task dir."""
        if self.args.dry_run:
            return None
        man = self.run_dir / "manifests"
        man.mkdir(parents=True, exist_ok=True)
        row_csv = man / f"{cand.submission_id}.csv"
        with row_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["worker_id", "submission_id", "task_id", "export_url"])
            w.writerow([cand.worker_id, cand.submission_id, cand.task_id, cand.export_url])
        out_dir = self.run_dir / "pipeline_out"
        await run_cmd([PY, "run_pipeline.py", "--input", row_csv, "--output", out_dir,
                       "--task-id", cand.task_id, "--rlhf", "--openclaw-repair"],
                      config.BASE_DIR, "run_pipeline")
        produced = next(out_dir.glob(f"**/{cand.submission_id}"), None)
        return produced

    # ----- run loop -----
    async def run(self, candidates: list[Candidate]):
        sem = asyncio.Semaphore(self.args.concurrency)
        self.write_progress()
        pending = [c for c in candidates
                   if self.items.get(c.submission_id, ItemResult(c.submission_id)).status
                   not in ("accepted", "quarantined")]
        i = 0
        while self.accepted < self.args.target and i < len(pending):
            # dispatch a window of candidates, then re-check the accepted count
            window = pending[i:i + self.args.concurrency]
            i += len(window)
            await asyncio.gather(*(self.process(c, sem) for c in window))
        final_zip = self.finalize()
        self.write_progress(done=True, final_zip=str(final_zip))
        return final_zip

    def finalize(self) -> Path:
        # review manifest
        man = self.run_dir / "review_manifest.csv"
        with man.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["submission_id", "status", "verdict", "attempts", "zip_path", "detail"])
            for it in self.items.values():
                w.writerow([it.submission_id, it.status, it.verdict, it.attempts, it.zip_path, it.detail])
        # roll-up zip of accumulated successes
        roll = self.run_dir / f"{self.run_dir.name}_accumulated.zip"
        with zipfile.ZipFile(roll, "w", zipfile.ZIP_DEFLATED) as zf:
            for zp in sorted(self.accumulated.glob("*.zip")):
                zf.write(zp, zp.name)
        # optional: upload deliverables to a GCS folder (one prefix per run)
        gcs_roll_up = gcs_prefix = None
        if getattr(self.args, "upload_prefix", None):
            base = self.args.upload_prefix.rstrip("/") + f"/{self.run_dir.name}"
            rc, _ = _gsutil("cp", str(roll), f"{base}/{roll.name}")
            if rc == 0:
                gcs_roll_up = f"{base}/{roll.name}"
                gcs_prefix = base
                # per-task zips under tasks/
                _gsutil("-m", "cp", *[str(p) for p in sorted(self.accumulated.glob("*.zip"))],
                        f"{base}/tasks/")
                _gsutil("cp", str(man), f"{base}/review_manifest.csv")
                log.info("uploaded deliverables -> %s", base)
            else:
                log.warning("GCS upload failed for %s", roll.name)
        # summary
        dist: dict[str, int] = {}
        for it in self.items.values():
            dist[it.status] = dist.get(it.status, 0) + 1
        summary = {
            "target": self.args.target, "accepted": self.accepted,
            "accept_bar": list(self.accept), "status_distribution": dist,
            "quarantined": [i.submission_id for i in self.items.values() if i.status == "quarantined"],
            "roll_up_zip": str(roll), "review_manifest": str(man),
            "gcs_roll_up": gcs_roll_up, "gcs_prefix": gcs_prefix,
            "finished_at": utc_stamp(),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        log.info("FINAL: %d/%d accepted; roll-up=%s%s", self.accepted, self.args.target,
                 roll.name, f" | gcs={gcs_prefix}" if gcs_prefix else "")
        return roll


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="OpenClaw Privacy delivery orchestrator")
    p.add_argument("--target", type=int, required=True, help="number of accepted deliverables to produce")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--built-root", type=Path, help="dir of already-assembled <uuid>/ task folders")
    src.add_argument("--candidates", type=Path, help="4-col manifest CSV (full pipeline per export)")
    src.add_argument("--bucket", help="GCS bucket to enumerate submission-id prefixes from")
    src.add_argument("--gcs", help="comma-separated submission ids to deliver from the live bucket "
                                   "(reuse path: processed trajectory + snapshot workspace + generated rlhf)")
    p.add_argument("--list-candidates", action="store_true", help="with --bucket: write manifest and use it")
    p.add_argument("--gcs-mode", choices=("complete", "reconstruct"), default="reconstruct",
                   help="with --bucket: 'complete' grabs only bucket-complete tasks (fast, no snapshot "
                        "rebuild); 'reconstruct' rebuilds workspace from raw export snapshots")
    p.add_argument("--min-privacy", type=int, default=4,
                   help="grab only tasks whose existing privacy-judge privacy_compliance >= this "
                        "(0 disables the good-filter)")
    p.add_argument("--run-dir", default=str(config.OUTPUT_DIR / f"delivery_run_{utc_stamp()}"))
    p.add_argument("--progress-file", default=None)
    p.add_argument("--accept", default=",".join(ACCEPT_DEFAULT), help="verdicts that count as success")
    p.add_argument("--max-recovery", type=int, default=2)
    p.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENT_TASKS)
    p.add_argument("--judge-model", default=getattr(config, "VERIFIER_MODEL", "claude-opus-4-7"))
    p.add_argument("--limit", type=int, default=0, help="cap candidate pool size")
    p.add_argument("--delivered-csv", default=None,
                   help="CSV of already-delivered task UUIDs to skip; accepted ids are appended to it")
    p.add_argument("--upload-prefix", default=None,
                   help="gs:// prefix to upload deliverables to (roll-up + per-task + manifest), "
                        "one subfolder per run. Empty = local only.")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="assemble only; skip recovery/judge LLM calls")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.built_root:
        candidates = candidates_from_built_root(Path(args.built_root))
    elif args.candidates:
        candidates = candidates_from_manifest(Path(args.candidates))
    elif args.gcs:
        candidates = [Candidate(submission_id=s.strip(), kind="gcs")
                      for s in args.gcs.split(",") if s.strip()]
    else:  # --bucket
        if args.gcs_mode == "complete":
            candidates = list_complete_gcs_candidates(args.bucket, args.limit)
        else:
            candidates = list_bucket_candidates(args.bucket, run_dir / "candidates.csv", args.limit)
    # drop already-delivered ids up front so the pool / --limit reflect real work
    if args.delivered_csv:
        delivered = load_delivered_ids(Path(args.delivered_csv))
        before = len(candidates)
        candidates = [c for c in candidates if c.submission_id not in delivered]
        log.info("skipping %d already-delivered candidate(s) from %s",
                 before - len(candidates), args.delivered_csv)
    if args.limit:
        candidates = candidates[:max(args.limit, args.target)]
    log.info("delivery run: target=%d, %d candidates, accept=%s, run-dir=%s",
             args.target, len(candidates), args.accept, run_dir)
    if not candidates:
        log.error("no candidates found")
        return 2

    orch = Orchestrator(args)
    asyncio.run(orch.run(candidates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
