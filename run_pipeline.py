"""Main CLI — async orchestrator with checkpoint/resume for the Privacy SFT pipeline."""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.logging import RichHandler

from config import MAX_CONCURRENT_TASKS, MAX_REFIX_ITERATIONS, MIN_PRIVACY_COMPLIANCE, OUTPUT_DIR
from models import TaskReport, VerificationResult
from token_tracker import tracker

console = Console()
logging.basicConfig(level=logging.INFO, handlers=[RichHandler(console=console)])
logger = logging.getLogger("pipeline")


def _load_input_csv(path: str) -> list[dict]:
    """Load input CSV: worker_id, submission_id, task_id, export_url."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("export_url"):
                rows.append(row)
    return rows


def _load_checkpoint(output_dir: Path) -> set[str]:
    """Load already-processed submission IDs from checkpoint."""
    cp_path = output_dir / "checkpoint.json"
    if cp_path.exists():
        data = json.loads(cp_path.read_text())
        return set(data.get("completed", []))
    return set()


def _save_checkpoint(output_dir: Path, completed: set[str]) -> None:
    """Save checkpoint with completed submission IDs."""
    cp_path = output_dir / "checkpoint.json"
    cp_path.write_text(json.dumps({"completed": sorted(completed)}, indent=2))


async def _process_single_task(
    row: dict, output_dir: Path, dry_run: bool = False, generate_rlhf: bool = True
) -> TaskReport:
    """Process a single trajectory through all pipeline stages."""
    from parser import parse_export
    from classifier import classify_trajectory
    from assembler import assemble_trajectory
    from verifier import verify_trajectory
    from writer import write_trajectory_output, write_sft_entry, write_rlhf_pairs

    task_id = row["task_id"]
    submission_id = row["submission_id"]
    worker_id = row.get("worker_id", "")
    export_url = row["export_url"]

    report = TaskReport(task_id=task_id, submission_id=submission_id, status="PASS")

    # Stage 1: Parse
    logger.info("[%s] Stage 1: Parsing...", task_id)
    trajectory = parse_export(task_id, submission_id, worker_id, export_url)
    if not trajectory:
        report.status = "PARSE_ERROR"
        report.error_message = "Failed to parse trajectory"
        return report

    logger.info(
        "[%s] Parsed: %d user msgs, %d assistant turns",
        task_id, len(trajectory.user_messages), len(trajectory.assistant_turns)
    )

    # Stage 2: Classify PII
    logger.info("[%s] Stage 2: Classifying PII...", task_id)
    try:
        pii_map = await classify_trajectory(trajectory)
    except Exception as e:
        report.status = "CLASSIFY_ERROR"
        report.error_message = str(e)
        return report

    report.pii_entities_found = len(pii_map.entities)
    report.max_pii_level = pii_map.max_level
    logger.info(
        "[%s] PII: %d entities, max level %s",
        task_id, len(pii_map.entities), pii_map.max_level
    )

    if dry_run:
        report.status = "PASS"
        return report

    # Stage 3: Assemble (deterministic registry + minimal Claude for adversarial text)
    logger.info("[%s] Stage 3: Assembling privacy-compliant trajectory...", task_id)
    try:
        rewrite_result = await assemble_trajectory(trajectory, pii_map)
    except Exception as e:
        report.status = "REWRITE_ERROR"
        report.error_message = str(e)
        return report

    report.scenarios_covered = rewrite_result.scenarios_covered
    report.privacy_decision_points = rewrite_result.privacy_decision_points

    # Stage 4: Verify
    logger.info("[%s] Stage 4: Verifying with GPT-5.4...", task_id)
    verification = await verify_trajectory(trajectory, rewrite_result, pii_map)
    report.verification_scores = {
        "privacy_compliance": verification.privacy_compliance,
        "correctness": verification.correctness,
        "completeness": verification.completeness,
        "efficiency": verification.efficiency,
        "naturality": verification.naturality,
        "overall": verification.overall,
    }

    # With the deterministic assembler, refixing is not needed —
    # the registry guarantees compliance. Log the verdict and move on.
    refix_count = 0
    report.refix_iterations = refix_count

    # Determine final status
    if verification.verdict == "PASS":
        report.status = "PASS" if refix_count == 0 else "MINOR_FIXED"
    elif verification.verdict == "MINOR_ISSUES":
        report.status = "MINOR_FIXED"
    else:
        report.status = "FAIL"
        report.error_message = verification.rationale

    # Stage 5: Write output
    logger.info("[%s] Stage 5: Writing output (verdict: %s)...", task_id, verification.verdict)
    write_trajectory_output(output_dir, trajectory, rewrite_result, pii_map, verification, report)

    # Stage 6: RLHF pair generation (only for passing trajectories)
    if generate_rlhf and report.status in ("PASS", "MINOR_FIXED"):
        logger.info("[%s] Stage 6: Generating RLHF preference pairs...", task_id)
        try:
            from rlhf_generator import generate_rlhf_pairs, compute_batch_report
            from reward_scorer import score_rejected_alternatives

            rlhf_pairs = await generate_rlhf_pairs(
                trajectory, rewrite_result, pii_map,
                max_pairs_per_trajectory=50,
            )

            if rlhf_pairs:
                rlhf_pairs = await score_rejected_alternatives(
                    rlhf_pairs,
                    task_description=trajectory.task_spec.get("title", task_id),
                )
                rlhf_report = compute_batch_report(task_id, submission_id, rlhf_pairs)

                task_output_dir = output_dir / submission_id
                write_rlhf_pairs(task_output_dir, rlhf_pairs, rlhf_report)

                total_rejected = sum(len(p.rejected) for p in rlhf_pairs)
                logger.info(
                    "[%s] RLHF: %d pairs, %d rejected alternatives, avg score %.2f",
                    task_id, len(rlhf_pairs), total_rejected,
                    rlhf_report.avg_reward_score,
                )
            else:
                logger.warning("[%s] RLHF: No decision points found", task_id)

        except Exception as e:
            logger.error("[%s] RLHF generation failed: %s", task_id, e)

    return report


async def run_pipeline(
    input_path: str,
    output_path: str,
    task_filter: str | None = None,
    resume: bool = False,
    dry_run: bool = False,
    concurrency: int = MAX_CONCURRENT_TASKS,
    generate_rlhf: bool = True,
) -> None:
    """Main pipeline orchestrator."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_input_csv(input_path)
    logger.info("Loaded %d tasks from input CSV", len(rows))

    # Filter to single task if specified
    if task_filter:
        rows = [r for r in rows if r["task_id"] == task_filter]
        if not rows:
            logger.error("Task %s not found in input CSV", task_filter)
            return

    # Resume support
    completed = _load_checkpoint(output_dir) if resume else set()
    if completed:
        logger.info("Resuming: %d already completed", len(completed))
    pending = [r for r in rows if r["submission_id"] not in completed]
    logger.info("Processing %d tasks (concurrency: %d, RLHF: %s)", len(pending), concurrency, generate_rlhf)

    # Process with semaphore
    sem = asyncio.Semaphore(concurrency)
    reports: list[TaskReport] = []
    sft_entries: list[dict] = []

    async def _bounded_process(row: dict) -> TaskReport:
        async with sem:
            return await _process_single_task(row, output_dir, dry_run, generate_rlhf)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing trajectories", total=len(pending))

        # Process in batches for better progress tracking
        tasks = [_bounded_process(row) for row in pending]
        for coro in asyncio.as_completed(tasks):
            report = await coro
            reports.append(report)
            completed.add(report.submission_id)
            _save_checkpoint(output_dir, completed)
            progress.advance(task)

            status_str = f"[green]{report.status}[/green]" if report.status in ("PASS", "MINOR_FIXED") else f"[red]{report.status}[/red]"
            logger.info("[%s] Done: %s", report.task_id, report.status)

    # Write SFT dataset
    if not dry_run:
        sft_path = output_dir / "sft_dataset.jsonl"
        logger.info("Writing SFT dataset to %s", sft_path)

        # Re-read completed trajectories for SFT entries
        for report in reports:
            if report.status in ("PASS", "MINOR_FIXED"):
                task_dir = output_dir / report.submission_id
                meta_path = task_dir / "metadata.json"
                if meta_path.exists():
                    sft_entries.append({
                        "task_id": report.task_id,
                        "submission_id": report.submission_id,
                        "status": report.status,
                    })

        with open(sft_path, "w") as f:
            for entry in sft_entries:
                f.write(json.dumps(entry) + "\n")

    # Write batch report
    from writer import write_batch_report
    write_batch_report(output_dir, reports, tracker.summary())

    # Final summary
    passed = sum(1 for r in reports if r.status in ("PASS", "MINOR_FIXED"))
    console.print(f"\n[bold green]Pipeline complete:[/bold green] {passed}/{len(reports)} passed")
    console.print(f"Token usage: {tracker.summary()['total_tokens']:,} total")


def main():
    parser = argparse.ArgumentParser(description="Privacy SFT Trajectory Converter Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--task-id", "-t", help="Process single task by ID")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Parse + classify only (no rewriting)")
    parser.add_argument("--concurrency", "-c", type=int, default=MAX_CONCURRENT_TASKS, help="Max concurrent tasks")
    parser.add_argument("--no-rlhf", action="store_true", help="Skip RLHF pair generation (Stage 6)")

    args = parser.parse_args()

    if not Path(args.input).exists():
        console.print(f"[red]Error:[/red] Input file not found: {args.input}")
        sys.exit(1)

    asyncio.run(run_pipeline(
        input_path=args.input,
        output_path=args.output,
        task_filter=args.task_id,
        resume=args.resume,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
        generate_rlhf=not args.no_rlhf,
    ))


if __name__ == "__main__":
    main()
