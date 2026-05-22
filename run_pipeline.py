"""Main CLI — async orchestrator with checkpoint/resume for the Privacy SFT pipeline.

Supports two modes:
  - Standard mode: reads CSV with export_url, downloads real trajectories from GCS
  - Synthetic mode (--synthetic): reads tasks_all.csv task definitions, generates
    trajectories purely from task_def + persona (no GCS required)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import sys
import uuid
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


def _deterministic_uuid(seed: str) -> str:
    """Generate a deterministic UUID5 from a seed string."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def _load_input_csv(path: str) -> list[dict]:
    """Load input CSV: worker_id, submission_id, task_id, export_url."""
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("export_url"):
                rows.append(row)
    if not rows:
        logger.warning("No rows with export_url found in %s", path)
    return rows


def _load_tasks_csv_for_synthetic(path: str) -> list[dict]:
    """Load tasks_all.csv and produce pipeline-compatible rows for synthetic mode.

    Generates deterministic submission_id and worker_id from the task_id.
    Maps P-prefixed task IDs to T-prefix for pipeline consistency.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = row.get("task_id", "").strip()
            if not task_id:
                continue
            t_task_id = task_id.replace("P-", "T-", 1) if task_id.startswith("P-") else task_id
            rows.append({
                "task_id": t_task_id,
                "submission_id": _deterministic_uuid(f"synth-{task_id}"),
                "worker_id": _deterministic_uuid(f"worker-{task_id}"),
                "export_url": "",
                "_synthetic": True,
            })
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
    row: dict, output_dir: Path, dry_run: bool = False, generate_rlhf: bool = False,
    single_shot: bool = False,
) -> TaskReport:
    """Process a single trajectory through all pipeline stages."""
    from parser import parse_export
    from classifier import classify_trajectory
    from assembler import assemble_trajectory
    from verifier import verify_trajectory
    from writer import write_trajectory_output, write_rlhf_pairs
    from task_context import get_task_definition, get_persona_for_task
    from synthetic_generator import check_trajectory_task_match, generate_synthetic_trajectory

    task_id = row["task_id"]
    submission_id = row["submission_id"]
    worker_id = row.get("worker_id", "")
    export_url = row.get("export_url", "")
    is_synthetic_mode = row.get("_synthetic", False)

    report = TaskReport(task_id=task_id, submission_id=submission_id, status="PASS")

    trajectory = None

    if is_synthetic_mode:
        # Synthetic-only mode: generate trajectory directly from task definition
        logger.info("[%s] Stage 1: Generating synthetic trajectory...", task_id)
        trajectory = await generate_synthetic_trajectory(task_id, submission_id, worker_id)
        if not trajectory:
            report.status = "SYNTH_ERROR"
            report.error_message = "Failed to generate synthetic trajectory"
            return report
        logger.info(
            "[%s] Synthetic: %d user msgs, %d assistant turns",
            task_id, len(trajectory.user_messages), len(trajectory.assistant_turns),
        )
    else:
        # Standard mode: parse from GCS export
        logger.info("[%s] Stage 1: Parsing...", task_id)
        trajectory = parse_export(task_id, submission_id, worker_id, export_url)
        if not trajectory:
            report.status = "PARSE_ERROR"
            report.error_message = "Failed to parse trajectory"
            return report

        # Update task_id if it was auto-detected by the parser
        if task_id == "AUTO" or not task_id:
            task_id = trajectory.task_id
            report.task_id = task_id
            logger.info("[AUTO] Resolved task_id to: %s", task_id)

        # Stage 1b: Enrich with task definition + persona from external data
        task_def = get_task_definition(task_id)
        if task_def:
            trajectory.task_spec = {
                "title": task_def.get("task_title", ""),
                "goal_summary": task_def.get("goal_summary", ""),
                "privacy_scenario": task_def.get("privacy_scenario", ""),
                "data_levels": task_def.get("data_levels", ""),
                "expected_privacy_actions": task_def.get("expected_privacy_actions", ""),
                "tool_tiers": task_def.get("tool_tiers", ""),
                "pii_fields_exercised": task_def.get("pii_fields_exercised", ""),
            }
            logger.info("[%s] Task context loaded: %s", task_id, task_def.get("goal_summary", "")[:80])

        if not trajectory.persona:
            ext_persona = get_persona_for_task(task_id)
            if ext_persona:
                trajectory.persona = ext_persona
                logger.info("[%s] Persona loaded: %s %s",
                            task_id, ext_persona.get("first_name", ""), ext_persona.get("last_name", ""))

        logger.info(
            "[%s] Parsed: %d user msgs, %d assistant turns",
            task_id, len(trajectory.user_messages), len(trajectory.assistant_turns)
        )

        # Stage 1c: Check trajectory-task alignment
        matches, match_reason = check_trajectory_task_match(trajectory)
        if not matches:
            logger.warning("[%s] Trajectory-task MISMATCH: %s", task_id, match_reason)
            logger.info("[%s] Generating synthetic trajectory from task definition...", task_id)
            synthetic = await generate_synthetic_trajectory(task_id, submission_id, worker_id)
            if synthetic:
                trajectory = synthetic
                logger.info(
                    "[%s] Synthetic trajectory: %d user msgs, %d assistant turns",
                    task_id, len(trajectory.user_messages), len(trajectory.assistant_turns),
                )
            else:
                logger.error("[%s] Synthetic generation failed, using original (will likely fail verification)", task_id)
        else:
            logger.info("[%s] Trajectory-task alignment: OK", task_id)

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

    # Stage 3: Rewrite trajectory for privacy compliance
    if single_shot:
        logger.info("[%s] Stage 3: Single-shot LLM rewrite (full context)...", task_id)
        try:
            from single_shot_rewriter import rewrite_trajectory_single_shot
            rewrite_result = await rewrite_trajectory_single_shot(trajectory, pii_map)
        except Exception as e:
            import traceback
            logger.error("[%s] SINGLE_SHOT_ERROR traceback:\n%s", task_id, traceback.format_exc())
            report.status = "REWRITE_ERROR"
            report.error_message = str(e)
            return report
    else:
        logger.info("[%s] Stage 3: Assembling privacy-compliant trajectory...", task_id)
        try:
            rewrite_result = await assemble_trajectory(trajectory, pii_map)
        except Exception as e:
            import traceback
            logger.error("[%s] REWRITE_ERROR traceback:\n%s", task_id, traceback.format_exc())
            report.status = "REWRITE_ERROR"
            report.error_message = str(e)
            return report

        # Stage 3.5: Trajectory reviewer (only in multi-stage mode)
        logger.info("[%s] Stage 3.5: Reviewing trajectory quality...", task_id)
        try:
            from trajectory_reviewer import review_and_fix
            rewrite_result = await review_and_fix(trajectory, rewrite_result, pii_map)
        except Exception as e:
            logger.warning("[%s] Reviewer failed (non-fatal): %s", task_id, e)

    report.scenarios_covered = rewrite_result.scenarios_covered
    report.privacy_decision_points = rewrite_result.privacy_decision_points

    # Stage 4: Verify
    logger.info("[%s] Stage 4: Verifying with Claude Opus 4.7...", task_id)
    verification = await verify_trajectory(trajectory, rewrite_result, pii_map)
    report.verification_scores = {
        "privacy_compliance": verification.privacy_compliance,
        "correctness": verification.correctness,
        "completeness": verification.completeness,
        "efficiency": verification.efficiency,
        "naturality": verification.naturality,
        "overall": verification.overall,
    }

    # Stage 4a: Synthetic FAIL retry — regenerate trajectory with verifier feedback
    is_synthetic = trajectory.task_spec.get("synthetic", False) or is_synthetic_mode
    synth_retry_count = 0
    _MAX_SYNTH_RETRIES = 2
    while (
        verification.verdict == "FAIL"
        and is_synthetic
        and synth_retry_count < _MAX_SYNTH_RETRIES
    ):
        synth_retry_count += 1
        logger.warning(
            "[%s] Synthetic FAIL (attempt %d/%d), regenerating with verifier feedback: %s",
            task_id, synth_retry_count, _MAX_SYNTH_RETRIES,
            verification.rationale[:200],
        )
        try:
            new_trajectory = await generate_synthetic_trajectory(
                task_id, submission_id, worker_id,
            )
            if not new_trajectory:
                logger.error("[%s] Retry regeneration failed", task_id)
                break
            trajectory = new_trajectory
            pii_map = await classify_trajectory(trajectory)
            rewrite_result = await assemble_trajectory(trajectory, pii_map)
            verification = await verify_trajectory(trajectory, rewrite_result, pii_map)
            logger.info(
                "[%s] Retry %d result: %s (privacy=%d/5, overall=%.1f)",
                task_id, synth_retry_count, verification.verdict,
                verification.privacy_compliance, verification.overall,
            )
        except Exception as e:
            logger.error("[%s] Retry %d failed: %s", task_id, synth_retry_count, e)
            break

    # Stage 4b: Refix loop — fix MINOR_ISSUES and re-verify
    refix_count = 0
    best_verification = verification
    best_rewrite = rewrite_result
    while verification.verdict == "MINOR_ISSUES" and refix_count < MAX_REFIX_ITERATIONS:
        refix_count += 1
        logger.info("[%s] Stage 4b: Refix iteration %d...", task_id, refix_count)
        try:
            from refixer import refix_trajectory
            rewrite_result = await refix_trajectory(
                trajectory, pii_map, rewrite_result, verification
            )
            verification = await verify_trajectory(trajectory, rewrite_result, pii_map)
            logger.info(
                "[%s] Re-verified: %s (privacy=%d/5, overall=%.1f)",
                task_id, verification.verdict,
                verification.privacy_compliance, verification.overall,
            )
            if verification.overall >= best_verification.overall:
                best_verification = verification
                best_rewrite = rewrite_result
            else:
                logger.warning(
                    "[%s] Refix regression (%.1f → %.1f), rolling back",
                    task_id, best_verification.overall, verification.overall,
                )
                verification = best_verification
                rewrite_result = best_rewrite
                break
        except Exception as e:
            logger.error("[%s] Refix failed: %s", task_id, e)
            break

    report.refix_iterations = refix_count

    report.verification_scores = {
        "privacy_compliance": verification.privacy_compliance,
        "correctness": verification.correctness,
        "completeness": verification.completeness,
        "efficiency": verification.efficiency,
        "naturality": verification.naturality,
        "overall": verification.overall,
    }

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

                rlhf_output_dir = output_dir / "_pipeline" / "rlhf"
                write_rlhf_pairs(rlhf_output_dir, rlhf_pairs, rlhf_report)

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
    generate_rlhf: bool = False,
    synthetic_mode: bool = False,
    single_shot: bool = False,
) -> None:
    """Main pipeline orchestrator."""
    from llm_retry import set_api_concurrency
    # Cap concurrent API calls: at most 2x the task concurrency, but max 20
    # to avoid Anthropic rate limits at scale
    api_concurrency = min(concurrency * 2, 20)
    set_api_concurrency(api_concurrency)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    if synthetic_mode:
        rows = _load_tasks_csv_for_synthetic(input_path)
        logger.info("Loaded %d task definitions for synthetic generation", len(rows))
    else:
        rows = _load_input_csv(input_path)
        logger.info("Loaded %d tasks from input CSV", len(rows))

    # Filter to single task if specified (supports both T- and P- prefix)
    if task_filter:
        alt_filter = task_filter.replace("P-", "T-", 1) if task_filter.startswith("P-") else task_filter.replace("T-", "P-", 1)
        rows = [r for r in rows if r["task_id"] in (task_filter, alt_filter)]
        if not rows:
            logger.error("Task %s not found in input CSV", task_filter)
            return

    # Resume support
    completed = _load_checkpoint(output_dir) if resume else set()
    if completed:
        logger.info("Resuming: %d already completed", len(completed))
    pending = [r for r in rows if r["submission_id"] not in completed]

    # Apply --limit if set (useful for testing synthetic mode on a subset)
    if hasattr(run_pipeline, '_limit') and run_pipeline._limit > 0:
        pending = pending[:run_pipeline._limit]

    logger.info("Processing %d tasks (concurrency: %d, RLHF: %s, synthetic: %s)",
                len(pending), concurrency, generate_rlhf, synthetic_mode)

    # Process with semaphore
    sem = asyncio.Semaphore(concurrency)
    reports: list[TaskReport] = []

    async def _bounded_process(row: dict) -> TaskReport:
        async with sem:
            return await _process_single_task(row, output_dir, dry_run, generate_rlhf, single_shot)

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

    # Write SFT dataset and batch report to _pipeline/ (not per-submission)
    pipeline_dir = output_dir / "_pipeline"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        from writer import write_sft_dataset
        logger.info("Writing Privacy SFT dataset to %s", pipeline_dir / "sft_dataset.jsonl")
        write_sft_dataset(pipeline_dir, reports)

    from writer import write_batch_report
    write_batch_report(pipeline_dir, reports, tracker.summary())

    # Final summary
    passed = sum(1 for r in reports if r.status in ("PASS", "MINOR_FIXED"))
    console.print(f"\n[bold green]Pipeline complete:[/bold green] {passed}/{len(reports)} passed")
    console.print(f"Token usage: {tracker.summary()['total_tokens']:,} total")


def main():
    parser = argparse.ArgumentParser(description="Privacy SFT Trajectory Converter Pipeline")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path (export CSV or tasks_all.csv with --synthetic)")
    parser.add_argument("--output", "-o", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--task-id", "-t", help="Process single task by ID (accepts both T- and P- prefix)")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--dry-run", action="store_true", help="Parse + classify only (no rewriting)")
    parser.add_argument("--concurrency", "-c", type=int, default=MAX_CONCURRENT_TASKS, help="Max concurrent tasks")
    parser.add_argument("--rlhf", action="store_true", help="Generate RLHF pairs after Privacy SFT output")
    parser.add_argument("--no-rlhf", action="store_true", help="Compatibility flag; RLHF is off unless --rlhf is set")
    parser.add_argument("--synthetic", "-s", action="store_true",
                        help="Synthetic-only mode: generate trajectories from task definitions (no GCS export required). "
                             "Input should be tasks_all.csv with task_id, goal_summary, etc.")
    parser.add_argument("--single-shot", action="store_true",
                        help="Use single-shot LLM rewrite (sends full trajectory + all context in one call)")
    parser.add_argument("--limit", "-n", type=int, default=0,
                        help="Limit to first N tasks (useful for testing synthetic mode)")

    args = parser.parse_args()

    if not Path(args.input).exists():
        console.print(f"[red]Error:[/red] Input file not found: {args.input}")
        sys.exit(1)

    run_pipeline._limit = args.limit

    asyncio.run(run_pipeline(
        input_path=args.input,
        output_path=args.output,
        task_filter=args.task_id,
        resume=args.resume,
        dry_run=args.dry_run,
        concurrency=args.concurrency,
        generate_rlhf=args.rlhf and not args.no_rlhf,
        synthetic_mode=args.synthetic,
        single_shot=args.single_shot,
    ))


if __name__ == "__main__":
    main()
