#!/usr/bin/env python3
"""Run and summarize Kimi-Audio LoRA evaluations.

The default mode still evaluates a model/task matrix through ``run.py``. The
``requested-benchmark`` suite launches the requested model/task/split jobs
in parallel; each job runs its three repetitions sequentially with per-question
concurrency fixed by ``--concurrency``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.models.kimi_lora import KIMI_LORA_MODEL_NAMES  # noqa: E402

ALL_TASKS: tuple[str, ...] = ("rad", "rsq", "ser_classify", "ser_intensity")
TASK_CHOICES: tuple[str, ...] = (*ALL_TASKS, "ser_classify_text")
SIMPLE_AUDIOQA_MODEL = "Kimi-Audio-LoRA-Simple-AudioQA"
ADAPTER_DIAGONAL: dict[str, str] = {
    "Kimi-Audio-LoRA-RAD": "rad",
    "Kimi-Audio-LoRA-RSQ": "rsq",
    "Kimi-Audio-LoRA-SER-Classify": "ser_classify",
    "Kimi-Audio-LoRA-SER-Intensity": "ser_intensity",
}
REQUESTED_BENCHMARK_JOBS: tuple[tuple[str, str, str], ...] = (
    ("Kimi-Audio-LoRA-RAD", "rad", "eval"),
    ("Kimi-Audio-LoRA-RSQ", "rsq", "eval"),
    ("Kimi-Audio-LoRA-SER-Classify", "ser_classify", "eval"),
    ("Kimi-Audio-LoRA-SER-Intensity", "ser_intensity", "eval"),
    ("Kimi-Audio-LoRA-RAD", "rsq", "all"),
    ("Kimi-Audio-LoRA-RAD", "ser_classify", "all"),
    ("Kimi-Audio-LoRA-RAD", "ser_intensity", "all"),
    (SIMPLE_AUDIOQA_MODEL, "rsq", "all"),
    (SIMPLE_AUDIOQA_MODEL, "ser_classify", "all"),
    (SIMPLE_AUDIOQA_MODEL, "ser_intensity", "all"),
    ("Kimi-Audio-LoRA-SER-Classify", "ser_classify_text", "eval"),
)


@dataclass(frozen=True)
class PairSpec:
    model: str
    task: str
    split: str


@dataclass
class JobState:
    index: int
    spec: PairSpec
    command: list[str]
    log_path: Path
    question_concurrency: int
    started_at: float | None = None
    finished_at: float | None = None
    process: subprocess.Popen | None = None
    log_handle: object | None = None
    status: str = "queued"
    returncode: int | None = None
    run_dirs: list[Path] = field(default_factory=list)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Kimi-Audio LoRA adapters and summarize accuracies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--suite", choices=["requested-benchmark"], default=None,
                   help="Named evaluation suite to run.")
    p.add_argument("--models", nargs="+", default=list(KIMI_LORA_MODEL_NAMES),
                   help="LoRA model names to evaluate in matrix mode.")
    p.add_argument("--tasks", nargs="+", default=list(ALL_TASKS),
                   choices=list(TASK_CHOICES),
                   help="Task slices to evaluate against in matrix mode.")
    p.add_argument("--split", default="eval", choices=["train", "eval", "all"],
                   help="Row-level split for matrix mode (default: eval).")
    p.add_argument("--backend", default="modal", choices=["docker", "modal"],
                   help="Adapter transport (default: modal).")
    p.add_argument("--diagonal-only", action="store_true",
                   help="Evaluate each adapter only on its matched task.")
    p.add_argument("--archive", default="archive")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--batch-name", default=None,
                   help="Batch directory name under --output-dir.")
    p.add_argument("--job-concurrency", type=int, default=None,
                   help="Number of model/task/split jobs to run in parallel.")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Per-question inference concurrency passed to run.py.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--runs", type=int, default=1,
                   help="Number of repetitions per pair job.")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit per run to N questions (useful for smoke).")
    p.add_argument("--position-swap", action="store_true",
                   help="Run the position-swap protocol on 2-clip tasks.")
    p.add_argument("--poll-interval", type=float, default=30.0,
                   help="Seconds between progress table refreshes.")
    p.add_argument("--summary-name", default=None,
                   help="Summary JSON filename (default: summary.json in batch dir).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned jobs and commands without running them.")
    p.add_argument("--promote-reference-trajectories", action="store_true",
                   help="Copy successful run dirs into reference_trajectories/.")
    p.add_argument("--write-reference-csv", action="store_true",
                   help="Write results/reference/kimi_lora_requested_benchmark.csv.")
    p.add_argument("--reference-trajectories-dir", default="reference_trajectories")
    p.add_argument("--reference-results-dir", default="results/reference")
    p.add_argument("--reference-csv-name", default="kimi_lora_requested_benchmark.csv")
    p.add_argument("--overwrite-reference", action="store_true",
                   help="Allow replacing existing promoted references/CSV.")
    return p.parse_args(argv)


def _build_specs(args: argparse.Namespace) -> list[PairSpec]:
    if args.suite == "requested-benchmark":
        return [PairSpec(*job) for job in REQUESTED_BENCHMARK_JOBS]

    specs: list[PairSpec] = []
    for model in args.models:
        if model not in KIMI_LORA_MODEL_NAMES:
            raise SystemExit(
                f"Unknown LoRA model {model!r}. Choose from: "
                f"{list(KIMI_LORA_MODEL_NAMES)}"
            )
        if args.diagonal_only:
            task = ADAPTER_DIAGONAL.get(model)
            if task and task in args.tasks:
                specs.append(PairSpec(model, task, args.split))
            continue
        for task in args.tasks:
            specs.append(PairSpec(model, task, args.split))
    return specs


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def _split_tag(split: str) -> str:
    return "" if split == "all" else f"-{split}"


def _run_pattern(spec: PairSpec, seed: int, backend: str) -> str:
    return f"{spec.model}-{spec.task}{_split_tag(spec.split)}-{backend}-seed{seed}-*"


def _question_concurrency_for_spec(args: argparse.Namespace, spec: PairSpec) -> int:
    if (
        args.suite == "requested-benchmark"
        and spec.split == "all"
        and spec.task in {"rsq", "ser_classify"}
    ):
        return 3
    return int(args.concurrency)


def _line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _find_run_dir(
    output_root: Path,
    spec: PairSpec,
    seed: int,
    backend: str,
    started_at: float | None,
) -> Path | None:
    candidates = sorted(
        output_root.glob(_run_pattern(spec, seed, backend)),
        key=lambda p: p.stat().st_mtime,
    )
    if started_at is not None:
        candidates = [p for p in candidates if p.stat().st_mtime >= started_at - 5]
    return candidates[-1] if candidates else None


def _job_commands(args: argparse.Namespace, specs: list[PairSpec], batch_root: Path) -> list[JobState]:
    logs_dir = batch_root / "logs"
    jobs: list[JobState] = []
    for index, spec in enumerate(specs, start=1):
        question_concurrency = _question_concurrency_for_spec(args, spec)
        log_name = (
            f"{index:02d}_{_safe_name(spec.model)}_"
            f"{spec.task}_{spec.split}.log"
        )
        command = [
            sys.executable,
            "run.py",
            "--model", spec.model,
            "--task", spec.task,
            "--split", spec.split,
            "--backend", args.backend,
            "--archive", args.archive,
            "--output-dir", args.output_dir,
            "--concurrency", str(question_concurrency),
            "--seed", str(args.seed),
            "--runs", str(args.runs),
        ]
        if args.max_samples is not None:
            command += ["--max-samples", str(args.max_samples)]
        if args.position_swap:
            command += ["--position-swap"]
        jobs.append(JobState(index, spec, command, logs_dir / log_name, question_concurrency))
    return jobs


def _job_progress(job: JobState, args: argparse.Namespace, output_root: Path) -> dict:
    completed_runs = 0
    current_lines = 0
    current_total = None
    latest_accuracy = None
    latest_run_dir = None
    run_dirs: list[Path] = []

    for offset in range(args.runs):
        seed = args.seed + offset
        run_dir = _find_run_dir(output_root, job.spec, seed, args.backend, job.started_at)
        if run_dir is None:
            continue
        run_dirs.append(run_dir)
        metrics = _read_json(run_dir / "metrics.json")
        if metrics:
            completed_runs += 1
            latest_accuracy = metrics.get("accuracy")
            latest_run_dir = run_dir
            continue

        config = _read_json(run_dir / "config.json")
        current_total = config.get("n_questions")
        latest_run_dir = run_dir
        response_count = _line_count(run_dir / "responses.jsonl")
        checkpoint_count = max(
            (_line_count(path) for path in run_dir.glob(".checkpoint*.jsonl")),
            default=0,
        )
        current_lines = max(response_count, checkpoint_count)

    job.run_dirs = run_dirs
    current_run = min(completed_runs + 1, args.runs) if job.status == "running" else completed_runs
    return {
        "completed_runs": completed_runs,
        "current_run": current_run,
        "current_lines": current_lines,
        "current_total": current_total,
        "latest_accuracy": latest_accuracy,
        "latest_run_dir": latest_run_dir,
    }


def _write_live_status(batch_root: Path, rows: list[dict]) -> None:
    status_json = batch_root / "status.json"
    status_csv = batch_root / "status.csv"
    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "jobs": rows,
    }
    status_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = [
        "index", "status", "model", "task", "split", "completed_runs",
        "current_run", "runs", "question_concurrency", "completed_questions",
        "total_questions", "latest_accuracy", "elapsed_s", "log_path",
        "latest_run_dir",
    ]
    with status_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _print_progress(
    jobs: list[JobState],
    args: argparse.Namespace,
    output_root: Path,
    batch_root: Path,
) -> None:
    print(f"\n=== Progress {time.strftime('%Y-%m-%d %H:%M:%S')} ===", flush=True)
    print("idx  status   runs  qconc  questions  acc    elapsed  job", flush=True)
    status_rows: list[dict] = []
    for job in jobs:
        progress = _job_progress(job, args, output_root)
        started_at = job.started_at or time.time()
        finished_at = job.finished_at or time.time()
        elapsed = max(0, int(finished_at - started_at)) if job.started_at else 0
        questions = "-"
        if progress["current_total"]:
            questions = f"{progress['current_lines']}/{progress['current_total']}"
        acc = "-"
        if progress["latest_accuracy"] is not None:
            acc = f"{progress['latest_accuracy']:.1f}"
        runs = f"{progress['completed_runs']}/{args.runs}"
        if job.status == "running":
            runs = f"{progress['current_run']}/{args.runs}"
        label = f"{job.spec.model} / {job.spec.task} / {job.spec.split}"
        latest_run_dir = progress["latest_run_dir"]
        status_rows.append({
            "index": job.index,
            "status": job.status,
            "model": job.spec.model,
            "task": job.spec.task,
            "split": job.spec.split,
            "completed_runs": progress["completed_runs"],
            "current_run": progress["current_run"],
            "runs": args.runs,
            "question_concurrency": job.question_concurrency,
            "completed_questions": progress["current_lines"],
            "total_questions": progress["current_total"],
            "latest_accuracy": progress["latest_accuracy"],
            "elapsed_s": elapsed,
            "log_path": str(job.log_path),
            "latest_run_dir": str(latest_run_dir) if latest_run_dir else None,
        })
        print(
            f"{job.index:02d}   {job.status:<7}  {runs:<5} {job.question_concurrency:<5} "
            f"{questions:<10} "
            f"{acc:<6} {elapsed:>6}s  {label}",
            flush=True,
        )
    _write_live_status(batch_root, status_rows)
    print(f"Live status: {batch_root / 'status.csv'} and {batch_root / 'status.json'}", flush=True)


def _run_jobs(
    jobs: list[JobState],
    args: argparse.Namespace,
    output_root: Path,
    batch_root: Path,
) -> None:
    max_parallel = max(1, int(args.job_concurrency or 1))
    pending = list(jobs)
    running: list[JobState] = []
    last_progress = 0.0

    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            job.log_handle = job.log_path.open("w", encoding="utf-8")
            job.log_handle.write("$ " + " ".join(job.command) + "\n\n")
            job.log_handle.flush()
            job.started_at = time.time()
            job.status = "running"
            job.process = subprocess.Popen(
                job.command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=job.log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=os.environ.copy(),
            )
            running.append(job)

        now = time.time()
        for job in list(running):
            assert job.process is not None
            returncode = job.process.poll()
            if returncode is None:
                continue
            job.returncode = returncode
            job.finished_at = time.time()
            job.status = "done" if returncode == 0 else "failed"
            if job.log_handle is not None:
                job.log_handle.close()
                job.log_handle = None
            running.remove(job)

        if now - last_progress >= args.poll_interval or (not pending and not running):
            _print_progress(jobs, args, output_root, batch_root)
            last_progress = now

        if pending or running:
            time.sleep(min(5.0, max(1.0, args.poll_interval)))


def _collect_rows(jobs: list[JobState], args: argparse.Namespace, output_root: Path) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    failed: list[dict] = []
    for job in jobs:
        if job.returncode not in (None, 0):
            failed.append({
                "model": job.spec.model,
                "task": job.spec.task,
                "split": job.spec.split,
                "returncode": job.returncode,
                "log_path": str(job.log_path),
            })

        for offset in range(args.runs):
            seed = args.seed + offset
            run_dir = _find_run_dir(output_root, job.spec, seed, args.backend, job.started_at)
            metrics = _read_json(run_dir / "metrics.json") if run_dir else {}
            row = {
                "model": job.spec.model,
                "task": job.spec.task,
                "split": job.spec.split,
                "backend": args.backend,
                "seed": seed,
                "run_index": offset,
                "status": "completed" if metrics else "missing_metrics",
                "question_concurrency": job.question_concurrency,
                "n": metrics.get("n"),
                "accuracy": metrics.get("accuracy"),
                "consistent_accuracy": metrics.get("consistent_accuracy"),
                "run_dir": str(run_dir.relative_to(output_root)) if run_dir else None,
                "log_path": str(job.log_path),
            }
            rows.append(row)
            if not metrics:
                failed.append({
                    "model": job.spec.model,
                    "task": job.spec.task,
                    "split": job.spec.split,
                    "seed": seed,
                    "error": "missing metrics.json",
                    "run_dir": row["run_dir"],
                    "log_path": str(job.log_path),
                })
    return rows, failed


def _group_stats(rows: list[dict]) -> dict[tuple[str, str, str, str], dict]:
    groups: dict[tuple[str, str, str, str], list[dict]] = {}
    for row in rows:
        if row.get("status") != "completed":
            continue
        key = (row["model"], row["task"], row["split"], row["backend"])
        groups.setdefault(key, []).append(row)

    stats: dict[tuple[str, str, str, str], dict] = {}
    for key, group_rows in groups.items():
        values = [float(row["accuracy"]) for row in group_rows if row.get("accuracy") is not None]
        ca_values = [
            float(row["consistent_accuracy"])
            for row in group_rows
            if row.get("consistent_accuracy") is not None
        ]
        stats[key] = {
            "group_count": len(group_rows),
            "accuracy_mean": statistics.mean(values) if values else None,
            "accuracy_std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
            "consistent_accuracy_mean": statistics.mean(ca_values) if ca_values else None,
            "consistent_accuracy_std": statistics.stdev(ca_values) if len(ca_values) > 1 else 0.0 if ca_values else None,
        }
    return stats


def _write_csv(path: Path, rows: list[dict], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "task", "split", "backend", "seed", "run_index", "status",
        "question_concurrency", "n", "accuracy", "consistent_accuracy", "run_dir",
        "reference_trajectory_dir", "group_count", "accuracy_mean",
        "accuracy_std", "consistent_accuracy_mean", "consistent_accuracy_std",
        "log_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _promote_references(rows: list[dict], args: argparse.Namespace, output_root: Path) -> list[dict]:
    reference_root = Path(args.reference_trajectories_dir)
    promoted: list[dict] = []
    for row in rows:
        if row.get("status") != "completed" or not row.get("run_dir"):
            continue
        source = output_root / row["run_dir"]
        if not (
            (source / "config.json").is_file()
            and (source / "metrics.json").is_file()
            and _line_count(source / "responses.jsonl") > 0
        ):
            row["reference_trajectory_dir"] = None
            row["reference_status"] = "skipped_incomplete"
            continue

        dest = reference_root / row["model"] / source.name
        if dest.exists():
            if not args.overwrite_reference:
                row["reference_trajectory_dir"] = str(dest)
                row["reference_status"] = "exists"
                promoted.append(row)
                continue
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, dest)
        row["reference_trajectory_dir"] = str(dest)
        row["reference_status"] = "promoted"
        promoted.append(row)
    return promoted


def _augment_group_stats(rows: list[dict]) -> None:
    stats = _group_stats(rows)
    for row in rows:
        group = stats.get((row["model"], row["task"], row["split"], row["backend"]), {})
        row.update(group)


def _format_table(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    cols = ["model", "task", "split", "seed", "n", "accuracy", "run_dir"]
    headers = {
        "model": "Model",
        "task": "Task",
        "split": "Split",
        "seed": "Seed",
        "n": "N",
        "accuracy": "Acc%",
        "run_dir": "Run dir",
    }

    def cell(row: dict, col: str) -> str:
        value = row.get(col)
        if value is None:
            return "-"
        if col == "accuracy" and isinstance(value, int | float) and not math.isnan(float(value)):
            return f"{value:.1f}"
        return str(value)

    widths = {col: max(len(headers[col]), *(len(cell(row, col)) for row in rows)) for col in cols}
    lines = ["  ".join(headers[col].ljust(widths[col]) for col in cols)]
    lines.append("  ".join("-" * widths[col] for col in cols))
    for row in rows:
        lines.append("  ".join(cell(row, col).ljust(widths[col]) for col in cols))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.suite == "requested-benchmark":
        args.runs = 3 if args.runs == 1 else args.runs
        args.job_concurrency = (
            len(REQUESTED_BENCHMARK_JOBS)
            if args.job_concurrency is None
            else args.job_concurrency
        )
        args.concurrency = int(args.concurrency)
    else:
        args.job_concurrency = 1 if args.job_concurrency is None else args.job_concurrency

    if args.promote_reference_trajectories:
        args.write_reference_csv = True

    specs = _build_specs(args)
    if not specs:
        print("No model/task/split jobs to run; check --models/--tasks/--diagonal-only.")
        return 1

    output_root = Path(args.output_dir)
    batch_name = args.batch_name or f"kimi_lora_batch_{int(time.time())}"
    batch_root = output_root / batch_name
    summary_path = batch_root / (args.summary_name or "summary.json")
    summary_csv_path = batch_root / "summary.csv"
    reference_csv_path = Path(args.reference_results_dir) / args.reference_csv_name

    jobs = _job_commands(args, specs, batch_root)
    print(
        f"Prepared {len(jobs)} Kimi LoRA jobs "
        f"(job_concurrency={args.job_concurrency}, runs={args.runs}, "
        f"default_per_question_concurrency={args.concurrency})."
    )
    for job in jobs:
        print(
            f"  {job.index:02d}. {job.spec.model} / {job.spec.task} / "
            f"{job.spec.split} (q_concurrency={job.question_concurrency})"
        )
        print(f"      log: {job.log_path}")
        print(f"      cmd: {' '.join(job.command)}")

    if args.dry_run:
        return 0

    batch_root.mkdir(parents=True, exist_ok=True)
    _run_jobs(jobs, args, output_root, batch_root)
    rows, failed = _collect_rows(jobs, args, output_root)
    _augment_group_stats(rows)

    promoted_rows: list[dict] = []
    if args.promote_reference_trajectories:
        promoted_rows = _promote_references(rows, args, output_root)

    _augment_group_stats(rows)
    _write_csv(summary_csv_path, rows, overwrite=True)
    if args.write_reference_csv:
        _write_csv(reference_csv_path, rows, overwrite=args.overwrite_reference)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "suite": args.suite,
                "backend": args.backend,
                "seed": args.seed,
                "runs": args.runs,
                "job_concurrency": args.job_concurrency,
                "default_per_question_concurrency": args.concurrency,
                "batch_root": str(batch_root),
                "summary_csv": str(summary_csv_path),
                "reference_csv": str(reference_csv_path) if args.write_reference_csv else None,
                "rows": rows,
                "failed": failed,
                "promoted": promoted_rows,
            },
            handle,
            indent=2,
        )

    print("\n=== Kimi-Audio LoRA results ===")
    print(_format_table(rows))
    print(f"\nSummary written to {summary_path}")
    print(f"Summary CSV written to {summary_csv_path}")
    if args.write_reference_csv:
        print(f"Reference CSV written to {reference_csv_path}")
    if args.promote_reference_trajectories:
        print(f"Promoted/existing reference trajectories: {len(promoted_rows)}")
    if failed:
        print("\nFailures:")
        for item in failed[:20]:
            print(f"  {item}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
