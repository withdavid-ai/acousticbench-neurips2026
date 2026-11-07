"""CLI entry point for the AcousticBench inference harness."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from loguru import logger


def _load_dotenv_if_present() -> None:
    """Auto-load ``.env`` from the repo root if python-dotenv is installed.

    This lets ``MODAL_ENVIRONMENT``, ``GEMINI_API_KEY``, ``HF_TOKEN``, and the
    other values in the user's ``.env`` flow into ``os.environ`` without an
    explicit ``set -a; source .env; set +a`` step. We never overwrite values
    that are already set in the parent environment.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if env_path.is_file():
        load_dotenv(dotenv_path=env_path, override=False)


_load_dotenv_if_present()

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = REPO_ROOT / "reference_trajectories"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AcousticBench inference harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Docker backend (default — no Modal account needed)
  python run.py --model Qwen3-Omni --task rsq --backend docker

  # Modal backend
  python run.py --model Qwen3-Omni --task rsq --backend modal

  # Parity test: run both backends and compare results
  python run.py --model Qwen3-Omni --task rsq --max-samples 20 --parity

  # Other examples
  python run.py --model Gemini-3.1-Pro --task rsq --position-swap
  python run.py --model Gemini-3.1-Pro --task all --concurrency 4
  python run.py --model Kimi-Audio-LoRA-RAD --task rad --split eval
  python run.py --model Qwen3-Omni --task all --runs 3 --seed 42 --compare-reference
  python run.py --list-models
  python run.py --verify-checksums --task all
""",
    )
    p.add_argument("--model", type=str, help="Model name (e.g. Gemini-3.1-Pro, Qwen3-Omni)")
    p.add_argument("--task", type=str, default="all",
                   choices=[
                       "rad",
                       "rsq",
                       "ser_classify",
                       "ser_classify_text",
                       "ser_intensity",
                       "all",
                   ],
                   help="Task slice to evaluate (default: all)")
    p.add_argument("--split", type=str, default="all",
                   choices=["train", "eval", "all"],
                   help="Row-level split to keep within the task slice (default: all)")
    p.add_argument("--archive", type=str, default="archive",
                   help="Path to the archive directory (default: ./archive)")
    p.add_argument("--output-dir", type=str, default="results",
                   help="Output directory for results (default: ./results)")
    p.add_argument("--backend", type=str, default="docker", choices=["docker", "modal"],
                   help="Backend for open-weight models (default: docker)")
    p.add_argument("--parity", action="store_true",
                   help="Run both docker and modal backends, compare results")
    p.add_argument("--modal-environment", type=str, default=None,
                   help="Modal environment that holds the deployed apps (sets "
                        "MODAL_ENVIRONMENT; default: that variable, else Modal's "
                        "default environment for your profile)")

    p.add_argument("--position-swap", action="store_true",
                   help="Run the position-swap protocol on the 2-clip questions "
                        "(rad, rsq, ser_intensity); single-clip tasks are rejected")
    p.add_argument("--runs", type=int, default=1,
                   help="Number of inference runs for error bars (default: 1)")
    p.add_argument("--seed", type=int, default=42,
                   help="Base seed; run i (0-based) uses seed+i. It is passed to the "
                        "model adapter, which forwards it to APIs that accept a seed "
                        "(decoding stays greedy) (default: 42)")
    p.add_argument("--extractor", type=str, default="llm",
                   choices=["llm", "deterministic-then-llm"],
                   help="Choice extractor: 'llm' sends every response to the judge "
                        "(paper protocol, default); 'deterministic-then-llm' tries the "
                        "regex extractor first and calls the judge only when it fails")

    p.add_argument("--concurrency", type=int, default=1,
                   help="Parallel inference threads (default: 1)")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Limit to N questions (for smoke testing)")
    p.add_argument("--continue-on-error", action="store_true", default=True,
                   help="Continue on inference errors (default: True)")
    p.add_argument("--checkpoint-every", type=int, default=20,
                   help="Save checkpoint every N items (default: 20)")

    p.add_argument("--compare-reference", action="store_true",
                   help="After running, compare results against reference_trajectories/ and print match rate")

    p.add_argument("--verify-checksums", action="store_true",
                   help="Verify audio SHA256 checksums, row counts, and "
                        "checksums/sha256sum.txt (if present) for --task, then exit")
    p.add_argument("--allow-missing-hashes", action="store_true",
                   help="With --verify-checksums, accept audio parts that carry no sha256")
    p.add_argument("--list-models", action="store_true",
                   help="List available models and exit")
    p.add_argument("--list-tasks", action="store_true",
                   help="List available tasks and exit")

    return p


def _run_single_backend(
    model_name: str,
    backend: str,
    questions: list[dict],
    args,
    run_dir: Path,
    tag_suffix: str = "",
) -> list[dict]:
    from .models import build_model_with_backend, set_backend
    from .runner import run_inference, save_results

    set_backend(backend)
    logger.info(f"Building model {model_name} with backend={backend} (seed={args.seed})")
    model = build_model_with_backend(model_name, backend, seed=args.seed)

    suffix = f"_{backend}{tag_suffix}"
    checkpoint = run_dir / f".checkpoint{suffix}.jsonl"
    results = run_inference(
        model, questions,
        concurrency=args.concurrency,
        checkpoint_path=checkpoint,
        checkpoint_every=args.checkpoint_every,
        continue_on_error=args.continue_on_error,
        extractor=args.extractor,
    )
    save_results(results, run_dir / f"responses{suffix}.jsonl")

    for ckpt in run_dir.glob(f".checkpoint{suffix}*.jsonl"):
        ckpt.unlink()

    return results


def _compare_results(
    docker_results: list[dict],
    modal_results: list[dict],
) -> dict:

    docker_by_id = {r["example_id"]: r for r in docker_results}
    modal_by_id = {r["example_id"]: r for r in modal_results}

    all_ids = sorted(set(docker_by_id) | set(modal_by_id))
    matches = 0
    mismatches = []
    docker_errors = 0
    modal_errors = 0
    total = 0

    for eid in all_ids:
        d = docker_by_id.get(eid)
        m = modal_by_id.get(eid)
        if d is None or m is None:
            continue
        total += 1

        d_status = d.get("response_status", "ok")
        m_status = m.get("response_status", "ok")
        if d_status == "error":
            docker_errors += 1
        if m_status == "error":
            modal_errors += 1
        if d_status == "error" or m_status == "error":
            continue

        d_choice = d.get("chosen")
        m_choice = m.get("chosen")

        if d_choice == m_choice:
            matches += 1
        else:
            mismatches.append({
                "example_id": eid,
                "docker_choice": d_choice,
                "modal_choice": m_choice,
                "docker_response": str(d.get("response_text", ""))[:200],
                "modal_response": str(m.get("response_text", ""))[:200],
            })

    compared = total - docker_errors - modal_errors
    match_rate = (matches / compared * 100) if compared > 0 else None

    docker_times = [r.get("wall_time_s", 0) for r in docker_results if r.get("wall_time_s")]
    modal_times = [r.get("wall_time_s", 0) for r in modal_results if r.get("wall_time_s")]

    return {
        "total_questions": total,
        "compared": compared,
        "matches": matches,
        "mismatches_count": len(mismatches),
        "match_rate_pct": round(match_rate, 1) if match_rate is not None else None,
        "docker_errors": docker_errors,
        "modal_errors": modal_errors,
        "mismatches": mismatches,
        "timing": {
            "docker_mean_s": round(sum(docker_times) / len(docker_times), 2) if docker_times else None,
            "modal_mean_s": round(sum(modal_times) / len(modal_times), 2) if modal_times else None,
        },
    }


def _run_parity(args, questions: list[dict], run_dir: Path):
    from .metrics import raw_accuracy

    logger.info("=== Parity test: running Docker backend ===")
    docker_results = _run_single_backend(
        args.model, "docker", questions, args, run_dir,
    )
    docker_metrics = raw_accuracy(docker_results)
    logger.info(f"Docker accuracy: {docker_metrics['accuracy']}% ({docker_metrics['correct']}/{docker_metrics['n']})")

    logger.info("=== Parity test: running Modal backend ===")
    modal_results = _run_single_backend(
        args.model, "modal", questions, args, run_dir,
    )
    modal_metrics = raw_accuracy(modal_results)
    logger.info(f"Modal accuracy: {modal_metrics['accuracy']}% ({modal_metrics['correct']}/{modal_metrics['n']})")

    comparison = _compare_results(docker_results, modal_results)
    comparison["docker_accuracy"] = docker_metrics
    comparison["modal_accuracy"] = modal_metrics

    with open(run_dir / "parity.json", "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info("=== Parity Results ===")
    logger.info(f"  Match rate: {comparison['match_rate_pct']}% ({comparison['matches']}/{comparison['compared']})")
    if comparison["mismatches"]:
        logger.info(f"  Mismatches: {comparison['mismatches_count']}")
        for mm in comparison["mismatches"][:5]:
            logger.info(f"    {mm['example_id']}: docker={mm['docker_choice']} modal={mm['modal_choice']}")
    logger.info(f"  Docker mean time: {comparison['timing']['docker_mean_s']}s")
    logger.info(f"  Modal mean time:  {comparison['timing']['modal_mean_s']}s")
    logger.info(f"  Saved to {run_dir / 'parity.json'}")


def _reference_seed_map(ref_dir: Path) -> dict[int, Path] | None:
    """Map seed -> reference run file when the reference records its seeds.

    Looks for a ``seeds`` list (run_1, run_2, ...) or ``per_run.<run>.seed``
    entries in ``summary.json`` / ``config.json``. Returns None otherwise.
    """
    for name in ("summary.json", "config.json"):
        path = ref_dir / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        seeds = data.get("seeds")
        if isinstance(seeds, list) and seeds:
            return {int(seed): ref_dir / f"run_{i}.jsonl" for i, seed in enumerate(seeds, start=1)}
        per_run = data.get("per_run")
        if isinstance(per_run, dict):
            mapping = {
                int(info["seed"]): ref_dir / f"{run_name}.jsonl"
                for run_name, info in per_run.items()
                if isinstance(info, dict) and info.get("seed") is not None
            }
            if mapping:
                return mapping
    return None


def _resolve_reference_run(ref_dir: Path, run_seed: int, run_index: int) -> tuple[Path | None, str]:
    """Pick the reference run to compare against and describe the mapping used."""
    seed_map = _reference_seed_map(ref_dir)
    if seed_map is not None:
        path = seed_map.get(run_seed)
        if path is None:
            return None, f"by seed: seed {run_seed} is not among the reference seeds {sorted(seed_map)}"
        return path, f"by seed ({run_seed} -> {path.name})"
    path = ref_dir / f"run_{run_index + 1}.jsonl"
    return path, f"by run index (run {run_index + 1} -> {path.name}; the reference records no seeds)"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _compare_to_reference(
    model_name: str,
    task: str,
    questions: list[dict],
    results: list[dict],
    run_seed: int,
    run_index: int,
) -> dict:
    """Compare ``chosen`` against reference_trajectories/<model>/<sub-task>/.

    References are stored per sub-task, so ``--task all`` is split back into
    its slices. The reference root is resolved from the repository, not the
    current working directory.
    """
    subset_by_id = {q["meta"]["example_id"]: q["subset"] for q in questions}
    groups: dict[str, list[dict]] = {}
    for r in results:
        subtask = task if task != "all" else subset_by_id.get(r["example_id"], "unknown")
        groups.setdefault(subtask, []).append(r)

    summary: dict[str, dict] = {}
    total_matches = 0
    total_compared = 0
    for subtask in sorted(groups):
        rows = groups[subtask]
        ref_dir = REFERENCE_ROOT / model_name / subtask
        ref_path, mapping = _resolve_reference_run(ref_dir, run_seed, run_index)
        entry: dict = {"mapping": mapping, "n_run": len(rows)}
        if ref_path is None or not ref_path.exists():
            shown = _display_path(ref_path or ref_dir)
            logger.warning(f"Reference not found for {subtask}: {shown} (mapped {mapping})")
            entry.update({"reference": shown, "found": False})
            summary[subtask] = entry
            continue

        ref_by_id = {}
        with ref_path.open() as f:
            for line in f:
                if line.strip():
                    ref_row = json.loads(line)
                    ref_by_id[ref_row["example_id"]] = ref_row
        # Only compare questions that exist in both (handles split mismatches)
        comparable = [r for r in rows if r["example_id"] in ref_by_id]
        entry.update({"reference": _display_path(ref_path), "found": True, "compared": len(comparable)})
        if not comparable:
            logger.warning(
                f"Reference comparison [{subtask}]: 0 overlapping IDs between your run "
                f"({len(rows)} questions) and {_display_path(ref_path)} ({len(ref_by_id)} questions). "
                f"Check --split flag; the reference may use a different split."
            )
        else:
            matches = sum(1 for r in comparable if ref_by_id[r["example_id"]].get("chosen") == r.get("chosen"))
            pct = round(matches / len(comparable) * 100, 1)
            entry.update({"matches": matches, "match_rate_pct": pct})
            total_matches += matches
            total_compared += len(comparable)
            logger.info(
                f"Reference comparison [{subtask}]: {matches}/{len(comparable)} ({pct}%) match "
                f"{_display_path(ref_path)} (mapped {mapping})"
            )
            if len(comparable) < len(rows):
                skipped = len(rows) - len(comparable)
                logger.info(f"  ({skipped} questions skipped: not in reference, likely different --split)")
        summary[subtask] = entry

    if len(groups) > 1 and total_compared:
        pct = round(total_matches / total_compared * 100, 1)
        logger.info(f"Reference comparison [overall]: {total_matches}/{total_compared} ({pct}%)")
    return summary


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)

    if args.list_tasks:
        from .loader import EXPECTED_ROW_COUNTS as n

        print("Available tasks:")
        print(f"  rad            - Relative Acoustic Discrimination ({n['rad']} questions, 2-clip binary)")
        print(f"  rsq            - Relative Speech Quality ({n['rsq']} questions, 2-clip binary)")
        print(f"  ser_classify   - Speech Emotion Recognition: Classification ({n['ser_classify']} questions, single-clip 2-way)")
        print(f"  ser_classify_text - SER Classification from transcript text only ({n['ser_classify_text']} questions)")
        print(f"  ser_intensity  - Speech Emotion Recognition: Intensity ({n['ser_intensity']} questions, 2-clip binary)")
        print(f"  all            - All five task slices combined ({n['all']} questions)")
        print("")
        print("Splits (per task slice, applied via --split):")
        print("  train          - Training half (used to fit Kimi-Audio LoRA adapters)")
        print("  eval           - Held-out half for unbiased evaluation")
        print("  all            - Both halves combined (default)")
        return

    from .loader import load_questions

    if args.verify_checksums:
        from .loader import CHECKSUM_MANIFEST, verify_archive

        logger.info(f"Verifying checksums for task={args.task}...")
        try:
            report = verify_archive(
                args.task, args.archive, allow_missing_hashes=args.allow_missing_hashes,
            )
        except (FileNotFoundError, ValueError) as e:
            logger.error(str(e))
            sys.exit(1)
        logger.info(f"All {report['n_questions']} questions passed checksum verification.")
        if report["manifest"]:
            logger.info(
                f"{report['manifest_files_checked']} files matched {report['manifest']}."
            )
        else:
            logger.warning(
                f"{CHECKSUM_MANIFEST} not found under {args.archive}; only per-row audio "
                f"hashes and row counts were checked."
            )
        return

    if args.list_models:
        from .models import list_models, set_backend
        set_backend(args.backend)
        print(f"Available models (backend={args.backend}):")
        for name, info in list_models().items():
            backends = ", ".join(info["backends"])
            print(f"  {name:30s} active={info['type']:8s} available=[{backends}]")
        return

    if not args.model:
        print("Error: --model is required (use --list-models to see options)")
        sys.exit(1)

    from .models import get_model_metadata, set_backend

    metadata = get_model_metadata(args.model)
    available_backends = metadata.get("backends") or []
    if available_backends and args.backend not in available_backends and not args.parity:
        new_backend = available_backends[0]
        logger.info(
            f"Model {args.model!r} has no {args.backend!r} adapter; "
            f"switching to backend={new_backend!r}."
        )
        args.backend = new_backend

    if args.modal_environment:
        os.environ["MODAL_ENVIRONMENT"] = args.modal_environment
    if args.parity or args.backend == "modal":
        modal_env = os.environ.get("MODAL_ENVIRONMENT")
        if modal_env:
            logger.info(f"Modal environment: {modal_env}")
        else:
            logger.info(
                "Modal environment: Modal's default for the active profile "
                "(set MODAL_ENVIRONMENT or --modal-environment to choose another)."
            )

    archive_root = Path(args.archive)
    output_root = Path(args.output_dir)

    logger.info(f"Loading questions: task={args.task} split={args.split}")
    questions = load_questions(args.task, archive_root, split=args.split)
    if args.max_samples:
        questions = questions[:args.max_samples]
    logger.info(f"Loaded {len(questions)} questions")

    if args.position_swap:
        from .position_swap import is_two_clip

        n_two_clip = sum(1 for q in questions if is_two_clip(q))
        if n_two_clip == 0:
            logger.error(
                f"--position-swap needs 2-clip questions, but task={args.task} has none "
                f"(ser_classify and ser_classify_text are single-clip or text-only)."
            )
            sys.exit(2)
        if n_two_clip < len(questions):
            logger.warning(
                f"--position-swap: {len(questions) - n_two_clip} single-clip or text-only "
                f"questions are not swapped and are excluded from Consistent Accuracy."
            )

    split_tag = "" if args.split == "all" else f"-{args.split}"

    if args.parity:
        run_tag = f"{args.model}-{args.task}{split_tag}-parity-{int(time.time())}"
        run_dir = output_root / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)
        _run_parity(args, questions, run_dir)
        return

    set_backend(args.backend)
    logger.info(f"Backend: {args.backend}")

    from .judge import DEFAULT_JUDGE_MODEL
    from .models import build_model
    from .runner import run_inference, save_results
    from .metrics import raw_accuracy, consistent_accuracy
    from .position_swap import generate_swapped_questions

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        run_tag = f"{args.model}-{args.task}{split_tag}-{args.backend}-seed{run_seed}-{int(time.time())}"
        run_dir = output_root / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        run_config = {
            "model": args.model,
            "task": args.task,
            "split": args.split,
            "backend": args.backend,
            "seed": run_seed,
            "run_index": run_idx,
            "total_runs": args.runs,
            "position_swap": args.position_swap,
            "extractor": args.extractor,
            "judge_model": DEFAULT_JUDGE_MODEL,
            "n_questions": len(questions),
            "concurrency": args.concurrency,
            "archive": str(archive_root),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(run_dir / "config.json", "w") as f:
            json.dump(run_config, f, indent=2)

        logger.info(f"Run {run_idx + 1}/{args.runs} (seed={run_seed}, tag={run_tag})")

        # Build per run so the run's seed reaches the adapter (BaseModel.seed).
        logger.info(f"Building model: {args.model} (seed={run_seed})")
        model = build_model(args.model, seed=run_seed)

        checkpoint = run_dir / ".checkpoint_original.jsonl"
        results = run_inference(
            model, questions,
            concurrency=args.concurrency,
            checkpoint_path=checkpoint,
            checkpoint_every=args.checkpoint_every,
            continue_on_error=args.continue_on_error,
            extractor=args.extractor,
        )
        save_results(results, run_dir / "responses.jsonl")

        metrics = raw_accuracy(results)
        logger.info(f"Raw accuracy: {metrics['accuracy']}% ({metrics['correct']}/{metrics['n']})")
        n_judge_errors = sum(1 for r in results if r.get("extraction_source") == "judge_error")
        if n_judge_errors:
            logger.warning(
                f"{n_judge_errors} responses could not be judged after retries and are "
                f"scored as unparsed; re-judge them with scripts/rejudge.py."
            )

        if args.position_swap:
            swapped_questions = generate_swapped_questions(questions)
            if args.max_samples:
                swapped_questions = swapped_questions[:args.max_samples]

            checkpoint_swap = run_dir / ".checkpoint_swap.jsonl"
            swap_results = run_inference(
                model, swapped_questions,
                concurrency=args.concurrency,
                checkpoint_path=checkpoint_swap,
                checkpoint_every=args.checkpoint_every,
                continue_on_error=args.continue_on_error,
                extractor=args.extractor,
            )
            save_results(swap_results, run_dir / "responses_swap.jsonl")

            ca_metrics = consistent_accuracy(results, swap_results, seed=run_seed)
            metrics.update(ca_metrics)
            logger.info(
                f"Consistent Accuracy: {ca_metrics['consistent_accuracy']}% "
                f"[{ca_metrics['ci_lo']}, {ca_metrics['ci_hi']}] "
                f"(inconsistency: {ca_metrics['inconsistency_rate']}%)"
            )

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results saved to {run_dir}")

        if args.compare_reference:
            comparison = _compare_to_reference(
                args.model, args.task, questions, results, run_seed, run_idx,
            )
            with open(run_dir / "reference_comparison.json", "w") as f:
                json.dump(comparison, f, indent=2)

        for ckpt in run_dir.glob(".checkpoint_*.jsonl"):
            ckpt.unlink()


if __name__ == "__main__":
    main()
