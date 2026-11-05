#!/usr/bin/env python3
"""Re-judge responses in existing run JSONL files through the LLM judge.

By default only rows without a parsed choice (``chosen`` is null, for example
after a judge outage) are re-judged; pass ``--all`` to re-judge every
non-empty response. The judge sees the real question text, looked up by
example_id in the archive (``--archive``), exactly as run.py sends it.

Results are written to ``--output-dir`` (default ``rejudged/``), mirroring the
``<model>/<task>/run_N.jsonl`` layout, together with a regenerated
``summary.json``. Pass ``--in-place`` to overwrite the input files instead.
The summary's ``std_accuracy`` is the sample standard deviation (ddof=1) of the
unrounded per-run accuracies, the convention the paper tables use.

Usage:
    # Re-judge unparsed rows of one file into rejudged/
    python scripts/rejudge.py results/<run_dir>/responses.jsonl

    # Re-judge every response of a model's reference runs, in place
    python scripts/rejudge.py --all --in-place reference_trajectories/GPT-4o/*/run_*.jsonl

    # Dry run (show what would change, don't write)
    python scripts/rejudge.py --dry-run reference_trajectories/GPT-4o/rad/run_1.jsonl

Requires: OPENAI_API_KEY exported in the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.judge import DEFAULT_JUDGE_MODEL, EXTRACTORS, extract_choice  # noqa: E402
from harness.loader import load_questions_by_id  # noqa: E402
from harness.runner import valid_labels_for_question  # noqa: E402


def needs_rejudge(row: dict, rejudge_all: bool) -> bool:
    response = row.get("response_text", "") or ""
    if not response.strip():
        return False
    return rejudge_all or row.get("chosen") is None


def rejudge_file(
    filepath: Path,
    questions_by_id: dict[str, dict],
    rejudge_all: bool = False,
    concurrency: int = 16,
    extractor: str = "llm",
) -> tuple[list[dict], dict]:
    results = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    targets = [i for i, r in enumerate(results) if needs_rejudge(r, rejudge_all)]
    changed = 0
    errors = 0
    missing_question = 0

    def process_one(idx: int) -> tuple[int, dict | None, str | None]:
        r = results[idx]
        example_id = r.get("example_id") or r.get("meta", {}).get("example_id", "")
        question = questions_by_id.get(example_id)
        if question is None:
            return idx, None, "missing_question"
        question_text = question["question"]
        try:
            extraction = extract_choice(
                r["response_text"],
                question_text,
                valid_labels_for_question(question_text),
                extractor=extractor,
            )
        except Exception as e:
            print(f"      Error on {example_id}: {e}")
            return idx, None, "error"
        return idx, extraction, None

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [executor.submit(process_one, i) for i in targets]
        done = 0
        for future in as_completed(futures):
            idx, extraction, problem = future.result()
            done += 1
            if problem == "missing_question":
                missing_question += 1
            elif problem == "error":
                errors += 1
            else:
                row = results[idx]
                new = extraction["label"]
                if (
                    new != row.get("chosen")
                    or row.get("extraction_source") != extraction["source"]
                ):
                    changed += 1
                row["chosen"] = new
                row["is_correct"] = new == row["correct"] if new else False
                row["extraction_source"] = extraction["source"]
                row["extraction_classification"] = extraction.get("classification")
                row["extraction_reason"] = extraction.get("reason", "")
                row.pop("judge_error", None)
            if done % 100 == 0:
                print(f"      {done}/{len(targets)}")

    stats = {
        "file": str(filepath),
        "total": len(results),
        "rejudged": len(targets) - missing_question - errors,
        "changed": changed,
        "errors": errors,
        "missing_question": missing_question,
    }
    return results, stats


def output_path_for(filepath: Path, output_dir: Path) -> Path:
    """Mirror <model>/<task>/<file> under output_dir."""
    parts = filepath.resolve().parts
    return output_dir.joinpath(*parts[-3:])


def write_results(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def regenerate_summary(task_dir: Path) -> dict | None:
    runs = {}
    for run_file in sorted(task_dir.glob("run_*.jsonl")):
        with run_file.open() as f:
            results = [json.loads(line) for line in f if line.strip()]
        n = len(results)
        if n == 0:
            continue
        correct = sum(1 for r in results if r.get("is_correct"))
        runs[run_file.stem] = {
            "accuracy": correct / n * 100,
            "correct": correct,
            "n": n,
            "unparseable": sum(1 for r in results if r.get("chosen") is None),
            "errors": sum(1 for r in results if r.get("response_status") == "error"),
        }

    if not runs:
        return None

    accs = np.array([r["accuracy"] for r in runs.values()], dtype=float)
    mean = float(np.mean(accs))
    # Sample SD (ddof=1) of unrounded per-run accuracies, as in the paper tables.
    std = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0
    half_width = 1.96 * std / np.sqrt(len(accs)) if len(accs) > 1 else 0.0
    summary = {
        "model": task_dir.parent.name,
        "task": task_dir.name,
        "n_runs": len(runs),
        "n_questions_per_run": next(iter(runs.values()))["n"],
        "mean_accuracy": round(mean, 2),
        "std_accuracy": round(std, 2),
        "std_ddof": 1,
        "ci_95_lo": round(mean - half_width, 2),
        "ci_95_hi": round(mean + half_width, 2),
        "judge_model": DEFAULT_JUDGE_MODEL,
        "per_run": runs,
    }

    with open(task_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"    Updated {task_dir / 'summary.json'}: {summary['mean_accuracy']}% ± {summary['std_accuracy']}% (sample SD)"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Re-judge responses through the LLM judge"
    )
    parser.add_argument("files", nargs="+", help="JSONL files to process")
    parser.add_argument(
        "--archive",
        default="archive",
        help="Archive root, used to look up each row's question text (default: ./archive)",
    )
    parser.add_argument(
        "--all",
        dest="rejudge_all",
        action="store_true",
        help="Re-judge every non-empty response, not only rows with chosen=null",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input files (and their summary.json)",
    )
    parser.add_argument(
        "--output-dir",
        default="rejudged",
        help="Where to write re-judged files when not --in-place (default: ./rejudged)",
    )
    parser.add_argument(
        "--extractor",
        default="llm",
        choices=list(EXTRACTORS),
        help="Choice extractor (default: llm, the paper protocol)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "Error: OPENAI_API_KEY is not set. Export it in your shell before running."
        )
        sys.exit(1)

    files = []
    for pattern in args.files:
        path = Path(pattern)
        if path.exists():
            files.append(path)
        else:
            files.extend(Path(".").glob(pattern))

    if not files:
        print("No files found")
        sys.exit(1)

    questions_by_id = load_questions_by_id(Path(args.archive))
    if not questions_by_id:
        print(
            f"Error: no questions found under {args.archive}/questions; pass --archive."
        )
        sys.exit(1)
    print(f"Loaded {len(questions_by_id)} questions from {args.archive}")

    output_dir = Path(args.output_dir)
    total_changed = 0
    task_dirs = set()

    for filepath in sorted(files):
        print(f"  {filepath}:")
        results, stats = rejudge_file(
            filepath,
            questions_by_id,
            rejudge_all=args.rejudge_all,
            concurrency=args.concurrency,
            extractor=args.extractor,
        )
        total_changed += stats["changed"]
        action = "would change" if args.dry_run else "changed"
        print(
            f"    {stats['rejudged']} re-judged, {stats['changed']}/{stats['total']} {action}, "
            f"{stats['errors']} judge errors, {stats['missing_question']} without a question"
        )
        if args.dry_run:
            continue
        target = filepath if args.in_place else output_path_for(filepath, output_dir)
        write_results(results, target)
        task_dirs.add(target.parent)

    if not args.dry_run:
        print("\nRegenerating summaries:")
        for task_dir in sorted(task_dirs):
            regenerate_summary(task_dir)

    print(f"\nTotal: {total_changed} extractions changed")


if __name__ == "__main__":
    main()
