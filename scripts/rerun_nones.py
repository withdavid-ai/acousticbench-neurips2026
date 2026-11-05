#!/usr/bin/env python3
"""Rerun questions where the model returned an unparseable response (chosen=None).

Reads a run JSONL, identifies None responses, reruns those questions through
the model, and patches the results in place. Can be run multiple times; each
pass only retries remaining Nones.

Questions are sent with the same interleaved message layout as run.py
("Clip A: <audio> Clip B: <audio> ..."), and ser_classify_text rows are
supported. Note that re-querying a row until it parses is a deviation from the
single-pass protocol used for the paper numbers, so report any use of it.

Usage:
    # Single file
    python scripts/rerun_nones.py --model GPT-4o reference_trajectories/GPT-4o/rad/run_1.jsonl

    # All files for a model
    python scripts/rerun_nones.py --model GPT-4o reference_trajectories/GPT-4o/*/run_*.jsonl

    # Dry run
    python scripts/rerun_nones.py --model GPT-4o --dry-run reference_trajectories/GPT-4o/rad/run_1.jsonl

    # With concurrency
    python scripts/rerun_nones.py --model GPT-4o --concurrency 8 reference_trajectories/GPT-4o/rad/run_1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.loader import load_questions_by_id
from harness.judge import extract_choice
from harness.models import build_model, set_backend
from harness.runner import valid_labels_for_question


def _load_questions_by_id(archive_root: Path) -> dict[str, dict]:
    """Load all questions (every task, including ser_classify_text) by example_id."""
    return load_questions_by_id(archive_root)


def build_message(question: dict) -> dict:
    """Build the model input exactly as harness.runner._run_single does."""
    return {
        "audio": question["audio_path"],
        "text": question["question"],
        "messages": question.get("messages"),
        "meta": question["meta"],
    }


def _run_single(model, question: dict) -> dict:
    """Run inference on a single question and extract the choice."""
    msg = build_message(question)
    t0 = time.perf_counter()
    trajectory = model(msg)
    wall_s = time.perf_counter() - t0

    response_text = trajectory.get("response_text", "") or ""
    response_status = trajectory.get("response_status", "ok")

    valid_labels = valid_labels_for_question(question["question"])

    extraction = extract_choice(response_text, question["question"], valid_labels)
    chosen = extraction["label"]
    is_correct = chosen == question["answer"] if chosen else False

    return {
        "response_text": response_text,
        "response_status": response_status,
        "chosen": chosen,
        "is_correct": is_correct,
        "extraction_source": extraction["source"],
        "wall_time_s": round(wall_s, 2),
    }


def rerun_file(
    filepath: Path,
    model,
    questions_by_id: dict[str, dict],
    dry_run: bool = False,
    concurrency: int = 1,
    max_retries: int = 3,
) -> dict:
    results = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    none_indices = [
        i for i, r in enumerate(results)
        if r.get("chosen") is None
    ]

    if not none_indices:
        return {"file": str(filepath), "total": len(results), "nones": 0, "resolved": 0, "still_none": 0}

    if dry_run:
        return {"file": str(filepath), "total": len(results), "nones": len(none_indices), "resolved": 0, "still_none": len(none_indices)}

    resolved = 0
    still_none_ids = list(none_indices)

    for attempt in range(1, max_retries + 1):
        if not still_none_ids:
            break

        print(f"    Attempt {attempt}/{max_retries}: {len(still_none_ids)} to retry")

        retry_items = []
        for idx in still_none_ids:
            eid = results[idx].get("example_id", results[idx].get("meta", {}).get("example_id", ""))
            question = questions_by_id.get(eid)
            if question:
                retry_items.append((idx, question))

        new_results = {}
        if concurrency <= 1:
            for idx, question in retry_items:
                new_results[idx] = _run_single(model, question)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(_run_single, model, question): idx
                    for idx, question in retry_items
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        new_results[idx] = future.result()
                    except Exception as e:
                        print(f"      Error on index {idx}: {e}")

        next_still_none = []
        for idx in still_none_ids:
            if idx not in new_results:
                next_still_none.append(idx)
                continue
            nr = new_results[idx]
            if nr["chosen"] is not None:
                results[idx]["response_text"] = nr["response_text"]
                results[idx]["response_status"] = nr["response_status"]
                results[idx]["chosen"] = nr["chosen"]
                results[idx]["is_correct"] = nr["is_correct"]
                results[idx]["extraction_source"] = f"rerun-attempt-{attempt}"
                results[idx]["wall_time_s"] = nr["wall_time_s"]
                resolved += 1
            else:
                next_still_none.append(idx)

        still_none_ids = next_still_none

    with open(filepath, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    return {
        "file": str(filepath),
        "total": len(results),
        "nones": len(none_indices),
        "resolved": resolved,
        "still_none": len(still_none_ids),
    }


def main():
    parser = argparse.ArgumentParser(description="Rerun None responses through the model")
    parser.add_argument("files", nargs="+", help="JSONL files to process")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--backend", type=str, default="docker", choices=["docker", "modal"])
    parser.add_argument("--archive", type=str, default="archive")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    set_backend(args.backend)

    print(f"Loading questions from {args.archive}...")
    questions_by_id = _load_questions_by_id(Path(args.archive))
    print(f"Loaded {len(questions_by_id)} questions")

    print(f"Building model: {args.model}")
    model = build_model(args.model)

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

    total_nones = 0
    total_resolved = 0

    for filepath in sorted(files):
        print(f"  {filepath}:")
        stats = rerun_file(
            filepath, model, questions_by_id,
            dry_run=args.dry_run,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
        )
        total_nones += stats["nones"]
        total_resolved += stats["resolved"]
        print(
            f"    {stats['nones']} nones → {stats['resolved']} resolved, "
            f"{stats['still_none']} still None"
        )

    print(f"\nTotal: {total_nones} nones, {total_resolved} resolved")


if __name__ == "__main__":
    main()
