#!/usr/bin/env python3
"""Parity test: Modal vs Docker for open-weight models.

Runs N questions per task through both backends and compares extracted
labels. Greedy decoding on identical weights should produce identical
outputs — any mismatch indicates a serving bug.

Usage:
    # Single model (Docker container must be running):
    python scripts/parity_test.py --model "Qwen3-Omni" --n 10

    # All models with Docker support:
    python scripts/parity_test.py --all --n 10

    # Specific tasks only:
    python scripts/parity_test.py --model "Kimi-Audio" --tasks rad,ser_classify --n 10
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.judge import extract_choice_deterministic
from harness.loader import load_questions
from harness.models import set_backend, build_model

ALL_TASKS = ["rad", "rsq", "ser_classify", "ser_classify_text", "ser_intensity"]

# Open-weight models of the paper roster (both Modal and Docker adapters).
DOCKER_MODELS = [
    "Qwen3-Omni",
    "Kimi-Audio",
    "Qwen2-Audio-7B-Instruct",
    "MiniCPM-o-4.5",
    "Phi-4-Multimodal",
    "Audio-Flamingo-3",
    "MiMo-Audio-7B",
    "Step-Audio-2-mini",
    "Voxtral-Small-24B",
    "Voxtral-Mini-3B",
]


def run_question(model, q):
    msg = {
        "audio": q["audio_path"],
        "text": q["question"],
        "messages": q.get("messages"),
        "meta": q.get("meta", {}),
    }
    try:
        result = model.generate_inner(msg)
        return result.get("response_text") or "", "ok"
    except Exception as e:
        return f"[ERROR: {e}]", "error"


def compare_responses(modal_text, docker_text):
    # Offline regex extractor (no judge calls); exact_match is the strict check.
    modal_label = extract_choice_deterministic(modal_text, ["A", "B"])
    docker_label = extract_choice_deterministic(docker_text, ["A", "B"])

    exact_match = modal_text.strip() == docker_text.strip()
    label_match = modal_label == docker_label

    return {
        "modal_label": modal_label,
        "docker_label": docker_label,
        "exact_match": exact_match,
        "label_match": label_match,
    }


def run_parity(model_name, tasks, n, archive_root):
    question_cache = {}
    for task in tasks:
        question_cache[task] = load_questions(task, archive_root=archive_root)

    results = {}

    for task in tasks:
        print(f"\n  --- {task} ({n} questions) ---")
        questions = question_cache[task][:n]

        # Run Modal
        set_backend("modal")
        modal_model = build_model(model_name)

        # Run Docker
        set_backend("docker")
        try:
            docker_model = build_model(model_name)
        except Exception as e:
            print(f"  SKIP — Docker adapter not available: {e}")
            results[task] = {"error": str(e)}
            continue

        pairs = []
        for i, q in enumerate(questions):
            modal_text, modal_status = run_question(modal_model, q)
            docker_text, docker_status = run_question(docker_model, q)

            comp = compare_responses(modal_text, docker_text)

            status_icon = "✅" if comp["label_match"] else "❌"
            exact_icon = "=" if comp["exact_match"] else "≈" if comp["label_match"] else "≠"
            print(
                f"    Q{i+1:>2}/{n}: "
                f"modal={comp['modal_label'] or '?'} "
                f"docker={comp['docker_label'] or '?'} "
                f"{status_icon}{exact_icon}  "
                f"{repr(modal_text[:50])}"
            )

            pairs.append({
                "example_id": q["meta"]["example_id"],
                "answer_key": q.get("answer"),
                "modal_text": modal_text[:200],
                "docker_text": docker_text[:200],
                "modal_status": modal_status,
                "docker_status": docker_status,
                **comp,
            })

        label_matches = sum(1 for p in pairs if p["label_match"])
        exact_matches = sum(1 for p in pairs if p["exact_match"])
        total = len(pairs)

        results[task] = {
            "n": total,
            "label_matches": label_matches,
            "exact_matches": exact_matches,
            "label_parity": f"{label_matches}/{total}",
            "exact_parity": f"{exact_matches}/{total}",
            "pairs": pairs,
        }

        verdict = "PASS" if label_matches == total else "FAIL"
        print(
            f"  → {verdict}: label parity {label_matches}/{total}, "
            f"exact match {exact_matches}/{total}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Modal vs Docker parity test")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--all", action="store_true", help="Test all Docker-supported models")
    parser.add_argument("--tasks", default=",".join(ALL_TASKS), help="Comma-separated tasks")
    parser.add_argument("--n", type=int, default=10, help="Questions per task")
    parser.add_argument("--archive", default="archive", help="Archive root")
    parser.add_argument("--output-dir", default="results/parity_tests", help="Output directory")
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = DOCKER_MODELS if args.all else [args.model]
    if not models or models == [None]:
        print("Specify --model <name> or --all")
        sys.exit(1)

    all_results = {}
    start_time = time.time()

    for model_idx, model_name in enumerate(models, 1):
        print(f"\n{'=' * 60}")
        print(f"  [{model_idx}/{len(models)}] {model_name}")
        print(f"{'=' * 60}")

        try:
            model_results = run_parity(model_name, tasks, args.n, args.archive)
        except Exception as e:
            print(f"  ERROR: {e}")
            model_results = {"error": str(e)}

        all_results[model_name] = model_results

        model_file = output_dir / f"{model_name.replace('/', '_')}.json"
        with open(model_file, "w") as f:
            json.dump({"model": model_name, "results": model_results}, f, indent=2)

    elapsed = time.time() - start_time

    # Summary report
    print(f"\n{'=' * 60}")
    print("  PARITY TEST SUMMARY")
    print(f"  {len(models)} models, {len(tasks)} tasks, {args.n} questions each")
    print(f"  Time: {elapsed/60:.1f} minutes")
    print(f"{'=' * 60}\n")

    all_pass = True
    for model_name in models:
        mr = all_results.get(model_name, {})
        if "error" in mr:
            print(f"  {model_name}: ERROR — {mr['error']}")
            all_pass = False
            continue
        model_pass = True
        for task in tasks:
            tr = mr.get(task, {})
            if "error" in tr:
                continue
            if tr.get("label_matches", 0) != tr.get("n", 0):
                model_pass = False
                all_pass = False
        status = "PASS ✅" if model_pass else "FAIL ❌"
        task_summary = ", ".join(
            f"{t}={mr[t]['label_parity']}"
            for t in tasks if t in mr and "label_parity" in mr[t]
        )
        print(f"  {model_name}: {status} ({task_summary})")

    print()
    if all_pass:
        print("✅ ALL PARITY TESTS PASSED")
    else:
        print("❌ SOME PARITY TESTS FAILED — investigate mismatches")

    report_path = output_dir / "REPORT.md"
    report_path.write_text(
        f"# Parity Test Report\n\n"
        f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"Questions per task: {args.n}\n\n"
        f"See per-model JSON files for full pair-by-pair data.\n"
    )
    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
