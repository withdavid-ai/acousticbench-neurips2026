"""Run one (task, arm, seed) LoRA training job on Modal.

Jobs use the shared Kimi launcher and training runtime. The encoder arm enables
`--lora-include-audio-encoder`; the decoder arm leaves it disabled. These are
local research tools, not authenticated provenance for the submitted results.
See docs/reproduction.md for scope and data-use restrictions.

Prerequisites, handled here rather than left as manual steps:
  * `prepare-model` -- materializes the base checkpoint once, shared by all runs
  * `prepare-data`  -- extracts semantic codes per bundle, trainer-agnostic

Both are idempotent and skipped when their outputs already exist on the volume.

Usage:
    # one run
    python -m reproduction.finetuning_lift.train --task rad --arm baseline --seed 1

    # a whole arm (4 tasks x 3 seeds)
    python -m reproduction.finetuning_lift.train --arm baseline --all

    python -m reproduction.finetuning_lift.train --arm baseline --all --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduction.finetuning_lift import config  # noqa: E402

LAUNCHER = "finetuning/kimi_audio/rad/modal/kimi_rad_finetune.py::main"
MODAL_BIN = "modal"


def _modal_env() -> dict[str, str]:
    return {**os.environ, "MODAL_ENVIRONMENT": config.MODAL_ENV}


def _run(cmd: list[str], *, dry_run: bool, label: str) -> None:
    print(f"\n=== {label} ===")
    if dry_run:
        print(f"    DRY RUN: {' '.join(cmd)}")
        return
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=_modal_env(), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed (exit {result.returncode})")


def prepare_model(*, dry_run: bool = False) -> None:
    """Materialize the base checkpoint on the volume. Shared by every run."""
    _run(
        [
            str(MODAL_BIN), "run", LAUNCHER,
            "--step", "prepare-model",
            "--modal-env", config.MODAL_ENV,
        ],
        dry_run=dry_run,
        label="prepare-model (shared)",
    )


def prepare_data(task: config.Task, *, dry_run: bool = False) -> None:
    """Extract semantic codes for one bundle. Identical across arms and seeds."""
    _run(
        [
            str(MODAL_BIN), "run", LAUNCHER,
            "--step", "prepare-data",
            "--modal-env", config.MODAL_ENV,
            "--bundle-name", task.bundle_name,
        ],
        dry_run=dry_run,
        label=f"prepare-data {task.label}",
    )


def train_one(
    task: config.Task,
    arm: config.Arm,
    seed: int,
    *,
    dry_run: bool = False,
) -> dict:
    """Launch a single training run."""
    recipe = config.recipe_for(arm, seed)
    name = config.run_name(task.key, arm.key, seed)

    cmd = [
        str(MODAL_BIN), "run", LAUNCHER,
        "--step", "train",
        "--modal-env", config.MODAL_ENV,
        "--bundle-name", task.bundle_name,
        "--run-name", name,
        "--trainer", "lora",
        # Table 9
        "--lora-r", str(recipe.lora_r),
        "--lora-alpha", str(recipe.lora_alpha),
        "--lora-dropout", str(recipe.lora_dropout),
        "--lora-target-modules", ",".join(recipe.lora_target_modules),
        "--learning-rate", str(recipe.learning_rate),
        "--weight-decay", str(recipe.weight_decay),
        "--warmup-ratio", str(recipe.warmup_ratio),
        "--per-device-train-batch-size", str(recipe.per_device_train_batch_size),
        "--gradient-accumulation-steps", str(recipe.gradient_accumulation_steps),
        "--model-max-length", str(recipe.model_max_length),
        "--logging-steps", str(recipe.logging_steps),
        "--save-steps", str(recipe.save_steps),
        "--max-steps", str(task.max_steps),
        "--seed", str(seed),
        # Disables the launcher's in-training held-out eval; see the note in
        # config.py. Scoring is evaluate.py, after training.
        "--heldout-evals-json", config.HELDOUT_EVALS_JSON,
        # The launcher defaults to W&B tracking and raises if WANDB_API_KEY is
        # absent on the Modal runner. Set explicitly here rather than changing that
        # default, which belongs to whoever does use W&B: this package logs nothing
        # to W&B, and its runs must not depend on a secret it never reads.
        "--tracking-backend", "none",
    ]
    # The single experimental variable.
    if arm.lora_include_audio_encoder:
        cmd += ["--lora-include-audio-encoder"]

    record = {
        "run_name": name,
        "task": task.key,
        "arm": arm.key,
        "seed": seed,
        "bundle": task.bundle_name,
        "lora_include_audio_encoder": arm.lora_include_audio_encoder,
        "max_steps": task.max_steps,
        "save_steps": recipe.save_steps,
        "command": " ".join(cmd),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    _run(cmd, dry_run=dry_run, label=f"train {name}")
    if dry_run:
        record["status"] = "dry-run"
        return record

    record["status"] = "ok"
    record["finished_at"] = datetime.now(timezone.utc).isoformat()

    out_dir = config.RESULTS_ROOT / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_record.json").write_text(json.dumps(record, indent=2))
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(config.TASKS))
    parser.add_argument("--arm", choices=sorted(config.ARMS), required=True)
    parser.add_argument("--seed", type=int, choices=config.SEEDS)
    parser.add_argument(
        "--all",
        action="store_true",
        help="All four tasks. Combine with --seed to run a single seed.",
    )
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.all and not (args.task and args.seed):
        parser.error("pass --all, or both --task and --seed")

    arm = config.ARMS[args.arm]
    tasks = list(config.TASKS.values()) if args.all else [config.TASKS[args.task]]
    # --seed narrows to one seed; without it, --all sweeps all three.
    seeds = (args.seed,) if args.seed else config.SEEDS

    print(f"Arm: {arm.label}")
    print(f"     {arm.description}")
    print(f"Runs: {len(tasks)} task(s) x {len(seeds)} seed(s) = {len(tasks) * len(seeds)}")

    if not args.skip_prepare:
        prepare_model(dry_run=args.dry_run)
        for task in tasks:
            prepare_data(task, dry_run=args.dry_run)

    records = [
        train_one(task, arm, seed, dry_run=args.dry_run)
        for task in tasks
        for seed in seeds
    ]

    print(f"\n{len(records)} run(s) launched for arm {arm.key}.")
    print("Next: python -m reproduction.finetuning_lift.evaluate --all --arm "
          f"{arm.key} --seed <n> --checkpoint <step>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
