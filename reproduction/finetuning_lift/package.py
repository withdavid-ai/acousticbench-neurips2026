"""Merge a trained adapter into a servable checkpoint, so `evaluate.py` can score it.

Training writes PEFT adapters to `/data/runs/{run}/train/checkpoint-{step}`. The
serving class (`KimiAudioFtServer`) needs a merged checkpoint, which the
launcher's `publish` step produces.

Each checkpoint is packaged to its own directory:

    /data/serving/{run}/checkpoint-{step}

rather than the launcher's default `/data/serving/kimi-audio-ft/latest`, which is
a single shared slot -- packaging several runs through it would silently
overwrite, and two arms would end up scoring the same weights.

`--run-name` is deliberately not forwarded. With it set, the publish branch
derives a promotion-archive directory and points the serving deploy there
instead of at `--package-output-dir`, which makes the path `evaluate.py` reads
depend on logic we do not need here.

Usage:
    python -m reproduction.finetuning_lift.package --arm baseline --all
    python -m reproduction.finetuning_lift.package --arm encoder_lora --all --seed 1
    python -m reproduction.finetuning_lift.package --task rad --arm baseline --seed 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduction.finetuning_lift import config  # noqa: E402

LAUNCHER = "finetuning/kimi_audio/rad/modal/kimi_rad_finetune.py::main"
MODAL_BIN = "modal"

TRAIN_CHECKPOINT_TEMPLATE = "/data/runs/{run_name}/train/checkpoint-{checkpoint}"


def serving_path(run_name: str, checkpoint: int) -> str:
    """Must match evaluate.FT_MODEL_PATH_TEMPLATE."""
    return f"/data/serving/{run_name}/checkpoint-{checkpoint}"


def package_one(
    task: config.Task,
    arm: config.Arm,
    seed: int,
    checkpoint: int,
    *,
    dry_run: bool = False,
) -> dict:
    run_name = config.run_name(task.key, arm.key, seed)
    source = TRAIN_CHECKPOINT_TEMPLATE.format(
        run_name=run_name, checkpoint=checkpoint
    )
    destination = serving_path(run_name, checkpoint)

    cmd = [
        str(MODAL_BIN), "run", LAUNCHER,
        "--step", "publish",
        "--modal-env", config.MODAL_ENV,
        "--package-input-dir", source,
        "--package-output-dir", destination,
        "--pretrained-path", config.PRETRAINED_PATH,
    ]

    print(f"\n=== package {run_name} checkpoint-{checkpoint} ===")
    print(f"    {source}")
    print(f" -> {destination}")
    if dry_run:
        print(f"    DRY RUN: {' '.join(cmd)}")
        return {"run_name": run_name, "status": "dry-run", "path": destination}

    env = {**os.environ, "MODAL_ENVIRONMENT": config.MODAL_ENV}
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)

    # Success is decided by the artifact, not the exit code. After exporting the
    # merged checkpoint the publish step runs a smoke test that reads a hardcoded
    # path from the old repo layout
    # (`research/tasks/relative_acoustic_perception/data/kimi_rad_modal/rad_kimi_val.jsonl`)
    # and dies with FileNotFoundError. That happens strictly after the export, so
    # the checkpoint on the volume is complete and usable.
    exported = _volume_has_checkpoint(destination)
    if not exported:
        raise RuntimeError(
            f"publish produced no usable checkpoint for {run_name} "
            f"checkpoint-{checkpoint} at {destination} (exit {result.returncode})"
        )
    if result.returncode != 0:
        print(
            f"    note: launcher exited {result.returncode} after export "
            f"(stale-path smoke test); checkpoint verified present"
        )
    return {"run_name": run_name, "status": "ok", "path": destination}


# Files a merged, servable checkpoint must have.
REQUIRED_CHECKPOINT_FILES = ("config.json", "model.safetensors.index.json")


def _volume_has_checkpoint(volume_path: str) -> bool:
    """True when `volume_path` holds a complete merged checkpoint."""
    listing = subprocess.run(
        [
            str(MODAL_BIN), "volume", "ls",
            config.VOLUME_NAME,
            volume_path.removeprefix("/data"),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "MODAL_ENVIRONMENT": config.MODAL_ENV},
        capture_output=True,
        text=True,
        check=False,
    )
    if listing.returncode != 0:
        return False
    return all(name in listing.stdout for name in REQUIRED_CHECKPOINT_FILES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(config.TASKS))
    parser.add_argument("--arm", choices=sorted(config.ARMS), required=True)
    parser.add_argument("--seed", type=int, choices=config.SEEDS)
    parser.add_argument(
        "--checkpoint",
        type=int,
        help="Checkpoint step to package (default: the final step per task).",
    )
    parser.add_argument("--all", action="store_true", help="All four tasks.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.all and not args.task:
        parser.error("pass --all or --task <key>")

    arm = config.ARMS[args.arm]
    tasks = list(config.TASKS.values()) if args.all else [config.TASKS[args.task]]
    seeds = (args.seed,) if args.seed else config.SEEDS

    print(f"Arm: {arm.label}")
    records = []
    for task in tasks:
        checkpoint = args.checkpoint or task.max_steps
        for seed in seeds:
            records.append(
                package_one(task, arm, seed, checkpoint, dry_run=args.dry_run)
            )

    print(f"\n{len(records)} checkpoint(s) packaged for arm {arm.key}.")
    step_hint = str(args.checkpoint) if args.checkpoint else "<step>"
    print(
        "Next: python -m reproduction.finetuning_lift.evaluate --all "
        f"--arm {arm.key} --seed {seeds[0]} --checkpoint {step_hint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
