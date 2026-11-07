"""Build each task's Kimi-Audio training bundle from the release, then upload it.

Two steps per task:

  1. Build   -- read the release's train split through `dataset.py`, copy the
                referenced clips into the bundle, and write the trainer's
                `rad_kimi_train.jsonl` with volume-side audio paths.
  2. Upload  -- `kimi_rad_finetune.py --step upload-bundle` syncs the bundle into
                the Modal volume and verifies the row counts remotely.

Prompts come from `dataset.py`, the same module `evaluate.py` uses, so the model
is trained on exactly the prompt format it is scored on. That equality is the
point of this module: the previous pipeline built training prompts and eval
prompts by different code paths, and the eval path silently dropped the
"Clip A:" / "Clip B:" labels.

No database and no S3: the release ships its own audio, and audio filenames are
content-addressed by the release's own directory layout, which is already unique
per example.

Usage:
    python -m reproduction.finetuning_lift.bundle --all
    python -m reproduction.finetuning_lift.bundle --task rad --skip-upload
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduction.finetuning_lift import config, dataset  # noqa: E402

MODAL_BIN = "modal"
LAUNCHER = "finetuning/kimi_audio/rad/modal/kimi_rad_finetune.py::main"

TRAIN_JSONL = "rad_kimi_train.jsonl"
VAL_JSONL = "rad_kimi_val.jsonl"
HELDOUT_JSONL = "heldout_eval_manifest.jsonl"
MANIFEST_JSON = "manifest.json"


def bundle_dir(task: config.Task) -> Path:
    return config.LOCAL_BUNDLE_ROOT / task.bundle_name


def _volume_audio_path(relpath: str) -> str:
    """Map a release-relative clip path to its path inside the Modal volume.

    The release nests clips per example (`audio/<task>/<example_id>/00_clip_a.wav`),
    so flattening on example_id keeps names unique without hashing.
    """
    parts = Path(relpath).parts
    return f"{config.VOLUME_AUDIO_ROOT}/{parts[-2]}__{parts[-1]}"


def _heldout_manifest_rows(task: config.Task, *, audio_dir: Path) -> list[dict]:
    """The held-out split in the trainer's manifest schema.

    The launcher requires this file to exist and wires it to the in-training
    checkpoint-eval hook. That hook rebuilds the prompt itself: it emits
    `meta.pre_audio_texts`, then each clip preceded by a reconstructed
    "Clip A:" / "Clip B:" label, then the `question` string. So the release's
    ordering is reproduced by putting the question in `pre_audio_texts` and the
    options block in `question`, with `content_order` leaving the options after
    the audio.

    Scoring for the paper comparison comes from `evaluate.py`, not from here.
    """
    rows = []
    for index, example in enumerate(dataset.load(task, split="eval")):
        text_parts = list(example.text_parts)
        options_block = text_parts[-1] if text_parts else ""
        pre_audio = [part for part in text_parts[:-1] if not part.endswith(":")]

        rows.append(
            {
                "index": index,
                "example_id": example.example_id,
                "audio_path": [
                    _volume_audio_path(relpath) for relpath in example.audio_relpaths
                ],
                "question": options_block,
                "answer": example.options.get(example.answer_key, example.answer_key),
                "answer_key": example.answer_key,
                "subset": dataset.subset_of(example),
                "meta": {
                    "example_id": example.example_id,
                    "benchmark_name": task.benchmark_name,
                    "category": example.category,
                    "dimension": example.dimension,
                    "answer_key": example.answer_key,
                    "pre_audio_texts": pre_audio,
                    "content_order": "audio-first",
                    # The uploader validates one local file per audio reference.
                    "local_audio_paths": [
                        str(audio_dir / Path(_volume_audio_path(relpath)).name)
                        for relpath in example.audio_relpaths
                    ],
                    "source": "authorized-data release, split=eval",
                },
            }
        )
    return rows


def build(task: config.Task, *, dry_run: bool = False) -> dict:
    """Write the task's training bundle from the release's train split."""
    examples = dataset.load(task, split="train")
    out_dir = bundle_dir(task)
    current = out_dir / "rad_exports" / "current"
    audio_dir = out_dir / "audio"

    print(f"\n=== build {task.label} ({len(examples)} train items) ===")
    if dry_run:
        print(f"    DRY RUN: would write {current / TRAIN_JSONL}")
        return {"task": task.key, "rows": len(examples), "status": "dry-run"}

    if len(examples) != task.train_count:
        raise RuntimeError(
            f"{task.label}: release has {len(examples)} train rows, "
            f"expected {task.train_count}"
        )

    for directory in (current, audio_dir):
        directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    copied = 0
    for example in examples:
        audio_path_for: dict[str, str] = {}
        local_audio_paths: list[str] = []
        for relpath in example.audio_relpaths:
            volume_path = _volume_audio_path(relpath)
            audio_path_for[relpath] = volume_path

            local_copy = audio_dir / Path(volume_path).name
            if not local_copy.exists():
                shutil.copy2(config.RELEASE_ROOT / relpath, local_copy)
                copied += 1
            local_audio_paths.append(str(local_copy))

        rows.append(
            {
                "task_type": "understanding",
                "conversation": dataset.kimi_conversation(
                    example, audio_path_for=audio_path_for
                ),
                "_meta": {
                    "example_id": example.example_id,
                    "category": example.category,
                    "benchmark_name": task.benchmark_name,
                    "dimension": example.dimension,
                    "subset": dataset.subset_of(example),
                    "answer_key": example.answer_key,
                    "prompt_digest": example.prompt_digest(),
                    # The uploader rewrites volume paths from these and asserts
                    # one local file per audio message.
                    "local_audio_paths": local_audio_paths,
                    "source": "authorized-data release, split=train",
                },
            }
        )

    with (current / TRAIN_JSONL).open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    # The trainer expects the file to exist. The paper holds out a separate eval
    # split and does not use a validation split, so this stays empty by design.
    (current / VAL_JSONL).write_text("")

    # The held-out manifest, plus its audio: the launcher validates that the file
    # exists, and any clip it names has to be resolvable inside the volume.
    heldout_rows = _heldout_manifest_rows(task, audio_dir=audio_dir)
    with (current / HELDOUT_JSONL).open("w") as handle:
        for row in heldout_rows:
            handle.write(json.dumps(row) + "\n")

    for example in dataset.load(task, split="eval"):
        for relpath in example.audio_relpaths:
            local_copy = audio_dir / Path(_volume_audio_path(relpath)).name
            if not local_copy.exists():
                shutil.copy2(config.RELEASE_ROOT / relpath, local_copy)
                copied += 1

    manifest = {
        "bundle_name": task.bundle_name,
        "task": task.key,
        "task_label": task.label,
        "benchmark_name": task.benchmark_name,
        "category": task.category,
        "clips_per_question": task.clips_per_question,
        "train_rows": len(rows),
        "val_rows": 0,
        "eval_rows_heldout": len(heldout_rows),
        "audio_files": copied,
        "release_root": str(config.RELEASE_ROOT),
        "prompt_source": "reproduction.finetuning_lift.dataset",
    }
    (current / MANIFEST_JSON).write_text(json.dumps(manifest, indent=2))

    print(f"    train    {len(rows)} rows")
    print(f"    heldout  {len(heldout_rows)} rows")
    print(f"    audio    {copied} clips -> {audio_dir.relative_to(REPO_ROOT)}")
    print(f"    -> {(current / TRAIN_JSONL).relative_to(REPO_ROOT)}")
    return {
        "task": task.key,
        "rows": len(rows),
        "heldout": len(heldout_rows),
        "audio": copied,
        "status": "ok",
    }


def upload(task: config.Task, *, dry_run: bool = False) -> None:
    """Sync the local bundle into the Modal volume and verify it remotely."""
    cmd = [
        str(MODAL_BIN), "run", LAUNCHER,
        "--step", "upload-bundle",
        "--modal-env", config.MODAL_ENV,
        "--bundle-name", task.bundle_name,
        "--staged-bundle-dir", str(bundle_dir(task).relative_to(REPO_ROOT)),
    ]

    print(f"\n=== upload {task.label} -> /bundles/{task.bundle_name} ===")
    if dry_run:
        print(f"    DRY RUN: {' '.join(cmd)}")
        return

    env = {**os.environ, "MODAL_ENVIRONMENT": config.MODAL_ENV}
    result = subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"upload failed for {task.label} (exit {result.returncode})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(config.TASKS), help="Single task.")
    parser.add_argument("--all", action="store_true", help="All four tasks.")
    parser.add_argument("--skip-build", action="store_true", help="Upload only.")
    parser.add_argument("--skip-upload", action="store_true", help="Build only.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions only.")
    args = parser.parse_args()

    if not args.task and not args.all:
        parser.error("pass --task <key> or --all")

    tasks = list(config.TASKS.values()) if args.all else [config.TASKS[args.task]]

    for task in tasks:
        if not args.skip_build:
            build(task, dry_run=args.dry_run)
        if not args.skip_upload:
            upload(task, dry_run=args.dry_run)

    print("\nAll bundles ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
