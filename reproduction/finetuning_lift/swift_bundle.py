"""Export a task's train split in ms-swift format, for Qwen3-Omni finetuning.

Kimi and Qwen need different training data formats, but they must be the *same
data with the same prompts*, or the cross-model comparison is meaningless. Both
exporters therefore read through `dataset.py`, which is the whole reason that
module exists.

ms-swift's schema (see docs/Customization/Custom-dataset):

    {"messages": [{"role": "user", "content": "...<audio>...<audio>..."},
                  {"role": "assistant", "content": "B. Clip B is louder"}],
     "audios": ["/abs/clip_a.wav", "/abs/clip_b.wav"]}

`<audio>` marks where each clip is inserted and `audios` supplies the paths in
order, so the interleaving that matters here -- "Clip A:" immediately before the
first clip, "Clip B:" before the second -- survives intact. This is the same
property that made the release format immune to the label-dropping bug: the
labels are ordinary text, not metadata on an audio object.

Usage:
    python -m reproduction.finetuning_lift.swift_bundle --all
    python -m reproduction.finetuning_lift.swift_bundle --task rad --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduction.finetuning_lift import config, dataset  # noqa: E402

AUDIO_TAG = "<audio>"

TRAIN_JSONL = "swift_train.jsonl"
HELDOUT_JSONL = "swift_heldout.jsonl"
MANIFEST_JSON = "manifest.json"


def bundle_dir(task: config.Task) -> Path:
    return config.LOCAL_BUNDLE_ROOT / f"{task.key}-{task.train_count}-swift"


def _volume_audio_path(relpath: str, bundle_name: str) -> str:
    """Where this clip will live inside the Modal training volume.

    Must agree with where `modal volume put` uploads the bundle, or training
    fails with a misleading `max_length` error (see config.py).
    """
    parts = Path(relpath).parts
    root = config.SWIFT_VOLUME_AUDIO_ROOT_TEMPLATE.format(bundle_name=bundle_name)
    return f"{root}/{parts[-2]}__{parts[-1]}"


def to_swift_record(
    example: dataset.Example,
    *,
    audio_path_for: dict[str, str],
    include_target: bool = True,
) -> dict:
    """One ms-swift training row, preserving the release's part order.

    Each audio part becomes an `<audio>` tag at exactly its original position,
    and `audios` lists the corresponding paths in the same order, so tag N binds
    to clip N.
    """
    rendered: list[str] = []
    audios: list[str] = []

    for part in example.parts:
        if part["type"] == "audio_file":
            rendered.append(AUDIO_TAG)
            audios.append(audio_path_for[part["path"]])
        else:
            rendered.append(part["text"])

    messages = []
    if example.system_prompt:
        messages.append({"role": "system", "content": example.system_prompt})
    messages.append({"role": "user", "content": "\n".join(rendered)})
    if include_target:
        messages.append({"role": "assistant", "content": example.target})

    record = {"messages": messages, "audios": audios}
    if len(audios) != example.parts_audio_count:
        raise RuntimeError(
            f"{example.example_id}: {len(audios)} audios for "
            f"{example.parts_audio_count} audio parts"
        )
    return record


def build(task: config.Task, *, dry_run: bool = False) -> dict:
    """Write the ms-swift bundle for one task."""
    train = dataset.load(task, split="train")
    heldout = dataset.load(task, split="eval")

    out_dir = bundle_dir(task)
    audio_dir = out_dir / "audio"

    print(f"\n=== build {task.label} (ms-swift) ===")
    print(f"    train {len(train)}  heldout {len(heldout)}")
    if dry_run:
        print(f"    DRY RUN: would write {out_dir / TRAIN_JSONL}")
        return {"task": task.key, "status": "dry-run"}

    if len(train) != task.train_count:
        raise RuntimeError(
            f"{task.label}: release has {len(train)} train rows, "
            f"expected {task.train_count}"
        )

    audio_dir.mkdir(parents=True, exist_ok=True)
    copied = 0

    def _rows(examples: list[dataset.Example], *, include_target: bool) -> list[dict]:
        nonlocal copied
        rows = []
        for example in examples:
            audio_path_for = {}
            for relpath in example.audio_relpaths:
                volume_path = _volume_audio_path(relpath, out_dir.name)
                audio_path_for[relpath] = volume_path
                local_copy = audio_dir / Path(volume_path).name
                if not local_copy.exists():
                    shutil.copy2(config.RELEASE_ROOT / relpath, local_copy)
                    copied += 1
            rows.append(
                to_swift_record(
                    example,
                    audio_path_for=audio_path_for,
                    include_target=include_target,
                )
            )
        return rows

    train_rows = _rows(train, include_target=True)
    heldout_rows = _rows(heldout, include_target=True)

    for name, rows in ((TRAIN_JSONL, train_rows), (HELDOUT_JSONL, heldout_rows)):
        with (out_dir / name).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    (out_dir / MANIFEST_JSON).write_text(
        json.dumps(
            {
                "bundle_name": bundle_dir(task).name,
                "task": task.key,
                "format": "ms-swift",
                "train_rows": len(train_rows),
                "heldout_rows": len(heldout_rows),
                "clips_per_question": task.clips_per_question,
                "audio_files": copied,
                "release_root": str(config.RELEASE_ROOT),
                "prompt_source": "reproduction.finetuning_lift.dataset",
            },
            indent=2,
        )
    )

    print(f"    audio {copied} clips -> {audio_dir.relative_to(REPO_ROOT)}")
    print(f"    -> {(out_dir / TRAIN_JSONL).relative_to(REPO_ROOT)}")
    return {
        "task": task.key,
        "train": len(train_rows),
        "heldout": len(heldout_rows),
        "audio": copied,
        "status": "ok",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(config.TASKS))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.task and not args.all:
        parser.error("pass --task <key> or --all")

    tasks = list(config.TASKS.values()) if args.all else [config.TASKS[args.task]]
    for task in tasks:
        build(task, dry_run=args.dry_run)

    print("\nms-swift bundles ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
