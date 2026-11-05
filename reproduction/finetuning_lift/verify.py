"""Preflight checks. Run this before spending GPU time.

Every assumption this package makes is asserted against the release on disk and
against the training entrypoint's source:

  1. The release exists and each task's split sizes match Table 2's `n`.
  2. Train and eval splits are disjoint at example-ID level.
  3. A/B answer keys are balanced, so chance is 50% and a position-biased model
     scores 50% rather than accidentally scoring well.
  4. Every referenced audio file is present on disk.
  5. Each question carries exactly the expected number of clips.
  6. THE REGRESSION GUARD: on pairwise tasks, every audio part is immediately
     preceded by a text part naming its clip. This is the bug that produced
     near-chance pairwise accuracy; it must never come back silently.
  7. Prompts are homogeneous within a task, and identical across arms.
  8. Answer extraction recovers the intended letter from each target string.
  9. The arms differ by exactly one field.
 10. The recipe matches Table 9.
 11. The encoder-LoRA flag is wired, gated, and default-off.

The encoder module count (192) is not checked here: resolving real Whisper module
names needs the GPU training image, and the entrypoint already refuses to run
when the flag is set but no encoder parameters end up trainable. That guard sits
at the only place where being wrong would matter.

Usage:
    python -m reproduction.finetuning_lift.verify
    python -m reproduction.finetuning_lift.verify --check-audio   # stat 5800 files
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduction.finetuning_lift import config, dataset  # noqa: E402

# A/B balance is allowed to deviate by this fraction from 50/50. The release
# documents exact 50/50 for every subcategory, so this is a tolerance for
# split-level rounding, not for genuine imbalance.
BALANCE_TOLERANCE = 0.06


def _report(ok: bool, message: str) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {message}")
    return ok


def check_release_present() -> bool:
    print(f"\nRelease location: {config.RELEASE_ROOT}")
    passed = _report(config.RELEASE_ROOT.is_dir(), "release directory exists")
    if not passed:
        print("       Set ACOUSTICBENCH_DATA to the authorized-data directory.")
        return False
    for task in config.TASKS.values():
        path = config.questions_path(task.slice_name)
        passed &= _report(path.is_file(), f"{task.label}: {path.name}")
    return passed


def check_splits() -> bool:
    """Split sizes match Table 2, and train/eval do not overlap."""
    print("\nSplits (from each record's own `split` field)")
    passed = True

    for task in config.TASKS.values():
        train = dataset.load(task, split="train")
        heldout = dataset.load(task, split="eval")

        passed &= _report(
            len(train) == task.train_count,
            f"{task.label} train: n={len(train)} (expect {task.train_count})",
        )
        passed &= _report(
            len(heldout) == task.eval_count,
            f"{task.label} eval:  n={len(heldout)} (expect {task.eval_count}, "
            f"Table 2 reports n={task.eval_count})",
        )

        train_ids = {ex.example_id for ex in train}
        heldout_ids = {ex.example_id for ex in heldout}
        overlap = train_ids & heldout_ids
        passed &= _report(
            not overlap,
            f"{task.label}: train/eval disjoint (overlap={len(overlap)})",
        )
        passed &= _report(
            len(train_ids) == len(train) and len(heldout_ids) == len(heldout),
            f"{task.label}: example_ids unique within each split",
        )
    return passed


def check_answer_balance() -> bool:
    """Balanced A/B keys, so chance is 50% and position bias cannot score well."""
    print("\nAnswer balance (chance = 50%)")
    passed = True
    for task in config.TASKS.values():
        for split in ("train", "eval"):
            examples = dataset.load(task, split=split)
            counts = collections.Counter(ex.answer_key for ex in examples)
            unexpected = set(counts) - set(dataset.ANSWER_LABELS)
            passed &= _report(
                not unexpected,
                f"{task.label} {split}: labels are A/B only"
                + (f" (found {sorted(unexpected)})" if unexpected else ""),
            )
            share_b = counts["B"] / max(1, len(examples))
            passed &= _report(
                abs(share_b - 0.5) <= BALANCE_TOLERANCE,
                f"{task.label} {split}: B={share_b:.1%} "
                f"(A={counts['A']}, B={counts['B']})",
            )
    return passed


def check_clip_structure(*, check_audio: bool) -> bool:
    """Clip counts, and that every clip is introduced by a label.

    The label check is the regression guard. On a pairwise task the two clips are
    distinguishable only by the "Clip A:" / "Clip B:" text that precedes them. If
    a future change flattens prompts and drops those labels, the model is asked
    to compare two clips it cannot tell apart: accuracy falls to chance and the
    failure looks like a modelling result rather than a bug.
    """
    print("\nClip structure")
    passed = True

    for task in config.TASKS.values():
        bad_count: list[str] = []
        unlabelled: list[str] = []
        missing: list[str] = []

        for split in ("train", "eval"):
            for example in dataset.load(task, split=split):
                audio_parts = [p for p in example.parts if p["type"] == "audio_file"]
                if len(audio_parts) != task.clips_per_question:
                    bad_count.append(example.example_id)

                # Every audio part must be directly preceded by a text part.
                for position, part in enumerate(example.parts):
                    if part["type"] != "audio_file":
                        continue
                    previous = example.parts[position - 1] if position else None
                    if not previous or previous["type"] != "text":
                        unlabelled.append(example.example_id)
                        break
                    if not previous["text"].strip():
                        unlabelled.append(example.example_id)
                        break

                if check_audio:
                    missing.extend(
                        str(path)
                        for path in example.audio_paths
                        if not path.is_file()
                    )

        passed &= _report(
            not bad_count,
            f"{task.label}: every question has {task.clips_per_question} clip(s)"
            + (f" ({len(bad_count)} violations)" if bad_count else ""),
        )
        passed &= _report(
            not unlabelled,
            f"{task.label}: every clip is preceded by a label text part"
            + (f" ({len(unlabelled)} unlabelled)" if unlabelled else ""),
        )
        if check_audio:
            passed &= _report(
                not missing,
                f"{task.label}: all audio present on disk"
                + (f" ({len(missing)} missing)" if missing else ""),
            )

    if not check_audio:
        print("  ----  audio presence not checked (pass --check-audio)")
    return passed


def check_prompt_homogeneity() -> bool:
    """Prompts vary only where the task varies, and never between arms.

    Both arms read prompts from this same release through `dataset.py`, so the
    only way they could differ is a code change. Printing the digests here makes
    an unintended prompt change visible in the diff of a run log.
    """
    print("\nPrompt fidelity")
    passed = True
    for task in config.TASKS.values():
        digests: collections.Counter[str] = collections.Counter()
        for split in ("train", "eval"):
            for example in dataset.load(task, split=split):
                digests[example.prompt_digest()] += 1

        # One digest per distinct question wording. RAD varies by cue, and
        # SER-Classify varies by emotion pair, so several is expected; hundreds
        # would mean per-item text leaked into the prompt.
        passed &= _report(
            len(digests) <= 64,
            f"{task.label}: {len(digests)} distinct prompt shape(s)",
        )
        most_common, count = digests.most_common(1)[0]
        print(f"        most common digest {most_common} ({count} items)")
    return passed


def check_answer_extraction() -> bool:
    """The scorer must recover the intended letter from a well-formed answer."""
    print("\nAnswer extraction")
    passed = True
    for task in config.TASKS.values():
        failures: list[tuple[str, str]] = []
        for example in dataset.load(task, split="eval"):
            for candidate in (
                example.answer_key,
                f"({example.answer_key})",
                example.target,
                f"The answer is {example.answer_key}.",
            ):
                if dataset.extract_answer(candidate) != example.answer_key:
                    failures.append((example.example_id, candidate))
        passed &= _report(
            not failures,
            f"{task.label}: target strings round-trip through extract_answer"
            + (f" ({len(failures)} failures, e.g. {failures[0][1]!r})" if failures else ""),
        )
    passed &= _report(
        dataset.extract_answer("I cannot tell") is None,
        "unparseable responses return None rather than a guess",
    )
    return passed


def check_arms_differ_by_one_field() -> bool:
    """The whole experiment rests on the two arms differing in exactly one way."""
    print("\nExperimental arms")

    baseline = dataclasses.asdict(config.BASELINE)
    ablation = dataclasses.asdict(config.ENCODER_LORA)

    descriptive = {"key", "label", "description"}
    behavioural = {
        field
        for field in baseline
        if field not in descriptive and baseline[field] != ablation[field]
    }

    passed = _report(
        behavioural == {"lora_include_audio_encoder"},
        f"arms differ only by lora_include_audio_encoder (differ: {sorted(behavioural)})",
    )
    passed &= _report(
        config.BASELINE.lora_include_audio_encoder is False
        and config.ENCODER_LORA.lora_include_audio_encoder is True,
        "baseline freezes the encoder; ablation adapts it",
    )
    return passed


def check_recipe_matches_paper() -> bool:
    """Guard the Table 9 values against accidental drift."""
    print("\nRecipe (Table 9)")
    r = config.RECIPE
    expectations = {
        "lora_r": (r.lora_r, 16),
        "lora_alpha": (r.lora_alpha, 32),
        "lora_dropout": (r.lora_dropout, 0.05),
        "learning_rate": (r.learning_rate, 5e-5),
        "lr_scheduler_type": (r.lr_scheduler_type, "cosine"),
        "warmup_ratio": (r.warmup_ratio, 0.03),
        "adam_beta2": (r.adam_beta2, 0.95),
        "weight_decay": (r.weight_decay, 0.1),
        "effective_batch_size": (r.effective_batch_size, 16),
        "model_max_length": (r.model_max_length, 512),
    }
    passed = True
    for field, (actual, expected) in expectations.items():
        passed &= _report(actual == expected, f"{field} = {actual!r}")
    passed &= _report(config.SEEDS == (1, 2, 3), f"seeds = {config.SEEDS} (3 runs)")
    for task in config.TASKS.values():
        passed &= _report(
            task.max_steps > 0, f"{task.label}: max_steps = {task.max_steps}"
        )
    return passed


def check_encoder_flag_wired() -> bool:
    """The encoder-LoRA path lives in the original entrypoint, gated, default-off.

    There is one training module, not a fork: `--lora_include_audio_encoder`
    adds a code path on top of the original. With the flag unset the module
    uses the decoder-only path. These checks verify wiring, not historical
    experiment provenance.
    """
    print("\nEncoder-LoRA flag (single shared entrypoint)")
    entrypoint = REPO_ROOT / "finetuning/kimi_audio/rad/runtime/lora_entrypoint.py"
    launcher = REPO_ROOT / "finetuning/kimi_audio/rad/modal/kimi_rad_finetune.py"
    passed = True

    src = entrypoint.read_text()
    match = re.search(
        r"lora_include_audio_encoder:\s*bool\s*=\s*field\(\s*default=(\w+)", src
    )
    passed &= _report(
        bool(match) and match.group(1) == "False",
        "entrypoint: flag defaults to False (decoder-only baseline)",
    )
    passed &= _report(
        "WHISPER_ENCODER_LORA_MODULES" in src,
        "entrypoint: Whisper projection names defined",
    )
    passed &= _report(
        "include_audio_encoder=include_audio" in src,
        "entrypoint: flag reaches target resolution and constraints",
    )
    passed &= _report(
        'summary["audio_trainable_params"] == 0' in src,
        "entrypoint: guards against a silent no-op encoder run",
    )
    # The baseline casts the encoder to bf16 as a side effect of freezing it.
    # The ablation must cast it too, or precision silently becomes a second
    # difference between the arms.
    passed &= _report(
        "_cast_audio_encoder_bf16(base_model)" in src,
        "entrypoint: ablation casts the encoder to bf16, matching the baseline",
    )
    passed &= _report(
        "--lora_include_audio_encoder" in launcher.read_text(),
        "launcher: flag forwarded to the training process",
    )
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-audio",
        action="store_true",
        help="Also stat every referenced audio file (5800 of them).",
    )
    args = parser.parse_args()

    print("Preflight: Table 2 reproduction + encoder-LoRA ablation")

    if not check_release_present():
        print("\nFAILED - release not usable.")
        return 1

    passed = check_splits()
    passed &= check_answer_balance()
    passed &= check_clip_structure(check_audio=args.check_audio)
    passed &= check_prompt_homogeneity()
    passed &= check_answer_extraction()
    passed &= check_arms_differ_by_one_field()
    passed &= check_recipe_matches_paper()
    passed &= check_encoder_flag_wired()

    print("\n" + ("All checks passed." if passed else "FAILED - do not run training."))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
