"""Score a model on a task's held-out split and compare against Table 2.

Prompts come from the same dataset module used to build training bundles,
preserving explicit clip labels. This research evaluator uses a deterministic
parser rather than the headline LLM judge. Unparsed responses count as incorrect
and are also reported separately; runs with excessive failures are rejected.
See docs/reproduction.md for limits on comparisons to the published results.

Usage:
    # Table 2 Base column: the un-finetuned model
    python -m reproduction.finetuning_lift.evaluate --task rad --arm base
    python -m reproduction.finetuning_lift.evaluate --all --arm base

    # a trained checkpoint
    python -m reproduction.finetuning_lift.evaluate \
        --task rad --arm baseline --seed 1 --checkpoint 1000
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduction.finetuning_lift import config, dataset  # noqa: E402

# Packaged-checkpoint layout on the training volume.
FT_MODEL_PATH_TEMPLATE = "/data/serving/{run_name}/checkpoint-{checkpoint}"

DEFAULT_CONCURRENCY = 8
PROGRESS_WIDTH = 30

# Above this share of failed requests the run is treated as broken rather than
# scored. Set low deliberately: a real eval has an error rate of zero.
MAX_ERROR_SHARE = 0.05

# Above this share of responses that yield no A/B label the run is also treated
# as broken. A missing checkpoint does not necessarily raise: the server answered
# every request and simply returned nothing parseable, so `errors` stayed at 0
# while `unparsed` hit 600/600. Chance on a two-way task is 50%, so a real eval
# cannot land near zero accuracy; anything past this threshold is a broken
# target, not a finding.
MAX_UNUSABLE_SHARE = 0.5


@dataclass(frozen=True)
class Target:
    """Which model to evaluate, and where its results belong."""

    model_key: str
    arm_key: str
    seed: int | None
    checkpoint: int | None
    # Distinguishes prompt variants of the same model so they land in separate
    # result directories instead of overwriting the canonical run.
    variant: str = ""
    # Which task's adapter to load. Empty means "the task being evaluated", i.e.
    # the specialist adapter on its own held-out split. Pointing it at a
    # different task turns the run into a transfer measurement: trained on A,
    # scored on B. This is what answers "why are the gains task-specific".
    adapter_task: str = ""

    @property
    def model(self) -> config.Model:
        return config.MODELS[self.model_key]

    def variant_for(self, task_key: str) -> str:
        """Effective variant tag for one evaluated task.

        A transfer run is a variant by construction: it shares
        (model, task, arm) with the canonical cell for the task it is scored on,
        so it has to stay out of the headline tables the same way the clip-swap
        control does.
        """
        if self.adapter_task and self.adapter_task != task_key:
            transfer = f"from-{self.adapter_task}"
            return f"{self.variant}-{transfer}" if self.variant else transfer
        return self.variant

    @property
    def is_base(self) -> bool:
        return self.arm_key == config.BASE_ARM_KEY

    def label_for(self, task_key: str) -> str:
        variant = self.variant_for(task_key)
        suffix = f" [{variant}]" if variant else ""
        if self.is_base:
            return f"STOCK {self.model.label} (no finetuning){suffix}"
        arm = config.ARMS[self.arm_key]
        nickname = arm.label.split(" ", 1)[0]
        return (
            f"{nickname} {self.model.label} seed{self.seed} "
            f"checkpoint-{self.checkpoint}{suffix}"
        )

    def slug(self, task_key: str) -> str:
        variant = self.variant_for(task_key)
        suffix = f"--{variant}" if variant else ""
        if self.is_base:
            return f"{task_key}--{self.model_key}--base{suffix}"
        return (
            f"{task_key}--{self.model_key}--{self.arm_key}-seed{self.seed}"
            f"-ckpt{self.checkpoint}{suffix}"
        )

    def model_path(self, task_key: str) -> str | None:
        """Where this target's trained weights live on the volume.

        The two models are served differently on purpose. Kimi is served from a
        merged checkpoint; Qwen is served as base weights plus a LoRA adapter,
        because merging a 30B model per checkpoint would cost ~60GB each and the
        adapter is ~50MB. Same contract to the caller either way.
        """
        if self.is_base:
            return None
        run_name = config.run_name(self.adapter_task or task_key, self.arm_key, self.seed)
        if self.model_key == config.QWEN3_OMNI.key:
            return config.SWIFT_ADAPTER_PATH_TEMPLATE.format(
                run_name=f"{run_name}-qwen3omni", checkpoint=self.checkpoint
            )
        return FT_MODEL_PATH_TEMPLATE.format(
            run_name=run_name, checkpoint=self.checkpoint
        )


def _remote_method(target: Target, task: config.Task):
    """Bind the Modal method for this target."""
    import modal

    model = target.model
    if target.is_base:
        remote_cls = modal.Cls.from_name(
            model.serving_app, model.serving_class,
            environment_name=config.MODAL_ENV,
        )
        return remote_cls().generate

    if not model.ft_serving_app:
        raise RuntimeError(
            f"{model.label} has no finetuned serving app configured"
        )
    remote_cls = modal.Cls.from_name(
        model.ft_serving_app, model.ft_serving_class,
        environment_name=config.MODAL_ENV,
    )
    # Kimi's server takes a merged checkpoint; Qwen's takes a LoRA adapter.
    # `Model.ft_path_param` names whichever this family expects.
    kwargs = {model.ft_path_param: target.model_path(task.key)}
    return remote_cls(**kwargs).generate


def _payload(example: dataset.Example, index: int) -> dict:
    """Build the serving payload. Clip order binds b64 order to part order."""
    audios = [
        {
            "data": base64.b64encode(path.read_bytes()).decode(),
            "format": path.suffix.lstrip(".").lower() or "wav",
        }
        for path in example.audio_paths
    ]

    return {
        "provider": "modal",
        "messages": dataset.wire_messages(example),
        "audios": audios,
        # Some adapters read the singular fields; keep them consistent.
        "audio_b64": audios[0]["data"],
        "audio_format": audios[0]["format"],
        "audio_paths": [str(path) for path in example.audio_paths],
        "meta": {
            # Caps generation on the finetuned server, which can otherwise hang
            # on exported checkpoints when max_new_tokens is unbounded.
            "task": "MQA",
            "example_id": example.example_id,
            "_eval_index": index,
        },
    }


def _score_one(method, example: dataset.Example, index: int) -> dict:
    """Send one question and grade the response."""
    started = time.perf_counter()
    record = {
        "index": index,
        "example_id": example.example_id,
        "dimension": example.dimension,
        "subset": dataset.subset_of(example),
        "answer_key": example.answer_key,
        "n_clips": len(example.audio_relpaths),
        "prompt_digest": example.prompt_digest(),
    }
    try:
        data = method.remote(item=_payload(example, index))
        response = data.get("response")
        predicted = dataset.extract_answer(response)
        record.update(
            {
                "response": response,
                "prompt": data.get("prompt"),
                "predicted": predicted,
                "correct": bool(predicted == example.answer_key),
                "parsed": predicted is not None,
                "status": "ok",
                "error": None,
            }
        )
    except Exception as exc:  # noqa: BLE001 - one bad item must not kill the run
        record.update(
            {
                "response": None,
                "prompt": None,
                "predicted": None,
                "correct": False,
                "parsed": False,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    record["wall_time_s"] = round(time.perf_counter() - started, 3)
    return record


def _summarize(task: config.Task, target: Target, records: list[dict]) -> dict:
    """Accuracy overall and per subset, plus the Table 2 comparison."""
    total = len(records)
    correct = sum(1 for r in records if r["correct"])
    parsed = sum(1 for r in records if r["parsed"])
    errors = sum(1 for r in records if r["status"] == "error")
    accuracy = 100.0 * correct / total if total else 0.0

    # Position bias is the tell for a broken pairwise prompt: balanced keys but
    # lopsided predictions. Reported always, so that failure can never hide.
    predicted = Counter(r["predicted"] for r in records if r["parsed"])
    gold = Counter(r["answer_key"] for r in records)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["subset"]].append(record)

    per_subset: dict[str, dict] = {}
    for subset, rows in sorted(grouped.items()):
        n = len(rows)
        subset_predicted = Counter(r["predicted"] for r in rows if r["parsed"])
        per_subset[subset] = {
            "n": n,
            "accuracy": round(100.0 * sum(1 for r in rows if r["correct"]) / n, 1),
            "gold_b_share": round(
                100.0 * sum(1 for r in rows if r["answer_key"] == "B") / n, 1
            ),
            "pred_b_share": round(
                100.0 * subset_predicted["B"] / max(1, sum(subset_predicted.values())), 1
            ),
        }

    reference = task.paper_base if target.is_base else task.paper_lora
    return {
        "task": task.key,
        "task_label": task.label,
        "model": target.model_key,
        "model_label": target.model.label,
        "arm": target.arm_key,
        "seed": target.seed,
        "checkpoint": target.checkpoint,
        "target_label": target.label_for(task.key),
        # Prompt variants (e.g. the clip-swap control) share task/model/arm with
        # the canonical run, so report.py needs this to keep them out of the
        # headline tables rather than silently overwriting the real number.
        "variant": target.variant_for(task.key),
        # Which task's adapter produced this run (empty = the evaluated task).
        "adapter_task": target.adapter_task or task.key,
        "n": total,
        "n_expected": task.eval_count,
        "correct": correct,
        "accuracy": round(accuracy, 1),
        "parsed": parsed,
        "unparsed": total - parsed,
        "errors": errors,
        "gold_a": gold["A"],
        "gold_b": gold["B"],
        "pred_a": predicted["A"],
        "pred_b": predicted["B"],
        "pred_b_share": round(
            100.0 * predicted["B"] / max(1, sum(predicted.values())), 1
        ),
        # Table 2 reports Kimi-Audio only, so a delta is meaningless for any
        # other model and is reported as None rather than a misleading number.
        "paper_reference": reference[0] if target.model_key == config.KIMI.key else None,
        "paper_reference_column": "Base" if target.is_base else "LoRA",
        "delta_vs_paper": (
            round(accuracy - reference[0], 1)
            if target.model_key == config.KIMI.key else None
        ),
        "per_subset": per_subset,
    }


def _print_summary(summary: dict) -> None:
    print(f"\n  {summary['task_label']}  ({summary['target_label']})")
    print(
        f"    accuracy      {summary['accuracy']:.1f}%  "
        f"({summary['correct']}/{summary['n']})"
    )
    if summary["paper_reference"] is not None:
        print(
            f"    paper {summary['paper_reference_column']:<5} "
            f"{summary['paper_reference']:.1f}%   "
            f"delta {summary['delta_vs_paper']:+.1f}"
        )
    gold_b_share = 100.0 * summary["gold_b"] / max(1, summary["n"])
    print(
        f"    predictions   A={summary['pred_a']} B={summary['pred_b']} "
        f"(B={summary['pred_b_share']:.0f}%, gold B={gold_b_share:.0f}%)"
    )
    if summary["unparsed"] or summary["errors"]:
        print(f"    unparsed      {summary['unparsed']}   errors {summary['errors']}")
    if len(summary["per_subset"]) > 1:
        print(f"    {'subset':<28}{'n':>5}{'acc':>8}{'predB':>8}")
        for subset, stats in summary["per_subset"].items():
            print(
                f"    {subset:<28}{stats['n']:>5}{stats['accuracy']:>8.1f}"
                f"{stats['pred_b_share']:>8.0f}"
            )


def _progress(done: int, total: int, correct: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    rate = done / elapsed if elapsed else 0.0
    eta_minutes = ((total - done) / rate / 60.0) if rate else 0.0
    filled = int(PROGRESS_WIDTH * done / total)
    bar = "#" * filled + "." * (PROGRESS_WIDTH - filled)
    print(
        f"\r    [{bar}] {done}/{total}  acc {100.0 * correct / done:5.1f}%  "
        f"eta {eta_minutes:4.1f}m",
        end="",
        flush=True,
    )


def evaluate(
    task: config.Task,
    target: Target,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    limit: int | None = None,
    system_prompt: str | None = None,
    swap_clips: bool = False,
) -> dict:
    """Evaluate one (task, target) pair and write results to disk."""
    examples = dataset.load(task, split="eval", system_prompt=system_prompt)
    if swap_clips:
        if task.clips_per_question != 2:
            raise ValueError(
                f"{task.label} has {task.clips_per_question} clip(s) per question; "
                "--swap-clips only applies to pairwise tasks"
            )
        examples = [dataset.swap_clip_order(example) for example in examples]
    if limit:
        examples = examples[:limit]

    out_dir = config.RESULTS_ROOT / "evals" / target.slug(task.key)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {task.label}: {target.label_for(task.key)} ===", flush=True)
    print(f"    {len(examples)} held-out items, concurrency {concurrency}", flush=True)
    print(f"    -> {out_dir.relative_to(REPO_ROOT)}", flush=True)
    print("    waiting for first response (cold start loads the model)...", flush=True)

    method = _remote_method(target, task)
    records: list[dict | None] = [None] * len(examples)
    correct = 0
    started = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_score_one, method, example, index): index
            for index, example in enumerate(examples)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records[futures[future]] = record
            correct += int(record["correct"])
            _progress(done, len(examples), correct, started)
    print()

    finished = [record for record in records if record is not None]
    summary = _summarize(task, target, finished)

    # A missing checkpoint makes every request fail, and the resulting 0.0 is a
    # perfectly well-formed summary. `_aggregate` pools seeds, so one such file
    # dragged the READOUT column from 96.3 to 48.1+/-136.2 and reported it as a
    # reproduction result. Predictions are still written for debugging, but the
    # summary that feeds the tables is not, so a broken run cannot look like a
    # finished one.
    n = summary["n"] or 1
    error_share = summary["errors"] / n
    unusable_share = (summary["errors"] + summary["unparsed"]) / n
    with (out_dir / "predictions.jsonl").open("w") as handle:
        for record in finished:
            handle.write(json.dumps(record) + "\n")
    if error_share > MAX_ERROR_SHARE or unusable_share > MAX_UNUSABLE_SHARE:
        raise RuntimeError(
            f"{task.label} / {target.label_for(task.key)}: "
            f"{summary['errors']}/{summary['n']} errored and "
            f"{summary['unparsed']}/{summary['n']} unparseable. No summary "
            f"written, so this cannot be pooled into the tables. Check that "
            f"{target.model_path(task.key)} exists on the volume."
        )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    _print_summary(summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(config.TASKS))
    parser.add_argument("--all", action="store_true", help="All four tasks.")
    parser.add_argument(
        "--model",
        default=config.DEFAULT_MODEL,
        choices=sorted(config.MODELS),
        help="Which model family to evaluate.",
    )
    parser.add_argument(
        "--swap-clips",
        action="store_true",
        help=(
            "Reverse the two clips, keeping labels in place, and flip the "
            "expected answer. Accuracy should hold if the model listens to both "
            "clips and invert if it is guessing by position. Pairwise tasks only."
        ),
    )
    parser.add_argument(
        "--arm",
        default=config.BASE_ARM_KEY,
        choices=[config.BASE_ARM_KEY, *sorted(config.ARMS)],
        help="'base' evaluates the un-finetuned model (Table 2 Base column).",
    )
    parser.add_argument("--seed", type=int, choices=config.SEEDS)
    parser.add_argument("--checkpoint", type=int, help="Checkpoint step to score.")
    parser.add_argument(
        "--adapter-task",
        choices=sorted(config.TASKS),
        help=(
            "Load this task's adapter instead of the evaluated task's, measuring "
            "transfer: trained on A, scored on B. Combine with --all to sweep one "
            "specialist across every task. The diagonal cell is skipped, since it "
            "is the canonical run and already scored."
        ),
    )
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, help="Score only the first N items.")
    parser.add_argument(
        "--with-observed-system-prompt",
        action="store_true",
        help=(
            "Prepend the expert system prompt seen in the pre-release training "
            "data. Absent from both Appendix D and the public release; use only "
            "to measure its effect."
        ),
    )
    args = parser.parse_args()

    if not args.task and not args.all:
        parser.error("pass --task <key> or --all")

    target = Target(
        model_key=args.model,
        arm_key=args.arm,
        seed=args.seed,
        checkpoint=args.checkpoint,
        variant="swapclips" if args.swap_clips else "",
        adapter_task=args.adapter_task or "",
    )
    if not target.is_base and (args.seed is None or args.checkpoint is None):
        parser.error(f"--arm {args.arm} requires --seed and --checkpoint")
    if args.adapter_task and target.is_base:
        parser.error("--adapter-task needs a trained arm; there is no base adapter")

    tasks = list(config.TASKS.values()) if args.all else [config.TASKS[args.task]]
    if args.adapter_task:
        kept = [task for task in tasks if task.key != args.adapter_task]
        if len(kept) != len(tasks):
            print(
                f"skipping {args.adapter_task}: the diagonal cell is the canonical "
                "run, already scored"
            )
        tasks = kept
        if not tasks:
            parser.error(
                f"--adapter-task {args.adapter_task} with --task {args.task} is the "
                "diagonal cell; nothing to transfer-evaluate"
            )
    system_prompt = (
        dataset.OBSERVED_RAD_SYSTEM_PROMPT if args.with_observed_system_prompt else None
    )

    summaries = [
        evaluate(
            task,
            target,
            concurrency=args.concurrency,
            limit=args.limit,
            system_prompt=system_prompt,
            swap_clips=args.swap_clips,
        )
        for task in tasks
    ]

    print(f"\n{'Task':<16}{'n':>6}{'ours':>8}{'paper':>8}{'delta':>8}")
    for summary in summaries:
        paper = summary["paper_reference"]
        delta = summary["delta_vs_paper"]
        paper_cell = f"{paper:>8.1f}" if paper is not None else f"{'--':>8}"
        delta_cell = f"{delta:>+8.1f}" if delta is not None else f"{'--':>8}"
        print(
            f"{summary['task_label']:<16}{summary['n']:>6}"
            f"{summary['accuracy']:>8.1f}{paper_cell}{delta_cell}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
