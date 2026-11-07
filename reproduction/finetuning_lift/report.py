"""Collect every eval summary on disk into the comparison tables.

Three tables, in the order the questions get asked:

  1. Reproduction  -- our base and LoRA numbers beside Table 2's, with deltas.
                      This is the gate: the baseline arm has to land on Table 2
                      before the ablation means anything.
  2. Ablation      -- baseline arm vs encoder-LoRA arm, the single-variable
                      comparison this whole package exists to produce.
  3. Checkpoints   -- accuracy per saved step, so a reproduction gap can be
                      attributed to step selection rather than confounded with
                      it (see the note on `Recipe.save_steps`).

Reads only `summary.json` files written by `evaluate.py`; runs nothing.

Usage:
    python -m reproduction.finetuning_lift.report
    python -m reproduction.finetuning_lift.report --markdown > REPORT.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduction.finetuning_lift import config  # noqa: E402

MISSING = "--"


def load_summaries(*, include_variants: bool = False) -> list[dict]:
    """Every eval summary written so far, in a stable order.

    Prompt-variant runs are excluded by default. A variant such as the clip-swap
    control shares (model, task, arm) with the canonical run, so including it
    silently overwrites the real number -- it did exactly that to Qwen's RAD STOCK
    figure, reporting the swapped 70.2 in place of 71.2.
    """
    root = config.RESULTS_ROOT / "evals"
    if not root.is_dir():
        return []
    summaries = []
    for path in sorted(root.glob("*/summary.json")):
        summary = json.loads(path.read_text())
        if summary.get("variant") and not include_variants:
            continue
        # Defence in depth: evaluate.py now refuses to write a summary for a run
        # where most requests failed, but any such file written before that guard
        # existed still scores 0.0 and would be pooled into the seed mean.
        n = summary.get("n") or 0
        unusable = summary.get("errors", 0) + summary.get("unparsed", 0)
        if n and unusable / n > 0.5:
            print(
                f"  skipping {path.parent.name}: {unusable}/{n} responses "
                "errored or were unparseable"
            )
            continue
        summaries.append(summary)
    return summaries


def _aggregate(summaries: list[dict]) -> dict[tuple[str, str, str], dict]:
    """Mean accuracy per (task, arm), pooled over seeds at the final checkpoint.

    Where several checkpoints were scored, the last one is used, matching the
    paper's reported configuration. `checkpoints()` shows the full curve.
    """
    by_key: dict[tuple[str, str, str], dict[int, dict]] = defaultdict(dict)
    for summary in summaries:
        # Results written before the model dimension existed are Kimi's.
        key = (summary.get("model", config.DEFAULT_MODEL), summary["task"], summary["arm"])
        seed = summary["seed"] if summary["seed"] is not None else 0
        checkpoint = summary["checkpoint"] or 0
        previous = by_key[key].get(seed)
        if previous is None or checkpoint >= (previous["checkpoint"] or 0):
            by_key[key][seed] = summary

    aggregated: dict[tuple[str, str, str], dict] = {}
    for key, per_seed in by_key.items():
        values = [s["accuracy"] for s in per_seed.values()]
        aggregated[key] = {
            "mean": statistics.fmean(values),
            # Table 2 reports +/-2 sample SD (Appendix L).
            "two_sd": 2 * statistics.stdev(values) if len(values) > 1 else 0.0,
            "n_seeds": len(values),
            "checkpoints": sorted(
                s["checkpoint"] for s in per_seed.values() if s["checkpoint"]
            ),
            "pred_b_share": statistics.fmean(
                s["pred_b_share"] for s in per_seed.values()
            ),
            "unparsed": sum(s["unparsed"] for s in per_seed.values()),
        }
    return aggregated


def _cell(entry: dict | None) -> str:
    if not entry:
        return MISSING
    if entry["n_seeds"] > 1:
        return f"{entry['mean']:.1f}±{entry['two_sd']:.1f}"
    return f"{entry['mean']:.1f}"


def _delta(entry: dict | None, reference: float) -> str:
    if not entry:
        return MISSING
    return f"{entry['mean'] - reference:+.1f}"


def reproduction_table(aggregated: dict, *, markdown: bool) -> list[str]:
    """Ours vs Table 2, for both the base model and the reproduced LoRA arm."""
    header = [
        "Task", "n", "STOCK", "paper", "Δ", "READOUT", "paper", "Δ", "Human",
    ]
    rows = [header]
    for task in config.TASKS.values():
        base = aggregated.get((config.KIMI.key, task.key, config.BASE_ARM_KEY))
        lora = aggregated.get((config.KIMI.key, task.key, config.BASELINE.key))
        rows.append([
            task.label,
            str(task.eval_count),
            _cell(base),
            f"{task.paper_base[0]:.1f}",
            _delta(base, task.paper_base[0]),
            _cell(lora),
            f"{task.paper_lora[0]:.1f}",
            _delta(lora, task.paper_lora[0]),
            f"{task.human:.1f}",
        ])
    return _render("1. Reproduction vs Table 2", rows, markdown=markdown)


def ablation_table(aggregated: dict, model: config.Model, *, markdown: bool) -> list[str]:
    """The single-variable comparison: frozen encoder vs adapted encoder."""
    header = [
        "Task", "n", "READOUT", "REWIRE", "Δ", "STOCK", "Lift (RO)", "Lift (RW)",
    ]
    rows = [header]
    for task in config.TASKS.values():
        base = aggregated.get((model.key, task.key, config.BASE_ARM_KEY))
        decoder = aggregated.get((model.key, task.key, config.BASELINE.key))
        encoder = aggregated.get((model.key, task.key, config.ENCODER_LORA.key))
        both = decoder and encoder
        rows.append([
            task.label,
            str(task.eval_count),
            _cell(decoder),
            _cell(encoder),
            f"{encoder['mean'] - decoder['mean']:+.1f}" if both else MISSING,
            _cell(base),
            f"{decoder['mean'] - base['mean']:+.1f}" if decoder and base else MISSING,
            f"{encoder['mean'] - base['mean']:+.1f}" if encoder and base else MISSING,
        ])
    return _render(
        f"2. Ablation on {model.label}: encoder LoRA "
        "(arms differ by exactly one field)",
        rows,
        markdown=markdown,
    )


def cross_model_table(
    aggregated: dict, models: list[config.Model], *, markdown: bool
) -> list[str]:
    """Every model side by side, so the encoder result can be read across them.

    The question this answers is whether adapting the audio encoder helps for
    reasons particular to one architecture, or generally.
    """
    header = ["Task", "n"]
    for model in models:
        header += [f"{model.key}:STOCK", f"{model.key}:READOUT", f"{model.key}:REWIRE", "Δ"]

    rows = [header]
    for task in config.TASKS.values():
        row = [task.label, str(task.eval_count)]
        for model in models:
            base = aggregated.get((model.key, task.key, config.BASE_ARM_KEY))
            decoder = aggregated.get((model.key, task.key, config.BASELINE.key))
            encoder = aggregated.get((model.key, task.key, config.ENCODER_LORA.key))
            row += [
                _cell(base),
                _cell(decoder),
                _cell(encoder),
                f"{encoder['mean'] - decoder['mean']:+.1f}"
                if decoder and encoder else MISSING,
            ]
        rows.append(row)
    return _render(
        "3. Encoder LoRA across architectures (Δ = REWIRE − READOUT)",
        rows,
        markdown=markdown,
    )


def checkpoint_table(summaries: list[dict], *, markdown: bool) -> list[str]:
    """Accuracy per saved checkpoint, so step selection stays visible."""
    trained = [s for s in summaries if s["checkpoint"]]
    if not trained:
        return []

    by_run: dict[tuple[str, str, str, int], dict[int, float]] = defaultdict(dict)
    for summary in trained:
        key = (
            summary.get("model", config.DEFAULT_MODEL),
            summary["task"],
            summary["arm"],
            summary["seed"],
        )
        by_run[key][summary["checkpoint"]] = summary["accuracy"]

    steps = sorted({s["checkpoint"] for s in trained})
    rows = [["Model", "Task", "Arm", "Seed", *[str(step) for step in steps]]]
    for (model_key, task_key, arm_key, seed), per_step in sorted(by_run.items()):
        rows.append([
            config.MODELS[model_key].label,
            config.TASKS[task_key].label,
            arm_key,
            str(seed),
            *[
                f"{per_step[step]:.1f}" if step in per_step else MISSING
                for step in steps
            ],
        ])
    return _render("4. Accuracy by checkpoint", rows, markdown=markdown)


def diagnostics(aggregated: dict) -> list[str]:
    """Surface the failure modes that previously masqueraded as results."""
    lines = ["", "Diagnostics", ""]
    flagged = False
    for (model_key, task_key, arm_key), entry in sorted(aggregated.items()):
        task = config.TASKS[task_key]
        model_label = config.MODELS[model_key].label
        notes = []
        # Balanced keys with lopsided predictions means the model is guessing by
        # position, which is what a dropped clip label looks like.
        if abs(entry["pred_b_share"] - 50.0) > 20.0:
            notes.append(
                f"position bias: predicts B {entry['pred_b_share']:.0f}% of the time"
            )
        if entry["unparsed"]:
            notes.append(f"{entry['unparsed']} unparseable response(s)")
        if entry["n_seeds"] < len(config.SEEDS) and arm_key != config.BASE_ARM_KEY:
            notes.append(f"only {entry['n_seeds']}/{len(config.SEEDS)} seeds scored")
        for note in notes:
            flagged = True
            lines.append(f"  {model_label} / {task.label} / {arm_key}: {note}")
    if not flagged:
        lines.append("  Nothing flagged.")
    return lines


def _render(title: str, rows: list[list[str]], *, markdown: bool) -> list[str]:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = ["", title, ""]
    if markdown:
        lines.append("| " + " | ".join(rows[0]) + " |")
        lines.append("|" + "|".join("---" for _ in rows[0]) + "|")
        for row in rows[1:]:
            lines.append("| " + " | ".join(row) + " |")
    else:
        for index, row in enumerate(rows):
            lines.append(
                "  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
            )
            if index == 0:
                lines.append("  " + "  ".join("-" * width for width in widths))
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--markdown", action="store_true", help="Emit markdown tables.")
    args = parser.parse_args()

    summaries = load_summaries()
    if not summaries:
        print("No eval summaries found. Run evaluate.py first.")
        return 1

    aggregated = _aggregate(summaries)

    models_present = [
        model for model in config.MODELS.values()
        if any(key[0] == model.key for key in aggregated)
    ]

    lines = [f"AcousticBench reproduction report ({len(summaries)} eval runs)"]
    lines += reproduction_table(aggregated, markdown=args.markdown)
    for model in models_present:
        lines += ablation_table(aggregated, model, markdown=args.markdown)
    if len(models_present) > 1:
        lines += cross_model_table(aggregated, models_present, markdown=args.markdown)
    lines += checkpoint_table(summaries, markdown=args.markdown)
    if not args.markdown:
        lines += diagnostics(aggregated)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
