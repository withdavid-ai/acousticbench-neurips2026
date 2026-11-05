"""Shared evaluation summaries and checkpoint-metrics row builders for Kimi RAD."""

from __future__ import annotations

from typing import Any

from finetuning.kimi_audio.common.run_layout import now_utc_iso

DEFAULT_CHECKPOINT_EVAL_METRIC_ROOT = "checkpoint_eval"


def checkpoint_eval_metric_prefix(
    *,
    eval_tag: str | None = None,
    wandb_metric_prefix: str | None = None,
) -> str:
    explicit_prefix = str(wandb_metric_prefix or "").strip().strip("/")
    if explicit_prefix:
        return explicit_prefix
    resolved_tag = str(eval_tag or "").strip().strip("/")
    if not resolved_tag:
        return DEFAULT_CHECKPOINT_EVAL_METRIC_ROOT
    return f"{DEFAULT_CHECKPOINT_EVAL_METRIC_ROOT}/{resolved_tag}"


def checkpoint_eval_metric_names(
    *,
    eval_tag: str | None = None,
    wandb_metric_prefix: str | None = None,
) -> dict[str, str]:
    prefix = checkpoint_eval_metric_prefix(
        eval_tag=eval_tag,
        wandb_metric_prefix=wandb_metric_prefix,
    )
    return {
        "checkpoint_step": f"{prefix}/checkpoint_step",
        "overall_accuracy_pct": f"{prefix}/overall_accuracy_pct",
        "accuracy_on_valid_pct": f"{prefix}/accuracy_on_valid_pct",
        "unparseable_or_unsure_pct": f"{prefix}/unparseable_or_unsure_pct",
        "step0_phase": f"{prefix}/step0_phase",
        "step0_wandb_ready": f"{prefix}/step0_wandb_ready",
    }


def simplified_checkpoint_eval_summary(summary_payload: dict[str, Any]) -> dict[str, Any]:
    total_examples = int(summary_payload.get("total_examples") or 0)
    accuracy_pct_total = summary_payload.get("accuracy_pct_total")
    accuracy_pct_parsed = summary_payload.get("accuracy_pct_parsed")
    choice_extraction_unsure_total = int(summary_payload.get("choice_extraction_unsure_total") or 0)
    choice_extraction_refusal_total = int(
        summary_payload.get("choice_extraction_refusal_total") or 0
    )
    unparseable_or_unsure_total = (
        choice_extraction_unsure_total + choice_extraction_refusal_total
    )
    return {
        "overall_accuracy_pct": accuracy_pct_total,
        "accuracy_on_valid_pct": accuracy_pct_parsed,
        "unparseable_or_unsure_pct": round(
            (unparseable_or_unsure_total / total_examples) * 100.0, 2
        )
        if total_examples
        else None,
    }


def summarize_performance_payload(performance_payload: dict[str, Any]) -> dict[str, Any]:
    task_payload = dict(performance_payload.get("performance") or {})
    total_examples = 0
    correct = 0
    choice_extraction_total = 0
    choice_extraction_deterministic_total = 0
    choice_extraction_llm_total = 0
    choice_extraction_unsure_total = 0
    choice_extraction_refusal_total = 0
    for task_metrics in task_payload.values():
        total_examples += int(task_metrics.get("total") or 0)
        correct += int(task_metrics.get("correct") or 0)
        choice_extraction_total += int(task_metrics.get("choice_extraction_total") or 0)
        choice_extraction_deterministic_total += int(
            task_metrics.get("choice_extraction_deterministic") or 0
        )
        choice_extraction_llm_total += int(task_metrics.get("choice_extraction_llm") or 0)
        choice_extraction_unsure_total += int(
            task_metrics.get("choice_extraction_unsure") or 0
        )
        choice_extraction_refusal_total += int(
            task_metrics.get("choice_extraction_refusal") or 0
        )
    parsed_eligible = max(
        0,
        total_examples - choice_extraction_unsure_total - choice_extraction_refusal_total,
    )
    summary = {
        "total_examples": total_examples,
        "valid_predictions": parsed_eligible,
        "correct": correct,
        "accuracy_pct_total": round((correct / total_examples) * 100.0, 2)
        if total_examples
        else None,
        "accuracy_pct_parsed": round((correct / parsed_eligible) * 100.0, 2)
        if parsed_eligible
        else None,
        "choice_extraction_total": choice_extraction_total,
        "choice_extraction_deterministic_total": choice_extraction_deterministic_total,
        "choice_extraction_llm_total": choice_extraction_llm_total,
        "choice_extraction_unsure_total": choice_extraction_unsure_total,
        "choice_extraction_refusal_total": choice_extraction_refusal_total,
    }
    summary.update(simplified_checkpoint_eval_summary(summary))
    return summary


def build_checkpoint_metrics_row(
    *,
    run_name: str,
    eval_name: str,
    repeat_index: int,
    benchmark_name: str,
    eval_method: str,
    model_name: str,
    checkpoint_dir: str,
    checkpoint_name: str,
    checkpoint_step: int,
    model_path_override: str | None,
    eval_tag: str | None = None,
    wandb_metric_prefix: str | None = None,
    persisted_checkpoint_dir: str | None = None,
    question_set_version: str | None,
    summary_path: str,
    combined_summary_path: str,
    summary_payload: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "run_name": run_name,
        "eval_name": eval_name,
        "repeat_index": repeat_index,
        "benchmark_name": benchmark_name,
        "eval_method": eval_method,
        "question_set_version": question_set_version,
        "model_name": model_name,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_name": checkpoint_name,
        "checkpoint_step": checkpoint_step,
        "model_path_override": model_path_override,
        "eval_tag": eval_tag,
        "wandb_metric_prefix": wandb_metric_prefix,
        "persisted_checkpoint_dir": persisted_checkpoint_dir,
        "summary_path": summary_path,
        "combined_summary_path": combined_summary_path,
        "created_at": now_utc_iso(),
    }
    for key in (
        "total_examples",
        "valid_predictions",
        "correct",
        "accuracy_pct_total",
        "accuracy_pct_parsed",
        "overall_accuracy_pct",
        "accuracy_on_valid_pct",
        "unparseable_or_unsure_pct",
        "choice_extraction_total",
        "choice_extraction_deterministic_total",
        "choice_extraction_llm_total",
        "choice_extraction_unsure_total",
        "choice_extraction_refusal_total",
    ):
        row[key] = summary_payload.get(key)
    return row
