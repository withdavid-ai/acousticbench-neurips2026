"""Compute evaluation metrics including Consistent Accuracy.

All metrics operate on lists of per-question result dicts with at minimum:
    {"example_id": str, "chosen": str | None, "correct": str, "is_correct": bool}
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .position_swap import classify_pair


def raw_accuracy(results: list[dict]) -> dict[str, Any]:
    """Compute single-pass accuracy with per-answer-key breakdown."""
    n = len(results)
    if n == 0:
        return {"accuracy": 0.0, "n": 0}

    correct = sum(1 for r in results if r.get("is_correct"))

    by_key: dict[str, list[bool]] = {}
    for r in results:
        key = r.get("correct", "?")
        by_key.setdefault(key, []).append(bool(r.get("is_correct")))

    per_key = {
        k: round(sum(v) / len(v) * 100, 1) if v else 0.0
        for k, v in sorted(by_key.items())
    }

    return {
        "accuracy": round(correct / n * 100, 1),
        "correct": correct,
        "n": n,
        "per_answer_key": per_key,
    }


def consistent_accuracy(
    original_results: list[dict],
    swapped_results: list[dict],
    seed: int = 42,
    n_bootstrap: int = 10_000,
    ci_level: float = 0.95,
) -> dict[str, Any]:
    """Compute Consistent Accuracy and related position-swap metrics.

    Pairs original and swapped results by example_id, classifies each pair,
    and computes CA, position consistency, bias direction, and bootstrap CIs.

    Args:
        original_results: Results from the original audio ordering.
        swapped_results: Results from the position-swapped ordering.
        seed: Random seed for bootstrap CIs.
        n_bootstrap: Number of bootstrap iterations.
        ci_level: Confidence level for CIs (e.g. 0.95 for 95%).

    Returns:
        Dict with CA, CIs, position consistency, bias direction, and
        per-category breakdowns.
    """
    swap_by_id = {}
    for r in swapped_results:
        # Only rows produced by swap_question carry original_example_id;
        # anything else has no swapped counterpart and is not paired.
        eid = (r.get("meta") or {}).get("original_example_id")
        if not eid:
            continue
        swap_by_id[eid] = r

    pairs = []
    for r in original_results:
        eid = r["meta"]["example_id"]
        if eid not in swap_by_id:
            continue
        sr = swap_by_id[eid]
        pairs.append({
            "example_id": eid,
            "orig_correct": bool(r.get("is_correct")),
            "orig_choice": r.get("chosen", ""),
            "swap_correct": bool(sr.get("is_correct")),
            "swap_choice": sr.get("chosen", ""),
        })

    n = len(pairs)
    if n == 0:
        return {"consistent_accuracy": 0.0, "n": 0, "paired": 0}

    categories = [
        classify_pair(
            p["orig_correct"], p["swap_correct"],
            p["orig_choice"], p["swap_choice"],
        )
        for p in pairs
    ]

    cc = sum(1 for c in categories if c == "consistent-correct")
    cw = sum(1 for c in categories if c == "consistent-wrong")
    ip = sum(1 for c in categories if c == "inconsistent-primacy")
    ir = sum(1 for c in categories if c == "inconsistent-recency")
    im = sum(1 for c in categories if c == "inconsistent-mixed")
    inconsistent = ip + ir + im

    ca = cc / n
    ca_arr = np.array([1 if c == "consistent-correct" else 0 for c in categories])
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(ca_arr, size=n, replace=True)
        boot[i] = sample.mean()
    alpha = (1 - ci_level) / 2
    ci_lo = float(np.percentile(boot, alpha * 100))
    ci_hi = float(np.percentile(boot, (1 - alpha) * 100))

    orig_correct = sum(1 for p in pairs if p["orig_correct"])
    swap_correct = sum(1 for p in pairs if p["swap_correct"])

    return {
        "consistent_accuracy": round(ca * 100, 1),
        "ci_lo": round(ci_lo * 100, 1),
        "ci_hi": round(ci_hi * 100, 1),
        "raw_accuracy": round(orig_correct / n * 100, 1),
        "swap_accuracy": round(swap_correct / n * 100, 1),
        "position_consistency": round((cc + cw) / n, 3),
        "inconsistency_rate": round(inconsistent / n * 100, 1),
        "positional_bias_direction": round(ip / inconsistent, 2) if inconsistent else None,
        "n": n,
        "n_consistent_correct": cc,
        "n_consistent_wrong": cw,
        "n_inconsistent_primacy": ip,
        "n_inconsistent_recency": ir,
        "n_inconsistent_mixed": im,
    }
