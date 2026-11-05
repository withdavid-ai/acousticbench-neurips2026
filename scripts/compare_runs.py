#!/usr/bin/env python3
"""Compare a rerun against the reference trajectories, cell by cell.

For every (model, task, split) cell in ``--candidate`` (a ``run.py`` results/
directory, a reference_trajectories-style root, or a mix) this reports:

  * the mean accuracy of the candidate runs and of the reference runs on the
    items both cover, and the difference (candidate minus reference)
  * item-level agreement on ``chosen`` between every candidate run and every
    reference run (mean and min over run pairs)
  * the reference's own run-to-run agreement on the same items (min over
    reference run pairs), used as the agreement baseline

and a verdict per cell (reproduction plan section 4, B2 tier T3):

  PASS            |delta mean| <= max(--min-tolerance, published +-2 sd) and the
                  mean cross agreement >= the reference's min run-to-run
                  agreement minus --agreement-slack
  FAIL_MEAN       the accuracy difference is outside the tolerance
  FAIL_AGREEMENT  only the item agreement is below the baseline
  NO_OVERLAP      the candidate and the reference share no example_id
  NO_REF          no reference directory for this (model, task); not a failure

The published +-2 sd comes from paper Table 7 (main cells), Table 10 (LoRA
diagonal and base Kimi-Audio on eval) or Table 3 (off-task LoRAs); cells not in
the paper fall back to the reference's own recomputed +-2 sd.

Items whose answer key differs between candidate and reference (``correct``
fields disagree, e.g. the 20 SER-Classify items with a stale A/B order in 15
models' references) are excluded from ``chosen`` agreement, because the same
answer is a different letter there. They are counted in the KeyDiff column and
compared on ``is_correct`` instead.

Usage:
    python scripts/compare_runs.py --candidate results
    python scripts/compare_runs.py --candidate results --models Qwen3-Omni --tasks rad rsq \\
        --json-out compare.json

Exit status: 0 if every compared cell passes, 1 otherwise, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_paper_tables import (  # noqa: E402
    CASCADE_DIR,
    Diagnostics,
    Group,
    Groups,
    Stat,
    cell_stat,
    load_groups,
    load_trajectory_layout,
    paper_error_bar,
    row_example_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FAILING = {"FAIL_MEAN", "FAIL_AGREEMENT", "NO_OVERLAP"}


@dataclass
class CellResult:
    model: str
    task: str
    split: str
    ref_split: str | None
    status: str
    n_common: int = 0
    n_candidate_ids: int = 0
    n_reference_ids: int = 0
    candidate_runs: int = 0
    reference_runs: int = 0
    candidate_mean: float | None = None
    reference_mean: float | None = None
    delta: float | None = None
    tolerance: float | None = None
    tolerance_source: str = ""
    agreement_mean: float | None = None
    agreement_min: float | None = None
    reference_agreement_min: float | None = None
    reference_agreement_mean: float | None = None
    key_mismatch_items: int = 0
    key_mismatch_is_correct_agreement: float | None = None
    notes: list[str] = field(default_factory=list)


def _index(group: Group) -> list[dict[str, dict[str, Any]]]:
    return [{row_example_id(row): row for row in run.rows} for run in group.runs]


def _agreement(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]],
               ids: set[str]) -> float | None:
    shared = [eid for eid in ids if eid in a and eid in b]
    if not shared:
        return None
    return 100.0 * sum(a[eid].get("chosen") == b[eid].get("chosen") for eid in shared) / len(shared)


def _is_correct_agreement(a: dict[str, dict[str, Any]], b: dict[str, dict[str, Any]],
                          ids: set[str]) -> float | None:
    shared = [eid for eid in ids if eid in a and eid in b]
    if not shared:
        return None
    return 100.0 * sum(bool(a[eid].get("is_correct")) == bool(b[eid].get("is_correct"))
                       for eid in shared) / len(shared)


def _pick_reference(refs: Groups, model: str, task: str, split: str) -> Group | None:
    if (model, task, split) in refs:
        return refs[(model, task, split)]
    for ref_split in ("all", "eval", "train"):
        if (model, task, ref_split) in refs:
            return refs[(model, task, ref_split)]
    return None


def compare_cell(candidate: Group, reference: Group | None, min_tolerance: float,
                 agreement_slack: float) -> CellResult:
    result = CellResult(candidate.model, candidate.task, candidate.split,
                        reference.split if reference else None, "NO_REF",
                        candidate_runs=len(candidate.runs))
    cand_ids = set(candidate.id_set())
    result.n_candidate_ids = len(cand_ids)
    if reference is None:
        result.notes.append("no reference trajectories for this model/task")
        return result
    ref_ids = set(reference.id_set())
    result.reference_runs = len(reference.runs)
    result.n_reference_ids = len(ref_ids)
    common = cand_ids & ref_ids
    result.n_common = len(common)
    if not common:
        result.status = "NO_OVERLAP"
        if candidate.model == CASCADE_DIR and candidate.task == "ser_classify":
            result.notes.append(
                "known: the Cascade ser_classify reference holds the ser_classify_text items "
                "(paper 42.9 was computed on those), so an audio rerun cannot be compared"
            )
        else:
            result.notes.append("check --split and the archive version")
        return result
    if common != ref_ids:
        result.notes.append(f"candidate covers {len(common)} of {len(ref_ids)} reference items")
    if result.candidate_runs < 3:
        result.notes.append(f"only {result.candidate_runs} candidate run(s)")

    frozen = frozenset(common)
    cand_stat: Stat | None = cell_stat(candidate, ids=frozen)
    ref_stat: Stat | None = cell_stat(reference, ids=frozen)
    assert cand_stat is not None and ref_stat is not None
    result.candidate_mean = cand_stat.mean
    result.reference_mean = ref_stat.mean
    result.delta = cand_stat.mean - ref_stat.mean

    # Which published cell (if any) these items correspond to.
    if common == ref_ids:
        effective_split: str | None = reference.split
    elif candidate.split == "eval" and common == cand_ids:
        effective_split = "eval"
    else:
        effective_split = None
    paper = (paper_error_bar(reference.model, reference.task, effective_split)
             if effective_split in ("all", "eval") else None)
    if paper is not None:
        result.tolerance, result.tolerance_source = max(min_tolerance, paper), "paper"
    else:
        result.tolerance, result.tolerance_source = max(min_tolerance, ref_stat.err), "ref"
    if result.tolerance == min_tolerance:
        result.tolerance_source = "floor"

    cand_index = _index(candidate)
    ref_index = _index(reference)
    key_mismatch = {
        eid for eid in common
        if any(eid in c and eid in r and c[eid].get("correct") != r[eid].get("correct")
               for c in cand_index for r in ref_index)
    }
    result.key_mismatch_items = len(key_mismatch)
    comparable = common - key_mismatch
    cross = [value for c, r in itertools.product(cand_index, ref_index)
             if (value := _agreement(c, r, comparable)) is not None]
    within = [value for a, b in itertools.combinations(ref_index, 2)
              if (value := _agreement(a, b, comparable)) is not None]
    if cross:
        result.agreement_mean = statistics.mean(cross)
        result.agreement_min = min(cross)
    if within:
        result.reference_agreement_min = min(within)
        result.reference_agreement_mean = statistics.mean(within)
    if key_mismatch:
        flips = [value for c, r in itertools.product(cand_index, ref_index)
                 if (value := _is_correct_agreement(c, r, key_mismatch)) is not None]
        result.key_mismatch_is_correct_agreement = statistics.mean(flips) if flips else None
        result.notes.append(
            f"{len(key_mismatch)} items have a different answer key in the reference; "
            "excluded from chosen agreement"
        )

    mean_ok = abs(result.delta) <= result.tolerance + 1e-9
    agreement_ok = True
    if result.reference_agreement_min is None:
        result.notes.append("reference has one run; agreement baseline unavailable")
    elif result.agreement_mean is not None:
        agreement_ok = result.agreement_mean + 1e-9 >= result.reference_agreement_min - agreement_slack
    if not mean_ok:
        result.status = "FAIL_MEAN"
    elif not agreement_ok:
        result.status = "FAIL_AGREEMENT"
    else:
        result.status = "PASS"
    return result


def compare(candidate_root: Path, refs_root: Path, models: list[str] | None = None,
            tasks: list[str] | None = None, min_tolerance: float = 1.0,
            agreement_slack: float = 0.0) -> list[CellResult]:
    diag = Diagnostics()
    candidates = load_groups(candidate_root, diag)
    references = load_trajectory_layout(refs_root, diag)
    results = []
    for (model, task, split), group in sorted(candidates.items()):
        if models and model not in models:
            continue
        if tasks and task not in tasks:
            continue
        reference = _pick_reference(references, model, task, split)
        results.append(compare_cell(group, reference, min_tolerance, agreement_slack))
    return results


def _f(value: float | None, digits: int = 2) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def print_results(results: list[CellResult]) -> None:
    headers = ("Model", "Task", "Split", "Runs c/r", "N", "Cand %", "Ref %", "Δ pp", "Tol pp",
               "Agree mean/min %", "Ref agree min %", "KeyDiff", "Status")
    rows = []
    for r in results:
        rows.append((
            r.model, r.task, r.split if r.ref_split in (None, r.split) else f"{r.split}->{r.ref_split}",
            f"{r.candidate_runs}/{r.reference_runs}", str(r.n_common), _f(r.candidate_mean),
            _f(r.reference_mean), "–" if r.delta is None else f"{r.delta:+.2f}",
            "–" if r.tolerance is None else f"{r.tolerance:.2f} ({r.tolerance_source})",
            f"{_f(r.agreement_mean, 1)}/{_f(r.agreement_min, 1)}", _f(r.reference_agreement_min, 1),
            str(r.key_mismatch_items), r.status,
        ))
    widths = [max(len(h), *(len(row[i]) for row in rows)) if rows else len(h) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(value.ljust(w) for value, w in zip(row, widths)).rstrip())
    noted = [r for r in results if r.notes]
    if noted:
        print()
        for r in noted:
            print(f"{r.model}/{r.task} ({r.split}): " + "; ".join(r.notes))
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print()
    print("Summary: " + ", ".join(f"{status} {n}" for status, n in sorted(counts.items())))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare rerun trajectories against reference_trajectories per (model, task).",
    )
    parser.add_argument("--candidate", type=Path, required=True,
                        help="results/ directory from run.py, or a reference_trajectories-style root")
    parser.add_argument("--refs", type=Path, default=REPO_ROOT / "reference_trajectories",
                        help="Reference trajectories root (default: <repo>/reference_trajectories)")
    parser.add_argument("--models", nargs="*", default=None, help="Only these model names")
    parser.add_argument("--tasks", nargs="*", default=None, help="Only these tasks")
    parser.add_argument("--min-tolerance", type=float, default=1.0,
                        help="Floor for the |delta mean| tolerance in points (default: %(default)s)")
    parser.add_argument("--agreement-slack", type=float, default=0.0,
                        help="Allowed shortfall (points) of mean cross agreement below the reference's "
                             "min run-to-run agreement (default: %(default)s)")
    parser.add_argument("--json-out", type=Path, default=None, help="Write per-cell results as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path in (args.candidate, args.refs):
        if not path.is_dir():
            logger.error(f"Directory not found: {path}")
            return 2
    results = compare(args.candidate, args.refs, args.models, args.tasks,
                      args.min_tolerance, args.agreement_slack)
    if not results:
        logger.error(f"No candidate trajectories found under {args.candidate}")
        return 2
    print_results(results)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps([asdict(r) for r in results], indent=2) + "\n", encoding="utf-8")
        logger.info(f"Wrote {args.json_out}")
    return 1 if any(r.status in FAILING for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
