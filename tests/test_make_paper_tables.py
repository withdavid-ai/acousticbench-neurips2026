"""Tests for scripts/make_paper_tables.py and scripts/compare_runs.py.

A tiny synthetic trajectory root is built in tmp_path with hand-checkable
numbers, so the aggregation rules (item-weighted RAD and SER, macro Overall,
sample sd, eval-split paired lift, Table 6 rates) are verified independently of
the shipped references. One integration test runs ``--check`` on the shipped
reference_trajectories when they are present.
"""

from __future__ import annotations

import json
import statistics
import sys
from fractions import Fraction
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import compare_runs  # noqa: E402
import make_paper_tables as mpt  # noqa: E402

CATEGORY = {
    "rad": "acoustic-perception",
    "rsq": "speech-quality",
    "ser_classify": "emotion-recognition",
    "ser_classify_text": "emotion-recognition",
    "ser_intensity": "emotion-recognition",
}
DIMENSION = {
    "rsq": "clean-vs-degraded-multilingual",
    "ser_classify": "primary-emotion-opposites-2way",
    "ser_classify_text": "primary-emotion-opposites-2way-text-only-gpt4o-transcript",
    "ser_intensity": "relative-emotional-intensity",
}
# rad items: 4 pitch, 2 loudness, 1 tempo, 1 dynamic-range (unequal on purpose,
# so item-weighted and dimension-macro averages differ).
RAD_DIMS = ["pitch"] * 4 + ["loudness"] * 2 + ["tempo"] + ["dynamic-range"]
SIZES = {"rad": 8, "rsq": 4, "ser_classify": 4, "ser_classify_text": 4, "ser_intensity": 2}

# Per-run correctness for Kimi-Audio (1 = correct). Index i is item task-i.
KIMI = {
    "rad": [[1, 1, 1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 1, 0, 1, 1], [1, 0, 0, 0, 0, 0, 1, 0]],
    "rsq": [[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
    "ser_classify": [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0]],
    "ser_intensity": [[1, 1], [1, 1], [1, 0]],
    "ser_classify_text": [[1, 1, 1, 1]] * 3,
}


def item_id(task: str, index: int) -> str:
    return f"{task}-{index:03d}"


def split_of(index: int) -> str:
    return "eval" if index % 2 == 0 else "train"


def make_row(task: str, index: int, correct: int, *, chosen: str | None = "use-correct",
             status: str = "ok", response: str = "A", with_split: bool = True) -> dict:
    key = "A" if index % 2 == 0 else "B"
    other = "B" if key == "A" else "A"
    if chosen == "use-correct":
        chosen = key if correct else other
    meta = {
        "example_id": item_id(task, index),
        "question_order": index,
        "category": CATEGORY[task],
        "dimension": RAD_DIMS[index] if task == "rad" else DIMENSION[task],
        "answer_key": key,
        "task": "MQA",
    }
    if with_split:
        meta["split"] = split_of(index)
    return {
        "example_id": item_id(task, index),
        "response_text": response,
        "response_status": status,
        "chosen": chosen,
        "correct": key,
        "is_correct": chosen == key,
        "extraction_source": "llm",
        "meta": meta,
    }


def write_runs(root: Path, model: str, task: str, runs: list[list[dict]], split: str | None = None) -> None:
    task_dir = root / model / task
    task_dir.mkdir(parents=True, exist_ok=True)
    for k, rows in enumerate(runs, start=1):
        with (task_dir / f"run_{k}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    summary = {"model": model, "task": task, "n_runs": len(runs)}
    if split:
        summary["split"] = split
    (task_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def build_root(root: Path) -> Path:
    for task, runs in KIMI.items():
        write_runs(root, "Kimi-Audio", task, [
            [make_row(task, i, c) for i, c in enumerate(run)] for run in runs
        ])
    # Specialist LoRA on the eval half of rad only (even indices), all correct.
    eval_idx = [i for i in range(SIZES["rad"]) if split_of(i) == "eval"]
    write_runs(root, "Kimi-Audio-LoRA-RAD", "rad",
               [[make_row("rad", i, 1) for i in eval_idx]] * 3, split="eval")
    # Gemini-3.1-Pro: all correct except one API error and one refusal in run_1 rad.
    for task in ("rad", "rsq", "ser_classify", "ser_intensity"):
        runs = []
        for k in range(3):
            rows = [make_row(task, i, 1) for i in range(SIZES[task])]
            if task == "rad" and k == 0:
                rows[0] = make_row("rad", 0, 0, chosen=None, status="error", response="")
                rows[1] = make_row("rad", 1, 0, chosen=None, response="Both clips are identical.")
            runs.append(rows)
        write_runs(root, "Gemini-3.1-Pro", task, runs)
    # Cascade ser_classify holding text-diagnostic items (the known release issue).
    write_runs(root, "Cascade-GPT4oTranscribe-GPT5.4", "ser_classify",
               [[make_row("ser_classify_text", i, 1) for i in range(4)]] * 3)
    return root


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


@pytest.fixture()
def built(tmp_path: Path):
    root = build_root(tmp_path / "trajectories")
    data, groups, expected, diag = mpt.build(root)
    return root, data, groups, expected, diag


def test_rad_is_item_weighted_and_dimension_columns(built) -> None:
    _, data, *_ = built
    stats = data.main["Kimi-Audio"]
    rad_runs = [100 * sum(run) / 8 for run in KIMI["rad"]]
    assert stats["rad"].per_run == pytest.approx(rad_runs)
    # Dimension-macro would give run_1 = mean(100, 50, 0, 100) = 62.5, not 75.
    assert rad_runs[0] == 75.0
    pitch_runs = [100 * sum(run[:4]) / 4 for run in KIMI["rad"]]
    assert stats["rad_pitch"].per_run == pytest.approx(pitch_runs)
    assert stats["rad_loudness"].per_run == pytest.approx([100 * sum(run[4:6]) / 2 for run in KIMI["rad"]])
    assert stats["rad_tempo"].per_run == pytest.approx([100.0 * run[6] for run in KIMI["rad"]])
    assert stats["rad_dynamics"].per_run == pytest.approx([100.0 * run[7] for run in KIMI["rad"]])


def test_ser_pools_items_and_excludes_text(built) -> None:
    _, data, *_ = built
    stats = data.main["Kimi-Audio"]
    ser_runs = [
        100 * (sum(c) + sum(i)) / (len(c) + len(i))
        for c, i in zip(KIMI["ser_classify"], KIMI["ser_intensity"])
    ]
    assert stats["ser"].per_run == pytest.approx(ser_runs)
    assert ser_runs[0] == pytest.approx(50.0)  # macro of (25, 100) would be 62.5
    assert stats["ser_classify_text"].mean == pytest.approx(100.0)


def test_overall_is_macro_of_rad_ser_rsq_with_sample_sd(built) -> None:
    _, data, *_ = built
    stats = data.main["Kimi-Audio"]
    rad = [100 * sum(run) / 8 for run in KIMI["rad"]]
    rsq = [100 * sum(run) / 4 for run in KIMI["rsq"]]
    ser = [100 * (sum(c) + sum(i)) / 6 for c, i in zip(KIMI["ser_classify"], KIMI["ser_intensity"])]
    overall = [(a + b + c) / 3 for a, b, c in zip(rad, ser, rsq)]
    mean, sd = mean_sd(overall)
    assert stats["overall"].mean == pytest.approx(mean)
    assert stats["overall"].err == pytest.approx(2 * sd)  # sample sd (ddof=1)
    assert stats["overall"].err != pytest.approx(2 * statistics.pstdev(overall))
    assert stats["overall"].exact_mean == sum(
        (Fraction(100 * sum(a), 8) + Fraction(100 * (sum(c) + sum(i)), 6) + Fraction(100 * sum(r), 4)) / 3
        for a, c, i, r in zip(KIMI["rad"], KIMI["ser_classify"], KIMI["ser_intensity"], KIMI["rsq"])
    ) / 3


def test_eval_split_lift_is_paired(built) -> None:
    _, data, _, expected, _ = built
    assert expected.eval["rad"] == frozenset(item_id("rad", i) for i in range(0, 8, 2))
    entry = data.lift["rad"]
    base_runs = [100 * sum(run[i] for i in range(0, 8, 2)) / 4 for run in KIMI["rad"]]
    assert entry["base"].per_run == pytest.approx(base_runs)
    assert entry["lora"].per_run == pytest.approx([100.0] * 3)
    lifts = [100.0 - b for b in base_runs]
    assert entry["lift"].mean == pytest.approx(statistics.mean(lifts))
    assert entry["lift"].err == pytest.approx(2 * statistics.stdev(lifts))
    assert entry["n_eval"] == 4


def test_table6_rates(built) -> None:
    _, data, *_ = built
    rad = data.parse["Gemini-3.1-Pro"]["rad"]
    assert rad["rows"] == 24 and rad["n_questions"] == 8
    assert (rad["parsed"], rad["refusal"], rad["api"]) == (22, 1, 1)
    assert rad["api_pct"] == pytest.approx(100 / 24)
    assert rad["refusal_by_dimension"] == {"pitch": 1}
    assert data.parse["Gemini-3.1-Pro"]["rsq"]["parsed_pct"] == pytest.approx(100.0)


def test_judge_error_counts_as_api() -> None:
    row = make_row("rad", 0, 0, chosen=None, response="Clip A is louder.")
    row["extraction_source"] = "judge_error"
    group = mpt.Group("M", "rad", "all", [mpt.Run("run_1", [row], "x")], "x")
    counts = mpt.parse_counts(group)
    assert (counts["parsed"], counts["refusal"], counts["api"]) == (0, 0, 1)


def test_cascade_style_id_mismatch_is_warned(built) -> None:
    *_, diag = built
    joined = "\n".join(diag.warnings)
    assert "Cascade-GPT4oTranscribe-GPT5.4/ser_classify" in joined
    assert "belong to ser_classify_text" in joined


def test_results_layout_with_task_all_and_seed_order(tmp_path: Path) -> None:
    results = tmp_path / "results"
    for seed, correct in ((43, 0), (42, 1)):
        run_dir = results / f"Kimi-Audio-all-docker-seed{seed}-1700000000"
        run_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps(
            {"model": "Kimi-Audio", "task": "all", "split": "all", "seed": seed}), encoding="utf-8")
        rows = [make_row("rad", i, correct) for i in range(8)] + [make_row("rsq", i, 1) for i in range(4)]
        (run_dir / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    groups = mpt.load_groups(results, mpt.Diagnostics())
    assert set(groups) == {("Kimi-Audio", "rad", "all"), ("Kimi-Audio", "rsq", "all")}
    rad = groups[("Kimi-Audio", "rad", "all")]
    assert [run.seed for run in rad.runs] == [42, 43]
    assert [run.name for run in rad.runs] == ["run_1", "run_2"]
    assert mpt.cell_stat(rad).per_run == pytest.approx([100.0, 0.0])


@pytest.mark.parametrize(
    ("paper", "value", "exact", "status"),
    [
        ("51.7", 207 / 400 * 100.0, Fraction(5175, 100), "OK"),   # float prints 51.7; exact tie noted
        ("51.7", 51.75, Fraction(5175, 100), "TIE"),              # prints 51.8 but is an exact tie
        ("51.8", 51.75, Fraction(5175, 100), "OK"),
        ("51.7", 51.80, None, "MISMATCH"),
        ("51.7", None, None, "MISSING"),
        (None, None, None, "OK"),
        ("+2.8", 2.75, Fraction(11, 4), "OK"),
    ],
)
def test_compare_cell_rounding(paper, value, exact, status) -> None:
    signed = bool(paper and paper.startswith("+"))
    cell = mpt.compare_cell("T", "row", "col", paper, value, exact, signed=signed)
    assert cell.status == status
    if exact is not None and status in ("OK", "TIE") and paper in ("51.7", "51.8"):
        assert "tie" in cell.note


def test_cli_writes_outputs(tmp_path: Path) -> None:
    root = build_root(tmp_path / "trajectories")
    out = tmp_path / "out"
    assert mpt.main(["--root", str(root), "--out", str(out)]) == 0
    for stem in ("table1", "table2", "table3", "table6", "table7", "table10", "table11"):
        assert (out / f"{stem}.md").is_file()
        assert "\\toprule" in (out / f"{stem}.tex").read_text(encoding="utf-8")
    payload = json.loads((out / "paper_tables.json").read_text(encoding="utf-8"))
    kimi = payload["tables"]["table1_table7"]["Kimi-Audio"]
    assert kimi["rad"]["per_run"] == pytest.approx([100 * sum(run) / 8 for run in KIMI["rad"]])
    assert any("ser_classify_text" in w for w in payload["warnings"])
    # A synthetic root cannot reproduce the paper.
    assert mpt.main(["--root", str(root), "--no-write", "--check"]) == 1


@pytest.mark.skipif(not (REPO_ROOT / "reference_trajectories").is_dir(),
                    reason="shipped reference_trajectories not present")
def test_check_passes_on_shipped_references(capsys) -> None:
    assert mpt.main(["--no-write", "--check"]) == 0
    out = capsys.readouterr().out
    assert "--check: PASS" in out
    assert "0 mismatches, 0 missing" in out


# ---------------------------------------------------------------------------
# compare_runs.py
# ---------------------------------------------------------------------------


def _copy_model(src_root: Path, dst_root: Path, model: str) -> None:
    for task_dir in (src_root / model).iterdir():
        target = dst_root / model / task_dir.name
        target.mkdir(parents=True, exist_ok=True)
        for path in task_dir.iterdir():
            (target / path.name).write_bytes(path.read_bytes())


def test_compare_runs_identical_passes(tmp_path: Path) -> None:
    refs = build_root(tmp_path / "refs")
    results = compare_runs.compare(refs, refs)
    assert results and all(r.status == "PASS" for r in results)


def test_compare_runs_key_flip_is_excluded_from_agreement(tmp_path: Path) -> None:
    refs = build_root(tmp_path / "refs")
    cand = tmp_path / "cand"
    _copy_model(refs, cand, "Kimi-Audio")
    # Same model answers, but item 3 has the opposite option order in the candidate
    # archive: key and chosen letter both flip, correctness is unchanged.
    task_dir = cand / "Kimi-Audio" / "ser_classify"
    for path in task_dir.glob("run_*.jsonl"):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["example_id"] == item_id("ser_classify", 3):
                flip = {"A": "B", "B": "A"}
                row["correct"] = flip[row["correct"]]
                row["chosen"] = flip[row["chosen"]]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    results = {(r.model, r.task): r for r in compare_runs.compare(cand, refs)}
    cell = results[("Kimi-Audio", "ser_classify")]
    assert cell.key_mismatch_items == 1
    assert cell.key_mismatch_is_correct_agreement is not None
    assert cell.status == "PASS"


def test_compare_runs_flags_mean_shift_and_no_overlap(tmp_path: Path) -> None:
    refs = build_root(tmp_path / "refs")
    cand = tmp_path / "cand"
    write_runs(cand, "Kimi-Audio", "rsq", [[make_row("rsq", i, 0) for i in range(4)]] * 3)
    write_runs(cand, "Cascade-GPT4oTranscribe-GPT5.4", "ser_classify",
               [[make_row("ser_classify", i, 1) for i in range(4)]] * 3)
    write_runs(cand, "Some-New-Model", "rad", [[make_row("rad", i, 1) for i in range(8)]] * 3)
    results = {(r.model, r.task): r for r in compare_runs.compare(cand, refs)}
    assert results[("Kimi-Audio", "rsq")].status == "FAIL_MEAN"
    assert results[("Cascade-GPT4oTranscribe-GPT5.4", "ser_classify")].status == "NO_OVERLAP"
    assert "known" in results[("Cascade-GPT4oTranscribe-GPT5.4", "ser_classify")].notes[0]
    assert results[("Some-New-Model", "rad")].status == "NO_REF"
    assert compare_runs.main(["--candidate", str(cand), "--refs", str(refs)]) == 1
