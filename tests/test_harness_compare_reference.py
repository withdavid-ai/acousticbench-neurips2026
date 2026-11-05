"""--compare-reference: repo-root resolution, per-sub-task lookup, seed mapping."""

from __future__ import annotations

import json
from pathlib import Path

from harness import cli


def _write_run(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps({"example_id": e, "chosen": c}) + "\n" for e, c in rows)
    )


def _q(example_id: str, subset: str) -> dict:
    return {"meta": {"example_id": example_id}, "subset": subset}


def _r(example_id: str, chosen: str) -> dict:
    return {"example_id": example_id, "chosen": chosen}


def test_reference_root_is_resolved_from_repo_not_cwd(monkeypatch, tmp_path):
    expected = Path(cli.__file__).resolve().parents[1] / "reference_trajectories"
    monkeypatch.chdir(tmp_path)
    assert cli.REFERENCE_ROOT.is_absolute()
    assert cli.REFERENCE_ROOT == expected


def test_run_index_mapping_when_reference_has_no_seeds(tmp_path):
    ref_dir = tmp_path / "M" / "rad"
    ref_dir.mkdir(parents=True)
    (ref_dir / "summary.json").write_text(
        json.dumps({"per_run": {"run_1": {"accuracy": 50}}})
    )
    path, mapping = cli._resolve_reference_run(ref_dir, run_seed=43, run_index=1)
    assert path == ref_dir / "run_2.jsonl"
    assert mapping.startswith("by run index")


def test_seed_mapping_when_reference_records_seeds(tmp_path):
    ref_dir = tmp_path / "M" / "ser_classify"
    ref_dir.mkdir(parents=True)
    (ref_dir / "summary.json").write_text(json.dumps({"seeds": [7, 1234, 9999]}))
    path, mapping = cli._resolve_reference_run(ref_dir, run_seed=1234, run_index=0)
    assert path == ref_dir / "run_2.jsonl"
    assert mapping.startswith("by seed")

    missing, mapping = cli._resolve_reference_run(ref_dir, run_seed=42, run_index=0)
    assert missing is None
    assert "not among the reference seeds" in mapping


def test_per_run_seed_entries_are_honoured(tmp_path):
    ref_dir = tmp_path / "M" / "rsq"
    ref_dir.mkdir(parents=True)
    (ref_dir / "summary.json").write_text(
        json.dumps(
            {
                "per_run": {
                    "run_1": {"seed": 42},
                    "run_2": {"seed": 43},
                    "run_3": {"seed": 44},
                },
            }
        )
    )
    path, _ = cli._resolve_reference_run(ref_dir, run_seed=44, run_index=0)
    assert path == ref_dir / "run_3.jsonl"


def test_task_all_compares_each_subtask_directory(monkeypatch, tmp_path):
    root = tmp_path / "reference_trajectories"
    _write_run(root / "M" / "rad" / "run_1.jsonl", [("r1", "A"), ("r2", "B")])
    _write_run(root / "M" / "ser_classify" / "run_1.jsonl", [("s1", "A")])
    monkeypatch.setattr(cli, "REFERENCE_ROOT", root)
    monkeypatch.chdir(tmp_path / "..")

    questions = [
        _q("r1", "rad"),
        _q("r2", "rad"),
        _q("s1", "ser_classify"),
        _q("i1", "ser_intensity"),
    ]
    results = [_r("r1", "A"), _r("r2", "A"), _r("s1", "A"), _r("i1", "B")]
    summary = cli._compare_to_reference(
        "M", "all", questions, results, run_seed=42, run_index=0
    )

    assert set(summary) == {"rad", "ser_classify", "ser_intensity"}
    assert summary["rad"]["matches"] == 1 and summary["rad"]["compared"] == 2
    assert summary["ser_classify"]["matches"] == 1
    assert summary["ser_intensity"]["found"] is False
    assert all(
        entry["mapping"].startswith("by run index") for entry in summary.values()
    )


def test_single_task_uses_task_directory(monkeypatch, tmp_path):
    root = tmp_path / "reference_trajectories"
    _write_run(root / "M" / "rsq" / "run_3.jsonl", [("q1", "B")])
    monkeypatch.setattr(cli, "REFERENCE_ROOT", root)
    summary = cli._compare_to_reference(
        "M", "rsq", [_q("q1", "rsq")], [_r("q1", "B")], 44, 2
    )
    assert summary["rsq"]["matches"] == 1
    assert summary["rsq"]["reference"].endswith("run_3.jsonl")
