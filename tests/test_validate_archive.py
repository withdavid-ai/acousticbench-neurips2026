"""Tests for scripts/validate_archive.py on a tiny synthetic archive.

The archive is built in tmp_path: 18 questions over the five task files, 26
unique 0.1 s WAV clips generated with numpy + the stdlib wave module, an
``acousticbench_all.jsonl`` union, ``checksums/sha256sum.txt`` and
``audio_files.csv``. Each test breaks one contract rule and checks that the
matching check reports it.
"""

from __future__ import annotations

import hashlib
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import validate_archive as va  # noqa: E402
from make_paper_tables import digest_lines  # noqa: E402

SAMPLE_RATE = 16000
CLIP_SECONDS = 0.1
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
RAD_DIMS = ["pitch", "pitch", "loudness", "loudness", "tempo", "tempo", "dynamic-range", "dynamic-range"]
SIZES = {"rad": 8, "rsq": 2, "ser_classify": 2, "ser_classify_text": 2, "ser_intensity": 2}
CLIPS = {"rad": 2, "rsq": 2, "ser_classify": 1, "ser_classify_text": 0, "ser_intensity": 2}
N_CLIPS = sum(SIZES[t] * CLIPS[t] for t in SIZES)  # 26


def write_wav(path: Path, freq: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(SAMPLE_RATE * CLIP_SECONDS)) / SAMPLE_RATE
    samples = (0.3 * np.sin(2 * np.pi * freq * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(samples.tobytes())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_archive(root: Path) -> Path:
    freq = 200.0
    rows_by_task: dict[str, list[dict]] = {}
    for task, size in SIZES.items():
        rows = []
        for i in range(size):
            eid = f"{task.replace('_', '-')}-{i:03d}"
            content: list[dict] = [{"type": "text", "text": "Which clip is the better match?"}]
            for c in range(CLIPS[task]):
                rel = f"audio/{task}/{eid}/{c:02d}_clip.wav"
                write_wav(root / rel, freq)
                freq += 37.0
                content.append({"type": "text", "text": f"Clip {'AB'[c]}:"})
                content.append({"type": "audio_file", "path": rel, "order": c, "sha256": sha256(root / rel)})
            content.append({"type": "text", "text": "Options:\nA. Clip A\nB. Clip B"})
            rows.append({
                "example_id": eid,
                "question_set": {"id": f"qs-{task}", "name": f"{task} set", "slice": task, "question_order": i},
                "category": CATEGORY[task],
                "dimension": RAD_DIMS[i] if task == "rad" else DIMENSION[task],
                "messages": [{"role": "user", "content": content}],
                "answer_key": "A" if i % 2 == 0 else "B",
                # Half eval / half train per task; rad alternates by dimension pair.
                "split": "eval" if (i // 2 if task == "rad" else i) % 2 == 0 else "train",
            })
        rows_by_task[task] = rows
    write_questions(root, rows_by_task)
    return root


def write_questions(root: Path, rows_by_task: dict[str, list[dict]]) -> None:
    qdir = root / "questions"
    qdir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for task, rows in rows_by_task.items():
        (qdir / f"acousticbench_{task}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        all_rows.extend(rows)
    (qdir / "acousticbench_all.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in all_rows), encoding="utf-8")
    (root / "checksums").mkdir(exist_ok=True)
    lines = [f"{sha256(path)}  {path.relative_to(root)}"
             for path in sorted(root.rglob("*")) if path.is_file() and "checksums" not in path.parts
             and path.name != "audio_files.csv"]
    (root / "checksums" / "sha256sum.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv_lines = ["example_id,released_path,sha256"]
    for rows in rows_by_task.values():
        for row in rows:
            for part in va.audio_parts(row):
                csv_lines.append(f"{row['example_id']},{part['path']},{part.get('sha256', '')}")
    (root / "audio_files.csv").write_text("\n".join(csv_lines) + "\n", encoding="utf-8")


def load_rows(root: Path) -> dict[str, list[dict]]:
    return {
        task: [json.loads(line)
               for line in (root / "questions" / f"acousticbench_{task}.jsonl").read_text().splitlines()]
        for task in SIZES
    }


def expectations(**overrides) -> va.Expectations:
    exp = va.Expectations(
        task_rows=dict(SIZES),
        eval_rows={task: size // 2 for task, size in SIZES.items()},
        audio_parts_per_row=dict(CLIPS),
        rad_dimension_rows={dim: 2 for dim in ("pitch", "loudness", "tempo", "dynamic-range")},
        unique_clips=N_CLIPS,
        paper_hours={"Total": N_CLIPS * CLIP_SECONDS / 3600},
        hours_tolerance=1e-6,
        contract=None,
        conflicts=(),
    )
    for key, value in overrides.items():
        setattr(exp, key, value)
    return exp


def statuses(report: va.Report) -> dict[str, str]:
    return {check.name: check.status for check in report.checks}


def check(report: va.Report, name: str) -> va.Check:
    return next(c for c in report.checks if c.name == name)


@pytest.fixture()
def archive(tmp_path: Path) -> Path:
    return build_archive(tmp_path / "archive")


def test_clean_archive_passes(archive: Path) -> None:
    report = va.run_checks(archive, refs=None, expectations=expectations(), workers=4)
    assert not report.failed, [(c.name, c.summary, c.details) for c in report.failed]
    s = statuses(report)
    for name in ("files", "row_counts", "all_union", "required_fields", "splits", "answer_balance",
                 "audio_integrity", "audio_uniqueness", "sha256sum_file", "label_literals",
                 "audio_durations", "pii_leak_scan"):
        assert s[name] == "PASS", (name, check(report, name).details)
    assert check(report, "audio_uniqueness").data["unique_contents"] == N_CLIPS
    census = check(report, "audio_format_census").data["census"]
    assert all("16000 Hz" in key for counts in census.values() for key in counts)


def test_durations_fall_back_to_wave(archive: Path, monkeypatch) -> None:
    monkeypatch.setattr(va, "soundfile", None)
    report = va.run_checks(archive, expectations=expectations(), workers=2)
    hours = check(report, "audio_durations").data["hours"]
    assert hours["Total"] == pytest.approx(N_CLIPS * CLIP_SECONDS / 3600)


def test_hash_mismatch_and_missing_sha_fail(archive: Path) -> None:
    rows = load_rows(archive)
    target = archive / rows["rad"][0]["messages"][0]["content"][2]["path"]
    data = bytearray(target.read_bytes())
    data[-1] ^= 0xFF
    target.write_bytes(bytes(data))
    del rows["rsq"][0]["messages"][0]["content"][2]["sha256"]
    write_questions_keep_checksums(archive, rows)
    report = va.run_checks(archive, expectations=expectations(), workers=2)
    integrity = check(report, "audio_integrity")
    assert integrity.status == "FAIL"
    assert integrity.data["sha256_mismatches"] == 1
    assert integrity.data["missing_sha256"] == 1
    assert check(report, "sha256sum_file").status == "FAIL"


def write_questions_keep_checksums(root: Path, rows_by_task: dict[str, list[dict]]) -> None:
    """Rewrite question files but keep the original checksum list (to detect tampering)."""
    saved = (root / "checksums" / "sha256sum.txt").read_text()
    write_questions(root, rows_by_task)
    (root / "checksums" / "sha256sum.txt").write_text(saved)


def test_label_literal_and_pii_are_flagged(archive: Path) -> None:
    rows = load_rows(archive)
    rows["ser_classify_text"][0]["messages"][0]["content"][0]["text"] = "He moved to Washington D.C. last year."
    rows["rsq"][1]["notes"] = "contact jane.doe@example.com, file /Users/jane/raw/clip.wav"
    rows["rad"][3]["worker_id"] = "W12345"
    write_questions(archive, rows)
    (archive / "audio" / "emolab_extra.txt").write_text("x")
    report = va.run_checks(archive, expectations=expectations(), workers=2)
    literals = check(report, "label_literals")
    assert literals.status == "FAIL" and literals.data["per_task"] == {"ser_classify_text": 1}
    leaks = check(report, "pii_leak_scan")
    assert leaks.status == "FAIL"
    assert leaks.data["severe"]["email"] >= 1 and leaks.data["severe"]["local_path"] >= 1
    assert leaks.data["heuristic"]["identity_like_field"] >= 1
    assert leaks.data["heuristic"]["source_corpus_name"] >= 1


def test_missing_split_in_all_and_union_mismatch(archive: Path) -> None:
    rows = load_rows(archive)
    write_questions(archive, rows)
    all_path = archive / "questions" / "acousticbench_all.jsonl"
    all_rows = [json.loads(line) for line in all_path.read_text().splitlines()]
    for row in all_rows:
        if row["question_set"]["slice"] == "ser_classify_text":
            row.pop("split")
    all_rows = all_rows[:-1]  # also drop one row
    all_path.write_text("".join(json.dumps(r) + "\n" for r in all_rows))
    report = va.run_checks(archive, expectations=expectations(), workers=2, skip_audio=True)
    splits = check(report, "splits")
    assert splits.status == "FAIL"
    assert any("ser_classify_text rows lack split" in line for line in splits.details)
    union = check(report, "all_union")
    assert union.status == "FAIL" and union.data["missing_from_all"] == 1


def test_imbalance_and_reuse_fail(archive: Path) -> None:
    rows = load_rows(archive)
    rows["rsq"][1]["answer_key"] = "A"
    reused = rows["rad"][0]["messages"][0]["content"][2]
    rows["rad"][1]["messages"][0]["content"][2] = dict(reused)
    write_questions(archive, rows)
    report = va.run_checks(archive, expectations=expectations(), workers=2)
    assert check(report, "answer_balance").status == "FAIL"
    uniqueness = check(report, "audio_uniqueness")
    assert uniqueness.status == "FAIL" and uniqueness.data["reused_paths"] == 1
    assert check(report, "audio_unreferenced").status == "WARN"


def test_release_contract_digests(archive: Path) -> None:
    rows = load_rows(archive)
    contract = {
        task: {
            "ids": digest_lines(r["example_id"] for r in task_rows),
            "eval_ids": digest_lines(r["example_id"] for r in task_rows if r["split"] == "eval"),
            "contract": digest_lines(f"{r['example_id']}\t{r['split']}\t{r['answer_key']}" for r in task_rows),
        }
        for task, task_rows in rows.items()
    }
    report = va.run_checks(archive, expectations=expectations(contract=contract), skip_audio=True)
    assert check(report, "release_contract").status == "PASS"
    rows["ser_classify"][0]["answer_key"], rows["ser_classify"][1]["answer_key"] = "B", "A"
    write_questions(archive, rows)
    report = va.run_checks(archive, expectations=expectations(contract=contract), skip_audio=True)
    failed = check(report, "release_contract")
    assert failed.status == "FAIL" and any("answer keys differ" in d for d in failed.details)


def _write_ref(root: Path, model: str, task: str, rows: list[dict], key_override: dict[str, str] | None = None,
               split: str | None = None) -> None:
    task_dir = root / model / task
    task_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for row in rows:
        key = (key_override or {}).get(row["example_id"], row["answer_key"])
        out.append({
            "example_id": row["example_id"], "chosen": key, "correct": key, "is_correct": True,
            "response_text": key, "response_status": "ok",
            "meta": {"example_id": row["example_id"], "dimension": row["dimension"], "split": row["split"],
                     "question_order": row["question_set"]["question_order"], "answer_key": key},
        })
    for k in (1, 2, 3):
        (task_dir / f"run_{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    if split:
        (task_dir / "summary.json").write_text(json.dumps({"split": split}))


def test_reference_cross_checks_and_conflict_report(archive: Path, tmp_path: Path) -> None:
    rows = load_rows(archive)
    refs = tmp_path / "refs"
    conflict_id = rows["ser_classify"][0]["example_id"]
    for task, task_rows in rows.items():
        _write_ref(refs, "Kimi-Audio", task, task_rows)
        _write_ref(refs, "Other-Model", task, task_rows,
                   key_override={conflict_id: "B"} if task == "ser_classify" else None)
    _write_ref(refs, "Kimi-Audio-LoRA-RAD", "rad", [r for r in rows["rad"] if r["split"] == "eval"], split="eval")
    _write_ref(refs, va.CASCADE_DIR, "ser_classify", rows["ser_classify_text"])
    exp = expectations(conflicts=((conflict_id, 0, rows["ser_classify"][0]["split"]),))
    report = va.run_checks(archive, refs=refs, expectations=exp, skip_audio=True)
    ids = check(report, "reference_ids")
    assert ids.status == "WARN"  # only the known Cascade exception
    assert any("known: Cascade" in d for d in ids.details)
    assert check(report, "reference_answer_keys").status == "PASS"
    assert check(report, "reference_split_meta").status == "PASS"
    conflict = check(report, "ser_classify_key_conflict")
    assert conflict.status == "WARN"
    assert conflict.data["disagree"] == ["Other-Model"]
    assert "Kimi-Audio" in conflict.data["agree"]


def test_cli_exit_codes_and_json(archive: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    # Default expectations are the 3,800-question release, so a tiny archive fails.
    code = va.main(["--archive", str(archive), "--no-refs", "--json-out", str(out), "--workers", "2"])
    assert code == 1
    payload = json.loads(out.read_text())
    assert payload["passed"] is False
    assert any(c["name"] == "row_counts" and c["status"] == "FAIL" for c in payload["checks"])
    assert va.main(["--archive", str(tmp_path / "missing")]) == 2


def test_duration_counts_repeated_clip_slots() -> None:
    refs = [va.AudioRef('rad', str(i), path, 'same-content')
            for i, path in enumerate(['a.wav', 'b.wav', 'a.wav'])]
    infos = {p: {'sha256': 'same-content', 'duration': 1200.0}
             for p in ['a.wav', 'b.wav']}
    report = va.Report('synthetic')
    va.check_durations(refs, infos, expectations(paper_hours={'Total': 1.0}), report)
    assert check(report, 'audio_durations').status == 'PASS'
    assert check(report, 'audio_durations').data['hours']['Total'] == 1.0


def test_missing_duration_is_not_reported_as_complete() -> None:
    refs = [va.AudioRef('rad', '1', 'a.wav', 'one'), va.AudioRef('rad', '2', 'b.wav', 'two')]
    infos = {'a.wav': {'sha256': 'one', 'duration': 3600.0}, 'b.wav': {}}
    report = va.Report('synthetic')
    va.check_durations(refs, infos, expectations(paper_hours={'Total': 1.0}), report)
    assert check(report, 'audio_durations').status == 'WARN'
