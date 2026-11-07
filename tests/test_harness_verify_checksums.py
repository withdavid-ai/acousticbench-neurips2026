"""--verify-checksums strictness, manifest checks, row counts, split inference."""

from __future__ import annotations

import hashlib
import json
import struct
import wave
from pathlib import Path

import pytest

from harness import cli, loader

N_PER_TASK = 4
SPEC = {
    "rad": (2, "acoustic-perception", "loudness"),
    "rsq": (2, "speech-quality", "clean-vs-degraded"),
    "ser_classify": (1, "emotion-recognition", "primary-emotion-opposites-2way"),
    "ser_intensity": (2, "emotion-recognition", "relative-emotional-intensity"),
    "ser_classify_text": (0, "emotion-recognition", "text-only"),
}


def _wav(path: Path, frames: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<h", 100) * frames)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_archive(
    root: Path, *, hash_audio: bool = True, text_split: bool = False
) -> Path:
    """Write a tiny archive with N_PER_TASK rows per task and an all.jsonl."""
    (root / "questions").mkdir(parents=True, exist_ok=True)
    all_rows = []
    for task, (n_audio, category, dimension) in SPEC.items():
        rows = []
        for i in range(N_PER_TASK):
            content = [{"type": "text", "text": "Which clip is louder?"}]
            for k in range(n_audio):
                rel = f"audio/{task}/{task}_{i}_{k}.wav"
                sha = _wav(root / rel, 1600 * (k + 1) + i)
                part = {"type": "audio_file", "path": rel, "order": k}
                if hash_audio:
                    part["sha256"] = sha
                content += [{"type": "text", "text": f"Clip {'AB'[k]}:"}, part]
            content.append({"type": "text", "text": "A. Clip A\nB. Clip B"})
            row = {
                "example_id": f"{task}-{i}",
                "messages": [{"role": "user", "content": content}],
                "question_set": {
                    "id": f"qs-{task}",
                    "name": task,
                    "slice": task,
                    "question_order": i,
                },
                "category": category,
                "dimension": dimension,
                "answer_key": "AB"[i % 2],
                "split": ["train", "eval"][i % 2],
            }
            if task == "ser_classify_text" and not text_split:
                row.pop("split")  # exercise split inference from ser_classify
            rows.append(row)
        (root / "questions" / f"acousticbench_{task}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
        all_rows += rows
    (root / "questions" / "acousticbench_all.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in all_rows)
    )
    return root


def write_manifest(root: Path) -> Path:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and "checksums" not in path.parts:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    manifest = root / "checksums" / "sha256sum.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(lines) + "\n")
    return manifest


@pytest.fixture
def small_counts(monkeypatch):
    counts = {task: N_PER_TASK for task in loader.SUBTASKS}
    counts["all"] = N_PER_TASK * len(loader.SUBTASKS)
    monkeypatch.setattr(loader, "EXPECTED_ROW_COUNTS", counts)
    return counts


def test_expected_counts_match_release():
    assert loader.EXPECTED_ROW_COUNTS == {
        "rad": 1200,
        "rsq": 1000,
        "ser_classify": 600,
        "ser_classify_text": 600,
        "ser_intensity": 400,
        "all": 3800,
    }
    assert "ser_classify_cascade" not in loader.TASK_SLICES


def test_verify_passes_on_clean_archive(tmp_path, small_counts):
    root = make_archive(tmp_path / "archive")
    write_manifest(root)
    report = loader.verify_archive("all", root)
    assert report["n_questions"] == small_counts["all"]
    assert report["manifest_files_checked"] > 0


def test_missing_hash_fails_unless_allowed(tmp_path, small_counts):
    root = make_archive(tmp_path / "archive", hash_audio=False)
    with pytest.raises(ValueError, match="has no sha256"):
        loader.verify_archive("rad", root)
    report = loader.verify_archive("rad", root, allow_missing_hashes=True)
    assert report["n_questions"] == N_PER_TASK


def test_audio_corruption_is_detected(tmp_path, small_counts):
    root = make_archive(tmp_path / "archive")
    (root / "audio" / "rsq" / "rsq_0_0.wav").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        loader.verify_archive("rsq", root)


def test_manifest_mismatch_on_question_file_is_detected(tmp_path, small_counts):
    root = make_archive(tmp_path / "archive")
    write_manifest(root)
    qfile = root / "questions" / "acousticbench_ser_intensity.jsonl"
    rows = [json.loads(line) for line in qfile.read_text().splitlines()]
    rows[0]["answer_key"] = (
        "B" if rows[0]["answer_key"] == "A" else "A"
    )  # re-keyed item
    qfile.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="sha256sum.txt"):
        loader.verify_archive("ser_intensity", root)


def test_row_count_mismatch_is_detected(tmp_path, monkeypatch):
    root = make_archive(tmp_path / "archive")
    monkeypatch.setattr(
        loader, "EXPECTED_ROW_COUNTS", {**loader.EXPECTED_ROW_COUNTS, "rad": 5}
    )
    with pytest.raises(ValueError, match="expected 5"):
        loader.verify_archive("rad", root)


def test_cli_verify_exits_nonzero_on_failure(tmp_path, small_counts):
    root = make_archive(tmp_path / "archive", hash_audio=False)
    with pytest.raises(SystemExit) as exc:
        cli.main(["--verify-checksums", "--task", "rad", "--archive", str(root)])
    assert exc.value.code == 1
    cli.main(
        [
            "--verify-checksums",
            "--task",
            "rad",
            "--archive",
            str(root),
            "--allow-missing-hashes",
        ]
    )


def test_ser_classify_text_split_is_inferred_inside_all(tmp_path):
    root = make_archive(tmp_path / "archive")
    text_alone = loader.load_questions("ser_classify_text", root, split="eval")
    in_all = [
        q
        for q in loader.load_questions("all", root, split="eval")
        if q["subset"] == "ser_classify_text"
    ]
    assert len(text_alone) == N_PER_TASK // 2
    assert sorted(q["meta"]["example_id"] for q in in_all) == sorted(
        q["meta"]["example_id"] for q in text_alone
    )
    assert all(q["split"] == "eval" for q in in_all)
