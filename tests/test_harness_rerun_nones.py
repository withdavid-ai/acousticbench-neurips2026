"""scripts/rerun_nones.py must send the same interleaved prompt as run.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import harness
from harness import runner
from harness.message_utils import ensure_messages

from test_harness_verify_checksums import make_archive

SCRIPTS = Path(harness.__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rerun_nones = _load_script("rerun_nones")


class _RecordingModel:
    def __init__(self):
        self.messages = []

    def __call__(self, msg):
        self.messages.append(msg)
        return {"response_text": "A", "response_status": "ok"}


def test_rerun_uses_interleaved_clip_layout(tmp_path, monkeypatch):
    root = make_archive(tmp_path / "archive")
    questions = rerun_nones._load_questions_by_id(root)
    question = questions["rad-0"]

    monkeypatch.setattr(
        rerun_nones, "extract_choice", lambda *a, **k: {"label": "A", "source": "llm"}
    )
    monkeypatch.setattr(
        runner, "extract_choice", lambda *a, **k: {"label": "A", "source": "llm"}
    )

    rerun_model = _RecordingModel()
    rerun_nones._run_single(rerun_model, question)
    run_model = _RecordingModel()
    runner._run_single(run_model, question)

    sent = rerun_model.messages[0]
    assert sent["messages"] == run_model.messages[0]["messages"]
    layout = [
        part["type"] if part["type"] == "audio" else part["text"]
        for part in ensure_messages(sent, audio_count=2)[0]["content"]
    ]
    assert layout == [
        "Which clip is louder?",
        "Clip A:",
        "audio",
        "Clip B:",
        "audio",
        "A. Clip A\nB. Clip B",
    ]


def test_rerun_loads_ser_classify_text_rows(tmp_path):
    root = make_archive(tmp_path / "archive")
    questions = rerun_nones._load_questions_by_id(root)
    assert "ser_classify_text-0" in questions
    assert {q["subset"] for q in questions.values()} == {
        "rad",
        "rsq",
        "ser_classify",
        "ser_classify_text",
        "ser_intensity",
    }
