"""Seed plumbing: --seed S, S+1, ... must reach the model adapters."""

from __future__ import annotations

import json
import struct
import wave
from pathlib import Path

import pytest

from harness import cli
from harness.models import base as base_module


def _write_wav(path: Path, seconds: float = 0.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<h", 0) * int(16000 * seconds))
    return path


def _question(example_id: str, audio_path: str) -> dict:
    return {
        "index": 0,
        "audio_path": audio_path,
        "question": "Which clip is louder?\nA. Clip A\nB. Clip B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Which clip is louder?"},
                    {"type": "audio", "audio_index": 0},
                    {"type": "text", "text": "A. Clip A\nB. Clip B"},
                ],
            }
        ],
        "answer": "A",
        "subset": "rad",
        "split": "eval",
        "meta": {"example_id": example_id, "answer_key": "A"},
    }


def _msg(question: dict) -> dict:
    """Model input as harness.runner builds it."""
    return {
        "audio": question["audio_path"],
        "text": question["question"],
        "messages": question["messages"],
        "meta": question["meta"],
    }


def test_base_model_seed_kwarg_reaches_request_meta(monkeypatch):
    monkeypatch.delenv("AUDIOBENCH_SEED", raising=False)
    model = base_module.BaseModel(seed=43)
    assert model.seed == 43
    assert model.sampling_request_meta()["_benchmark_seed"] == 43


def test_cli_builds_each_run_with_its_own_seed(monkeypatch, tmp_path):
    built_seeds: list[int | None] = []

    class FakeModel:
        def __init__(self, seed):
            self.seed = seed

    def fake_build_model(name, **kwargs):
        built_seeds.append(kwargs.get("seed"))
        return FakeModel(kwargs.get("seed"))

    seen_models = []

    def fake_run_inference(model, questions, **kwargs):
        seen_models.append(model.seed)
        return [
            {
                "example_id": q["meta"]["example_id"],
                "chosen": "A",
                "correct": q["answer"],
                "is_correct": True,
                "extraction_source": "llm",
                "meta": q["meta"],
            }
            for q in questions
        ]

    import harness.loader
    import harness.models
    import harness.runner

    monkeypatch.setattr(
        harness.models, "get_model_metadata", lambda name: {"backends": ["api"]}
    )
    monkeypatch.setattr(harness.models, "build_model", fake_build_model)
    monkeypatch.setattr(harness.runner, "run_inference", fake_run_inference)
    monkeypatch.setattr(
        harness.loader,
        "load_questions",
        lambda task, archive, split="all", **kw: [_question("q1", "unused.wav")],
    )

    out = tmp_path / "results"
    cli.main(
        [
            "--model",
            "Fake",
            "--task",
            "rad",
            "--runs",
            "3",
            "--seed",
            "7",
            "--output-dir",
            str(out),
        ]
    )

    assert built_seeds == [7, 8, 9]
    assert seen_models == [7, 8, 9]
    configs = sorted(
        json.loads(p.read_text())["seed"] for p in out.glob("*/config.json")
    )
    assert configs == [7, 8, 9]


def test_gpt4o_sends_seed_and_dated_snapshot(monkeypatch, tmp_path):
    pytest.importorskip("openai")
    from harness.models.gpt4o import GPT4oAudio

    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    model = GPT4oAudio(seed=44, require_submission_payload=False)
    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            raise RuntimeError("stop before network")

    class _Client:
        chat = type("Chat", (), {"completions": _Completions()})()

    model.client = _Client()
    audio = _write_wav(tmp_path / "a.wav")
    result = model.generate_inner(_msg(_question("q1", str(audio))))

    assert result["response_status"] == "error"
    assert captured["model"] == "gpt-4o-audio-preview-2025-06-03"
    assert captured["seed"] == 44
    assert captured["temperature"] == 0.0


def test_gemini_does_not_send_seed_when_sdk_lacks_field(monkeypatch, tmp_path):
    genai = pytest.importorskip("google.generativeai")
    from harness.models import gemini

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    model = gemini.Gemini31Pro(seed=42, require_submission_payload=False)
    assert model.model_name == "gemini-3.1-pro-preview"

    captured = {}

    def fake_generate_content(contents, generation_config=None):
        captured["generation_config"] = generation_config
        raise RuntimeError("stop before network")

    monkeypatch.setattr(model.model, "generate_content", fake_generate_content)
    audio = _write_wav(tmp_path / "a.wav")
    model.generate_inner(_msg(_question("q1", str(audio))))

    config = captured["generation_config"]
    assert config["temperature"] == 0.0
    if gemini._generation_config_supports_seed(genai):
        assert config["seed"] == 42
    else:
        # An unknown `seed` key would make every request fail in this SDK.
        assert "seed" not in config


def test_gemini_model_ids_match_paper_table5():
    from harness.models.gemini import GEMINI_MODEL_IDS

    assert GEMINI_MODEL_IDS == {
        "Gemini-2.5-Flash-Audio": "gemini-2.5-flash",
        "Gemini-2.5-Pro": "gemini-2.5-pro",
        "Gemini-3-Flash": "gemini-3-flash-preview",
        "Gemini-3.1-Pro": "gemini-3.1-pro-preview",
    }


def test_cascade_reasoner_is_dated_snapshot():
    from harness.models.openai_cascade import OpenAICascadeTranscribeGPT54 as Cascade

    assert Cascade.DEFAULT_ANSWER_MODEL == "gpt-5.4-2026-03-05"
    assert Cascade.DEFAULT_TRANSCRIBE_MODEL == "gpt-4o-transcribe"
    # The dated name must still take the reasoning-model request path.
    assert Cascade._is_new_api_model(Cascade.DEFAULT_ANSWER_MODEL)
