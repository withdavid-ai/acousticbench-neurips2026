"""--position-swap must not crash on single-clip or text-only questions."""

from __future__ import annotations

import pytest

from harness import cli
from harness.metrics import consistent_accuracy
from harness.position_swap import generate_swapped_questions


def _q(example_id: str, audio, answer: str = "A") -> dict:
    return {
        "audio_path": audio,
        "question": "Q\nA. a\nB. b",
        "answer": answer,
        "subset": "x",
        "meta": {"example_id": example_id, "answer_key": answer},
    }


def _result(q: dict, chosen: str) -> dict:
    return {
        "example_id": q["meta"]["example_id"],
        "chosen": chosen,
        "correct": q["answer"],
        "is_correct": chosen == q["answer"],
        "meta": q["meta"],
    }


def test_only_two_clip_questions_are_swapped():
    questions = [_q("pair", ["a.wav", "b.wav"]), _q("single", "c.wav"), _q("text", [])]
    swapped = generate_swapped_questions(questions)
    assert [q["meta"]["original_example_id"] for q in swapped] == ["pair"]
    assert swapped[0]["audio_path"] == ["b.wav", "a.wav"]
    assert swapped[0]["answer"] == "B"


def test_consistent_accuracy_ignores_unpaired_rows():
    pair = _q("pair", ["a.wav", "b.wav"])
    single = _q("single", "c.wav")
    swapped_pair = generate_swapped_questions([pair])[0]
    original = [_result(pair, "A"), _result(single, "A")]
    # A legacy pass-through row (no original_example_id) must not raise KeyError.
    swapped = [_result(swapped_pair, "B"), _result(single, "A")]
    metrics = consistent_accuracy(original, swapped, n_bootstrap=50)
    assert metrics["n"] == 1
    assert metrics["consistent_accuracy"] == 100.0


def test_cli_rejects_single_clip_task_before_inference(monkeypatch, tmp_path):
    import harness.loader
    import harness.models

    def fail_build(*args, **kwargs):
        raise AssertionError("model must not be built for a rejected swap run")

    monkeypatch.setattr(
        harness.models, "get_model_metadata", lambda name: {"backends": ["api"]}
    )
    monkeypatch.setattr(harness.models, "build_model", fail_build)
    monkeypatch.setattr(
        harness.loader,
        "load_questions",
        lambda task, archive, split="all", **kw: [_q("s1", "c.wav"), _q("s2", "d.wav")],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "--model",
                "Fake",
                "--task",
                "ser_classify",
                "--position-swap",
                "--output-dir",
                str(tmp_path),
            ]
        )
    assert exc.value.code == 2
