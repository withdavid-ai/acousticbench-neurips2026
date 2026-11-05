"""Judge retry/backoff, pinned snapshot, deterministic extractor, rejudge.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import requests

import harness
from harness import judge, runner

from test_harness_verify_checksums import make_archive


class _Resp:
    def __init__(
        self, status: int, payload: dict | None = None, headers: dict | None = None
    ):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _ok(label: str) -> _Resp:
    content = json.dumps(
        {"classification": "choice", "selected_label": label, "notes": ""}
    )
    return _Resp(200, {"choices": [{"message": {"content": content}}]})


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(judge.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    return sleeps


def test_judge_uses_dated_snapshot(no_sleep, monkeypatch):
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent.update(json)
        return _ok("B")

    monkeypatch.setattr(requests, "post", fake_post)
    out = judge.extract_choice("I pick B", "Q", ["A", "B"])
    assert judge.DEFAULT_JUDGE_MODEL == "gpt-4o-mini-2024-07-18"
    assert sent["model"] == "gpt-4o-mini-2024-07-18"
    assert out["label"] == "B" and out["source"] == "llm"


def test_judge_retries_transient_failures(no_sleep, monkeypatch):
    replies = [
        requests.exceptions.ConnectionError("reset"),
        _Resp(429, {"error": "rate"}, headers={"Retry-After": "3"}),
        _Resp(200, {"error": "no choices"}),
        _ok("A"),
    ]

    def fake_post(*args, **kwargs):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(requests, "post", fake_post)
    out = judge.extract_choice("Clip A", "Q", ["A", "B"])
    assert out["label"] == "A"
    assert out["judge_attempts"] == 4
    assert len(no_sleep) == 3 and no_sleep[1] >= 3


def test_judge_gives_up_after_bounded_attempts(no_sleep, monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(503, {"error": "down"}))
    with pytest.raises(judge.JudgeError, match="after 5 attempts"):
        judge.extract_choice("A", "Q", ["A", "B"])
    assert len(no_sleep) == judge.JUDGE_MAX_ATTEMPTS - 1


def test_judge_does_not_retry_auth_errors(no_sleep, monkeypatch):
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: _Resp(401, {"error": "bad key"})
    )
    with pytest.raises(judge.JudgeError, match="HTTP 401"):
        judge.extract_choice("A", "Q", ["A", "B"])
    assert no_sleep == []


def test_runner_keeps_response_when_judge_fails(monkeypatch):
    def broken_judge(*args, **kwargs):
        raise judge.JudgeError("judge down")

    monkeypatch.setattr(runner, "extract_choice", broken_judge)
    item = {
        "audio_path": "a.wav",
        "question": "Q\nA. a\nB. b",
        "messages": None,
        "answer": "A",
        "meta": {"example_id": "e1"},
    }
    row = runner._run_single(
        lambda msg: {"response_text": "The answer is A", "response_status": "ok"}, item
    )
    assert row["response_text"] == "The answer is A"
    assert row["response_status"] == "ok"
    assert row["extraction_source"] == "judge_error"
    assert row["chosen"] is None and row["is_correct"] is False


def test_deterministic_then_llm_extractor(monkeypatch):
    calls = []
    monkeypatch.setattr(
        judge,
        "extract_choice_llm",
        lambda *a, **k: calls.append(a) or {"label": "B", "source": "llm"},
    )
    det = judge.extract_choice(
        "A. Clip A is louder", "Q", ["A", "B"], extractor="deterministic-then-llm"
    )
    assert det == {"label": "A", "classification": "choice", "source": "deterministic"}
    assert calls == []
    fallback = judge.extract_choice(
        "The second one", "Q", ["A", "B"], extractor="deterministic-then-llm"
    )
    assert fallback["source"] == "llm" and len(calls) == 1
    # Default protocol never short-circuits.
    judge.extract_choice("A", "Q", ["A", "B"])
    assert len(calls) == 2


def _load_rejudge():
    path = Path(harness.__file__).resolve().parents[1] / "scripts" / "rejudge.py"
    spec = importlib.util.spec_from_file_location("rejudge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rejudge_passes_question_and_targets_null_rows(tmp_path, monkeypatch):
    rejudge = _load_rejudge()
    root = make_archive(tmp_path / "archive")
    questions = rejudge.load_questions_by_id(root)
    seen = []

    def fake_extract(response, question, labels, extractor="llm"):
        seen.append(question)
        return {"label": "A", "source": "llm", "classification": "choice"}

    monkeypatch.setattr(rejudge, "extract_choice", fake_extract)
    run_dir = tmp_path / "refs" / "M" / "rad"
    run_dir.mkdir(parents=True)
    rows = [
        {
            "example_id": "rad-0",
            "response_text": "first",
            "chosen": None,
            "correct": "A",
            "is_correct": False,
        },
        {
            "example_id": "rad-1",
            "response_text": "second",
            "chosen": "B",
            "correct": "B",
            "is_correct": True,
        },
    ]
    run_file = run_dir / "run_1.jsonl"
    run_file.write_text("".join(json.dumps(r) + "\n" for r in rows))

    results, stats = rejudge.rejudge_file(run_file, questions, concurrency=1)
    assert stats["rejudged"] == 1
    assert seen == [questions["rad-0"]["question"]] and seen[0].strip()
    assert results[0]["chosen"] == "A" and results[0]["is_correct"] is True
    assert results[1]["chosen"] == "B"

    out = rejudge.output_path_for(run_file, tmp_path / "rejudged")
    assert out == tmp_path / "rejudged" / "M" / "rad" / "run_1.jsonl"
    rejudge.write_results(results, out)
    assert (
        json.loads(run_file.read_text().splitlines()[0])["chosen"] is None
    )  # input untouched


def test_rejudge_summary_uses_sample_sd(tmp_path):
    rejudge = _load_rejudge()
    task_dir = tmp_path / "M" / "rad"
    task_dir.mkdir(parents=True)
    for run, n_correct in (("run_1", 1), ("run_2", 2), ("run_3", 2)):
        rows = [{"chosen": "A", "is_correct": i < n_correct} for i in range(3)]
        (task_dir / f"{run}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    summary = rejudge.regenerate_summary(task_dir)
    # Runs: 33.33, 66.67, 66.67 -> sample SD 19.25 (population SD would be 15.71).
    assert summary["std_ddof"] == 1
    assert summary["std_accuracy"] == 19.25
    assert summary["mean_accuracy"] == 55.56


def test_only_final_thinking_segment_is_used():
    from harness.judge import _strip_thinking
    assert _strip_thinking("<think>A</think><think>B</think>A") == "A"
