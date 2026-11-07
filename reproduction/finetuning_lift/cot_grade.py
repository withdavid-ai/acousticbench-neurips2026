"""Grade the acoustic descriptions produced in the CoT condition.

`cot.py` scores only the final A/B letter, which cannot distinguish two very
different failures: describing the audio correctly and then choosing wrongly, or
describing it wrongly in the first place. Reviewer 2 asks whether two-choice
accuracy reflects open-ended acoustic ability, and that question is about the
description, not the letter.

So this module reads the stored CoT responses and asks, per item, what the
description itself claims. Four quantities come out of that:

  commits          the description states a direction at all
  description acc  the stated direction matches the gold key
  consistency      the stated direction matches the letter the model returned
  decision loss    description right, final letter wrong

**The judge never sees the answer key and never decides correctness.** It only
reports which clip the description favours, which is close to an extraction task.
Correctness is computed here, against the release's own key. That matters because
the judge and one of the graded models are both Gemini: a judge asked "is this
right?" could prefer its own output, whereas a judge asked "what does this text
claim?" has very little room to.

Raw judge verdicts are written to disk, so re-aggregating never re-calls the API.

Usage:
    python -m reproduction.finetuning_lift.cot_grade --all
    python -m reproduction.finetuning_lift.cot_grade --report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduction.finetuning_lift import config, cot, dataset  # noqa: E402

GRADED_MODELS = ("kimi", "gemini")

# Long enough for any real description. The degenerate Kimi generations run to
# 21k characters of one repeated sentence; truncating them costs no information
# and keeps the judge call bounded.
MAX_DESCRIPTION_CHARS = 4000

JUDGE_MODEL = cot.GEMINI_MODEL
DEFAULT_CONCURRENCY = 8

JUDGE_PROMPT = """You are extracting a claim from a piece of text. You are NOT \
judging whether the claim is right, and you are not being shown the correct \
answer.

A model was played two audio clips (or one, for emotion classification) and asked \
to describe what it heard and then choose between two options. Below are the \
options it was given and the text it produced. Report only what the TEXT claims.

Options:
{options}

Text:
\"\"\"
{description}
\"\"\"

Answer with a single JSON object, no other output:
{{
  "direction": "A" | "B" | "none",
  "asserts_identical": true | false,
  "describes_audio": true | false
}}

"direction" is which option the text ultimately favours, or "none" if it declines, \
hedges without choosing, or says it cannot tell.
"asserts_identical" is true if the text claims the clips are the same, equal, or \
indistinguishable on the property in question.
"describes_audio" is true if the text makes at least one concrete observation \
about the audio (loudness, pitch, tempo, noise, tone of voice, and so on) rather \
than only restating the question or refusing."""


@dataclass(frozen=True)
class Cell:
    task_key: str
    model: str

    @property
    def name(self) -> str:
        return f"{self.task_key}--{self.model}--cot"

    @property
    def source_dir(self) -> Path:
        return config.RESULTS_ROOT / "cot" / self.name

    @property
    def out_dir(self) -> Path:
        return config.RESULTS_ROOT / "cot_grade" / self.name


def _options_block(example: dataset.Example) -> str:
    """The Options lines from the item's own prompt, verbatim."""
    for part in reversed(example.parts):
        if part["type"] != "text":
            continue
        text = part["text"]
        if "Options:" in text:
            block = text.split("Options:", 1)[1]
            # Drop the trailing answer-format instruction.
            lines = [
                line.strip()
                for line in block.splitlines()
                if line.strip() and line.strip()[:2] in ("A.", "B.")
            ]
            if lines:
                return "\n".join(lines)
    raise RuntimeError(f"no Options block in {example.example_id}")


def _parse_judge(raw: str) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text.strip("`")
        text = text.removeprefix("json").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    direction = str(obj.get("direction", "none")).strip().upper()
    if direction not in ("A", "B"):
        direction = None
    return {
        "direction": direction,
        "asserts_identical": bool(obj.get("asserts_identical")),
        "describes_audio": bool(obj.get("describes_audio")),
    }


def _judge_one(client, record: dict, options: str) -> dict:
    from google.genai import types

    description = (record.get("response") or "")[:MAX_DESCRIPTION_CHARS]
    prompt = JUDGE_PROMPT.format(options=options, description=description)
    out = {
        "example_id": record["example_id"],
        "answer_key": record["answer_key"],
        "model_letter": record.get("predicted"),
        "model_letter_correct": bool(record.get("correct")),
        "response_chars": record.get("response_chars"),
        "truncated_for_judge": len(record.get("response") or "") > MAX_DESCRIPTION_CHARS,
    }
    try:
        response = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        verdict = _parse_judge(response.text or "")
        if verdict is None:
            out |= {"status": "unparsed_judge", "raw": (response.text or "")[:400]}
        else:
            out |= {"status": "ok", **verdict}
    except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
        out |= {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return out


def grade(cell: Cell, *, concurrency: int, limit: int | None) -> dict:
    source = cell.source_dir / "predictions.jsonl"
    if not source.exists():
        raise FileNotFoundError(f"no stored CoT responses at {source}")
    records = [json.loads(line) for line in source.open()]
    if limit:
        records = records[:limit]

    task = config.TASKS[cell.task_key]
    options_by_id = {
        example.example_id: _options_block(example) for example in cot.subset(task)
    }

    client = cot._gemini_client()
    print(f"\n=== grading {cell.name}: {len(records)} descriptions ===", flush=True)
    started = time.perf_counter()
    verdicts: list[dict | None] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_judge_one, client, record, options_by_id[record["example_id"]]): i
            for i, record in enumerate(records)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            verdicts[futures[future]] = future.result()
            if done % 25 == 0 or done == len(records):
                rate = done / max(1e-9, time.perf_counter() - started)
                print(f"    {done}/{len(records)}  {rate:.1f}/s", flush=True)

    finished = [v for v in verdicts if v is not None]
    cell.out_dir.mkdir(parents=True, exist_ok=True)
    with (cell.out_dir / "verdicts.jsonl").open("w") as handle:
        for verdict in finished:
            handle.write(json.dumps(verdict) + "\n")
    summary = _summarize(cell, finished)
    (cell.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _summarize(cell: Cell, verdicts: list[dict]) -> dict:
    ok = [v for v in verdicts if v["status"] == "ok"]
    n = len(ok)
    committed = [v for v in ok if v["direction"]]
    desc_right = [v for v in committed if v["direction"] == v["answer_key"]]
    with_letter = [v for v in ok if v["model_letter"]]
    consistent = [
        v for v in with_letter if v["direction"] and v["direction"] == v["model_letter"]
    ]
    # Description right but the returned letter wrong: perception fine, decision or
    # formatting lost it. The reverse is a letter that beat its own reasoning.
    desc_right_letter_wrong = [
        v
        for v in ok
        if v["direction"] == v["answer_key"] and not v["model_letter_correct"]
    ]
    desc_wrong_letter_right = [
        v
        for v in ok
        if v["direction"] and v["direction"] != v["answer_key"] and v["model_letter_correct"]
    ]

    def pct(part: list, whole: int) -> float | None:
        return round(100.0 * len(part) / whole, 1) if whole else None

    return {
        "cell": cell.name,
        "task": cell.task_key,
        "model": cell.model,
        "judge_model": JUDGE_MODEL,
        "n": len(verdicts),
        "judged_ok": n,
        "judge_unparsed": sum(1 for v in verdicts if v["status"] == "unparsed_judge"),
        "judge_errors": sum(1 for v in verdicts if v["status"] == "error"),
        "commits_pct": pct(committed, n),
        "asserts_identical_pct": pct([v for v in ok if v["asserts_identical"]], n),
        "describes_audio_pct": pct([v for v in ok if v["describes_audio"]], n),
        # Accuracy of the description itself, over items where it committed.
        "description_accuracy": pct(desc_right, len(committed)),
        # Same quantity over all items, abstentions counted as incorrect, which is
        # the convention the letter-level numbers already use.
        "description_accuracy_all": pct(desc_right, n),
        "letter_accuracy_all": pct([v for v in ok if v["model_letter_correct"]], n),
        "self_consistency": pct(consistent, len(with_letter)),
        "desc_right_letter_wrong": len(desc_right_letter_wrong),
        "desc_wrong_letter_right": len(desc_wrong_letter_right),
        "truncated_for_judge": sum(1 for v in verdicts if v.get("truncated_for_judge")),
    }


def load_summaries() -> list[dict]:
    root = config.RESULTS_ROOT / "cot_grade"
    if not root.is_dir():
        return []
    return [json.loads(p.read_text()) for p in sorted(root.glob("*/summary.json"))]


def report() -> None:
    summaries = load_summaries()
    if not summaries:
        print("no graded cells yet")
        return
    print("\nAcoustic description quality in the CoT condition")
    print(f"  judge: {summaries[0]['judge_model']} (claim extraction only)\n")
    head = (
        f"  {'cell':30s}{'n':>5}{'commits':>9}{'desc acc':>10}"
        f"{'desc acc*':>11}{'letter*':>9}{'consist':>9}"
    )
    print(head)
    print("  " + "-" * (len(head) - 2))
    for s in sorted(summaries, key=lambda x: (x["task"], x["model"])):
        def cell(v):
            return f"{v:>9.1f}" if isinstance(v, (int, float)) else f"{'--':>9}"
        print(
            f"  {s['cell']:30s}{s['judged_ok']:>5}"
            f"{cell(s['commits_pct'])}{cell(s['description_accuracy']):>10}"
            f"{cell(s['description_accuracy_all']):>11}"
            f"{cell(s['letter_accuracy_all'])}{cell(s['self_consistency'])}"
        )
    print("\n  desc acc  = description direction correct, over items it committed on")
    print("  desc acc* = same, over all items (abstentions counted incorrect)")
    print("  letter*   = final A/B accuracy over all items, for comparison")
    print("  consist   = description direction agrees with the returned letter")
    print("\n  Splitting perception from decision:")
    for s in sorted(summaries, key=lambda x: (x["task"], x["model"])):
        print(
            f"    {s['cell']:30s} described correctly but answered wrong: "
            f"{s['desc_right_letter_wrong']:>3}   "
            f"answered right despite a wrong description: "
            f"{s['desc_wrong_letter_right']:>3}   "
            f"asserts clips identical: {s['asserts_identical_pct']}%"
        )
    stale = [s for s in summaries if s["judge_errors"] or s["judge_unparsed"]]
    if stale:
        print("\n  judge failures (excluded from the rates above):")
        for s in stale:
            print(
                f"    {s['cell']:30s} errors={s['judge_errors']} "
                f"unparsed={s['judge_unparsed']}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(cot.COT_TASKS))
    parser.add_argument("--all", action="store_true", help="All CoT tasks.")
    parser.add_argument("--model", choices=GRADED_MODELS, help="Default: both.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, help="Grade only the first N items.")
    parser.add_argument("--report", action="store_true", help="Print from disk only.")
    args = parser.parse_args()

    if args.report:
        report()
        return 0
    if not args.task and not args.all:
        parser.error("pass --task <key> or --all (or --report)")

    tasks = list(cot.COT_TASKS) if args.all else [args.task]
    models = [args.model] if args.model else list(GRADED_MODELS)
    for task_key in tasks:
        for model in models:
            grade(
                Cell(task_key, model),
                concurrency=args.concurrency,
                limit=args.limit,
            )
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
