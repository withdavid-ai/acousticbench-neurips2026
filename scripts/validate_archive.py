#!/usr/bin/env python3
"""Validate an AcousticBench dataset archive against the v1.0 release contract.

Implements the archive contract checks of the camera-ready reproduction plan
(section 3, A1) that can be computed from the archive itself, plus optional
cross-checks against the reference trajectories:

  * question files present; row counts per task (rad 1,200, rsq 1,000,
    ser_classify 600, ser_classify_text 600, ser_intensity 400) and
    ``acousticbench_all.jsonl`` == the exact union of the per-task files
  * required fields (``messages``, ``question_set.{id,question_order,slice}``,
    ``category``, ``dimension``, ``answer_key``), audio parts per row
  * ``split`` on every row (including ser_classify_text inside all.jsonl),
    eval halves rad 600 / rsq 500 / ser_classify 300 / ser_intensity 200
  * answer keys A/B exactly 50/50 per task and per RAD dimension
  * example_id sets, eval splits and answer keys against the release contract
    digests, and (with ``--refs``) against every reference trajectory directory,
    including the 20 SER-Classify items whose key differs across model families
  * every ``audio_file`` exists inside the archive, carries a sha256, and the
    hash matches (hashed in parallel); ``checksums/sha256sum.txt`` if present
  * unique clip count and hash de-duplication (5,355 unique recordings across
    5,800 clip slots; reuse is confined to RAD), unreferenced audio files
  * total duration per task family vs the paper (RAD 1.9 h, SER 7.5 h,
    RSQ 2.2 h) and a sample-rate / format census (needs ``soundfile``; falls
    back to the stdlib ``wave`` module for WAV; skipped otherwise)
  * a literal ``C.`` / ``C)`` / ``(C)`` in prompt text (widens the judge's label
    set to A-D, see harness/runner.py)
  * a PII / leak heuristic scan of the JSONL, ``audio_files.csv``, other
    metadata files and all filenames (emails, local paths, cloud URIs,
    internal names, identity-like fields)

Statuses: FAIL (contract violated), WARN (needs a human look), INFO, SKIP, PASS.

Usage:
    python scripts/validate_archive.py --archive archive
    python scripts/validate_archive.py --archive archive --refs reference_trajectories \\
        --json-out archive_report.json

Exit status: 0 if no check FAILs, 1 if any check FAILs, 2 on usage errors.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import wave
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Iterable
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_paper_tables import (  # noqa: E402
    CASCADE_DIR,
    RAD_DIMENSIONS,
    RELEASE_CONTRACT,
    TASK_FILES,
    TASKS,
    Diagnostics,
    digest_lines,
    load_trajectory_layout,
    row_example_id,
    row_meta,
    task_from_fields,
)

try:  # Optional: richer audio metadata. The stdlib wave module is the fallback.
    import soundfile  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - depends on the environment
    soundfile = None

REPO_ROOT = Path(__file__).resolve().parents[1]
ALL_FILE = "acousticbench_all.jsonl"
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac", ".webm"}
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".md", ".yaml", ".yml"}
LABEL_LITERALS = ("C.", "C)", "(C)")

# The 20 SER-Classify items (question_order 558-599) whose answer key is "A"
# in the Kimi-family reference runs (canonical archive) and "B" in the other
# 15 audio models' runs (a stale A/B option order). See audit section 0.3.
# (example_id, question_order, split)
SER_CLASSIFY_CONFLICTS: tuple[tuple[str, int, str], ...] = (
    ("a6f1957c-5685-52d5-bcba-de13abc53987", 558, "eval"),
    ("1c1c71d8-1def-59e2-b78c-7b1283bfd371", 559, "eval"),
    ("f4fb3a18-d8b9-5398-809d-9a74232cf83a", 567, "train"),
    ("fd67b1a2-5fcd-5b56-a455-27dbdb73b0c3", 568, "train"),
    ("a54f8e15-71b0-54e0-8f9f-e183ab147921", 569, "train"),
    ("1d07cdde-9c44-5e03-bf95-9c5688872f41", 570, "train"),
    ("a3d14f69-517d-5a0b-945a-f055cc978123", 571, "eval"),
    ("786bd9de-4c3d-5b28-9197-7c7a043d19ac", 574, "eval"),
    ("e7e46004-2031-5bab-8b22-89aca366df6d", 575, "train"),
    ("2994540d-aea1-59fd-ad3a-9d3bc9e6328d", 577, "train"),
    ("554f0639-e1a9-5ae3-a774-8f6d5cb21f04", 579, "train"),
    ("69c571eb-58b0-5252-903f-754328ff8ebe", 580, "eval"),
    ("78aefa17-42b8-58db-aad5-557cce486407", 582, "train"),
    ("ad3cbfa5-4460-5cca-9f60-75cba149f29d", 586, "train"),
    ("4c836ed0-c2bc-52ec-af56-26def44462bb", 592, "train"),
    ("81d840d9-39fd-5332-b94e-47a2ca87b3c4", 593, "eval"),
    ("80767fc3-0f7a-5974-ac20-52066b7d37c3", 594, "eval"),
    ("5f5df3c3-8396-583b-92fc-72b23c98b462", 595, "eval"),
    ("0ca59f3b-4d11-5433-a519-9826083193d8", 596, "eval"),
    ("6b9a5ba7-6ed9-50f0-ad19-df0c718a74f4", 599, "train"),
)
CONFLICT_CANONICAL_KEY = "A"


@dataclass
class Expectations:
    """What a valid archive must contain. Defaults are the paper's v1.0 release."""

    task_rows: dict[str, int] = field(default_factory=lambda: {
        "rad": 1200, "rsq": 1000, "ser_classify": 600, "ser_classify_text": 600, "ser_intensity": 400,
    })
    eval_rows: dict[str, int] = field(default_factory=lambda: {
        "rad": 600, "rsq": 500, "ser_classify": 300, "ser_classify_text": 300, "ser_intensity": 200,
    })
    audio_parts_per_row: dict[str, int] = field(default_factory=lambda: {
        "rad": 2, "rsq": 2, "ser_classify": 1, "ser_classify_text": 0, "ser_intensity": 2,
    })
    rad_dimension_rows: dict[str, int] | None = field(default_factory=lambda: {
        "pitch": 400, "loudness": 400, "tempo": 200, "dynamic-range": 200,
    })
    # The paper distinguishes distinct content from clip slots.
    unique_clips: int | None = 5355
    require_no_reuse: bool = False
    # Fig. 2 hours per task family; tolerance covers one-decimal rounding.
    paper_hours: dict[str, float] | None = field(default_factory=lambda: {
        "RAD": 1.9, "SER": 7.5, "RSQ": 2.2, "Total": 11.6,
    })
    hours_tolerance: float = 0.05
    contract: dict[str, dict[str, Any]] | None = field(default_factory=lambda: RELEASE_CONTRACT)
    conflicts: tuple[tuple[str, int, str], ...] = SER_CLASSIFY_CONFLICTS
    conflict_key: str = CONFLICT_CANONICAL_KEY


@dataclass
class Check:
    name: str
    status: str
    summary: str
    details: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Report:
    archive: str
    checks: list[Check] = field(default_factory=list)

    def add(self, name: str, status: str, summary: str, details: Iterable[str] = (),
            **data: Any) -> Check:
        check = Check(name, status, summary, list(details), data)
        self.checks.append(check)
        return check

    @property
    def failed(self) -> list[Check]:
        return [check for check in self.checks if check.status == "FAIL"]

    def counts(self) -> Counter[str]:
        return Counter(check.status for check in self.checks)

    def to_json(self) -> dict[str, Any]:
        return {
            "archive": self.archive,
            "summary": dict(self.counts()),
            "passed": not self.failed,
            "checks": [asdict(check) for check in self.checks],
        }


def _limit(items: Iterable[str], n: int = 10) -> list[str]:
    items = list(items)
    return items[:n] + ([f"... and {len(items) - n} more"] if len(items) > n else [])


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------


@dataclass
class QuestionFile:
    task: str
    path: Path
    rows: list[dict[str, Any]]
    parse_errors: list[str]
    derived: bool = False  # rows taken from acousticbench_all.jsonl because the task file is missing


def load_question_file(task: str, path: Path) -> QuestionFile:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_no}: {exc}")
                continue
            if not isinstance(row, dict):
                errors.append(f"{path.name}:{line_no}: not a JSON object")
                continue
            rows.append(row)
    return QuestionFile(task, path, rows, errors)


def user_text(row: dict[str, Any]) -> str:
    """Prompt text exactly as harness/loader.py builds it (user text parts joined)."""
    for message in row.get("messages") or []:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                str(part.get("text", "")) for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
    return ""


def audio_parts(row: dict[str, Any]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for message in row.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        parts.extend(
            part for part in content if isinstance(part, dict) and part.get("type") == "audio_file"
        )
    return parts


def row_task(row: dict[str, Any], default: str) -> str:
    question_set = row.get("question_set") if isinstance(row.get("question_set"), dict) else {}
    slice_name = question_set.get("slice")
    if slice_name in TASKS:
        return str(slice_name)
    return task_from_fields(row.get("category"), row.get("dimension")) or default


def question_order(row: dict[str, Any]) -> Any:
    question_set = row.get("question_set")
    return question_set.get("question_order") if isinstance(question_set, dict) else None


# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------


def check_files(archive: Path, report: Report) -> dict[str, QuestionFile]:
    files: dict[str, QuestionFile] = {}
    missing = []
    for task in (*TASKS, "all"):
        name = ALL_FILE if task == "all" else TASK_FILES[task]
        path = archive / "questions" / name
        if path.is_file():
            files[task] = load_question_file(task, path)
        else:
            missing.append(f"questions/{name}")
    optional = {
        name: (archive / name).is_file()
        for name in ("audio_files.csv", "checksums/sha256sum.txt", "dataset-metadata.json")
    }
    report.add(
        "files",
        "FAIL" if missing else "PASS",
        f"{len(files)} of {len(TASKS) + 1} question files present"
        + (f"; missing: {', '.join(missing)}" if missing else ""),
        [f"{name}: {'present' if present else 'absent'}" for name, present in optional.items()],
        missing=missing,
        optional=optional,
    )
    errors = [error for qfile in files.values() for error in qfile.parse_errors]
    report.add(
        "jsonl_parse",
        "FAIL" if errors else "PASS",
        f"{len(errors)} unparseable JSONL lines",
        _limit(errors),
    )
    if "all" in files:
        # Keep checking content when a per-task file is missing: take that
        # task's rows from all.jsonl (by question_set.slice or category/dimension).
        for task in TASKS:
            if task not in files:
                rows = [row for row in files["all"].rows if row_task(row, "") == task]
                files[task] = QuestionFile(task, files["all"].path, rows, [], derived=True)
        derived = [task for task in TASKS if files[task].derived]
        if derived:
            report.add("derived_task_views", "INFO",
                       f"per-task checks for {', '.join(derived)} use rows from {ALL_FILE}")
    return files


def check_row_counts(files: dict[str, QuestionFile], exp: Expectations, report: Report) -> None:
    details = []
    bad = False
    expected_all = sum(exp.task_rows.values())
    for task in (*TASKS, "all"):
        if task not in files:
            continue
        n = len(files[task].rows)
        expected = expected_all if task == "all" else exp.task_rows[task]
        ok = n == expected and not files[task].derived
        bad |= not ok
        source = f" (from {ALL_FILE}; task file missing)" if files[task].derived else ""
        details.append(f"{task}: {n} rows{source} (expected {expected}){'' if ok else '  <-- MISMATCH'}")
    report.add("row_counts", "FAIL" if bad else "PASS",
               "row counts per task" + (" do not match" if bad else " match the release"), details)


def check_duplicates(files: dict[str, QuestionFile], report: Report) -> None:
    details = []
    owner: dict[str, str] = {}
    cross = []
    for task in TASKS:
        if task not in files:
            continue
        counts = Counter(str(row.get("example_id") or "") for row in files[task].rows)
        dups = [eid for eid, count in counts.items() if count > 1]
        empty = counts.get("", 0)
        if dups:
            details.append(f"{task}: {len(dups)} duplicated example_ids, e.g. {dups[0]}")
        if empty:
            details.append(f"{task}: {empty} rows without example_id")
        for eid in counts:
            if eid and eid in owner and owner[eid] != task:
                cross.append(f"{eid} in {owner[eid]} and {task}")
            owner.setdefault(eid, task)
    details.extend(f"cross-task duplicate: {item}" for item in _limit(cross))
    report.add("duplicate_ids", "FAIL" if details else "PASS",
               "example_ids are unique within and across tasks" if not details
               else "duplicate or missing example_ids", details)


def check_all_union(files: dict[str, QuestionFile], report: Report) -> None:
    if "all" not in files:
        report.add("all_union", "SKIP", f"{ALL_FILE} missing")
        return
    if any(files.get(task) is None or files[task].derived for task in TASKS):
        report.add("all_union", "FAIL", "cannot verify: per-task question files are missing")
        return
    per_task: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        for row in files[task].rows:
            per_task[str(row.get("example_id"))] = row
    all_rows: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in files["all"].rows:
        eid = str(row.get("example_id"))
        duplicates += eid in all_rows
        all_rows[eid] = row
    only_all = sorted(set(all_rows) - set(per_task))
    only_tasks = sorted(set(per_task) - set(all_rows))
    differing = []
    for eid in sorted(set(all_rows) & set(per_task)):
        a, b = all_rows[eid], per_task[eid]
        if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            fields = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            differing.append(f"{eid}: fields differ: {', '.join(fields)}")
    bad = bool(only_all or only_tasks or differing or duplicates)
    details = []
    if only_all:
        details.append(f"{len(only_all)} IDs only in {ALL_FILE}, e.g. {only_all[0]}")
    if only_tasks:
        details.append(f"{len(only_tasks)} IDs missing from {ALL_FILE}, e.g. {only_tasks[0]}")
    if duplicates:
        details.append(f"{duplicates} duplicate IDs inside {ALL_FILE}")
    details.extend(_limit(differing))
    report.add("all_union", "FAIL" if bad else "PASS",
               f"{ALL_FILE} is the exact union of the per-task files" if not bad
               else f"{ALL_FILE} differs from the union of the per-task files", details,
               only_in_all=len(only_all), missing_from_all=len(only_tasks), differing_rows=len(differing))


def check_required_fields(files: dict[str, QuestionFile], exp: Expectations, report: Report) -> None:
    problems: Counter[str] = Counter()
    examples: dict[str, str] = {}
    warnings: Counter[str] = Counter()
    warn_examples: dict[str, str] = {}

    def flag(kind: str, eid: str, severe: bool = True) -> None:
        bucket, store = (problems, examples) if severe else (warnings, warn_examples)
        bucket[kind] += 1
        store.setdefault(kind, eid)

    for task, qfile in files.items():
        for row in qfile.rows:
            eid = str(row.get("example_id") or "<missing>")
            tag = f"{task}:{eid}"
            if not isinstance(row.get("example_id"), str) or not row.get("example_id"):
                flag("example_id missing or not a string", tag)
            messages = row.get("messages")
            if not isinstance(messages, list) or not any(
                isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), list)
                for m in messages
            ):
                flag("messages lacks a user message with a content list", tag)
            for message in messages if isinstance(messages, list) else []:
                content = message.get("content") if isinstance(message, dict) else None
                for part in content if isinstance(content, list) else []:
                    if not isinstance(part, dict) or part.get("type") not in ("text", "audio_file"):
                        flag("content part with unknown type", tag)
                    elif part["type"] == "text" and not isinstance(part.get("text"), str):
                        flag("text part without a string 'text'", tag)
            question_set = row.get("question_set")
            if not isinstance(question_set, dict):
                flag("question_set missing", tag)
            else:
                for key in ("id", "slice"):
                    if not question_set.get(key):
                        flag(f"question_set.{key} missing", tag)
                order = question_set.get("question_order")
                if not isinstance(order, int) or isinstance(order, bool):
                    flag("question_set.question_order missing or not an int", tag)
                slice_name = question_set.get("slice")
                if task != "all" and slice_name and slice_name != task:
                    flag("question_set.slice != file task", tag, severe=False)
            for key in ("category", "dimension"):
                if not isinstance(row.get(key), str) or not row.get(key):
                    flag(f"{key} missing", tag)
            if row.get("answer_key") not in ("A", "B"):
                flag("answer_key not 'A' or 'B'", tag)
            if task == "all":
                continue
            implied = task_from_fields(row.get("category"), row.get("dimension"))
            if implied != task:
                flag(f"category/dimension imply {implied}, not the file task", tag, severe=False)
            n_parts = len(audio_parts(row))
            if n_parts != exp.audio_parts_per_row[task]:
                flag(f"{task}: {n_parts} audio parts (expected {exp.audio_parts_per_row[task]})", tag)
            for part in audio_parts(row):
                if not isinstance(part.get("path"), str) or not part.get("path"):
                    flag("audio_file part without a path", tag)
                if not isinstance(part.get("order"), int):
                    flag("audio_file part without an int order", tag, severe=False)
    details = [f"{kind}: {count} rows (e.g. {examples[kind]})" for kind, count in problems.most_common()]
    details += [f"[warn] {kind}: {count} rows (e.g. {warn_examples[kind]})" for kind, count in warnings.most_common()]
    status = "FAIL" if problems else ("WARN" if warnings else "PASS")
    report.add("required_fields", status,
               "all rows carry the fields the loader requires" if not problems
               else f"{sum(problems.values())} field problems", details)


def check_splits(files: dict[str, QuestionFile], exp: Expectations, report: Report) -> dict[str, dict[str, str]]:
    """Split present and valid everywhere; eval halves; text split mirrors audio split."""
    splits: dict[str, dict[str, str]] = {}
    problems = []
    details = []
    for task, qfile in files.items():
        missing = [str(row.get("example_id")) for row in qfile.rows if row.get("split") not in ("train", "eval")]
        if missing and not qfile.derived:
            problems.append(f"{qfile.path.name}: {len(missing)} rows without a valid split, e.g. {missing[0]}")
            if task == "all":
                text_missing = [
                    str(row.get("example_id")) for row in qfile.rows
                    if row.get("split") not in ("train", "eval") and row_task(row, "") == "ser_classify_text"
                ]
                if text_missing:
                    problems.append(
                        f"{ALL_FILE}: {len(text_missing)} ser_classify_text rows lack split; the loader "
                        "does not infer it for --task all, so they vanish under --split eval"
                    )
        if task == "all":
            continue
        splits[task] = {str(row.get("example_id")): str(row.get("split")) for row in qfile.rows}
        counts = Counter(splits[task].values())
        expected_eval = exp.eval_rows[task]
        ok = counts.get("eval", 0) == expected_eval and counts.get("train", 0) == exp.task_rows[task] - expected_eval
        details.append(
            f"{task}: eval {counts.get('eval', 0)} / train {counts.get('train', 0)} "
            f"(expected {expected_eval} / {exp.task_rows[task] - expected_eval})" + ("" if ok else "  <-- MISMATCH")
        )
        if not ok:
            problems.append(f"{task}: eval/train sizes differ from the release")
    if "ser_classify" in files and "ser_classify_text" in files:
        audio_by_order = {question_order(r): r.get("split") for r in files["ser_classify"].rows}
        mismatched = [
            str(r.get("example_id")) for r in files["ser_classify_text"].rows
            if question_order(r) in audio_by_order and r.get("split") != audio_by_order[question_order(r)]
        ]
        details.append(f"ser_classify_text split mirrors ser_classify by question_order: "
                       f"{'yes' if not mismatched else f'no ({len(mismatched)} rows differ)'}")
        if mismatched:
            problems.append(f"ser_classify_text: {len(mismatched)} rows have a different split than the "
                            f"ser_classify row with the same question_order, e.g. {mismatched[0]}")
    report.add("splits", "FAIL" if problems else "PASS",
               "split present on every row; eval halves match" if not problems else "split problems",
               problems + details)
    return splits


def check_balance(files: dict[str, QuestionFile], exp: Expectations, report: Report) -> None:
    problems = []
    details = []
    for task in TASKS:
        if task not in files:
            continue
        rows = files[task].rows
        keys = Counter(row.get("answer_key") for row in rows)
        if keys.get("A", 0) != keys.get("B", 0):
            problems.append(f"{task}: A {keys.get('A', 0)} / B {keys.get('B', 0)} (not 50/50)")
        by_split = Counter((row.get("split"), row.get("answer_key")) for row in rows)
        details.append(
            f"{task}: A {keys.get('A', 0)} / B {keys.get('B', 0)}; eval A {by_split[('eval', 'A')]} / "
            f"B {by_split[('eval', 'B')]}; train A {by_split[('train', 'A')]} / B {by_split[('train', 'B')]}"
        )
        if task == "rad":
            dims = Counter((str(row.get("dimension")), row.get("answer_key")) for row in rows)
            dim_counts = Counter(str(row.get("dimension")) for row in rows)
            unknown = sorted(set(dim_counts) - set(RAD_DIMENSIONS))
            if unknown:
                problems.append(f"rad: dimensions outside {RAD_DIMENSIONS}: {unknown}")
            for dim in RAD_DIMENSIONS:
                a, b = dims[(dim, "A")], dims[(dim, "B")]
                expected = (exp.rad_dimension_rows or {}).get(dim)
                if a != b:
                    problems.append(f"rad/{dim}: A {a} / B {b} (not 50/50)")
                if expected is not None and a + b != expected:
                    problems.append(f"rad/{dim}: {a + b} rows (expected {expected})")
                details.append(f"rad/{dim}: A {a} / B {b}")
    report.add("answer_balance", "FAIL" if problems else "PASS",
               "answer keys are 50/50 per task and per RAD dimension" if not problems
               else "answer-key balance problems", problems + details)


def check_contract(files: dict[str, QuestionFile], exp: Expectations, report: Report) -> None:
    if exp.contract is None:
        report.add("release_contract", "SKIP", "no contract digests configured")
        return
    problems = []
    details = []
    for task in TASKS:
        if task not in files:
            continue
        rows = files[task].rows
        ids = [str(row.get("example_id")) for row in rows]
        evals = [str(row.get("example_id")) for row in rows if row.get("split") == "eval"]
        lines = [f"{row.get('example_id')}\t{row.get('split')}\t{row.get('answer_key')}" for row in rows]
        contract = exp.contract[task]
        flags = {
            "ids": digest_lines(ids) == contract["ids"],
            "eval_ids": digest_lines(evals) == contract["eval_ids"],
            "keys": digest_lines(lines) == contract["contract"],
        }
        details.append(f"{task}: " + ", ".join(f"{k} {'match' if v else 'DIFFER'}" for k, v in flags.items()))
        if not flags["ids"]:
            problems.append(f"{task}: example_id set differs from the release (and from the reference trajectories)")
        elif not flags["eval_ids"]:
            problems.append(f"{task}: eval split differs from the release / LoRA reference runs")
        elif not flags["keys"]:
            hint = ""
            if task == "ser_classify":
                by_id = {str(row.get("example_id")): row.get("answer_key") for row in rows}
                conflict_keys = Counter(by_id.get(eid) for eid, _, _ in exp.conflicts)
                hint = (f"; keys on the {len(exp.conflicts)} known conflict items: {dict(conflict_keys)} "
                        f"(release contract expects {exp.conflict_key}, the Kimi-family key)")
            problems.append(f"{task}: answer keys differ from the release contract{hint}")
    report.add("release_contract", "FAIL" if problems else "PASS",
               "example_ids, eval split and answer keys match the v1.0 release contract" if not problems
               else "archive differs from the v1.0 release contract", problems + details)


def check_label_literals(files: dict[str, QuestionFile], report: Report) -> None:
    hits = []
    per_task: Counter[str] = Counter()
    for task in TASKS:
        for row in files.get(task, QuestionFile(task, Path(), [], [])).rows:
            text = user_text(row)
            found = [literal for literal in LABEL_LITERALS if literal in text]
            if found:
                per_task[task] += 1
                index = text.find(found[0])
                snippet = text[max(0, index - 30): index + 30].replace("\n", " ")
                hits.append(f"{task}:{row.get('example_id')}: {found} in '...{snippet}...'")
    report.add(
        "label_literals",
        "FAIL" if hits else "PASS",
        "no literal 'C.', 'C)' or '(C)' in prompt text" if not hits
        else f"{len(hits)} prompts contain 'C.'/'C)'/'(C)', which widens the judge label set to A-D",
        _limit(hits, 20),
        per_task=dict(per_task),
    )


# ---------------------------------------------------------------------------
# Audio checks
# ---------------------------------------------------------------------------


def _safe_resolve(archive: Path, rel: str) -> Path | None:
    if not rel or os.path.isabs(rel) or ".." in Path(rel).parts:
        return None
    resolved = (archive / rel).resolve()
    try:
        resolved.relative_to(archive.resolve())
    except ValueError:
        return None
    return resolved


def inspect_audio(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path)}
    try:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
                size += len(chunk)
        info["sha256"] = digest.hexdigest()
        info["bytes"] = size
    except OSError as exc:
        info["error"] = f"read failed: {exc}"
        return info
    if soundfile is not None:
        try:
            meta = soundfile.info(str(path))
            info.update(duration=float(meta.frames) / meta.samplerate if meta.samplerate else None,
                        samplerate=meta.samplerate, channels=meta.channels,
                        format=meta.format, subtype=meta.subtype)
            return info
        except Exception as exc:  # noqa: BLE001 - report and fall through to wave
            info["audio_error"] = f"soundfile: {exc}"
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                rate = handle.getframerate()
                info.update(duration=handle.getnframes() / rate if rate else None, samplerate=rate,
                            channels=handle.getnchannels(), format="WAV",
                            subtype=f"PCM_{8 * handle.getsampwidth()}")
                info.pop("audio_error", None)
        except (wave.Error, EOFError, OSError) as exc:
            info["audio_error"] = f"wave: {exc}"
    return info


@dataclass
class AudioRef:
    task: str
    example_id: str
    path: str
    sha256: str | None


def collect_audio_refs(files: dict[str, QuestionFile]) -> list[AudioRef]:
    refs = []
    for task in TASKS:
        for row in files.get(task, QuestionFile(task, Path(), [], [])).rows:
            for part in audio_parts(row):
                refs.append(AudioRef(task, str(row.get("example_id")), str(part.get("path") or ""),
                                     part.get("sha256") if isinstance(part.get("sha256"), str) else None))
    return refs


def check_audio(archive: Path, files: dict[str, QuestionFile], exp: Expectations, workers: int,
                report: Report) -> dict[str, dict[str, Any]]:
    refs = collect_audio_refs(files)
    unsafe = sorted({ref.path for ref in refs if _safe_resolve(archive, ref.path) is None})
    unique_paths = sorted({ref.path for ref in refs if ref.path not in unsafe})
    resolved = {rel: _safe_resolve(archive, rel) for rel in unique_paths}
    missing = sorted(rel for rel, path in resolved.items() if path is None or not path.is_file())
    present = {rel: path for rel, path in resolved.items() if path is not None and path.is_file()}

    logger.info(f"Hashing {len(present)} audio files with {workers} threads")
    infos: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rel, info in zip(present, pool.map(inspect_audio, present.values())):
            infos[rel] = info

    no_hash = [f"{ref.task}:{ref.example_id}:{ref.path}" for ref in refs if not ref.sha256]
    mismatched = [
        f"{ref.task}:{ref.example_id}:{ref.path}: expected {ref.sha256[:12]}..., "
        f"got {infos[ref.path].get('sha256', '?')[:12]}..."
        for ref in refs
        if ref.sha256 and ref.path in infos and infos[ref.path].get("sha256") != ref.sha256.lower()
    ]
    read_errors = [f"{rel}: {info['error']}" for rel, info in infos.items() if "error" in info]
    problems = []
    if unsafe:
        problems.append(f"{len(unsafe)} audio paths are absolute or escape the archive, e.g. {unsafe[0]}")
    if missing:
        problems.append(f"{len(missing)} referenced audio files are missing, e.g. {missing[0]}")
    if no_hash:
        problems.append(f"{len(no_hash)} audio parts carry no sha256 (run.py --verify-checksums skips them silently)")
    if mismatched:
        problems.append(f"{len(mismatched)} audio parts fail sha256 verification")
    if read_errors:
        problems.append(f"{len(read_errors)} audio files could not be read")
    report.add(
        "audio_integrity",
        "FAIL" if problems else "PASS",
        f"{len(refs)} audio parts, {len(unique_paths)} unique paths; all exist, carry sha256 and verify"
        if not problems else "audio integrity problems",
        problems + _limit(no_hash, 5) + _limit(mismatched, 10) + _limit(read_errors, 5),
        audio_parts=len(refs), unique_paths=len(unique_paths), missing=len(missing),
        missing_sha256=len(no_hash), sha256_mismatches=len(mismatched),
    )

    # Uniqueness and reuse.
    path_uses = Counter(ref.path for ref in refs)
    reused_paths = {path: n for path, n in path_uses.items() if n > 1}
    by_hash: dict[str, set[str]] = defaultdict(set)
    for rel, info in infos.items():
        if info.get("sha256"):
            by_hash[info["sha256"]].add(rel)
    dup_content = {digest: paths for digest, paths in by_hash.items() if len(paths) > 1}
    tasks_by_path: dict[str, set[str]] = defaultdict(set)
    for ref in refs:
        tasks_by_path[ref.path].add(ref.task)
    cross_task = sorted(path for path, tasks in tasks_by_path.items() if len(tasks) > 1)
    unique_content = len(by_hash)
    details = [
        f"audio slots referenced: {len(refs)}",
        f"unique paths: {len(unique_paths)}; unique contents (sha256): {unique_content}",
        f"paths used by more than one slot: {len(reused_paths)}; shared across tasks: {len(cross_task)}",
        f"distinct paths with byte-identical content: {sum(len(p) for p in dup_content.values())} "
        f"in {len(dup_content)} groups",
    ]
    details += _limit((f"reused path {p} x{n}" for p, n in sorted(reused_paths.items())), 5)
    details += _limit((f"identical content: {sorted(p)[:3]}" for p in dup_content.values()), 5)
    problems = []
    if exp.unique_clips is not None and unique_content != exp.unique_clips:
        problems.append(f"{unique_content} unique clips by content (paper: {exp.unique_clips:,})")
    if exp.require_no_reuse and (reused_paths or dup_content):
        problems.append("clips are reused but this validation configuration prohibits reuse")
    if not exp.require_no_reuse:
        for paths in dup_content.values():
            if set().union(*(tasks_by_path[p] for p in paths)) != {"rad"}:
                problems.append("duplicate content outside RAD")
                break
    report.add("audio_uniqueness", "FAIL" if problems else "PASS",
               f"{unique_content} unique clips; reuse checked against task policy" if not problems else "; ".join(problems),
               problems + details, unique_contents=unique_content, reused_paths=len(reused_paths),
               duplicate_content_groups=len(dup_content), cross_task_paths=len(cross_task))

    # Unreferenced audio on disk.
    on_disk = sorted(
        str(path.relative_to(archive)) for path in archive.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    unreferenced = sorted(set(on_disk) - set(unique_paths))
    report.add("audio_unreferenced", "WARN" if unreferenced else "PASS",
               f"{len(unreferenced)} audio files on disk are not referenced by any question",
               _limit(unreferenced, 10), unreferenced=len(unreferenced))

    check_durations(refs, infos, exp, report)
    return infos


def check_durations(refs: list[AudioRef], infos: dict[str, dict[str, Any]], exp: Expectations,
                    report: Report) -> None:
    have_duration = [info for info in infos.values() if info.get("duration") is not None]
    if not have_duration:
        report.add("audio_durations", "SKIP",
                   "no durations available (install soundfile; WAV falls back to the wave module)")
        report.add("audio_format_census", "SKIP", "no audio metadata available")
        return
    families = {"RAD": {"rad"}, "SER": {"ser_classify", "ser_intensity"}, "RSQ": {"rsq"},
                "Total": {"rad", "rsq", "ser_classify", "ser_intensity", "ser_classify_text"}}
    task_paths: dict[str, set[str]] = defaultdict(set)
    for ref in refs:
        task_paths[ref.task].add(ref.path)
    details = []
    problems = []
    hours: dict[str, float] = {}
    for family, tasks in families.items():
        # Published hours sum clip slots, including repeated RAD recordings.
        durations: list[float] = []
        unknown = 0
        for ref in refs:
            if ref.task not in tasks:
                continue
            info = infos.get(ref.path, {})
            if info.get("duration") is None:
                unknown += 1
                continue
            durations.append(float(info["duration"]))
        hours[family] = sum(durations) / 3600.0
        paper = (exp.paper_hours or {}).get(family)
        line = f"{family}: {hours[family]:.3f} h over {len(durations)} clip slots"
        if unknown:
            line += f" ({unknown} clips without a duration)"
            problems.append(f"{family}: incomplete duration metadata; install soundfile")
        if paper is not None:
            line += f"; paper {paper} h"
            if abs(hours[family] - paper) > exp.hours_tolerance:
                line += "  <-- differs"
                problems.append(f"{family}: {hours[family]:.2f} h vs paper {paper} h")
        details.append(line)
    for task in TASKS:
        clips = {infos[p].get("sha256") or p: float(infos[p]["duration"]) for p in task_paths.get(task, ())
                 if p in infos and infos[p].get("duration") is not None}
        if clips:
            seconds = list(clips.values())
            details.append(f"  {task}: {sum(seconds) / 3600:.3f} h, {len(seconds)} clips, "
                           f"{min(seconds):.2f}-{max(seconds):.2f} s per clip")
    report.add("audio_durations", "WARN" if problems else "PASS",
               "durations match the paper's hours" if not problems
               else "durations differ from the paper's hours", problems + details, hours=hours)

    census: dict[str, Counter[str]] = defaultdict(Counter)
    for ref in refs:
        info = infos.get(ref.path)
        if not info or info.get("samplerate") is None:
            continue
        census[ref.task][f"{info.get('format')}/{info.get('subtype')} {info.get('samplerate')} Hz "
                         f"{info.get('channels')}ch"] += 1
    lines = [f"{task}: " + ", ".join(f"{k} x{v}" for k, v in counts.most_common()) for task, counts in census.items()]
    rates = {key.split(" ")[1] for counts in census.values() for key in counts}
    report.add("audio_format_census", "INFO",
               f"sample rates in use: {', '.join(sorted(rates, key=lambda r: int(r) if r.isdigit() else 0))} Hz "
               "(native sample rates; see the paper appendix)", lines,
               census={task: dict(counts) for task, counts in census.items()})


def check_checksum_file(archive: Path, infos: dict[str, dict[str, Any]], workers: int,
                        report: Report) -> None:
    path = archive / "checksums" / "sha256sum.txt"
    if not path.is_file():
        report.add("sha256sum_file", "SKIP", "checksums/sha256sum.txt not present")
        return
    entries: dict[str, str] = {}
    malformed = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        match = re.match(r"^([0-9a-fA-F]{64})\s+\*?(.+)$", line.strip())
        if not match:
            malformed.append(f"line {line_no}: {line[:80]}")
            continue
        entries[match.group(2).strip()] = match.group(1).lower()
    to_hash = {}
    missing = []
    for rel in entries:
        target = _safe_resolve(archive, rel)
        if target is None or not target.is_file():
            missing.append(rel)
        elif rel not in infos:
            to_hash[rel] = target
    computed = {rel: info.get("sha256") for rel, info in infos.items()}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for rel, info in zip(to_hash, pool.map(inspect_audio, to_hash.values())):
            computed[rel] = info.get("sha256")
    mismatched = [rel for rel, digest in entries.items() if rel in computed and computed[rel] != digest]
    unlisted_audio = sorted(set(infos) - set(entries))
    questions_listed = sorted(rel for rel in entries if rel.startswith("questions/"))
    problems = []
    if malformed:
        problems.append(f"{len(malformed)} malformed lines")
    if missing:
        problems.append(f"{len(missing)} listed files are missing, e.g. {missing[0]}")
    if mismatched:
        problems.append(f"{len(mismatched)} listed files fail verification, e.g. {mismatched[0]}")
    details = [
        f"{len(entries)} entries; {len(entries) - len(missing) - len(mismatched)} verified",
        f"question files covered: {', '.join(questions_listed) if questions_listed else 'none'}",
        f"referenced audio not listed: {len(unlisted_audio)}",
    ]
    status = "FAIL" if problems else ("WARN" if unlisted_audio or not questions_listed else "PASS")
    report.add("sha256sum_file", status,
               "checksums/sha256sum.txt verifies" if not problems else "checksums/sha256sum.txt problems",
               problems + details + _limit(malformed, 5) + _limit(mismatched, 5),
               entries=len(entries), missing=len(missing), mismatched=len(mismatched),
               unlisted_audio=len(unlisted_audio))


# ---------------------------------------------------------------------------
# PII / leak heuristics
# ---------------------------------------------------------------------------

SEVERE_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
    "local_path": re.compile(
        r"(?<![\w.])(?:/Users/|/home/|/private/|/var/folders/|/tmp/|/mnt/|/root/|/opt/|/workspace/|"
        r"/data/|[A-Za-z]:\\)"
    ),
    "cloud_uri": re.compile(r"\b(?:s3|s3a|gs|gcs|az|abfss?|wasbs?|hf|modal)://", re.IGNORECASE),
}
SOFT_PATTERNS: dict[str, re.Pattern[str]] = {
    "url": re.compile(r"https?://[^\s\"'<>)]+"),
    "internal_name": re.compile(
        r"(?i)(?<![a-z])(?:withdavid|davidai|david[-_ ]ai|private-workspace|private-review-account|"
        r"modal\.run|slack\.com|notion\.so|linear\.app|internal)(?![a-z])"
    ),
    "source_corpus_name": re.compile(
        r"(?i)(?<![a-z])(?:emolab|iemocap|tess|msp[-_ ]?podcast|msp[-_ ]?improv|crema[-_ ]?d|ravdess|"
        r"savee|meld|librispeech|commonvoice|common[-_ ]voice|vctk)(?![a-z])"
    ),
    "phone": re.compile(r"(?<![\w+])\+\d{1,3}[\s.-]?\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}(?!\w)"),
    "worker_like_token": re.compile(
        r"(?i)(?<![a-z])(?:worker|rater|annotator|labeler|participant|contributor|user|speaker|actor|"
        r"session|prolific|mturk)[-_]?(?:id)?[-_]?[0-9a-f]{3,}"
    ),
    "long_digit_run": re.compile(r"(?<![0-9A-Fa-f])\d{9,}(?![0-9A-Fa-f])"),
}
UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
IDENTITY_KEY = re.compile(
    r"(?i)^(?:.*_)?(?:worker|rater|annotator|labeler|contributor|participant|respondent|prolific|"
    r"mturk|turker|user|username|speaker|actor|voice_?actor|email|e_?mail|phone|full_?name|"
    r"first_?name|last_?name|session|uploader|author|ip|device)(?:_?(?:id|name|email|hash))?$"
)


def _walk_strings(value: Any, path: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_strings(child, f"{path}.{key}" if path else str(key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_strings(child, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def _walk_keys(value: Any, path: str = "") -> Iterable[tuple[str, str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield child_path, str(key), child
            yield from _walk_keys(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_keys(child, f"{path}[{index}]")


class LeakScanner:
    """Aggregates pattern hits by distinct matched value (with one example location each)."""

    def __init__(self) -> None:
        self.values: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.where: dict[tuple[str, str, str], str] = {}

    def _hit(self, severity: str, kind: str, value: str, where: str) -> None:
        self.values[(severity, kind)][value] += 1
        self.where.setdefault((severity, kind, value), where)

    def scan_text(self, text: str, where: str, severe_ok: bool = False) -> None:
        for kind, pattern in SEVERE_PATTERNS.items():
            for match in pattern.finditer(text):
                self._hit("WARN" if severe_ok else "FAIL", kind, match.group(0), where)
        # UUIDs (example_ids, question-set ids) are expected; drop them before
        # the heuristic patterns so their digit groups are not reported.
        text = UUID_PATTERN.sub(" ", text)
        for kind, pattern in SOFT_PATTERNS.items():
            for match in pattern.finditer(text):
                self._hit("WARN", kind, match.group(0), where)

    def scan_key(self, path: str, key: str, value: Any, where: str) -> None:
        if path.startswith("question_set.") or not IDENTITY_KEY.match(key):
            return
        sample = value if isinstance(value, (str, int, float)) else type(value).__name__
        generic_path = re.sub(r"\[\d+\]", "[]", path)
        self._hit("WARN", "identity_like_field", f"{generic_path} = {str(sample)[:60]!r}", where)

    def summary(self, severity: str) -> dict[str, Counter[str]]:
        return {kind: counts for (sev, kind), counts in self.values.items() if sev == severity}

    def describe(self, severity: str, kind: str, counts: Counter[str], top: int = 8) -> list[str]:
        lines = [f"[{severity}] {kind}: {sum(counts.values())} occurrences, {len(counts)} distinct values"]
        for value, count in counts.most_common(top):
            lines.append(f"    {value!r} x{count} (e.g. {self.where[(severity, kind, value)]})")
        if len(counts) > top:
            lines.append(f"    ... and {len(counts) - top} more distinct values")
        return lines


def check_leaks(archive: Path, files: dict[str, QuestionFile], report: Report) -> None:
    scanner = LeakScanner()
    for qfile in files.values():
        if qfile.derived:
            continue
        for row in qfile.rows:
            where = f"{qfile.path.name}:{row.get('example_id')}"
            for path, text in _walk_strings(row):
                if path.endswith("sha256"):
                    continue
                scanner.scan_text(text, f"{where}:{path}")
            for path, key, value in _walk_keys(row):
                scanner.scan_key(path, key, value, where)

    csv_path = archive / "audio_files.csv"
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for column in reader.fieldnames or []:
                scanner.scan_key(column, column, "<column>", "audio_files.csv header")
            for line_no, record in enumerate(reader, start=2):
                for column, cell in record.items():
                    if cell:
                        scanner.scan_text(str(cell), f"audio_files.csv:{line_no}:{column}")

    question_dir = (archive / "questions").resolve()
    for path in sorted(archive.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(archive))
        scanner.scan_text(rel, f"filename {rel}")
        if (path.suffix.lower() in TEXT_SUFFIXES and path.resolve().parent != question_dir
                and path.name not in ("audio_files.csv", "sha256sum.txt")
                and path.stat().st_size < 50 * 1024 * 1024):
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            # Metadata files (dataset card, provenance) may legitimately hold a
            # contact email or URL, so their hits are warnings, not failures.
            for line_no, line in enumerate(text.splitlines(), start=1):
                scanner.scan_text(line, f"{rel}:{line_no}", severe_ok=True)

    severe = scanner.summary("FAIL")
    soft = scanner.summary("WARN")
    details = []
    for kind, counts in sorted(severe.items()):
        details.extend(scanner.describe("FAIL", kind, counts))
    for kind, counts in sorted(soft.items()):
        details.extend(scanner.describe("WARN", kind, counts))
    status = "FAIL" if severe else ("WARN" if soft else "PASS")
    report.add(
        "pii_leak_scan",
        status,
        "no emails, local paths, cloud URIs or identity-like fields found" if status == "PASS"
        else f"{sum(sum(c.values()) for c in severe.values())} severe and "
             f"{sum(sum(c.values()) for c in soft.values())} heuristic hits (review by hand)",
        details,
        severe={kind: sum(counts.values()) for kind, counts in severe.items()},
        heuristic={kind: sum(counts.values()) for kind, counts in soft.items()},
    )


# ---------------------------------------------------------------------------
# Reference trajectory cross-checks and the SER-Classify key conflict
# ---------------------------------------------------------------------------


def check_against_refs(files: dict[str, QuestionFile], refs: Path | None, exp: Expectations,
                       report: Report) -> None:
    conflict_ids = {eid for eid, _, _ in exp.conflicts}
    archive_rows: dict[str, tuple[str, dict[str, Any]]] = {}
    for task in TASKS:
        for row in files.get(task, QuestionFile(task, Path(), [], [])).rows:
            archive_rows[str(row.get("example_id"))] = (task, row)
    task_ids = {task: {eid for eid, (t, _) in archive_rows.items() if t == task} for task in TASKS}
    eval_ids = {task: {eid for eid in ids if archive_rows[eid][1].get("split") == "eval"}
                for task, ids in task_ids.items()}

    if refs is None or not refs.is_dir():
        keys = Counter(archive_rows.get(eid, ("", {}))[1].get("answer_key") for eid in conflict_ids)
        report.add("reference_ids", "SKIP", "no --refs directory; reference cross-checks skipped")
        report.add("ser_classify_key_conflict", "INFO",
                   f"archive keys on the {len(conflict_ids)} known conflict items: {dict(keys)} "
                   f"(Kimi-family reference runs used {exp.conflict_key}; 15 other models used B)")
        return

    diag = Diagnostics()
    groups = load_trajectory_layout(refs, diag)
    id_problems, id_known, id_ok = [], [], 0
    split_bad: Counter[str] = Counter()
    meta_bad: Counter[str] = Counter()
    key_conflict_dirs: dict[str, set[str]] = {}
    key_other: dict[str, set[str]] = {}
    ref_keys: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for (model, task, split), group in sorted(groups.items()):
        label = f"{model}/{task}" + ("" if split == "all" else f" ({split})")
        ids = group.id_set()
        target = eval_ids[task] if split == "eval" else task_ids[task]
        if ids == target:
            id_ok += 1
        else:
            other = [t for t in TASKS if t != task and task_ids[t] and ids <= task_ids[t]]
            message = (f"{label}: {len(ids & target)}/{len(target)} archive IDs present, "
                       f"{len(ids - set(archive_rows))} IDs unknown to the archive"
                       + (f"; IDs are the {other[0]} set" if other else ""))
            if model == CASCADE_DIR and task == "ser_classify" and other == ["ser_classify_text"]:
                id_known.append(message + " (known: Cascade ser_classify holds the text-diagnostic items)")
            else:
                id_problems.append(message)
        conflicts_here: set[str] = set()
        others_here: set[str] = set()
        for run in group.runs:
            for row in run.rows:
                eid = row_example_id(row)
                if eid not in archive_rows:
                    continue
                _, arow = archive_rows[eid]
                meta = row_meta(row)
                if meta.get("split") in ("train", "eval") and meta["split"] != arow.get("split"):
                    split_bad[label] += 1
                if meta.get("dimension") is not None and meta.get("dimension") != arow.get("dimension"):
                    meta_bad[f"{label} dimension"] += 1
                if meta.get("question_order") is not None and meta.get("question_order") != question_order(arow):
                    meta_bad[f"{label} question_order"] += 1
                key = row.get("correct") or meta.get("answer_key")
                ref_keys[eid][str(key)].add(model)
                if key != arow.get("answer_key"):
                    (conflicts_here if eid in conflict_ids else others_here).add(eid)
        if conflicts_here:
            key_conflict_dirs[label] = conflicts_here
        if others_here:
            key_other[label] = others_here

    report.add(
        "reference_ids",
        "FAIL" if id_problems else ("WARN" if id_known else "PASS"),
        f"{id_ok} of {len(groups)} reference directories match the archive ID sets exactly"
        + (f"; {len(id_problems)} do not" if id_problems else "")
        + (f"; {len(id_known)} known exception" if id_known else ""),
        _limit(id_problems, 20) + id_known,
    )
    problems = [f"{label}: {n} rows with a different meta.split" for label, n in split_bad.items()]
    problems += [f"{label}: {n} rows differ" for label, n in meta_bad.items()]
    report.add("reference_split_meta", "FAIL" if problems else "PASS",
               "reference meta.split / dimension / question_order agree with the archive" if not problems
               else "reference metadata disagrees with the archive", _limit(problems, 20))
    other_lines = [f"{label}: {len(ids)} items, e.g. {sorted(ids)[0]}" for label, ids in sorted(key_other.items())]
    report.add("reference_answer_keys", "FAIL" if key_other else "PASS",
               "reference `correct` labels agree with archive answer keys outside the known conflict items"
               if not key_other else "reference `correct` labels disagree with archive answer keys",
               _limit(other_lines, 20))

    # The 20 known conflict items: archive key vs every reference model.
    lines = []
    items = []
    archive_keys: Counter[str] = Counter()
    for eid, order, split in exp.conflicts:
        arow = archive_rows.get(eid, ("", {}))[1]
        archive_keys[str(arow.get("answer_key"))] += 1
        by_key = {key: sorted(models) for key, models in ref_keys.get(eid, {}).items()}
        summary = "; ".join(f"{key}: {len(models)} models" for key, models in sorted(by_key.items()))
        lines.append(f"q{order} {eid} ({split}): archive {arow.get('answer_key')}; refs {summary or 'none'}")
        items.append({"example_id": eid, "question_order": order, "split": split,
                      "archive_key": arow.get("answer_key"), "reference_keys": by_key})
    absent = [eid for eid, _, _ in exp.conflicts if eid not in archive_rows]
    if absent:
        report.add("ser_classify_key_conflict", "FAIL",
                   f"{len(absent)} of the {len(exp.conflicts)} known conflict items are not in the archive",
                   _limit(absent, 5))
        return
    disagree = sorted({label.split("/")[0] for label in key_conflict_dirs})
    covering = {model for eid in conflict_ids for models in ref_keys.get(eid, {}).values() for model in models}
    agree = sorted(covering - set(disagree))
    headline = (
        f"archive keys on the {len(exp.conflicts)} conflict items: {dict(archive_keys)}; "
        f"{len(disagree)} reference models disagree on these items, {len(agree)} agree"
    )
    details = [
        f"models agreeing with the archive on all conflict items: {', '.join(agree) or 'none'}",
        f"models disagreeing: {', '.join(disagree) or 'none'}",
        "a perfect reproduction of a disagreeing model shows ~20/600 (3.3%) `chosen` mismatches "
        "vs its reference on ser_classify",
        *lines,
    ]
    report.add("ser_classify_key_conflict", "WARN" if disagree else "PASS", headline, details,
               archive_keys=dict(archive_keys), agree=agree, disagree=disagree, items=items)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_checks(archive: Path, refs: Path | None = None, expectations: Expectations | None = None,
               workers: int = 8, skip_audio: bool = False) -> Report:
    exp = expectations or Expectations()
    report = Report(str(archive))
    files = check_files(archive, report)
    if not files:
        return report
    check_row_counts(files, exp, report)
    check_duplicates(files, report)
    check_all_union(files, report)
    check_required_fields(files, exp, report)
    check_splits(files, exp, report)
    check_balance(files, exp, report)
    check_contract(files, exp, report)
    check_against_refs(files, refs, exp, report)
    check_label_literals(files, report)
    if skip_audio:
        report.add("audio_integrity", "SKIP", "--skip-audio")
    else:
        infos = check_audio(archive, files, exp, workers, report)
        check_checksum_file(archive, infos, workers, report)
    check_leaks(archive, files, report)
    return report


def print_report(report: Report) -> None:
    print(f"AcousticBench archive validation: {report.archive}")
    print("=" * 78)
    for check in report.checks:
        print(f"[{check.status:<4}] {check.name}: {check.summary}")
        for line in check.details:
            print(f"         {line}")
    print("=" * 78)
    counts = report.counts()
    print("Summary: " + ", ".join(f"{status} {counts.get(status, 0)}"
                                  for status in ("PASS", "FAIL", "WARN", "INFO", "SKIP")))
    print("RESULT: " + ("FAIL" if report.failed else "PASS (no hard failures)"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an AcousticBench archive against the v1.0 release contract.",
    )
    parser.add_argument("--archive", type=Path, required=True,
                        help="Archive root containing questions/ and audio/")
    default_refs = REPO_ROOT / "reference_trajectories"
    parser.add_argument("--refs", type=Path, default=default_refs if default_refs.is_dir() else None,
                        help="Reference trajectories root for ID/key cross-checks "
                             "(default: the repo's reference_trajectories/ if present)")
    parser.add_argument("--no-refs", action="store_true", help="Skip reference cross-checks")
    parser.add_argument("--json-out", type=Path, default=None, help="Write the full report as JSON")
    parser.add_argument("--workers", type=int, default=min(32, (os.cpu_count() or 4) * 2),
                        help="Threads for hashing audio (default: %(default)s)")
    parser.add_argument("--skip-audio", action="store_true",
                        help="Skip audio hashing, durations and format checks")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.archive.is_dir():
        logger.error(f"Archive directory not found: {args.archive}")
        return 2
    refs = None if args.no_refs else args.refs
    report = run_checks(args.archive, refs, workers=args.workers, skip_audio=args.skip_audio)
    print_report(report)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report.to_json(), indent=2, default=str) + "\n", encoding="utf-8")
        logger.info(f"Wrote {args.json_out}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
