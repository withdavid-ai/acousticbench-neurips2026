#!/usr/bin/env python3
"""Regenerate the AcousticBench paper tables from row-level trajectories.

Rebuilds paper Tables 1, 2, 3, 6, 7 and 10 (plus the constant human Table 11)
from trajectory files and writes LaTeX (booktabs), Markdown and a JSON file of
raw values. ``--check`` compares every published cell against the values
transcribed from the paper and exits nonzero on any mismatch beyond rounding.
This is an offline, zero-cost check: no model or judge is called.

Inputs (``--root``; both layouts may be mixed under one root):
  * ``reference_trajectories/<Model>/<task>/run_<k>.jsonl`` (+ ``summary.json``)
  * ``results/<run_tag>/{config.json,responses.jsonl}`` as written by ``run.py``.
    Runs made with ``--task all`` are split back into tasks using
    ``meta.category`` / ``meta.dimension``. Runs of one (model, task, split)
    are ordered by seed, then timestamp, and named run_1, run_2, ...

Aggregation (identical to the authors' generator that produced the paper):
  * Per-run accuracy = ``correct / n * 100.0`` over every row of the cell.
    Abstentions, unparseable outputs and API errors count as incorrect.
  * RAD is item-weighted over its 1,200 rows. The Pitch / Loud. / Tempo / Dyn.
    columns filter RAD rows by ``meta.dimension``.
  * SER is item-weighted over SER-Classify (audio) + SER-Intensity rows of the
    same run index (600 + 400 items). Classify (text) is a diagnostic and is
    excluded from every aggregate.
  * Overall = mean(RAD, SER, RSQ) per run index, then the mean over runs.
  * Error bars are 2 x the SAMPLE standard deviation (ddof=1) of the per-run
    values. Population sd does not reproduce paper Table 7 (for example
    Gemini-3.1-Pro RAD is 90.0 +- 0.5 with ddof=1 and +- 0.4 with ddof=0).
    Runs are paired across tasks by run index (run_1 with run_1, ...).
  * Tables 2 and 10 use the held-out ``eval`` split: base Kimi-Audio rows are
    filtered to eval IDs, the specialist LoRA uses its eval runs, and the lift
    is mean +- 2 sd of the paired per-run differences (LoRA run k minus base
    run k).
  * Table 3 uses split ``all`` for base Kimi-Audio and both off-task LoRAs;
    the delta is the paired per-run difference.
  * Table 6 pools every row of every run of the (model, task) cell:
      API     = ``response_status == "error"``, an empty response, or a judge
                failure (``extraction_source == "judge_error"``)
      Parsed  = a definite label (``chosen`` is not null)
      Refusal = everything else (a non-empty response the judge could not map
                to a label: explicit declines and off-label answers)
    Rates are percentages of all rows; N is the per-run question count.

Rounding: cells are printed with ``format(value, ".1f")`` (``.2f`` for
Table 6) of the float computed exactly as above, which is how the paper's
numbers were produced. Fifteen published means are exact ties at one decimal
(for example Step-Audio-2-mini SER-Intensity, 207/400 = 51.75). For a tie the
printed digit follows the binary float, not a round-half rule: 207/400*100.0 is
51.74999999999999 and prints 51.7, while Kimi-Audio Pitch (80.25) prints 80.2
and Gemini-2.5-Flash-Audio Intensity (80.75) prints 80.8. Any other rounding
rule changes some of these digits. The exact rational value of every mean is
tracked, so ``--check`` lists every tie instead of hiding it, and it accepts a
cell when the printed string equals the paper or when the exact value lies
within half a unit of the paper's last digit (a tie that may print either way).

Known data issue surfaced as a warning, not hidden: the reference directory
``Cascade-GPT4oTranscribe-GPT5.4/ser_classify`` holds the 600
``ser_classify_text`` example IDs, not the audio SER-Classify IDs. The paper's
Cascade Classify (audio) cell (42.9) was computed from these rows, so the
tables use them as published and the script reports the mismatch.

Usage:
    python scripts/make_paper_tables.py                    # refs -> generated_tables/
    python scripts/make_paper_tables.py --check            # also diff against the paper
    python scripts/make_paper_tables.py --root results --out generated_tables/rerun
    python scripts/make_paper_tables.py --archive archive  # take splits from the dataset

Not reproduced: paper citation markers ([41] etc.); the Human row and Table 11
are constants from the human study (paper Table 11), not computed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from collections.abc import Callable, Iterable
from typing import Any

from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "reference_trajectories"
DEFAULT_OUT = REPO_ROOT / "generated_tables"

TASKS: tuple[str, ...] = ("rad", "rsq", "ser_classify", "ser_classify_text", "ser_intensity")
TASK_FILES: dict[str, str] = {task: f"acousticbench_{task}.jsonl" for task in TASKS}
TASK_SIZES: dict[str, int] = {
    "rad": 1200,
    "rsq": 1000,
    "ser_classify": 600,
    "ser_classify_text": 600,
    "ser_intensity": 400,
}
EVAL_SIZES: dict[str, int] = {task: size // 2 for task, size in TASK_SIZES.items()}
VALID_SPLITS: tuple[str, ...] = ("train", "eval")
RAD_DIMENSIONS: tuple[str, ...] = ("pitch", "loudness", "tempo", "dynamic-range")

# Digests of the released v1.0 question contract, derived from the reference
# trajectories (identical ID sets across all full-coverage model directories,
# except Cascade ser_classify). Each digest is
#   sha256("\n".join(sorted(lines)) + "\n")
# where a line is the example_id ("ids", "eval_ids") or
# "<example_id>\t<split>\t<answer_key>" ("contract"). The answer key is the
# one used by the Kimi-family runs, which ran on the canonical archive (it
# differs from 15 other models on 20 SER-Classify items; see validate_archive.py).
RELEASE_CONTRACT: dict[str, dict[str, Any]] = {
    "rad": {
        "n": 1200,
        "n_eval": 600,
        "ids": "e12163bbf1cfaa6a414b8371c37398eff2c67098845f4d60bb46350606a34ea6",
        "eval_ids": "756e83b5829e17e1968fee3bc9d92d4008db04fa0d4e98632fdfa25fd87890c4",
        "contract": "d27d9c4d13e18635077ec9c9b9c430a65c1d707b4ade66070ab92179db984813",
    },
    "rsq": {
        "n": 1000,
        "n_eval": 500,
        "ids": "9f7659f8ab654818572938ec69da6d2334fc276c3ec1fa0d6c905c727edc1a4b",
        "eval_ids": "710c646683560d1db45dd4b022388732ea6b4d4e1c9b8dcb85e856cfd0dbd894",
        "contract": "bd1a87a3b80e6cbaa65a621e0c10381d0a91471ba92169109b38056c6c2822d1",
    },
    "ser_classify": {
        "n": 600,
        "n_eval": 300,
        "ids": "433facba91a158e2b42db94f4899dc346449688eaf9d1ce8b9a92165cd4a6bf2",
        "eval_ids": "e6e8db900772437258816b29a1549c7990cbb0c0f8b06417a0e09205f70033a9",
        "contract": "e551a7f810311b8a2991b1fa130ec771271a26e9e2f854266b5cb2916a8bb04c",
    },
    "ser_classify_text": {
        "n": 600,
        "n_eval": 300,
        "ids": "ed81ab0d830dab90c145b95c4399a9c27007f88771040bb6c25a323f46fec4bc",
        "eval_ids": "762cf29d98d4dbf44ca31f1d5faf2345b52ed27a3f5884d828da11e72b6cdbd9",
        "contract": "9cf501fb52755e8d3b6ca68b2d2e90ecfaa6a34615cd785cc17e1a3b9b666840",
    },
    "ser_intensity": {
        "n": 400,
        "n_eval": 200,
        "ids": "e3bf5ba9b3d53e669a263478e7da49af0a2ed7445add6a3b4042c1a2f5a2883d",
        "eval_ids": "4953951440fa1d6e5ba19e76458885375b61fc82d0dec889c31b30d5ed32507c",
        "contract": "2800db7e84af9c3c7ffad7d57e9a271e0c562a20fbdcd0155200a4abdae9a451",
    },
}

# (trajectory directory name, display name) in paper Table 1 order.
PAPER_ROSTER: tuple[tuple[str, str], ...] = (
    ("Gemini-3.1-Pro", "Gemini-3.1-Pro"),
    ("Gemini-2.5-Pro", "Gemini-2.5-Pro"),
    ("Gemini-2.5-Flash-Audio", "Gemini-2.5-Flash-Audio"),
    ("Gemini-3-Flash", "Gemini-3-Flash"),
    ("Qwen3-Omni", "Qwen3-Omni"),
    ("Kimi-Audio", "Kimi-Audio"),
    ("MiniCPM-o-4.5", "MiniCPM-o-4.5"),
    ("GPT-4o", "GPT-4o-Audio"),
    ("Step-Audio-2-mini", "Step-Audio-2-mini"),
    ("Voxtral-Small-24B", "Voxtral-Small-24B"),
    ("MiMo-Audio-7B", "MiMo-Audio-7B"),
    ("Phi-4-Multimodal", "Phi-4-Multimodal"),
    ("Qwen2-Audio-7B-Instruct", "Qwen2-Audio-7B-Instruct"),
    ("Voxtral-Mini-3B", "Voxtral-Mini-3B"),
    ("Audio-Flamingo-3", "Audio-Flamingo-3"),
    ("Cascade-GPT4oTranscribe-GPT5.4", "Cascade (GPT-5.4)"),
)
DISPLAY_NAMES: dict[str, str] = dict(PAPER_ROSTER)
CASCADE_DIR = "Cascade-GPT4oTranscribe-GPT5.4"
BASE_MODEL = "Kimi-Audio"
# Paper Table 1 footnote 1: these systems require audio, so text-only items get a silent file.
SILENT_AUDIO_FOOTNOTE: frozenset[str] = frozenset({"GPT-4o", "MiMo-Audio-7B"})
LORA_DIAGONAL: dict[str, str] = {
    "rad": "Kimi-Audio-LoRA-RAD",
    "ser_classify": "Kimi-Audio-LoRA-SER-Classify",
    "ser_intensity": "Kimi-Audio-LoRA-SER-Intensity",
    "rsq": "Kimi-Audio-LoRA-RSQ",
}
LIFT_TASK_LABELS: dict[str, str] = {
    "rad": "RAD",
    "ser_classify": "SER-Classify",
    "ser_intensity": "SER-Intensity",
    "rsq": "RSQ",
}
TRANSFER_ROWS: tuple[tuple[str, str], ...] = (
    ("Kimi-Audio-LoRA-RAD", "RAD LoRA"),
    ("Kimi-Audio-LoRA-Simple-AudioQA", "Simple AudioQA LoRA"),
)
TRANSFER_TASKS: tuple[tuple[str, str], ...] = (
    ("ser_intensity", "SER-Intensity (pairwise)"),
    ("ser_classify", "SER-Classify (single-clip)"),
    ("rsq", "RSQ (pairwise)"),
)
PARSE_TASKS: tuple[tuple[str, str], ...] = (
    ("rad", "RAD"),
    ("rsq", "RSQ"),
    ("ser_classify", "SER-Classify"),
)

# Table 1 / 7 columns: (metric key, header).
MAIN_METRICS: tuple[tuple[str, str], ...] = (
    ("overall", "Overall"),
    ("rad", "RAD"),
    ("rad_pitch", "Pitch"),
    ("rad_loudness", "Loud."),
    ("rad_tempo", "Tempo"),
    ("rad_dynamics", "Dyn."),
    ("ser", "SER"),
    ("ser_classify", "Classify (audio)"),
    ("ser_classify_text", "Classify (text)"),
    ("ser_intensity", "Intensity"),
    ("rsq", "RSQ"),
)
DIMENSION_METRICS: dict[str, str] = {
    "rad_pitch": "pitch",
    "rad_loudness": "loudness",
    "rad_tempo": "tempo",
    "rad_dynamics": "dynamic-range",
}
# The published PDF bolds the best model mean in every column except
# Classify (text), although the caption says "per column". Matched here.
UNBOLDED_MAIN_METRICS: frozenset[str] = frozenset({"ser_classify_text"})

# Paper Table 11 (human accuracy; rating-weighted over latest human attempts).
# These come from the human study, not from model trajectories, and are
# reproduced verbatim: (category, accuracy %, ratings, questions).
HUMAN_TABLE11: tuple[tuple[str, str, str, str], ...] = (
    ("Overall (macro)", "93.49", "5,134", "3,200"),
    ("RAD", "99.03", "1,036", "1,200"),
    ("Pitch", "98.32", "357", "400"),
    ("Loudness", "99.49", "393", "400"),
    ("Tempo", "99.46", "186", "200"),
    ("Dynamic range", "99.00", "100", "200"),
    ("SER", "87.84", "2,910", "1,000"),
    ("Classify (audio)", "87.14", "2,433", "600"),
    ("Intensity", "91.40", "477", "400"),
    ("RSQ", "93.60", "1,188", "1,000"),
)
# Human row of Tables 1 and 7 = Table 11 rounded to one decimal.
HUMAN_MAIN: dict[str, str | None] = {
    "overall": "93.5",
    "rad": "99.0",
    "rad_pitch": "98.3",
    "rad_loudness": "99.5",
    "rad_tempo": "99.5",
    "rad_dynamics": "99.0",
    "ser": "87.8",
    "ser_classify": "87.1",
    "ser_classify_text": None,
    "ser_intensity": "91.4",
    "rsq": "93.6",
}
HUMAN_LIFT: dict[str, str] = {"rad": "99.0", "ser_classify": "87.1", "ser_intensity": "91.4", "rsq": "93.6"}

# ---------------------------------------------------------------------------
# Published values, transcribed from the submitted PDF (audit paper_claims.md
# section 1). Strings keep the printed precision. "-" marks an empty cell.
# Camera-ready values: the Table 1/7 and Table 6 cells affected by the
# 2026-10-12 reruns (MiMo-Audio-7B SER-Classify and SER-Intensity with greedy
# decoding; the 20 SER-Classify items with question_order 558 to 599 for 12
# systems) are the camera-ready values, not the submitted ones. The 62 changed
# cells and their submitted values are listed in
# docs/camera-ready/audit/rerun_2026-10.md.
# ---------------------------------------------------------------------------

# Table 7 (means equal Table 1). Columns follow MAIN_METRICS; "mean+-err".
_PAPER_TABLE7 = """
Gemini-3.1-Pro                 84.5+-0.2 90.0+-0.5 95.1+-0.3 82.4+-2.3 98.2+-2.1 86.7+-0.6 79.6+-0.4 75.9+-0.7 55.1+-1.3 85.1+-0.8 83.9+-0.7
Gemini-2.5-Pro                 77.4+-0.6 74.1+-0.8 90.2+-0.0 64.3+-2.9 91.8+-1.2 43.3+-0.6 74.2+-1.1 74.1+-1.6 52.9+-1.9 74.5+-1.0 83.8+-0.4
Gemini-2.5-Flash-Audio         75.3+-0.9 70.5+-0.9 80.0+-1.0 76.3+-2.0 70.3+-5.0 40.0+-1.0 73.0+-1.1 67.9+-1.3 49.4+-1.1 80.8+-2.3 82.5+-1.2
Gemini-3-Flash                 73.7+-0.4 73.1+-0.8 91.8+-0.6 58.1+-4.1 67.8+-1.5 70.7+-4.0 73.8+-1.0 68.3+-1.3 49.7+-1.0 82.0+-1.8 74.3+-0.8
Qwen3-Omni                     69.4+-0.2 70.7+-0.0 85.0+-0.0 71.0+-0.0 67.5+-0.0 44.5+-0.0 69.5+-0.6 64.0+-0.0 48.0+-0.0 77.8+-1.4 68.0+-0.8
Kimi-Audio                     66.2+-0.0 70.4+-0.0 80.2+-0.0 79.0+-0.0 61.0+-0.0 43.0+-0.0 72.5+-0.0 74.0+-0.3 53.3+-0.0 70.2+-0.5 55.7+-0.1
MiniCPM-o-4.5                  59.0+-0.3 56.9+-0.3 69.8+-0.3 50.8+-0.5 50.0+-0.0 50.0+-0.0 59.8+-0.3 66.2+-0.5 42.7+-0.2 50.2+-0.0 60.3+-0.5
GPT-4o                         58.1+-1.3 57.5+-2.8 69.2+-2.8 52.2+-6.5 50.0+-0.0 52.0+-3.0 63.6+-0.8 63.2+-2.0 48.4+-1.7 64.2+-1.3 53.3+-0.8
Step-Audio-2-mini              57.3+-0.1 53.0+-0.1 58.6+-0.3 49.6+-0.3 51.5+-0.0 50.0+-0.0 58.8+-0.2 63.6+-0.4 55.8+-0.0 51.7+-0.0 60.1+-0.6
Voxtral-Small-24B              55.1+-0.4 50.0+-0.0 50.0+-0.0 50.0+-0.0 50.0+-0.0 50.0+-0.0 56.9+-0.6 52.7+-0.7 49.2+-0.0 63.2+-0.5 58.5+-0.8
MiMo-Audio-7B                  54.1+-0.1 50.5+-0.3 51.3+-1.0 50.1+-0.3 50.3+-0.6 50.0+-0.0 61.7+-0.0 60.5+-0.0 51.3+-0.0 63.5+-0.0 50.0+-0.0
Phi-4-Multimodal               51.7+-0.0 50.7+-0.0 50.0+-0.0 52.0+-0.0 50.0+-0.0 50.0+-0.0 53.8+-0.0 53.8+-0.0 50.2+-0.0 53.8+-0.0 50.7+-0.0
Qwen2-Audio-7B-Instruct        51.3+-0.0 49.9+-0.0 50.0+-0.0 49.8+-0.0 50.0+-0.0 50.0+-0.0 54.1+-0.0 56.5+-0.0 51.8+-0.0 50.5+-0.0 50.0+-0.0
Voxtral-Mini-3B                51.7+-0.1 49.3+-0.3 49.4+-0.6 50.0+-0.0 47.0+-1.0 50.0+-0.0 55.1+-0.0 58.3+-0.0 50.5+-0.0 50.2+-0.0 50.6+-0.0
Audio-Flamingo-3               51.2+-0.1 49.9+-0.0 49.5+-0.0 50.0+-0.0 50.5+-0.0 50.0+-0.0 52.6+-0.5 54.4+-0.8 56.0+-0.0 50.0+-0.0 50.9+-0.2
Cascade-GPT4oTranscribe-GPT5.4 51.5+-1.1 52.7+-1.2 53.0+-8.3 50.9+-4.8 53.7+-8.3 54.7+-4.0 48.1+-3.3 42.9+-2.7 - 55.9+-4.2 53.6+-1.1
"""

# Table 2 (base, LoRA, delta) and Table 10 (base+-, LoRA+-, lift+-).
_PAPER_TABLE2: dict[str, tuple[str, str, str]] = {
    "rad": ("69.2", "98.7", "+29.5"),
    "ser_classify": ("70.3", "75.2", "+4.9"),
    "ser_intensity": ("72.3", "92.5", "+20.2"),
    "rsq": ("54.8", "98.4", "+43.6"),
}
_PAPER_TABLE10: dict[str, tuple[str, str, str]] = {
    "rad": ("69.2+-0.0", "98.7+-0.0", "29.5+-0.0"),
    "ser_classify": ("70.3+-0.7", "75.2+-0.4", "4.9+-0.4"),
    "ser_intensity": ("72.3+-0.6", "92.5+-0.0", "20.2+-0.6"),
    "rsq": ("54.8+-0.8", "98.4+-0.0", "43.6+-0.8"),
}
# Table 3: row -> task -> (mean+-err, delta or None).
_PAPER_TABLE3: dict[str, dict[str, tuple[str, str | None]]] = {
    BASE_MODEL: {
        "ser_intensity": ("70.2+-0.5", None),
        "ser_classify": ("74.0+-0.3", None),
        "rsq": ("55.7+-0.1", None),
    },
    "Kimi-Audio-LoRA-RAD": {
        "ser_intensity": ("77.9+-0.3", "+7.7"),
        "ser_classify": ("73.7+-0.2", "-0.3"),
        "rsq": ("61.3+-0.2", "+5.6"),
    },
    "Kimi-Audio-LoRA-Simple-AudioQA": {
        "ser_intensity": ("73.0+-0.5", "+2.8"),
        "ser_classify": ("73.1+-0.5", "-0.9"),
        "rsq": ("57.0+-0.2", "+1.3"),
    },
}
# Table 6: model dir -> "parsed/refusal/api" for RAD, RSQ, SER-Classify.
_PAPER_TABLE6 = """
Gemini-3.1-Pro          99.56/0.28/0.17  100.00/0.00/0.00 100.00/0.00/0.00
Gemini-2.5-Pro          97.81/2.19/0.00  100.00/0.00/0.00 100.00/0.00/0.00
Gemini-2.5-Flash-Audio  99.14/0.86/0.00  99.90/0.10/0.00  100.00/0.00/0.00
Gemini-3-Flash          100.00/0.00/0.00 100.00/0.00/0.00 100.00/0.00/0.00
Qwen3-Omni              100.00/0.00/0.00 100.00/0.00/0.00 100.00/0.00/0.00
Kimi-Audio              100.00/0.00/0.00 100.00/0.00/0.00 100.00/0.00/0.00
MiniCPM-o-4.5           100.00/0.00/0.00 100.00/0.00/0.00 100.00/0.00/0.00
GPT-4o                  99.78/0.22/0.00  100.00/0.00/0.00 100.00/0.00/0.00
Step-Audio-2-mini       100.00/0.00/0.00 100.00/0.00/0.00 100.00/0.00/0.00
Voxtral-Small-24B       100.00/0.00/0.00 100.00/0.00/0.00 99.94/0.06/0.00
MiMo-Audio-7B           100.00/0.00/0.00 100.00/0.00/0.00 100.00/0.00/0.00
Phi-4-Multimodal        100.00/0.00/0.00 100.00/0.00/0.00 100.00/0.00/0.00
Qwen2-Audio-7B-Instruct 100.00/0.00/0.00 100.00/0.00/0.00 100.00/0.00/0.00
Voxtral-Mini-3B         100.00/0.00/0.00 100.00/0.00/0.00 100.00/0.00/0.00
Audio-Flamingo-3        100.00/0.00/0.00 100.00/0.00/0.00 100.00/0.00/0.00
"""
# Table 6 footnote (Gemini-2.5-Pro RAD refusals): total refusals, the only
# dimension they occur on, distinct questions affected, questions refused in
# every run.
_PAPER_TABLE6_FOOTNOTE: dict[str, Any] = {
    "model": "Gemini-2.5-Pro",
    "task": "rad",
    "refusals": 79,
    "dimensions": ["loudness"],
    "questions": 52,
    "all_runs": 3,
}


def _parse_pm(cell: str) -> tuple[str, str] | None:
    if cell == "-":
        return None
    mean, err = cell.split("+-")
    return mean, err


def _parse_table7() -> dict[str, dict[str, tuple[str, str] | None]]:
    parsed: dict[str, dict[str, tuple[str, str] | None]] = {}
    for line in _PAPER_TABLE7.strip().splitlines():
        parts = line.split()
        model, cells = parts[0], parts[1:]
        if len(cells) != len(MAIN_METRICS):
            raise ValueError(f"Bad paper Table 7 line for {model}: {len(cells)} cells")
        parsed[model] = {
            metric: _parse_pm(cell) for (metric, _), cell in zip(MAIN_METRICS, cells)
        }
    return parsed


def _parse_table6() -> dict[str, dict[str, tuple[str, str, str]]]:
    parsed: dict[str, dict[str, tuple[str, str, str]]] = {}
    for line in _PAPER_TABLE6.strip().splitlines():
        parts = line.split()
        model, cells = parts[0], parts[1:]
        parsed[model] = {
            task: tuple(cell.split("/"))  # type: ignore[misc]
            for (task, _), cell in zip(PARSE_TASKS, cells)
        }
    return parsed


PAPER_TABLE7 = _parse_table7()
PAPER_TABLE6 = _parse_table6()


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@dataclass
class Diagnostics:
    """Collects warnings so they can be logged, returned and written to JSON."""

    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        logger.warning(message)

    def note(self, message: str) -> None:
        self.notes.append(message)
        logger.info(message)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

JsonRow = dict[str, Any]


@dataclass
class Run:
    name: str
    rows: list[JsonRow]
    source: str
    seed: int | None = None


@dataclass
class Group:
    """All runs of one (model, task, declared split) cell."""

    model: str
    task: str
    split: str
    runs: list[Run]
    source: str

    def id_set(self) -> frozenset[str]:
        return frozenset(row_example_id(row) for run in self.runs for row in run.rows)


Groups = dict[tuple[str, str, str], Group]


def iter_jsonl(path: Path) -> Iterable[JsonRow]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def row_example_id(row: JsonRow) -> str:
    example_id = row.get("example_id")
    if example_id:
        return str(example_id)
    meta = row.get("meta")
    if isinstance(meta, dict) and meta.get("example_id"):
        return str(meta["example_id"])
    return ""


def row_meta(row: JsonRow) -> dict[str, Any]:
    meta = row.get("meta")
    return meta if isinstance(meta, dict) else {}


def task_from_fields(category: object, dimension: object) -> str | None:
    """Map AcousticBench category/dimension values to a task slug."""
    category = str(category or "")
    dimension = str(dimension or "")
    if category == "acoustic-perception":
        return "rad"
    if category == "speech-quality":
        return "rsq"
    if category == "emotion-recognition":
        if "text-only" in dimension:
            return "ser_classify_text"
        if "intensity" in dimension:
            return "ser_intensity"
        return "ser_classify"
    return None


def infer_row_task(row: JsonRow) -> str | None:
    meta = row_meta(row)
    return task_from_fields(meta.get("category"), meta.get("dimension"))


def normalize_dimension(value: object) -> str:
    dim = str(value or "").lower().replace("_", "-").replace(" ", "-")
    if dim in RAD_DIMENSIONS:
        return dim
    if "pitch" in dim:
        return "pitch"
    if "dynamic" in dim or "envelope" in dim:
        return "dynamic-range"
    if "loud" in dim:
        return "loudness"
    if "tempo" in dim or "speed" in dim or "time-stretch" in dim:
        return "tempo"
    return dim


def _run_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(\d+)$", path.stem)
    return (int(match.group(1)) if match else 0, path.stem)


def _declared_split(summary: dict[str, Any], runs: list[Run]) -> str:
    split = summary.get("split")
    if split in ("all", "train", "eval"):
        return str(split)
    splits = {row_meta(row).get("split") for run in runs for row in run.rows}
    if len(splits) == 1 and next(iter(splits)) in VALID_SPLITS:
        return str(next(iter(splits)))
    return "all"


def load_trajectory_layout(root: Path, diag: Diagnostics) -> Groups:
    """Load ``<root>/<Model>/<task>/run_*.jsonl``."""
    groups: Groups = {}
    if not root.is_dir():
        return groups
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith((".", "_")):
            continue
        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.name not in TASKS:
                continue
            run_files = sorted(task_dir.glob("run_*.jsonl"), key=_run_sort_key)
            if not run_files:
                continue
            summary: dict[str, Any] = {}
            summary_path = task_dir / "summary.json"
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    diag.warn(f"{summary_path}: invalid JSON, ignored")
            runs = [
                Run(name=path.stem, rows=list(iter_jsonl(path)), source=str(path))
                for path in run_files
            ]
            split = _declared_split(summary, runs)
            key = (model_dir.name, task_dir.name, split)
            groups[key] = Group(model_dir.name, task_dir.name, split, runs, str(task_dir))
    return groups


def load_results_layout(root: Path, diag: Diagnostics) -> Groups:
    """Load ``run.py`` output: ``<root>/**/<run_tag>/{config.json,responses.jsonl}``."""
    pending: dict[tuple[str, str, str], list[tuple[tuple[Any, ...], Run]]] = defaultdict(list)
    if not root.is_dir():
        return {}
    for responses_path in sorted(root.rglob("responses.jsonl")):
        run_dir = responses_path.parent
        config: dict[str, Any] = {}
        config_path = run_dir / "config.json"
        if config_path.is_file():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                diag.warn(f"{config_path}: invalid JSON")
        model = str(config.get("model") or "")
        if not model:
            diag.warn(f"{run_dir}: no config.json with a model name, skipped")
            continue
        declared_task = str(config.get("task") or "all")
        split = str(config.get("split") or "all")
        seed = config.get("seed")
        rows = list(iter_jsonl(responses_path))
        by_task: dict[str, list[JsonRow]] = defaultdict(list)
        if declared_task in TASKS:
            by_task[declared_task] = rows
        else:
            if declared_task not in ("all",):
                diag.warn(
                    f"{run_dir}: task {declared_task!r} is not a paper task; rows are "
                    "assigned to tasks by meta.category/dimension"
                )
            unknown = 0
            for row in rows:
                task = infer_row_task(row)
                if task is None:
                    unknown += 1
                    continue
                by_task[task].append(row)
            if unknown:
                diag.warn(f"{run_dir}: {unknown} rows with an unknown task were ignored")
        sort_key = (
            seed if isinstance(seed, int) else 10**9,
            str(config.get("timestamp") or ""),
            run_dir.name,
        )
        for task, task_rows in by_task.items():
            run = Run(name=run_dir.name, rows=task_rows, source=str(run_dir),
                      seed=seed if isinstance(seed, int) else None)
            pending[(model, task, split)].append((sort_key, run))

    groups: Groups = {}
    for (model, task, split), items in pending.items():
        items.sort(key=lambda item: item[0])
        runs = []
        for index, (_, run) in enumerate(items, start=1):
            runs.append(Run(name=f"run_{index}", rows=run.rows, source=run.source, seed=run.seed))
        groups[(model, task, split)] = Group(model, task, split, runs, str(root))
    return groups


def load_groups(root: Path, diag: Diagnostics) -> Groups:
    groups = load_trajectory_layout(root, diag)
    for key, group in load_results_layout(root, diag).items():
        if key in groups:
            diag.warn(
                f"{key}: present in both trajectory and results layouts under {root}; "
                "keeping the trajectory layout"
            )
            continue
        groups[key] = group
    return groups


# ---------------------------------------------------------------------------
# Expected ID sets and validation
# ---------------------------------------------------------------------------


def digest_lines(lines: Iterable[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(lines)) + "\n").encode("utf-8")).hexdigest()


@dataclass
class ExpectedIds:
    full: dict[str, frozenset[str]]
    eval: dict[str, frozenset[str]]
    source: str


def _question_order(raw: JsonRow) -> int | None:
    question_set = raw.get("question_set")
    if isinstance(question_set, dict) and question_set.get("question_order") is not None:
        try:
            return int(question_set["question_order"])
        except (TypeError, ValueError):
            return None
    return None


def expected_ids_from_archive(archive: Path, diag: Diagnostics) -> ExpectedIds:
    full: dict[str, frozenset[str]] = {}
    eval_ids: dict[str, frozenset[str]] = {}
    split_by_order: dict[int, str] = {}
    for task in ("rad", "rsq", "ser_classify", "ser_intensity", "ser_classify_text"):
        path = archive / "questions" / TASK_FILES[task]
        if not path.is_file():
            diag.warn(f"--archive: missing {path}")
            continue
        ids: set[str] = set()
        evals: set[str] = set()
        for raw in iter_jsonl(path):
            example_id = str(raw.get("example_id") or "")
            split = raw.get("split")
            order = _question_order(raw)
            if task == "ser_classify" and split in VALID_SPLITS and order is not None:
                split_by_order[order] = str(split)
            if split not in VALID_SPLITS and task == "ser_classify_text" and order is not None:
                split = split_by_order.get(order)
            ids.add(example_id)
            if split == "eval":
                evals.add(example_id)
        full[task] = frozenset(ids)
        eval_ids[task] = frozenset(evals)
    return ExpectedIds(full, eval_ids, f"archive {archive}")


def group_row_task(group: Group) -> str | None:
    """Majority task implied by the rows' category/dimension."""
    counts = Counter(infer_row_task(row) for run in group.runs for row in run.rows)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def expected_ids_from_groups(groups: Groups, diag: Diagnostics) -> ExpectedIds:
    """Derive per-task ID sets as the consensus over full-coverage groups.

    A group only votes for a task when its rows' category/dimension agree with
    the task directory it sits in (this excludes Cascade ser_classify, whose
    rows are ser_classify_text items). Eval IDs come from ``meta.split`` and
    from groups declared as eval runs.
    """
    full: dict[str, frozenset[str]] = {}
    eval_ids: dict[str, frozenset[str]] = {}
    for task in TASKS:
        votes: Counter[frozenset[str]] = Counter()
        split_labels: dict[str, set[str]] = defaultdict(set)
        declared_eval: set[str] = set()
        for (_model, group_task, split), group in groups.items():
            if group_task != task or group_row_task(group) != task:
                continue
            if split == "all":
                votes[group.id_set()] += 1
            elif split == "eval":
                declared_eval |= group.id_set()
            for run in group.runs:
                for row in run.rows:
                    label = row_meta(row).get("split")
                    if label in VALID_SPLITS:
                        split_labels[row_example_id(row)].add(str(label))
        if votes:
            full[task] = votes.most_common(1)[0][0]
        conflicting = sorted(eid for eid, labels in split_labels.items() if len(labels) > 1)
        if conflicting:
            diag.warn(
                f"{task}: {len(conflicting)} example_ids carry both train and eval in "
                f"meta.split (e.g. {conflicting[0]})"
            )
        evals = {eid for eid, labels in split_labels.items() if labels == {"eval"}}
        evals |= declared_eval
        if task in full:
            evals &= full[task]
        eval_ids[task] = frozenset(evals)
    return ExpectedIds(full, eval_ids, "consensus of the trajectory directories")


def compare_with_release_contract(expected: ExpectedIds, diag: Diagnostics) -> dict[str, dict[str, bool]]:
    status: dict[str, dict[str, bool]] = {}
    for task in TASKS:
        contract = RELEASE_CONTRACT[task]
        ids = expected.full.get(task)
        evals = expected.eval.get(task)
        status[task] = {
            "ids": bool(ids) and digest_lines(ids) == contract["ids"],
            "eval_ids": bool(evals) and digest_lines(evals) == contract["eval_ids"],
        }
    matched = [task for task, flags in status.items() if all(flags.values())]
    if len(matched) == len(TASKS):
        diag.note(
            f"Expected ID and eval-split sets ({expected.source}) match the v1.0 release "
            "contract for every task"
        )
    else:
        for task, flags in status.items():
            if task not in expected.full:
                continue
            if not all(flags.values()):
                diag.warn(
                    f"{task}: expected ID set from {expected.source} does not match the v1.0 "
                    f"release contract (ids match: {flags['ids']}, eval ids match: "
                    f"{flags['eval_ids']})"
                )
    return status


def validate_groups(groups: Groups, expected: ExpectedIds, diag: Diagnostics) -> None:
    """Warn about run-level inconsistencies and directories with unexpected IDs."""
    for (model, task, split), group in sorted(groups.items()):
        label = f"{model}/{task}" + ("" if split == "all" else f" (split {split})")
        run_sets = [frozenset(row_example_id(row) for row in run.rows) for run in group.runs]
        for run, run_set in zip(group.runs, run_sets):
            if len(run_set) != len(run.rows):
                diag.warn(f"{label} {run.name}: {len(run.rows) - len(run_set)} duplicate example_ids")
        if len(set(run_sets)) > 1:
            diag.warn(f"{label}: runs cover different example_id sets")
        row_task = group_row_task(group)
        if row_task is not None and row_task != task:
            diag.warn(
                f"{label}: rows are {row_task} items by meta.category/dimension, but sit "
                f"under task {task!r}"
            )
        target = expected.eval.get(task) if split == "eval" else expected.full.get(task)
        if split == "train" or not target:
            continue
        ids = group.id_set()
        if ids == target:
            continue
        overlap = len(ids & target)
        message = (
            f"{label}: example_ids do not match the expected {task} "
            f"{'eval ' if split == 'eval' else ''}ID set ({overlap}/{len(target)} expected IDs "
            f"present, {len(ids - target)} unexpected)"
        )
        for other in TASKS:
            other_ids = expected.full.get(other)
            if other != task and other_ids and ids <= other_ids:
                message += f"; all {len(ids)} IDs belong to {other}"
        if ids < target:
            message += "; partial coverage"
        diag.warn(message)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

Predicate = Callable[[JsonRow], bool]


@dataclass(frozen=True)
class RunScore:
    correct: int
    n: int

    @property
    def accuracy(self) -> float:
        # Same float expression as the generator behind the paper; see module docstring.
        return self.correct / self.n * 100.0

    @property
    def exact(self) -> Fraction:
        return Fraction(100 * self.correct, self.n)


@dataclass(frozen=True)
class Stat:
    """Mean and +-2 sample sd of per-run values (accuracy or derived)."""

    run_names: tuple[str, ...]
    per_run: tuple[float, ...]
    per_run_exact: tuple[Fraction, ...]
    n_items: tuple[int, ...] = ()
    correct: tuple[int, ...] = ()

    @property
    def mean(self) -> float:
        return statistics.mean(self.per_run)

    @property
    def exact_mean(self) -> Fraction:
        return sum(self.per_run_exact, Fraction(0)) / len(self.per_run_exact)

    @property
    def sd(self) -> float:
        return statistics.stdev(self.per_run) if len(self.per_run) > 1 else 0.0

    @property
    def err(self) -> float:
        return 2.0 * self.sd

    @property
    def n_runs(self) -> int:
        return len(self.per_run)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "mean": self.mean,
            "mean_exact": str(self.exact_mean),
            "sd_sample": self.sd,
            "err_2sd": self.err,
            "n_runs": self.n_runs,
            "runs": list(self.run_names),
            "per_run": list(self.per_run),
        }
        if self.n_items:
            payload["n_items"] = list(self.n_items)
        if self.correct:
            payload["correct"] = list(self.correct)
        return payload


def score_rows(rows: Iterable[JsonRow], predicate: Predicate | None = None,
               ids: frozenset[str] | None = None) -> RunScore | None:
    correct = 0
    n = 0
    for row in rows:
        if ids is not None and row_example_id(row) not in ids:
            continue
        if predicate is not None and not predicate(row):
            continue
        n += 1
        if bool(row.get("is_correct")):
            correct += 1
    return RunScore(correct, n) if n else None


def run_scores(group: Group | None, predicate: Predicate | None = None,
               ids: frozenset[str] | None = None) -> dict[str, RunScore]:
    if group is None:
        return {}
    scores: dict[str, RunScore] = {}
    for run in group.runs:
        score = score_rows(run.rows, predicate, ids)
        if score is not None:
            scores[run.name] = score
    return scores


def stat_from_scores(scores: dict[str, RunScore]) -> Stat | None:
    if not scores:
        return None
    names = tuple(sorted(scores, key=_name_key))
    return Stat(
        run_names=names,
        per_run=tuple(scores[name].accuracy for name in names),
        per_run_exact=tuple(scores[name].exact for name in names),
        n_items=tuple(scores[name].n for name in names),
        correct=tuple(scores[name].correct for name in names),
    )


def _name_key(name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", name)
    return (int(match.group(1)) if match else 0, name)


def cell_stat(group: Group | None, predicate: Predicate | None = None,
              ids: frozenset[str] | None = None) -> Stat | None:
    return stat_from_scores(run_scores(group, predicate, ids))


def pooled_stat(groups: list[Group | None], diag: Diagnostics | None = None,
                label: str = "") -> Stat | None:
    """Item-weighted accuracy over several groups, per run index."""
    per_group = [run_scores(group) for group in groups]
    if any(not scores for scores in per_group):
        return None
    common = set.intersection(*(set(scores) for scores in per_group))
    union = set.union(*(set(scores) for scores in per_group))
    if diag is not None and common != union:
        diag.warn(f"{label}: runs {sorted(union - common)} are missing for some tasks; dropped")
    pooled = {
        name: RunScore(
            sum(scores[name].correct for scores in per_group),
            sum(scores[name].n for scores in per_group),
        )
        for name in common
    }
    return stat_from_scores(pooled)


def macro_stat(stats: list[Stat | None], diag: Diagnostics | None = None,
               label: str = "") -> Stat | None:
    """Per run index, the unweighted mean of several per-run values."""
    if any(stat is None for stat in stats):
        return None
    maps = [dict(zip(stat.run_names, zip(stat.per_run, stat.per_run_exact))) for stat in stats]  # type: ignore[union-attr]
    common = set.intersection(*(set(values) for values in maps))
    union = set.union(*(set(values) for values in maps))
    if diag is not None and common != union:
        diag.warn(f"{label}: runs {sorted(union - common)} are missing for some tasks; dropped")
    if not common:
        return None
    names = tuple(sorted(common, key=_name_key))
    per_run = tuple(statistics.mean(values[name][0] for values in maps) for name in names)
    per_run_exact = tuple(
        sum((values[name][1] for values in maps), Fraction(0)) / len(maps) for name in names
    )
    return Stat(run_names=names, per_run=per_run, per_run_exact=per_run_exact)


def paired_delta(after: Stat | None, before: Stat | None, diag: Diagnostics | None = None,
                 label: str = "") -> Stat | None:
    """Per-run differences, pairing runs in run-index order."""
    if after is None or before is None:
        return None
    if diag is not None and after.n_runs != before.n_runs:
        diag.warn(
            f"{label}: {after.n_runs} vs {before.n_runs} runs; pairing the first "
            f"{min(after.n_runs, before.n_runs)} in run order"
        )
    pairs = list(zip(after.per_run, before.per_run))
    exact_pairs = list(zip(after.per_run_exact, before.per_run_exact))
    if not pairs:
        return None
    names = tuple(f"{a}-{b}" for a, b in zip(after.run_names, before.run_names))
    return Stat(
        run_names=names,
        per_run=tuple(a - b for a, b in pairs),
        per_run_exact=tuple(a - b for a, b in exact_pairs),
    )


# ---------------------------------------------------------------------------
# Table computations
# ---------------------------------------------------------------------------


def dimension_predicate(dimension: str) -> Predicate:
    return lambda row: normalize_dimension(row_meta(row).get("dimension")) == dimension


def main_stats(groups: Groups, model: str, diag: Diagnostics) -> dict[str, Stat | None]:
    def group(task: str) -> Group | None:
        return groups.get((model, task, "all"))

    stats: dict[str, Stat | None] = {
        "rad": cell_stat(group("rad")),
        "rsq": cell_stat(group("rsq")),
        "ser_classify": cell_stat(group("ser_classify")),
        "ser_classify_text": cell_stat(group("ser_classify_text")),
        "ser_intensity": cell_stat(group("ser_intensity")),
    }
    for metric, dimension in DIMENSION_METRICS.items():
        stats[metric] = cell_stat(group("rad"), dimension_predicate(dimension))
    stats["ser"] = pooled_stat([group("ser_classify"), group("ser_intensity")], diag, f"{model} SER")
    stats["overall"] = macro_stat([stats["rad"], stats["ser"], stats["rsq"]], diag, f"{model} Overall")
    return stats


def eval_group_stat(groups: Groups, model: str, task: str, expected: ExpectedIds,
                    diag: Diagnostics) -> Stat | None:
    eval_ids = expected.eval.get(task)
    if not eval_ids:
        diag.warn(f"{task}: no eval-split IDs available; Table 2/10 cell left empty")
        return None
    group = groups.get((model, task, "eval")) or groups.get((model, task, "all"))
    if group is None:
        return None
    stat = cell_stat(group, ids=eval_ids)
    if stat is not None and any(n != len(eval_ids) for n in stat.n_items):
        diag.warn(
            f"{model}/{task}: eval-split runs cover {sorted(set(stat.n_items))} of "
            f"{len(eval_ids)} eval items"
        )
    return stat


@dataclass
class TableData:
    main: dict[str, dict[str, Stat | None]]
    lift: dict[str, dict[str, Any]]
    transfer: dict[str, dict[str, dict[str, Stat | None]]]
    parse: dict[str, dict[str, dict[str, Any]]]
    roster: list[str]
    extra_models: list[str]


def parse_counts(group: Group | None) -> dict[str, Any] | None:
    if group is None:
        return None
    total = parsed = refusal = api = 0
    refusal_by_dimension: Counter[str] = Counter()
    refused_runs: Counter[str] = Counter()
    for run in group.runs:
        for row in run.rows:
            total += 1
            empty = not str(row.get("response_text") or "").strip()
            # A judge failure (extraction_source "judge_error", written by newer
            # harness versions) is an upstream API loss, not a model refusal.
            upstream = (
                row.get("response_status") == "error"
                or row.get("extraction_source") in ("error", "judge_error")
                or "judge_error" in row
            )
            if row.get("chosen") is not None:
                parsed += 1
            elif upstream or empty:
                api += 1
            else:
                refusal += 1
                refusal_by_dimension[str(row_meta(row).get("dimension"))] += 1
                refused_runs[row_example_id(row)] += 1
    n_runs = len(group.runs)
    return {
        "n_questions": max((len(run.rows) for run in group.runs), default=0),
        "n_runs": n_runs,
        "rows": total,
        "parsed": parsed,
        "refusal": refusal,
        "api": api,
        "parsed_pct": 100.0 * parsed / total if total else None,
        "refusal_pct": 100.0 * refusal / total if total else None,
        "api_pct": 100.0 * api / total if total else None,
        "refusal_by_dimension": dict(refusal_by_dimension),
        "refused_questions": len(refused_runs),
        "refused_in_all_runs": sum(1 for count in refused_runs.values() if count == n_runs),
    }


def compute_tables(groups: Groups, expected: ExpectedIds, diag: Diagnostics,
                   order: str = "paper") -> TableData:
    present_models = {model for model, _, _ in groups}
    roster = [model for model, _ in PAPER_ROSTER if model in present_models]
    missing = [model for model, _ in PAPER_ROSTER if model not in present_models]
    if missing:
        diag.warn(f"Paper roster models with no trajectories: {', '.join(missing)}")
    used = set(roster) | set(LORA_DIAGONAL.values()) | {model for model, _ in TRANSFER_ROWS}
    extra = sorted(present_models - used)
    if extra:
        diag.note(f"Not part of the paper tables (ignored): {', '.join(extra)}")

    main = {model: main_stats(groups, model, diag) for model in roster}
    if order == "overall":
        def sort_key(model: str) -> tuple[bool, float]:
            overall = main[model]["overall"]
            return (model == CASCADE_DIR, -(overall.mean if overall else float("-inf")))
        roster.sort(key=sort_key)

    lift: dict[str, dict[str, Any]] = {}
    for task, adapter in LORA_DIAGONAL.items():
        base = eval_group_stat(groups, BASE_MODEL, task, expected, diag)
        lora = eval_group_stat(groups, adapter, task, expected, diag)
        lift[task] = {
            "adapter": adapter,
            "n_eval": len(expected.eval.get(task, ())),
            "base": base,
            "lora": lora,
            "lift": paired_delta(lora, base, diag, f"Table 10 {task}"),
        }

    transfer: dict[str, dict[str, dict[str, Stat | None]]] = {}
    base_stats = {
        task: cell_stat(groups.get((BASE_MODEL, task, "all"))) for task, _ in TRANSFER_TASKS
    }
    transfer[BASE_MODEL] = {task: {"stat": base_stats[task], "delta": None} for task, _ in TRANSFER_TASKS}
    for adapter, _ in TRANSFER_ROWS:
        transfer[adapter] = {}
        for task, _ in TRANSFER_TASKS:
            stat = cell_stat(groups.get((adapter, task, "all")))
            transfer[adapter][task] = {
                "stat": stat,
                "delta": paired_delta(stat, base_stats[task], diag, f"Table 3 {adapter} {task}"),
            }

    parse = {
        model: {task: parse_counts(groups.get((model, task, "all"))) for task, _ in PARSE_TASKS}
        for model in roster
        if model != CASCADE_DIR
    }
    return TableData(main, lift, transfer, parse, roster, extra)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def fmt(value: float | None, dp: int = 1) -> str:
    if value is None:
        return "–"
    text = format(value, f".{dp}f")
    return text[1:] if text.startswith("-") and float(text) == 0 else text


def fmt_signed(value: float | None, dp: int = 1) -> str:
    if value is None:
        return "–"
    text = fmt(value, dp)
    return text if text.startswith("-") else f"+{text}"


def fmt_delta_display(value: float | None, dp: int = 1) -> str:
    """Signed value with a typographic minus, as printed in Tables 2 and 3."""
    return fmt_signed(value, dp).replace("-", "−")


def fmt_pm(stat: Stat | None, dp: int = 1) -> str:
    if stat is None:
        return "–"
    return f"{fmt(stat.mean, dp)} ± {fmt(stat.err, dp)}"


@dataclass
class RenderedTable:
    key: str
    title: str
    caption: str
    label: str
    headers: list[str]
    rows: list[list[str]]
    align: str
    bold: set[tuple[int, int]] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    midrule_after: set[int] = field(default_factory=set)


_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_LATEX_SYMBOLS = {
    "±": r"$\pm$",
    "Δ": r"$\Delta$",
    "–": "--",
    "−": "$-$",
    "¹": r"$^{1}$",
    "†": r"$^{\dagger}$",
}


def latex_escape(text: str) -> str:
    escaped = "".join(_LATEX_SPECIAL.get(char, char) for char in text)
    for symbol, replacement in _LATEX_SYMBOLS.items():
        escaped = escaped.replace(symbol, replacement)
    return escaped


def _split_bold(cell: str) -> tuple[str, str]:
    """Bold covers the value only, not a trailing "(delta)" annotation, as in the paper."""
    head, sep, tail = cell.partition(" (")
    return head, (sep + tail) if sep else ""


def render_markdown(table: RenderedTable) -> str:
    lines = [f"**{table.title}**", "", f"_{table.caption}_", ""]
    lines.append("| " + " | ".join(table.headers) + " |")
    lines.append("|" + "|".join("---" if a == "l" else "---:" for a in table.align) + "|")
    for r, row in enumerate(table.rows):
        cells = []
        for c, cell in enumerate(row):
            if (r, c) in table.bold:
                head, tail = _split_bold(cell)
                cell = f"**{head}**{tail}"
            cells.append(cell)
        lines.append("| " + " | ".join(cells) + " |")
    if table.notes:
        lines.append("")
        lines.extend(table.notes)
    return "\n".join(lines) + "\n"


def render_latex(table: RenderedTable, source: str) -> str:
    lines = [
        f"% {table.title}. Generated by scripts/make_paper_tables.py from {source}.",
        "% Requires \\usepackage{booktabs}. Do not edit by hand; rerun the script.",
        "\\begin{table}[t]",
        "  \\centering",
        f"  \\caption{{{latex_escape(table.caption)}}}",
        f"  \\label{{{table.label}}}",
        "  \\small",
        f"  \\begin{{tabular}}{{{table.align}}}",
        "    \\toprule",
        "    " + " & ".join(latex_escape(header) for header in table.headers) + " \\\\",
        "    \\midrule",
    ]
    for r, row in enumerate(table.rows):
        cells = []
        for c, cell in enumerate(row):
            if (r, c) in table.bold:
                head, tail = _split_bold(cell)
                cells.append(f"\\textbf{{{latex_escape(head)}}}{latex_escape(tail)}")
            else:
                cells.append(latex_escape(cell))
        lines.append("    " + " & ".join(cells) + " \\\\")
        if r in table.midrule_after:
            lines.append("    \\midrule")
    lines.append("    \\bottomrule")
    lines.append("  \\end{tabular}")
    for note in table.notes:
        lines.append(f"  \\par\\smallskip{{\\footnotesize {latex_escape(note)}}}")
    lines.append("\\end{table}")
    return "\n".join(lines) + "\n"


def _bold_best(rows_values: list[list[float | None]], row_offset: int,
               skip_cols: set[int]) -> set[tuple[int, int]]:
    """Bold the best displayed (1 dp) value per column; ties are all bold."""
    bold: set[tuple[int, int]] = set()
    if not rows_values:
        return bold
    for c in range(len(rows_values[0])):
        if c in skip_cols:
            continue
        shown = [(r, float(fmt(values[c]))) for r, values in enumerate(rows_values) if values[c] is not None]
        if not shown:
            continue
        best = max(value for _, value in shown)
        bold |= {(r + row_offset, c + 1) for r, value in shown if value == best}
    return bold


def display_name(model: str, footnotes: bool = False) -> str:
    name = DISPLAY_NAMES.get(model, model)
    if footnotes and model in SILENT_AUDIO_FOOTNOTE:
        name += "¹"
    return name


def build_table1(data: TableData) -> RenderedTable:
    headers = ["System"] + [header for _, header in MAIN_METRICS]
    rows = [["Human"] + [HUMAN_MAIN[metric] or "–" for metric, _ in MAIN_METRICS]]
    values: list[list[float | None]] = []
    for model in data.roster:
        stats = data.main[model]
        means = [stats[m].mean if stats[m] else None for m, _ in MAIN_METRICS]
        rows.append([display_name(model, footnotes=True)] + [fmt(mean) for mean in means])
        values.append(means)
    skip = {i for i, (metric, _) in enumerate(MAIN_METRICS) if metric in UNBOLDED_MAIN_METRICS}
    return RenderedTable(
        key="table1",
        title="Table 1: Accuracy (%) on AcousticBench",
        caption=(
            "Overall is the macro-average of RAD, SER, and RSQ. Classify-Text is a "
            "transcript-only diagnostic omitted from aggregates. Error bars are in Table 7."
        ),
        label="tab:main-results",
        headers=headers,
        rows=rows,
        align="l" + "r" * len(MAIN_METRICS),
        bold=_bold_best(values, 1, skip),
        notes=[
            "¹ GPT-4o-Audio and MiMo-Audio-7B both require an audio input, so we pass in a "
            "silent audio file for text-only items.",
            "Human row: paper Table 11 (rating-weighted), rounded to one decimal.",
        ],
        midrule_after={0},
    )


def build_table7(data: TableData) -> list[RenderedTable]:
    parts = (
        ("table7a", "Overall and RAD", MAIN_METRICS[:6],
         ["System", "Overall", "RAD Agg.", "Pitch", "Loud.", "Tempo", "Dyn."]),
        ("table7b", "SER and RSQ", MAIN_METRICS[6:],
         ["System", "SER Agg.", "Classify (audio)", "Classify (text)", "Intensity", "RSQ"]),
    )
    tables = []
    for key, subtitle, metrics, headers in parts:
        rows = [["Human"] + [HUMAN_MAIN[metric] or "–" for metric, _ in metrics]]
        values: list[list[float | None]] = []
        for model in data.roster:
            stats = data.main[model]
            rows.append([display_name(model)] + [fmt_pm(stats[m]) for m, _ in metrics])
            values.append([stats[m].mean if stats[m] else None for m, _ in metrics])
        skip = {i for i, (metric, _) in enumerate(metrics) if metric in UNBOLDED_MAIN_METRICS}
        tables.append(RenderedTable(
            key=key,
            title=f"Table 7: Run-to-run variance for Table 1 ({subtitle})",
            caption=(
                "Accuracy means (%) with ±2 sample standard deviations across runs; ±0.0 "
                "indicates a deterministic run or a single completed run. Human cells match "
                "Table 11 (one-decimal rounding). Bold marks the best model mean per column."
            ),
            label=f"tab:variance-{'rad' if key == 'table7a' else 'ser'}",
            headers=headers,
            rows=rows,
            align="l" + "r" * len(metrics),
            bold=_bold_best(values, 1, skip),
            midrule_after={0},
        ))
    return tables


def build_table2(data: TableData) -> RenderedTable:
    rows: list[list[str]] = []
    bold: set[tuple[int, int]] = set()
    for r, task in enumerate(LORA_DIAGONAL):
        entry = data.lift[task]
        base, lora, lift = entry["base"], entry["lora"], entry["lift"]
        human = HUMAN_LIFT[task]
        rows.append([
            f"{LIFT_TASK_LABELS[task]} (n = {entry['n_eval']})",
            fmt(base.mean if base else None),
            f"{fmt(lora.mean if lora else None)} ({fmt_delta_display(lift.mean if lift else None)})",
            human,
        ])
        shown = {1: base.mean if base else None, 2: lora.mean if lora else None, 3: float(human)}
        shown_rounded = {c: float(fmt(v)) for c, v in shown.items() if v is not None}
        if shown_rounded:
            best = max(shown_rounded.values())
            bold |= {(r, c) for c, v in shown_rounded.items() if v == best}
    return RenderedTable(
        key="table2",
        title="Table 2: Task-specific LoRA performance improvement on held-out set",
        caption=(
            "Held-out eval split (half of each task, split by question ID). Human is the "
            "full-set human accuracy (Table 11). Error bars in Table 10."
        ),
        label="tab:lora-lift",
        headers=["Task", "Base (Kimi-Audio)", "LoRA (Δ)", "Human"],
        rows=rows,
        align="lrrr",
        bold=bold,
    )


def build_table10(data: TableData) -> RenderedTable:
    rows = []
    for task in LORA_DIAGONAL:
        entry = data.lift[task]
        rows.append([
            f"{LIFT_TASK_LABELS[task]} (n={entry['n_eval']})",
            entry["adapter"],
            fmt_pm(entry["base"]),
            fmt_pm(entry["lora"]),
            fmt_pm(entry["lift"]),
        ])
    return RenderedTable(
        key="table10",
        title="Table 10: Per-run detail for Table 2",
        caption=(
            "Specialist LoRA on the held-out eval split. Mean accuracy (%) over runs with "
            "±2 sd error bars; Lift uses paired per-run differences. Not directly comparable "
            "to Table 1."
        ),
        label="tab:lora-lift-detail",
        headers=["Task", "Adapter", "Kimi-Audio (reference)", "Specialist LoRA", "Lift (paired Δ)"],
        rows=rows,
        align="llrrr",
    )


def build_table3(data: TableData) -> RenderedTable:
    rows = []
    labels = [(BASE_MODEL, "Base Kimi-Audio")] + list(TRANSFER_ROWS)
    for model, label in labels:
        row = [label]
        for task, _ in TRANSFER_TASKS:
            entry = data.transfer[model][task]
            cell = fmt_pm(entry["stat"])
            if entry["delta"] is not None:
                cell += f" (Δ {fmt_delta_display(entry['delta'].mean)})"
            row.append(cell)
        rows.append(row)
    bold: set[tuple[int, int]] = set()
    for c, (task, _) in enumerate(TRANSFER_TASKS, start=1):
        candidates = [
            (r, data.transfer[model][task]) for r, (model, _) in enumerate(labels) if r > 0
        ]
        winners = [
            (r, float(fmt(entry["stat"].mean))) for r, entry in candidates
            if entry["stat"] is not None and entry["delta"] is not None and entry["delta"].mean > 0
        ]
        if winners:
            best = max(value for _, value in winners)
            bold |= {(r, c) for r, value in winners if value == best}
    return RenderedTable(
        key="table3",
        title="Table 3: Off-task accuracy (%) for two matched LoRAs",
        caption=(
            "Full test pools (split all). Δ is the lift in points over matched base "
            "Kimi-Audio (paired per run). Values are means across runs with ±2 standard "
            "deviations."
        ),
        label="tab:transfer",
        headers=["", *[header for _, header in TRANSFER_TASKS]],
        rows=rows,
        align="lrrr",
        bold=bold,
    )


def table6_footnote(data: TableData) -> tuple[str, str] | None:
    """(model, task) whose refusals get the dagger, and the footnote text."""
    model, task = _PAPER_TABLE6_FOOTNOTE["model"], _PAPER_TABLE6_FOOTNOTE["task"]
    counts = data.parse.get(model, {}).get(task)
    if not counts or not counts["refusal"]:
        return None
    dims = ", ".join(f"{dim} {n}" for dim, n in sorted(counts["refusal_by_dimension"].items()))
    text = (
        f"† {display_name(model)} {task.upper()}: {counts['refusal']} refusals (by dimension: "
        f"{dims}) on {counts['refused_questions']} distinct questions; "
        f"{counts['refused_in_all_runs']} refused in all {counts['n_runs']} runs."
    )
    return model, text


def build_table6(data: TableData) -> RenderedTable:
    rows: list[list[str]] = []
    midrules: set[int] = set()
    footnote = table6_footnote(data)
    for task, task_label in PARSE_TASKS:
        for model in data.parse:
            counts = data.parse[model][task]
            if counts is None:
                rows.append([display_name(model), task_label, "–", "–", "–", "–"])
                continue
            refusal = fmt(counts["refusal_pct"], 2)
            if footnote and footnote[0] == model and task == _PAPER_TABLE6_FOOTNOTE["task"]:
                refusal += "†"
            rows.append([
                display_name(model),
                task_label,
                f"{counts['n_questions']:,}",
                fmt(counts["parsed_pct"], 2),
                refusal,
                fmt(counts["api_pct"], 2),
            ])
        midrules.add(len(rows) - 1)
    midrules.discard(len(rows) - 1)
    return RenderedTable(
        key="table6",
        title="Table 6: Per-model output parseability on the three main tasks",
        caption=(
            "N is the per-task question count; rates pool all rows of all runs. Parsed: a "
            "definite A/B label. Refusal: an explicit decline or off-label response. API: "
            "upstream errors or empty responses. Parsed + Refusal + API = 100% up to rounding."
        ),
        label="tab:parse-rates",
        headers=["System", "Task", "N", "Parsed (%)", "Refusal (%)", "API (%)"],
        rows=rows,
        align="llrrrr",
        notes=[footnote[1]] if footnote else [],
        midrule_after=midrules,
    )


def build_table11() -> RenderedTable:
    rows = [list(row) for row in HUMAN_TABLE11]
    return RenderedTable(
        key="table11",
        title="Table 11: Human accuracy on AcousticBench",
        caption=(
            "Constants from the human study (not computed from trajectories). Overall is the "
            "macro-average of RAD, SER, and RSQ; other rows are rating-weighted averages over "
            "latest human attempts."
        ),
        label="tab:human",
        headers=["Category", "Accuracy (%)", "Ratings", "Questions"],
        rows=rows,
        align="lrrr",
    )


# ---------------------------------------------------------------------------
# --check against the published paper
# ---------------------------------------------------------------------------


@dataclass
class CheckCell:
    table: str
    row: str
    column: str
    paper: str
    ours: str
    value: float | None
    status: str
    note: str = ""


def _decimals(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def _is_tie(exact: Fraction, dp: int) -> bool:
    scaled = exact * 10**dp
    return (scaled * 2).denominator == 1 and (scaled * 2).numerator % 2 == 1


def compare_cell(table: str, row: str, column: str, paper: str | None,
                 value: float | None, exact: Fraction | None = None,
                 signed: bool = False) -> CheckCell:
    paper = "-" if paper is None else paper
    if paper == "-":
        if value is None:
            return CheckCell(table, row, column, "–", "–", None, "OK")
        return CheckCell(table, row, column, "–", fmt(value), value, "EXTRA",
                         "paper has no value for this cell")
    dp = _decimals(paper)
    if value is None:
        return CheckCell(table, row, column, paper, "–", None, "MISSING")
    ours = fmt_signed(value, dp) if signed else fmt(value, dp)
    note = ""
    if exact is not None and _is_tie(exact, dp):
        note = f"exact value {float(exact):.{dp + 2}f} is a rounding tie at {dp} dp"
    if ours == paper or float(ours) == float(paper):
        return CheckCell(table, row, column, paper, ours, value, "OK", note)
    if abs(value - float(paper)) <= 0.5 * 10**-dp + 1e-9:
        return CheckCell(table, row, column, paper, ours, value, "TIE",
                         note or "within half a unit of the printed digit")
    return CheckCell(table, row, column, paper, ours, value, "MISMATCH", note)


def run_check(data: TableData) -> list[CheckCell]:
    cells: list[CheckCell] = []
    for model, columns in PAPER_TABLE7.items():
        stats = data.main.get(model)
        name = display_name(model)
        for metric, header in MAIN_METRICS:
            paper = columns[metric]
            stat = stats.get(metric) if stats else None
            if paper is None:
                cells.append(compare_cell("T1/T7", name, header, None, stat.mean if stat else None))
                continue
            cells.append(compare_cell("T1/T7", name, header, paper[0],
                                      stat.mean if stat else None,
                                      stat.exact_mean if stat else None))
            cells.append(compare_cell("T7", name, f"{header} ±2sd", paper[1],
                                      stat.err if stat else None))

    for task, (base, lora, delta) in _PAPER_TABLE2.items():
        entry = data.lift[task]
        label = LIFT_TASK_LABELS[task]
        cells.append(compare_cell("T2", label, "Base", base, _mean(entry["base"]), _exact(entry["base"])))
        cells.append(compare_cell("T2", label, "LoRA", lora, _mean(entry["lora"]), _exact(entry["lora"])))
        cells.append(compare_cell("T2", label, "Δ", delta, _mean(entry["lift"]), _exact(entry["lift"]), signed=True))
    for task, triple in _PAPER_TABLE10.items():
        entry = data.lift[task]
        label = LIFT_TASK_LABELS[task]
        for column, cell, stat in zip(("Base", "LoRA", "Lift"), triple,
                                      (entry["base"], entry["lora"], entry["lift"])):
            mean, err = cell.split("+-")
            cells.append(compare_cell("T10", label, column, mean, _mean(stat), _exact(stat)))
            cells.append(compare_cell("T10", label, f"{column} ±2sd", err, stat.err if stat else None))

    for model, tasks in _PAPER_TABLE3.items():
        row_label = dict([(BASE_MODEL, "Base Kimi-Audio"), *TRANSFER_ROWS])[model]
        for task, (cell, delta) in tasks.items():
            entry = data.transfer.get(model, {}).get(task, {"stat": None, "delta": None})
            stat = entry["stat"]
            mean, err = cell.split("+-")
            header = dict(TRANSFER_TASKS)[task]
            cells.append(compare_cell("T3", row_label, header, mean, _mean(stat), _exact(stat)))
            cells.append(compare_cell("T3", row_label, f"{header} ±2sd", err, stat.err if stat else None))
            if delta is not None:
                cells.append(compare_cell("T3", row_label, f"{header} Δ", delta,
                                          _mean(entry["delta"]), _exact(entry["delta"]), signed=True))

    for model, tasks in PAPER_TABLE6.items():
        name = display_name(model)
        for task, task_label in PARSE_TASKS:
            counts = data.parse.get(model, {}).get(task)
            for column, paper in zip(("Parsed", "Refusal", "API"), tasks[task]):
                value = counts[f"{column.lower()}_pct"] if counts else None
                cells.append(compare_cell("T6", name, f"{task_label} {column}", paper, value))

    expected = _PAPER_TABLE6_FOOTNOTE
    counts = data.parse.get(expected["model"], {}).get(expected["task"])
    observed = None
    if counts:
        observed = {
            "refusals": counts["refusal"],
            "dimensions": sorted(counts["refusal_by_dimension"]),
            "questions": counts["refused_questions"],
            "all_runs": counts["refused_in_all_runs"],
        }
    for key in ("refusals", "dimensions", "questions", "all_runs"):
        paper_value = expected[key]
        ours_value = observed[key] if observed else None
        status = "OK" if ours_value == paper_value else ("MISSING" if observed is None else "MISMATCH")
        cells.append(CheckCell("T6 footnote", display_name(expected["model"]), key,
                               str(paper_value), str(ours_value), None, status))
    return cells


def _mean(stat: Stat | None) -> float | None:
    return stat.mean if stat is not None else None


def _exact(stat: Stat | None) -> Fraction | None:
    return stat.exact_mean if stat is not None else None


def print_check(cells: list[CheckCell], verbose: bool = False) -> int:
    counts = Counter(cell.status for cell in cells)
    shown = [cell for cell in cells if verbose or cell.status != "OK" or cell.note]
    if shown:
        headers = ("Table", "Row", "Column", "Paper", "Ours", "Raw", "Status", "Note")
        table_rows = [
            (cell.table, cell.row, cell.column, cell.paper, cell.ours,
             "" if cell.value is None else f"{cell.value:.4f}", cell.status, cell.note)
            for cell in shown
        ]
        widths = [max(len(h), *(len(row[i]) for row in table_rows)) for i, h in enumerate(headers)]
        print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
        print("  ".join("-" * w for w in widths))
        for row in table_rows:
            print("  ".join(value.ljust(w) for value, w in zip(row, widths)).rstrip())
        print()
    failures = counts.get("MISMATCH", 0) + counts.get("MISSING", 0)
    exact_ties = sum(1 for cell in cells if cell.status == "OK" and "rounding tie" in cell.note)
    print(
        f"--check: {len(cells)} published cells compared: {counts.get('OK', 0)} OK "
        f"({exact_ties} of them exact rounding ties printed as the paper prints them), "
        f"{counts.get('TIE', 0)} accepted ties printed differently, {counts.get('EXTRA', 0)} "
        f"extra, {counts.get('MISMATCH', 0)} mismatches, {counts.get('MISSING', 0)} missing."
    )
    print("--check: PASS" if failures == 0 else "--check: FAIL")
    return failures


# ---------------------------------------------------------------------------
# Paper error bars for other scripts (compare_runs.py)
# ---------------------------------------------------------------------------


def paper_error_bar(model: str, task: str, split: str) -> float | None:
    """Published +-2 sd for a (model, task, split) cell, or None if not in the paper."""
    if split == "eval":
        for diag_task, adapter in LORA_DIAGONAL.items():
            if diag_task != task:
                continue
            if model == adapter:
                return float(_PAPER_TABLE10[task][1].split("+-")[1])
            if model == BASE_MODEL:
                return float(_PAPER_TABLE10[task][0].split("+-")[1])
        return None
    if model in _PAPER_TABLE3 and model != BASE_MODEL and task in _PAPER_TABLE3[model]:
        return float(_PAPER_TABLE3[model][task][0].split("+-")[1])
    columns = PAPER_TABLE7.get(model)
    if columns and task in columns and columns[task] is not None:
        return float(columns[task][1])  # type: ignore[index]
    return None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def tables_to_json(data: TableData) -> dict[str, Any]:
    def stat_json(stat: Stat | None) -> dict[str, Any] | None:
        return stat.to_json() if stat is not None else None

    return {
        "table1_table7": {
            model: {
                "display_name": display_name(model),
                **{metric: stat_json(data.main[model][metric]) for metric, _ in MAIN_METRICS},
            }
            for model in data.roster
        },
        "table2_table10": {
            task: {
                "adapter": entry["adapter"],
                "n_eval": entry["n_eval"],
                "base": stat_json(entry["base"]),
                "lora": stat_json(entry["lora"]),
                "lift": stat_json(entry["lift"]),
            }
            for task, entry in data.lift.items()
        },
        "table3": {
            model: {
                task: {"stat": stat_json(entry["stat"]), "delta": stat_json(entry["delta"])}
                for task, entry in tasks.items()
            }
            for model, tasks in data.transfer.items()
        },
        "table6": data.parse,
        "table11": [
            {"category": c, "accuracy_pct": a, "ratings": r, "questions": q}
            for c, a, r, q in HUMAN_TABLE11
        ],
        "human_row": HUMAN_MAIN,
    }


def selected_tables(data: TableData, wanted: set[str]) -> list[RenderedTable]:
    tables: list[RenderedTable] = []
    if "1" in wanted:
        tables.append(build_table1(data))
    if "2" in wanted:
        tables.append(build_table2(data))
    if "3" in wanted:
        tables.append(build_table3(data))
    if "6" in wanted:
        tables.append(build_table6(data))
    if "7" in wanted:
        tables.extend(build_table7(data))
    if "10" in wanted:
        tables.append(build_table10(data))
    if "11" in wanted:
        tables.append(build_table11())
    return tables


def write_outputs(out: Path, tables: list[RenderedTable], payload: dict[str, Any],
                  source: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    by_file: dict[str, list[RenderedTable]] = defaultdict(list)
    for table in tables:
        by_file[table.key.rstrip("ab") if table.key.startswith("table7") else table.key].append(table)
    for stem, group in by_file.items():
        (out / f"{stem}.md").write_text("\n".join(render_markdown(t) for t in group), encoding="utf-8")
        (out / f"{stem}.tex").write_text("\n".join(render_latex(t, source) for t in group), encoding="utf-8")
    (out / "paper_tables.md").write_text(
        f"<!-- Generated by scripts/make_paper_tables.py from {source}. -->\n\n"
        + "\n".join(render_markdown(t) for t in tables),
        encoding="utf-8",
    )
    (out / "paper_tables.tex").write_text("\n".join(render_latex(t, source) for t in tables), encoding="utf-8")
    (out / "paper_tables.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return path.name


def build(root: Path, archive: Path | None = None, order: str = "paper",
          diag: Diagnostics | None = None) -> tuple[TableData, Groups, ExpectedIds, Diagnostics]:
    """Load trajectories, validate IDs and compute every table (library entry point)."""
    diag = diag or Diagnostics()
    groups = load_groups(root, diag)
    if archive is not None:
        expected = expected_ids_from_archive(archive, diag)
    else:
        expected = expected_ids_from_groups(groups, diag)
    compare_with_release_contract(expected, diag)
    validate_groups(groups, expected, diag)
    data = compute_tables(groups, expected, diag, order=order)
    return data, groups, expected, diag


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate AcousticBench paper Tables 1, 2, 3, 6, 7, 10, 11 from trajectories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1].split("Not reproduced")[0] if __doc__ else None,
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Trajectory root: reference_trajectories/ layout and/or run.py results/ "
                             "layout (default: <repo>/reference_trajectories)")
    parser.add_argument("--archive", type=Path, default=None,
                        help="Optional dataset root; take ID sets and eval split from "
                             "questions/*.jsonl instead of the trajectories")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output directory for .tex/.md/.json (default: <repo>/generated_tables)")
    parser.add_argument("--tables", nargs="+", default=["1", "2", "3", "6", "7", "10", "11"],
                        choices=["1", "2", "3", "6", "7", "10", "11"],
                        help="Tables to write (default: all)")
    parser.add_argument("--order", choices=("paper", "overall"), default="paper",
                        help="Row order: the paper's, or by recomputed Overall (Cascade last)")
    parser.add_argument("--check", action="store_true",
                        help="Compare against the published paper values; exit 1 on any mismatch")
    parser.add_argument("--check-verbose", action="store_true",
                        help="With --check, print every compared cell, not only exceptions")
    parser.add_argument("--no-write", action="store_true", help="Do not write output files")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root: Path = args.root
    if not root.is_dir():
        logger.error(f"Trajectory root not found: {root}")
        return 2
    data, groups, expected, diag = build(root, args.archive, args.order)
    if not groups:
        logger.error(f"No trajectories found under {root}")
        return 2
    logger.info(f"Loaded {len(groups)} (model, task, split) cells from {root}")

    tables = selected_tables(data, set(args.tables))
    for table in tables:
        print(render_markdown(table))

    check_failures = 0
    check_cells: list[CheckCell] = []
    if args.check:
        check_cells = run_check(data)
        check_failures = print_check(check_cells, verbose=args.check_verbose)

    if not args.no_write:
        source = _display_path(root)
        payload = {
            "generator": "scripts/make_paper_tables.py",
            "source": source,
            "id_sets": expected.source,
            "aggregation": {
                "error_bar": "2 x sample standard deviation (ddof=1) over runs",
                "rad": "item-weighted over all RAD rows",
                "ser": "item-weighted over SER-Classify (audio) + SER-Intensity rows",
                "overall": "per run mean(RAD, SER, RSQ), then mean over runs",
                "classify_text": "diagnostic, excluded from aggregates",
            },
            "tables": tables_to_json(data),
            "warnings": diag.warnings,
            "notes": diag.notes,
        }
        if args.check:
            payload["check"] = [cell.__dict__ for cell in check_cells]
        write_outputs(args.out, tables, payload, source)
        logger.info(f"Wrote {len(tables)} tables to {args.out}")

    if diag.warnings:
        logger.warning(f"{len(diag.warnings)} warnings (see above)")
    return 1 if check_failures else 0


if __name__ == "__main__":
    sys.exit(main())
