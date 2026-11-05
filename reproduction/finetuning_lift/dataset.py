"""The one data layer. Both training and evaluation read prompts from here.

Clip A/B labels are ordinary text parts in the release schema. This module
preserves their document order for both training and evaluation.

Usage:
    from reproduction.finetuning_lift import dataset

    train = dataset.load(config.TASKS["rad"], split="train")
    heldout = dataset.load(config.TASKS["rad"], split="eval")
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

from reproduction.finetuning_lift import config

# The release documents exactly two answer labels for every task.
ANSWER_LABELS: tuple[str, ...] = ("A", "B")

_OPTION_LINE = re.compile(r"^\s*([AB])\.\s+(.*?)\s*$")

# Observed in the pre-release training data for RAD, absent from both the paper's
# Appendix D and the public release. Not applied by default: the release is what
# a third party can reproduce, and an undocumented prompt would be an unstated
# difference from the published artifact. Exposed so its effect can be measured
# rather than guessed -- see `load(..., system_prompt=...)`.
OBSERVED_RAD_SYSTEM_PROMPT = (
    "You are an expert audio engineer and acoustic analyst with decades of "
    "professional experience in music production, speech processing, and "
    "psychoacoustics. You have an exceptional ear for perceiving subtle "
    "differences in audio properties such as pitch, tempo, loudness, and "
    "dynamic range. Listen carefully to the provided audio clips and give "
    "precise, confident answers based on your expert analysis."
)


@dataclass(frozen=True)
class Example:
    """One benchmark question, with its prompt parts in document order."""

    example_id: str
    task_key: str
    split: str
    category: str
    dimension: str
    question_order: int
    answer_key: str

    # Release message parts, verbatim and in order: alternating text and
    # audio_file entries. Never reordered, never flattened.
    parts: tuple[dict[str, Any], ...]

    system_prompt: str | None = None

    @property
    def audio_relpaths(self) -> tuple[str, ...]:
        return tuple(
            part["path"] for part in self.parts if part["type"] == "audio_file"
        )

    @property
    def audio_paths(self) -> tuple[Path, ...]:
        return tuple(config.RELEASE_ROOT / rel for rel in self.audio_relpaths)

    @property
    def parts_audio_count(self) -> int:
        return sum(1 for part in self.parts if part["type"] == "audio_file")

    @property
    def text_parts(self) -> tuple[str, ...]:
        return tuple(
            part["text"] for part in self.parts if part["type"] == "text"
        )

    @property
    def question(self) -> str:
        """The task question, which the release always places first."""
        return self.text_parts[0] if self.text_parts else ""

    @property
    def options(self) -> dict[str, str]:
        """Answer letter -> option text, parsed from the trailing options block."""
        parsed: dict[str, str] = {}
        for line in self.text_parts[-1].splitlines() if self.text_parts else []:
            match = _OPTION_LINE.match(line)
            if match:
                parsed[match.group(1)] = match.group(2)
        return parsed

    @property
    def target(self) -> str:
        """The assistant string to train on, e.g. "B. Clip B is louder"."""
        option = self.options.get(self.answer_key)
        return f"{self.answer_key}. {option}" if option else self.answer_key

    def prompt_digest(self) -> str:
        """Stable fingerprint of the rendered prompt, audio paths excluded.

        Audio paths are replaced by their position so the digest tracks prompt
        wording and part ordering only.
        """
        tokens: list[str] = []
        if self.system_prompt:
            tokens.append(f"system:{self.system_prompt}")
        for part in self.parts:
            if part["type"] == "audio_file":
                tokens.append("<AUDIO>")
            else:
                tokens.append(part["text"])
        payload = "\n".join(tokens).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _iter_records(slice_name: str) -> Iterator[dict[str, Any]]:
    path = config.questions_path(slice_name)
    if not path.is_file():
        raise FileNotFoundError(
            f"release file not found: {path}\n"
            f"Set ACOUSTICBENCH_DATA to the authorized-data directory."
        )
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load(
    task: config.Task,
    *,
    split: str,
    system_prompt: str | None = None,
) -> list[Example]:
    """Load one task's split from the release, in file order.

    `split` is read from each record's own `split` field, so the release defines
    the partition and this package never re-splits anything.
    """
    if split not in {"train", "eval"}:
        raise ValueError(f"split must be 'train' or 'eval', got {split!r}")

    examples: list[Example] = []
    for record in _iter_records(task.slice_name):
        if record["split"] != split:
            continue

        found_slice = record["question_set"]["slice"]
        if found_slice != task.slice_name:
            raise ValueError(
                f"{task.label}: record {record['example_id']} has slice "
                f"{found_slice!r}, expected {task.slice_name!r}"
            )

        content = record["messages"][0]["content"]
        examples.append(
            Example(
                example_id=record["example_id"],
                task_key=task.key,
                split=split,
                category=record["category"],
                dimension=record["dimension"],
                question_order=record["question_set"]["question_order"],
                answer_key=record["answer_key"],
                parts=tuple(content),
                system_prompt=system_prompt,
            )
        )
    return examples


def swap_clip_order(example: Example) -> Example:
    """Reverse the two clips while leaving the labels in place, flipping the answer.

    The decisive test for whether a model actually consumes both clips of a
    pairwise question. The prompt text is byte-identical; only which audio sits
    behind "Clip A:" and which behind "Clip B:" changes, so the correct answer
    flips.

    A model that listens scores about the same either way. A model that is
    guessing by position -- the failure mode that produced 85% "B" and
    near-chance accuracy when clip labels were being dropped -- scores
    roughly `100 - x`, because its fixed answer is now wrong exactly as often as
    it was previously right.

    Raises on anything that is not a two-clip question; there is nothing to swap.
    """
    audio_positions = [
        index
        for index, part in enumerate(example.parts)
        if part["type"] == "audio_file"
    ]
    if len(audio_positions) != 2:
        raise ValueError(
            f"{example.example_id}: swap needs exactly 2 clips, "
            f"found {len(audio_positions)}"
        )

    first, second = audio_positions
    parts = list(example.parts)
    parts[first], parts[second] = parts[second], parts[first]

    flipped = {"A": "B", "B": "A"}[example.answer_key]
    return dataclasses.replace(example, parts=tuple(parts), answer_key=flipped)


@lru_cache(maxsize=1)
def subset_index() -> dict[str, str]:
    """example_id -> fine-grained subset, from the release audio inventory.

    The question JSONL carries `dimension` (loudness, pitch, ...) but the paper's
    per-cue tables also split by source family (speech / instrument /
    synthetic). That distinction lives in `audio_files.csv` as `audio_category`.
    """
    path = config.RELEASE_ROOT / "audio_files.csv"
    if not path.is_file():
        return {}
    index: dict[str, str] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            subset = (row.get("audio_category") or "").strip()
            if subset:
                index.setdefault(row["example_id"], subset)
    return index


def subset_of(example: Example) -> str:
    """Finest available grouping for reporting, falling back to `dimension`."""
    return subset_index().get(example.example_id) or example.dimension


# --------------------------------------------------------------------------
# Rendering -- the two consumers
# --------------------------------------------------------------------------


def wire_messages(example: Example) -> list[dict[str, Any]]:
    """Messages for the Modal serving contract (evaluation).

    Emits canonical parts with explicit `audio_index` rather than relying on the
    server's implicit ordering, so the i-th audio part always binds to the i-th
    encoded clip. Text parts pass through verbatim, which is what keeps
    "Clip A:" / "Clip B:" in front of their clips.
    """
    content: list[dict[str, Any]] = []
    audio_index = 0
    for part in example.parts:
        if part["type"] == "audio_file":
            content.append({"type": "audio_file", "audio_index": audio_index})
            audio_index += 1
        else:
            content.append({"type": "text", "text": part["text"]})

    messages: list[dict[str, Any]] = []
    if example.system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": example.system_prompt}],
            }
        )
    messages.append({"role": "user", "content": content})
    return messages


def kimi_conversation(
    example: Example,
    *,
    audio_path_for: dict[str, str],
    include_target: bool = True,
) -> list[dict[str, str]]:
    """Conversation in Kimi-Audio's native training format.

    `audio_path_for` maps release-relative audio paths to the paths the trainer
    will see (inside the Modal volume). Part order is preserved exactly as in
    `wire_messages`, so the model trains on the prompt it is evaluated on.
    """
    conversation: list[dict[str, str]] = []
    if example.system_prompt:
        conversation.append(
            {
                "role": "user",
                "message_type": "text",
                "content": example.system_prompt,
            }
        )

    for part in example.parts:
        if part["type"] == "audio_file":
            conversation.append(
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": audio_path_for[part["path"]],
                }
            )
        else:
            conversation.append(
                {
                    "role": "user",
                    "message_type": "text",
                    "content": part["text"],
                }
            )

    if include_target:
        conversation.append(
            {
                "role": "assistant",
                "message_type": "text",
                "content": example.target,
            }
        )
    return conversation


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

# Ordered most-specific first: a bare letter is the documented output format, but
# models often restate the option, and "Clip B" must not be read as choosing A
# merely because the letter "A" appears elsewhere in the sentence.
_ANSWER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*\(?([AB])\)?\s*[.):]?\s*$"),
    re.compile(r"^\s*\(?([AB])\)?\s*[.):]\s+"),
    re.compile(r"\banswer\s*(?:is)?\s*[:\-]?\s*\(?([AB])\)?\b", re.IGNORECASE),
    re.compile(r"\boption\s*\(?([AB])\)?\b", re.IGNORECASE),
    re.compile(r"\bclip\s+([AB])\b", re.IGNORECASE),
    re.compile(r"^\s*\(?([AB])\)?\b"),
)


def extract_answer(response: str) -> str | None:
    """Pull the chosen letter out of a model response, or None if ambiguous.

    Returns None rather than guessing so that unparseable generations are
    reported as such instead of being silently scored as wrong -- an
    instruction-following failure and an acoustic failure are different results.
    """
    text = (response or "").strip()
    if not text:
        return None
    for pattern in _ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    return None
