"""Position-swap protocol for pairwise audio comparison tasks.

Generates position-swapped variants of 2-clip questions in memory,
runs inference on both orderings, and classifies each (question, model)
pair into consistent-correct, consistent-wrong, or inconsistent.
"""

from __future__ import annotations

import copy

FLIP = {"A": "B", "B": "A"}


def swap_question(question: dict) -> dict:
    """Create a position-swapped variant of a 2-clip question.

    Audio paths are swapped between positions. Labels stay tied to
    position (first audio is always "Clip A"). Answer key flips.
    """
    q = copy.deepcopy(question)

    audio = q["audio_path"]
    if not isinstance(audio, list) or len(audio) != 2:
        raise ValueError(
            f"Position swap requires exactly 2 audio paths, "
            f"got {type(audio).__name__}: {audio}"
        )
    q["audio_path"] = [audio[1], audio[0]]
    q["answer"] = FLIP.get(q["answer"], q["answer"])
    q["meta"] = dict(q.get("meta", {}))
    q["meta"]["answer_key"] = q["answer"]
    q["meta"]["position_swapped"] = True
    q["meta"]["original_example_id"] = q["meta"].get("example_id", "")
    q["meta"]["example_id"] = q["meta"]["example_id"] + "-posswap"

    return q


def is_two_clip(question: dict) -> bool:
    audio = question.get("audio_path")
    return isinstance(audio, list) and len(audio) == 2


def generate_swapped_questions(questions: list[dict]) -> list[dict]:
    """Generate position-swapped variants for all 2-clip questions.

    Single-clip and text-only questions (ser_classify, ser_classify_text)
    have no position to swap, so they are left out of the swapped set and
    therefore out of Consistent Accuracy.
    """
    return [swap_question(q) for q in questions if is_two_clip(q)]


def classify_pair(
    orig_correct: bool,
    swap_correct: bool,
    orig_choice: str,
    swap_choice: str,
) -> str:
    """Classify a (question, model) pair from both orderings.

    Returns one of:
        "consistent-correct"    — right in both orderings
        "consistent-wrong"      — wrong in both orderings
        "inconsistent-primacy"  — model picked first-position both times
        "inconsistent-recency"  — model picked second-position both times
        "inconsistent-mixed"    — other inconsistent pattern
    """
    if orig_correct and swap_correct:
        return "consistent-correct"
    if not orig_correct and not swap_correct:
        return "consistent-wrong"

    orig_picked_first = orig_choice == "A"
    swap_picked_first = swap_choice == "A"

    if orig_picked_first and swap_picked_first:
        return "inconsistent-primacy"
    if not orig_picked_first and not swap_picked_first:
        return "inconsistent-recency"
    return "inconsistent-mixed"
