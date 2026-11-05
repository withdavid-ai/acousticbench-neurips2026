from __future__ import annotations


def preserve_db_question_text(prompt: str | None) -> str:
    """Return DB-authored question text unchanged."""
    return "" if prompt is None else str(prompt)
