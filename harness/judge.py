"""Choice extraction from free-text model responses via LLM judge.

By default every non-empty response is sent to an LLM judge
(``gpt-4o-mini-2024-07-18``) for extraction. The judge understands intent and
handles chain-of-thought, refusals, and verbose responses.

The opt-in ``deterministic-then-llm`` extractor first tries the regex
patterns below and only calls the judge when none of them match. It is the
extractor that produced the reference rows whose ``extraction_source`` is
``"deterministic"``; it is not the paper's default protocol.

Transient judge failures (network errors, timeouts, HTTP 429/5xx, malformed
replies) are retried with exponential backoff before an error is raised.

Requires: OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any

from loguru import logger

# Dated snapshot behind the `gpt-4o-mini` alias used for the reference runs.
DEFAULT_JUDGE_MODEL = "gpt-4o-mini-2024-07-18"
OPENAI_CHAT_COMPLETIONS_URL = "https://api.openai.com/v1/chat/completions"

EXTRACTORS = ("llm", "deterministic-then-llm")
DEFAULT_EXTRACTOR = "llm"

JUDGE_MAX_ATTEMPTS = 5
JUDGE_BACKOFF_BASE_S = 2.0
JUDGE_BACKOFF_MAX_S = 60.0
JUDGE_TIMEOUT_S = 30
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

# Regex extractor from the original harness, used for the reference rows
# labelled `extraction_source: "deterministic"`.
CHOICE_PATTERNS = [
    re.compile(r"^\s*\(?([A-D])\)?\s*$", re.IGNORECASE),
    re.compile(r"^\s*\(?([A-D])\)?[\s.:,]", re.IGNORECASE),
    re.compile(
        r"\b(?:answer|choice|option)\s*(?:is|:)\s*\**\(?([A-D])\)?\**", re.IGNORECASE
    ),
    re.compile(
        r"\(?([A-D])\)?\s*(?:is\s+(?:the\s+)?(?:correct|best|right|higher))",
        re.IGNORECASE,
    ),
    re.compile(r"\*\*\(?([A-D])\)?\*\*", re.IGNORECASE),
    re.compile(
        r"\b(?:choose|select|pick)\s+(?:option\s+)?\(?([A-D])\)?", re.IGNORECASE
    ),
    re.compile(r"\bis\s+([A-D])[.\s,]", re.IGNORECASE),
]

LLM_EXTRACTION_PROMPT = """You are extracting which option a model selected in a multiple-choice task.
Do not judge correctness. Only identify the option the model intended to pick.

Rules:
- Return exactly one JSON object and nothing else.
- `classification` must be one of: `choice`, `unsure`, `refusal`.
- `selected_label` must be one of the provided labels or null.
- If `classification` is `choice`, `selected_label` must be filled.
- If `classification` is `unsure` or `refusal`, `selected_label` must be null.
- Prefer explicit option letters when present.
- If the answer paraphrases one option, map it to that option.
- If the answer refuses, says it cannot determine, says both are equal, or hedges, set `classification` to `refusal`.
- If the answer does not clearly choose an option, set `classification` to `unsure`.
- If the model's response contains chain-of-thought or thinking tags, focus ONLY on the final answer after any </think> tags.

Question:
{question}

Options:
{options_block}

Model answer:
{response}

Return JSON: {{ "classification": ..., "selected_label": ..., "notes": ... }}"""


def _strip_thinking(response: str) -> str:
    if "</think>" in response:
        return response.rsplit("</think>", 1)[1].strip()
    return response


class JudgeError(RuntimeError):
    """The LLM judge could not produce an extraction after all retries."""


class _RetryableJudgeError(Exception):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


def extract_choice_deterministic(
    response: str,
    valid_labels: list[str] | None = None,
) -> str | None:
    """Return the label matched by the first applicable regex, else None."""
    if not response or not response.strip():
        return None

    text = response.strip()
    if valid_labels is None:
        valid_labels = ["A", "B", "C", "D"]
    valid_set = {label.upper() for label in valid_labels}

    for pattern in CHOICE_PATTERNS:
        match = pattern.search(text)
        if match:
            label = match.group(1).upper()
            if label in valid_set:
                return label

    if len(text) == 1 and text.upper() in valid_set:
        return text.upper()

    return None


def _retry_after_seconds(resp) -> float | None:
    value = resp.headers.get("Retry-After") if resp is not None else None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _post_judge_request(api_key: str, payload: dict[str, Any]) -> str:
    """Send one judge request and return the raw message content."""
    import requests

    try:
        resp = requests.post(
            OPENAI_CHAT_COMPLETIONS_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=payload,
            timeout=JUDGE_TIMEOUT_S,
        )
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        raise _RetryableJudgeError(f"{type(e).__name__}: {e}") from e

    if resp.status_code in _RETRYABLE_STATUS:
        raise _RetryableJudgeError(
            f"HTTP {resp.status_code}: {resp.text[:200]}",
            retry_after=_retry_after_seconds(resp),
        )
    if resp.status_code >= 400:
        raise JudgeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

    try:
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
        raise _RetryableJudgeError(f"malformed judge response: {e}") from e


def extract_choice_llm(
    response: str,
    question: str,
    valid_labels: list[str],
    model: str = DEFAULT_JUDGE_MODEL,
    max_attempts: int = JUDGE_MAX_ATTEMPTS,
) -> dict[str, Any]:
    import json

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is required for the LLM judge. "
            "Set it in your environment before running evaluations."
        )

    valid_set = {label.upper() for label in valid_labels}
    options_block = "\n".join(f"{label}: Option {label}" for label in sorted(valid_set))

    judged_response = _strip_thinking(response)

    prompt = LLM_EXTRACTION_PROMPT.format(
        question=question,
        options_block=options_block,
        response=judged_response[:3000],
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_completion_tokens": 300,
        "response_format": {"type": "json_object"},
    }

    last_error: Exception | None = None
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        try:
            raw = _post_judge_request(api_key, payload)
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise _RetryableJudgeError(
                    f"judge returned non-object JSON: {raw[:200]}"
                )
        except JudgeError as e:
            logger.error(f"LLM judge extraction failed (not retryable): {e}")
            raise JudgeError(
                f"LLM judge extraction failed: {e}. "
                "Check OPENAI_API_KEY and the judge model name."
            ) from e
        except (_RetryableJudgeError, json.JSONDecodeError) as e:
            last_error = e
            if attempt >= max_attempts:
                break
            retry_after = getattr(e, "retry_after", None)
            delay = min(
                JUDGE_BACKOFF_MAX_S, JUDGE_BACKOFF_BASE_S * (2 ** (attempt - 1))
            )
            if retry_after is not None:
                delay = min(JUDGE_BACKOFF_MAX_S, max(delay, retry_after))
            logger.warning(
                f"LLM judge call failed (attempt {attempt}/{max_attempts}): {e}; "
                f"retrying in {delay:.1f}s"
            )
            time.sleep(delay)
            continue

        classification = str(parsed.get("classification", "")).strip().lower()
        if classification not in {"choice", "unsure", "refusal"}:
            classification = "choice" if parsed.get("selected_label") else "unsure"

        label = None
        if classification == "choice":
            raw_label = parsed.get("selected_label")
            if isinstance(raw_label, str):
                label = raw_label.strip().upper()
            if label not in valid_set:
                label = None
                classification = "unsure"

        notes = str(parsed.get("notes", "")).strip()

        return {
            "label": label,
            "classification": classification,
            "source": "llm",
            "reason": notes,
            "judge_model": model,
            "judge_attempts": attempt,
        }

    logger.error(
        f"LLM judge extraction failed after {max_attempts} attempts: {last_error}"
    )
    raise JudgeError(
        f"LLM judge extraction failed after {max_attempts} attempts: {last_error}. "
        "Check OPENAI_API_KEY and network connectivity."
    ) from last_error


def extract_choice(
    response: str,
    question: str = "",
    valid_labels: list[str] | None = None,
    extractor: str = DEFAULT_EXTRACTOR,
) -> dict[str, Any]:
    """Extract a choice label from a model response.

    ``extractor="llm"`` (default, the paper protocol) sends every non-empty
    response to the LLM judge. ``extractor="deterministic-then-llm"`` returns
    the regex match when there is one (``source="deterministic"``) and calls
    the judge otherwise. Raises ``JudgeError`` if the judge still fails after
    bounded retries, so a transient outage is never scored silently.
    """
    if extractor not in EXTRACTORS:
        raise ValueError(
            f"Unknown extractor {extractor!r}; choose from {list(EXTRACTORS)}"
        )
    if valid_labels is None:
        valid_labels = ["A", "B"]

    if not response or not response.strip():
        return {"label": None, "source": "empty"}

    if extractor == "deterministic-then-llm":
        label = extract_choice_deterministic(response, valid_labels)
        if label is not None:
            return {
                "label": label,
                "classification": "choice",
                "source": "deterministic",
            }

    return extract_choice_llm(response, question, valid_labels)
