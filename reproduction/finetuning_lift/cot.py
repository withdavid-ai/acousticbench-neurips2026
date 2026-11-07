"""Does asking for a description before the answer help? Closed vs open model.

The native prompt, with its answer-format line swapped for a reasoning
instruction. Nothing else changes:

    Which clip sounds louder?
    Clip A:
    [audio]
    Clip B:
    [audio]
    Options:
    A. Clip A is louder
    B. Clip B is louder

    Describe the acoustic quality of the provided audio, and then make the
    judgment of whether A or B is louder.

The last line replaces "Answer with only the letter A or B." so the model is not
given contradictory instructions. "louder" is taken from the item's own option A,
because RAD varies across louder / higher in pitch / faster / wider dynamic range.

Run against the direct prompt the rest of this package uses, on Gemini 3.1 Pro
(closed, reasoning at max) and Kimi-Audio-7B (open). Four cells per task, so the
CoT effect is separable from the model difference:

                    direct      CoT
    Gemini 3.1 Pro    x          x
    Kimi-Audio-7B     x          x

Only the three pairwise tasks. The prompt presupposes two clips being compared,
and SER-Classify has one clip whose A/B are emotion labels rather than clips, so
the same intervention does not exist there.

Two details that matter more than they look:

* **Extraction reads the last judgment, not the first mention.** A CoT response
  says things like "Clip A sounds brighter ... Clip B has more energy ...
  therefore B". Scanning for the first A or B would score the description instead
  of the answer, and would do it silently.
* **Kimi's server caps generation at 16 tokens** when `meta.task == "MQA"`, which
  is right for a bare letter and fatal for a description. The CoT condition drops
  that flag.

`thinking_level="high"` is the maximum the API accepts; `max` and `ultra` are
rejected, so high is max effort for this model.

Usage:
    python -m reproduction.finetuning_lift.cot \
        --model gemini --condition cot --all
    python -m reproduction.finetuning_lift.cot --report
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproduction.finetuning_lift import config, dataset  # noqa: E402

# All four tasks. SER-Classify was initially excluded on the grounds that the
# template compares two clips while its A/B are emotion labels, but the actual
# intervention is "describe before judging", which applies to it perfectly well.
# Its predicate falls back to the generic "is the answer", since there is no
# "Clip A is X" option text to derive one from.
COT_TASKS = ("rad", "ser_classify", "ser_intensity", "rsq")

# Fixed subset, same items for every model and condition, so the comparison is
# paired rather than four independent samples.
SUBSET_SIZE = 200
SUBSET_SEED = 20260726

# Models the report iterates over. Gemini is the closed system; the rest are
# Modal-hosted open models, and a second open model exists here as a control on
# the pipeline itself rather than as a headline result.
COT_MODELS = ("kimi", "qwen3omni", "gemini")

GEMINI_MODEL = "gemini-3.1-pro-preview"
# "high" is the ceiling: the API rejects "max" and "ultra".
GEMINI_THINKING_LEVEL = "high"

# Per-request deadline, milliseconds. Generous: at max thinking a single RAD item
# takes tens of seconds.
GEMINI_TIMEOUT_MS = 240_000

# Generous ceiling for CoT generation. Gemini's longest description is ~1.5k
# characters, so this is several times the longest legitimate response.
MODAL_MAX_NEW_TOKENS = 1024

RESULTS_SUBDIR = "cot"


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------


# The release prompt ends with this line on every pairwise task, verified across
# all three. The CoT condition swaps it for the reasoning instruction so the two
# conditions differ by exactly that, and the model is never told both to describe
# and to answer with only a letter.
DIRECT_ANSWER_LINE = "Answer with only the letter A or B."

COT_TEMPLATE = (
    "Describe the acoustic quality of the provided audio, and then make the "
    "judgment of whether A or B {predicate}."
)

_CLIP_A_PREFIX = re.compile(r"^clip\s+a\s+(is|has)\s+", re.IGNORECASE)


def comparative_for(example: dataset.Example) -> str:
    """The predicate for "whether A or B ...", taken from the item's own options.

    Derived rather than hardcoded. RAD alone varies across louder / higher in
    pitch / faster / the wider dynamic range, and the verb varies with it: three
    read "Clip A is X" but dynamic range reads "Clip A has X". Carrying the verb
    through keeps the instruction grammatical, which matters because the prompt is
    the experiment.
    """
    option = (example.options.get("A") or "").strip().rstrip(".")
    match = _CLIP_A_PREFIX.match(option)
    if not match:
        return "is the answer"
    verb = match.group(1).lower()
    rest = option[match.end():].strip()
    return f"{verb} {rest}" if rest else "is the answer"


ANSWER_FIRST_TEMPLATE = (
    "Answer with only the letter A or B, and then explain your reasoning."
)

# Requires free text before the answer while requiring nothing of the audio. If a
# model degrades here too, the cause is generating text before committing, not
# acoustic reasoning -- which is the confound the CoT condition alone cannot rule
# out.
PLACEBO_TEMPLATE = (
    "Restate the two answer options in your own words, and then answer with only "
    "the letter A or B."
)


def _replace_answer_line(example: dataset.Example, instruction: str) -> list[dict]:
    """The native prompt with its answer-format line swapped for `instruction`."""
    parts: list[dict] = []
    audio_index = 0
    replaced = False
    for part in example.parts:
        if part["type"] == "audio_file":
            parts.append({"type": "audio_file", "audio_index": audio_index})
            audio_index += 1
            continue
        text = part["text"]
        if DIRECT_ANSWER_LINE in text:
            text = text.replace(DIRECT_ANSWER_LINE, instruction)
            replaced = True
        parts.append({"type": "text", "text": text})
    if not replaced:
        raise RuntimeError(
            f"{example.example_id}: expected the prompt to contain "
            f"{DIRECT_ANSWER_LINE!r}; refusing to guess where the instruction goes"
        )
    return parts


def answer_first_parts(example: dataset.Example) -> list[dict]:
    """Control: the letter first, the explanation after."""
    return _replace_answer_line(example, ANSWER_FIRST_TEMPLATE)


def placebo_parts(example: dataset.Example) -> list[dict]:
    """Control: free text before the answer, but nothing about the audio."""
    return _replace_answer_line(example, PLACEBO_TEMPLATE)


# CoT plus an explicit forcing clause. Note this is NOT a single-line swap from
# the direct prompt: it adds a sentence the direct condition never had, so it is
# only interpretable against `cot`, not against `direct`.
FORCED_SUFFIX = (
    " You must choose either A or B, even if the difference is small. Do not"
    " answer that the clips are the same."
)


def cot_forced_parts(example: dataset.Example) -> list[dict]:
    """CoT, but the model is told it may not decline.

    Disambiguates the abstentions. Under plain CoT, Kimi-Audio answers only the
    items it judges answerable and scores 67.3 on them -- a self-selected set. What
    it would score on the items it declined is the question that decides whether
    "these clips sound identical" is an honest report of an absent percept or a
    behavioural quirk masking usable signal.
    """
    instruction = COT_TEMPLATE.format(predicate=comparative_for(example))
    return _replace_answer_line(example, instruction + FORCED_SUFFIX)


def _framed_parts(example: dataset.Example, instruction: str | None) -> list[dict]:
    """Move the task framing from after the audio to before it.

    The release prompt is: question, "Clip A:", audio, "Clip B:", audio, then a
    final part carrying the Options block and the answer instruction. So the
    instruction sits behind roughly six seconds of audio tokens. This variant puts
    that whole trailing block directly after the question stem, before either clip.

    Note the scope: this moves the Options block along with the instruction, so it
    tests "task framing before vs after the audio" rather than the instruction
    sentence in isolation. Splitting them would leave the instruction referring to
    options the model has not been shown yet.

    `instruction` replaces the answer-format line, or None keeps it as-is.
    """
    trailing_index = max(
        i for i, part in enumerate(example.parts) if part["type"] == "text"
    )
    trailing = example.parts[trailing_index]["text"]
    if instruction is not None:
        if DIRECT_ANSWER_LINE not in trailing:
            raise RuntimeError(
                f"{example.example_id}: expected {DIRECT_ANSWER_LINE!r} in the "
                "trailing text part"
            )
        trailing = trailing.replace(DIRECT_ANSWER_LINE, instruction)

    head: list[dict] = []
    body: list[dict] = []
    audio_index = 0
    for i, part in enumerate(example.parts):
        if i == trailing_index:
            continue
        if part["type"] == "audio_file":
            body.append({"type": "audio_file", "audio_index": audio_index})
            audio_index += 1
        elif not head:
            head.append({"type": "text", "text": part["text"]})
        else:
            body.append({"type": "text", "text": part["text"]})
    return [*head, {"type": "text", "text": trailing}, *body]


def framed_parts(example: dataset.Example) -> list[dict]:
    """Framing before the audio, answer-format line unchanged."""
    return _framed_parts(example, None)


def framed_cot_parts(example: dataset.Example) -> list[dict]:
    """Framing before the audio, with the CoT instruction."""
    return _framed_parts(example, COT_TEMPLATE.format(predicate=comparative_for(example)))


def cot_parts(example: dataset.Example) -> list[dict]:
    """The native prompt with its answer-format line replaced by the CoT instruction.

    Everything else is byte-identical to the direct condition, clip labels and
    ordering included. An earlier version rebuilt the prompt from its pieces,
    which confounded the reasoning instruction with a prompt rewrite and made any
    difference uninterpretable.
    """
    instruction = COT_TEMPLATE.format(predicate=comparative_for(example))

    parts: list[dict] = []
    audio_index = 0
    replaced = False
    for part in example.parts:
        if part["type"] == "audio_file":
            parts.append({"type": "audio_file", "audio_index": audio_index})
            audio_index += 1
            continue
        text = part["text"]
        if DIRECT_ANSWER_LINE in text:
            text = text.replace(DIRECT_ANSWER_LINE, instruction)
            replaced = True
        parts.append({"type": "text", "text": text})

    if not replaced:
        raise RuntimeError(
            f"{example.example_id}: expected the prompt to contain "
            f"{DIRECT_ANSWER_LINE!r}; refusing to guess where the instruction goes"
        )
    return parts


def direct_parts(example: dataset.Example) -> list[dict]:
    """The prompt used everywhere else, unchanged."""
    return dataset.wire_messages(example)[0]["content"]


# --------------------------------------------------------------------------
# CoT-aware extraction
# --------------------------------------------------------------------------

# Explicit verdicts, searched for the LAST occurrence. Ordered by how
# unambiguous they are.
_VERDICT_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "Answer: A", "**Judgment:**\nA. Fear", "Verdict - B"
    re.compile(
        r"\b(?:answer|judgment|judgement|verdict|conclusion|choice|option)\b"
        r"[^A-Za-z0-9]{0,12}(?:is\b[^A-Za-z0-9]{0,6})?\(?([AB])\)?(?![A-Za-z])",
        re.IGNORECASE,
    ),
    # "... the emotion conveyed more strongly is:\nB. Neutral"
    re.compile(r"\bis\s*:?\s*\n?\s*\(?([AB])\)?[.):]\s", re.IGNORECASE),
    # A line that restates the chosen option: "A. Sadness", "**B. Neutral**"
    re.compile(r"^[^A-Za-z0-9]{0,4}\(?([AB])\)?[.):]\s+\S", re.MULTILINE),
    # "Therefore ... A". Window widened from 40: the real sentences run longer.
    re.compile(r"\b(?:therefore|thus|hence|so)\b[^.]{0,140}?\b([AB])\b", re.IGNORECASE),
    re.compile(
        r"\bclip\s+([AB])\s+is\s+(?:the\s+)?(?:louder|higher|faster|cleaner|more)",
        re.IGNORECASE,
    ),
    # Bare letter alone on its line.
    re.compile(r"^\s*\(?([AB])\)?\s*[.):]?\s*$", re.MULTILINE),
)

# Phrases where A and B appear because the model is echoing the question, not
# answering it. Matching inside these produced a confident wrong label on
# "we cannot determine whether A or B is the answer".
_ECHO_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"whether\s+A\s+or\s+B", re.IGNORECASE),
    re.compile(r"\bA\s+or\s+B\b", re.IGNORECASE),
    re.compile(r"only\s+the\s+letter\s+A\s+or\s+B", re.IGNORECASE),
    # Explicit double negatives. "A or B" above does not cover "A nor B", so
    # "therefore, neither A nor B can be chosen" fell through to the loose
    # "therefore ... ([AB])" rule and was scored as A. That is the same failure
    # as the original fallback bug: a stated non-answer becoming a scored guess,
    # which by chance lands on the right letter about half the time and
    # therefore inflates accuracy rather than only misclassifying it.
    re.compile(r"neither\s+A\s+nor\s+B", re.IGNORECASE),
    re.compile(r"between\s+A\s+and\s+B", re.IGNORECASE),
)

# The model declining to judge, rather than judging wrongly. Tracked separately:
# for one of the models below this is the dominant behaviour and reporting it as
# "incorrect" would hide the actual result.
_REFUSAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"cannot (?:directly )?(?:listen|hear|access|determine|provide|make)", re.IGNORECASE),
    re.compile(r"(?:unable|not able) to (?:listen|hear|access|determine|provide)", re.IGNORECASE),
    re.compile(r"(?:audio|clips?) (?:are |is |were )?not provided", re.IGNORECASE),
    re.compile(r"does not contain any (?:actual )?audio", re.IGNORECASE),
    re.compile(r"as an AI language model", re.IGNORECASE),
    re.compile(r"impossible to (?:make|determine)", re.IGNORECASE),
)


def is_refusal(response: str) -> bool:
    """True when the model declined to judge rather than judging."""
    text = response or ""
    return any(pattern.search(text) for pattern in _REFUSAL_PATTERNS)


# A bare leading letter, the expected shape of an answer-first response.
_LEADING_LETTER = __import__("re").compile(r"^\s*\(?([AB])\)?\s*(?:[.):,\-]|\s|$)")


def _mask_echoes(response: str) -> str:
    """Strip markdown emphasis and blank out question echoes.

    Shared by both free-text extractors so they cannot drift apart. Markdown
    emphasis around a verdict ("**A. Fear**") blocks every pattern that expects the
    letter to follow its cue directly; echoes like "whether A or B" otherwise let a
    verdict pattern match inside the restated question.
    """
    text = (response or "").strip()
    if not text:
        return ""
    text = text.replace("**", "").replace("__", "")
    for pattern in _ECHO_PATTERNS:
        text = pattern.sub(lambda m: "_" * len(m.group(0)), text)
    return text


def extract_cot_answer(response: str) -> str | None:
    """Pull the final verdict out of a free-text CoT response.

    Takes the *last* explicit verdict, because the description that precedes the
    answer mentions both clips by name. `dataset.extract_answer` is first-match
    and is correct for the direct condition, where the whole response is a letter.

    Returns None when there is no explicit verdict, rather than guessing from the
    last letter mentioned. That fallback looked reasonable and was actively
    harmful: on a refusal reading "we cannot determine whether A or B is the
    answer" it returned A, turning a non-answer into a scored guess.
    """
    masked = _mask_echoes(response)
    if not masked:
        return None

    for pattern in _VERDICT_PATTERNS:
        matches = list(pattern.finditer(masked))
        if matches:
            return matches[-1].group(1).upper()
    return None


def extract_answer_first(response: str) -> str | None:
    """The letter the model committed to *before* explaining.

    Last-match extraction is wrong here by construction: the explanation that
    follows the answer mentions both letters, so scanning from the end would score
    the reasoning instead of the answer. Reads the leading span, then falls back to
    the earliest explicit verdict.
    """
    if not response:
        return None
    masked = _mask_echoes(response)
    # The instruction asks for the letter first, so the common case is a bare
    # leading "A" or "B", optionally followed by punctuation or a line break. Match
    # that explicitly: `dataset.extract_answer` expects a letter-only response and
    # does not fire on "A\n\nBecause ...".
    lead = _LEADING_LETTER.match(masked)
    if lead:
        return lead.group(1).upper()
    head = masked[:60]
    direct = dataset.extract_answer(head)
    if direct:
        return direct
    earliest: tuple[int, str] | None = None
    for pattern in _VERDICT_PATTERNS:
        match = pattern.search(masked)
        if match and (earliest is None or match.start() < earliest[0]):
            earliest = (match.start(), match.group(1).upper())
    return earliest[1] if earliest else None


# An explicit verdict cue, e.g. "Answer: A". Reused by the placebo extractor.
_VERDICT_CUE = re.compile(
    r"\b(?:answer|judgment|judgement|verdict|conclusion|choice)\b"
    r"[^A-Za-z0-9]{0,12}(?:is\b[^A-Za-z0-9]{0,6})?\(?([AB])\)?(?![A-Za-z])",
    re.IGNORECASE,
)
_BARE_LETTER_LINE = re.compile(r"^\s*\(?([AB])\)?\s*[.):,]?\s*$")


def extract_placebo_answer(response: str) -> str | None:
    """The answer in the placebo condition, where the model first restates options.

    `extract_cot_answer` cannot be used here. The restatement produces lines like
    "A. Clip A is louder" and "B. Clip B is louder", which match its
    restated-option pattern; since it returns the last match of the highest-priority
    pattern that fires, it read the second restated option as the verdict and
    labelled 100% of Qwen3-Omni's responses "B".

    So: prefer an explicit cue, else a bare letter on the final line. A response
    that only restates options, with no separate answer, is genuinely ambiguous
    between restatement and answer and is left unparsed rather than guessed.
    """
    text = _mask_echoes(response)
    if not text:
        return None
    cues = list(_VERDICT_CUE.finditer(text))
    if cues:
        return cues[-1].group(1).upper()
    for line in reversed([l for l in text.splitlines() if l.strip()]):
        bare = _BARE_LETTER_LINE.match(line)
        if bare:
            return bare.group(1).upper()
        break  # only the final non-empty line counts as the answer slot
    return None


# Every condition, as (prompt builder, answer extractor). One mapping rather than
# two parallel dicts: a previous version kept them separate and an edit added three
# conditions to the builder side only, so each new condition raised KeyError at
# dispatch. Extraction is not interchangeable between conditions -- the answer sits
# at the start in `answer_first`, at the end elsewhere, and the placebo's
# restatement mimics a verdict -- so the pairing has to be explicit.
CONDITIONS: dict[str, tuple] = {
    "direct": (direct_parts, dataset.extract_answer),
    "cot": (cot_parts, extract_cot_answer),
    # The answer precedes the explanation, so read forwards, not backwards.
    "answer_first": (answer_first_parts, extract_answer_first),
    # NOT extract_cot_answer: the restatement mimics a verdict.
    "placebo": (placebo_parts, extract_placebo_answer),
    # Instruction position variants; output shape matches their unframed twins.
    "framed": (framed_parts, dataset.extract_answer),
    "framed_cot": (framed_cot_parts, extract_cot_answer),
    # CoT plus a clause forbidding abstention.
    "cot_forced": (cot_forced_parts, extract_cot_answer),
}

# Conditions whose responses are free text and therefore re-scoreable from disk.
FREE_TEXT_CONDITIONS = tuple(k for k in CONDITIONS if k != "direct")


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


def _modal_method(model: config.Model):
    """Bind the un-finetuned serving method for any Modal-hosted model.

    Kept model-agnostic on purpose: the point of running a second open model is to
    check that the CoT behaviour is a property of the model and not of this
    pipeline, which only holds if both go through the identical code path.
    """
    import modal

    remote_cls = modal.Cls.from_name(
        model.serving_app,
        model.serving_class,
        environment_name=config.MODAL_ENV,
    )
    return remote_cls().generate


def _modal_call(method, example: dataset.Example, parts: list[dict], condition: str) -> str:
    audios = [
        {
            "data": base64.b64encode(path.read_bytes()).decode(),
            "format": path.suffix.lstrip(".").lower() or "wav",
        }
        for path in example.audio_paths
    ]
    meta = {
        "example_id": example.example_id,
        # Qwen3-Omni's server defaults thinker_max_new_tokens to 256, which
        # truncates a CoT response mid-word -- often exactly at "**Judgment:**",
        # so the verdict is cut off and the item scores as unparsed. That produced
        # a spurious -71.0 on SER-Intensity. Set identically for both conditions:
        # it cannot bind on direct, where responses are a single character, so the
        # prompt remains the only difference between them. Ignored by servers that
        # do not read it, including Kimi's, which is already unbounded.
        "_thinker_max_new_tokens": MODAL_MAX_NEW_TOKENS,
    }
    if condition == "direct":
        # Caps generation at 16 tokens server-side, which is what a bare letter
        # needs. Omitted for CoT: the same cap would truncate the description.
        meta["task"] = "MQA"
    payload = {
        "provider": "modal",
        "messages": [{"role": "user", "content": parts}],
        "audios": audios,
        "audio_b64": audios[0]["data"],
        "audio_format": audios[0]["format"],
        "audio_paths": [str(p) for p in example.audio_paths],
        "meta": meta,
    }
    data = method.remote(item=payload)
    return data.get("response") or ""


def _gemini_client():
    import os

    from google import genai

    api_key = (
        os.environ.get("GOOGLE_GENAI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        raise EnvironmentError(
            "Set GEMINI_API_KEY in your shell."
        )
    return genai.Client(api_key=api_key)


def _gemini_call(client, example: dataset.Example, parts: list[dict], condition: str) -> str:
    from google.genai import types

    audio_bytes = [path.read_bytes() for path in example.audio_paths]
    request_parts = []
    for part in parts:
        if part["type"] == "audio_file":
            request_parts.append(
                types.Part.from_bytes(
                    data=audio_bytes[part["audio_index"]], mime_type="audio/wav"
                )
            )
        else:
            request_parts.append(types.Part(text=part["text"]))

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[types.Content(role="user", parts=request_parts)],
        config=types.GenerateContentConfig(
            temperature=0.0,
            thinking_config=types.ThinkingConfig(
                thinking_level=GEMINI_THINKING_LEVEL
            ),
            # Without this, a hung request has no client-side deadline: four calls
            # stalled the final 4 of 200 items for nearly two hours while the
            # progress line kept displaying its last ETA.
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
        ),
    )
    return response.text or ""


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


def subset(task: config.Task) -> list[dataset.Example]:
    """A fixed random subset of the held-out split, identical across runs."""
    examples = dataset.load(task, split="eval")
    if len(examples) <= SUBSET_SIZE:
        return examples
    rng = random.Random(f"{SUBSET_SEED}-{task.key}")
    return sorted(
        rng.sample(examples, SUBSET_SIZE), key=lambda e: e.question_order
    )


def run(
    task: config.Task,
    model_key: str,
    condition: str,
    *,
    concurrency: int,
    limit: int | None,
) -> dict:
    examples = subset(task)
    if limit:
        examples = examples[:limit]

    build_parts, extract = CONDITIONS[condition]

    if model_key == "gemini":
        client = _gemini_client()
        call = lambda ex, parts: _gemini_call(client, ex, parts, condition)  # noqa: E731
        label = f"Gemini-3.1-Pro (thinking={GEMINI_THINKING_LEVEL})"
    else:
        model = config.MODELS[model_key]
        method = _modal_method(model)
        call = lambda ex, parts: _modal_call(method, ex, parts, condition)  # noqa: E731
        label = model.label

    out_dir = config.RESULTS_ROOT / RESULTS_SUBDIR / f"{task.key}--{model_key}--{condition}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {task.label}: {label} / {condition} ===", flush=True)
    print(f"    {len(examples)} items, concurrency {concurrency}", flush=True)

    def score_one(index: int, example: dataset.Example) -> dict:
        record = {
            "index": index,
            "example_id": example.example_id,
            "dimension": example.dimension,
            "answer_key": example.answer_key,
        }
        try:
            response = call(example, build_parts(example))
            predicted = extract(response)
            record.update(
                {
                    "response": response,
                    "response_chars": len(response),
                    "predicted": predicted,
                    "correct": bool(predicted == example.answer_key),
                    "parsed": predicted is not None,
                    "refused": is_refusal(response),
                    "status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001
            record.update(
                {
                    "response": None, "response_chars": 0, "predicted": None,
                    "correct": False, "parsed": False, "refused": False,
                    "status": f"error: {type(exc).__name__}: {exc}"[:300],
                }
            )
        return record

    records: list[dict | None] = [None] * len(examples)
    correct = 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(score_one, i, ex): i for i, ex in enumerate(examples)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records[futures[future]] = record
            correct += int(record["correct"])
            elapsed = time.perf_counter() - started
            eta = (len(examples) - done) / (done / elapsed) / 60 if elapsed else 0
            print(
                f"\r    {done}/{len(examples)}  acc {100 * correct / done:5.1f}%  "
                f"eta {eta:4.1f}m",
                end="", flush=True,
            )
    print()

    finished = [r for r in records if r is not None]
    total = len(finished)
    parsed = sum(1 for r in finished if r["parsed"])
    errors = sum(1 for r in finished if r["status"] != "ok")
    refused = sum(1 for r in finished if r.get("refused"))
    mean_chars = (
        sum(r["response_chars"] for r in finished) / total if total else 0
    )
    summary = {
        "task": task.key,
        "task_label": task.label,
        "model": model_key,
        "model_label": label,
        "condition": condition,
        "n": total,
        "correct": sum(1 for r in finished if r["correct"]),
        "accuracy": round(100.0 * sum(1 for r in finished if r["correct"]) / total, 1)
        if total else None,
        "parsed": parsed,
        "unparsed": total - parsed,
        "refused": refused,
        "refusal_rate": round(100.0 * refused / total, 1) if total else None,
        "errors": errors,
        "mean_response_chars": round(mean_chars, 1),
        "subset_seed": SUBSET_SEED,
    }
    with (out_dir / "predictions.jsonl").open("w") as handle:
        for record in finished:
            handle.write(json.dumps(record) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"    accuracy {summary['accuracy']}%  "
        f"({summary['correct']}/{total})  unparsed {summary['unparsed']}  "
        f"refused {refused}  errors {errors}  mean response {mean_chars:.0f} chars"
    )
    return summary


def rescore() -> int:
    """Recompute every CoT summary from stored responses.

    Extraction is a measurement choice, and the raw responses are on disk, so a
    change to it must never require calling the models again. This exists because
    the first extractor scored 95 of Gemini's 200 SER-Classify answers as
    unparsed when every one of them stated a verdict; it just wrote them as
    "B. Neutral" or "**Judgment:**\nA. Fear" rather than a bare letter.
    """
    root = config.RESULTS_ROOT / RESULTS_SUBDIR
    changed = 0
    for summary_path in sorted(root.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text())
        condition = summary["condition"]
        if condition not in FREE_TEXT_CONDITIONS:
            continue
        extractor = CONDITIONS[condition][1]
        predictions_path = summary_path.parent / "predictions.jsonl"
        rows = [json.loads(line) for line in predictions_path.open() if line.strip()]
        for row in rows:
            row["predicted"] = extractor(row.get("response") or "")
            row["parsed"] = row["predicted"] is not None
            row["correct"] = bool(row["predicted"] == row["answer_key"])
            row["refused"] = is_refusal(row.get("response") or "")
        total = len(rows)
        correct = sum(1 for r in rows if r["correct"])
        parsed = sum(1 for r in rows if r["parsed"])
        refused = sum(1 for r in rows if r["refused"])
        before = summary["accuracy"]
        summary.update(
            {
                "correct": correct,
                "accuracy": round(100.0 * correct / total, 1) if total else None,
                "parsed": parsed,
                "unparsed": total - parsed,
                "refused": refused,
                "refusal_rate": round(100.0 * refused / total, 1) if total else None,
            }
        )
        with predictions_path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        summary_path.write_text(json.dumps(summary, indent=2))
        print(
            f"  {summary_path.parent.name:<32} {before} -> {summary['accuracy']}  "
            f"unparsed {total - parsed}"
        )
        changed += 1
    print(f"\nrescored {changed} CoT cells from stored responses")
    return 0


def report(*, markdown: bool = False) -> int:
    root = config.RESULTS_ROOT / RESULTS_SUBDIR
    if not root.is_dir():
        print("No CoT results yet.")
        return 1
    found: dict[tuple[str, str, str], dict] = {}
    for path in sorted(root.glob("*/summary.json")):
        s = json.loads(path.read_text())
        found[(s["task"], s["model"], s["condition"])] = s

    header = ["Task", "n", "Kimi direct", "Kimi CoT", "Δ", "Gemini direct", "Gemini CoT", "Δ"]
    rows = [header]
    for task_key in COT_TASKS:
        task = config.TASKS[task_key]
        row = [task.label, "--"]
        for model in COT_MODELS:
            direct = found.get((task_key, model, "direct"))
            cot = found.get((task_key, model, "cot"))
            if direct and row[1] == "--":
                row[1] = str(direct["n"])
            row.append(f"{direct['accuracy']:.1f}" if direct else "--")
            row.append(f"{cot['accuracy']:.1f}" if cot else "--")
            row.append(
                f"{cot['accuracy'] - direct['accuracy']:+.1f}"
                if direct and cot else "--"
            )
        rows.append(row)

    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    print()
    print("Chain-of-thought vs direct prompting")
    print()
    if markdown:
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join("---" for _ in header) + "|")
        for row in rows[1:]:
            print("| " + " | ".join(row) + " |")
    else:
        for index, row in enumerate(rows):
            print("  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
            if index == 0:
                print("  " + "  ".join("-" * w for w in widths))

    print()
    print("Response length (mean chars) and parse failures:")
    for (task_key, model, condition), s in sorted(found.items()):
        print(
            f"  {config.TASKS[task_key].label:<15}{model:<8}{condition:<8}"
            f"{s['mean_response_chars']:>8.0f} chars  unparsed={s['unparsed']}"
            f"  refused={s.get('refused', 0)}  errors={s['errors']}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=COT_TASKS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model", choices=("gemini", *sorted(config.MODELS)))
    parser.add_argument(
        "--condition",
        choices=tuple(CONDITIONS),
    )
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report", action="store_true")
    parser.add_argument(
        "--rescore", action="store_true",
        help="Recompute CoT summaries from stored responses, no model calls.",
    )
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    if args.rescore:
        return rescore()
    if args.report:
        return report(markdown=args.markdown)
    if not args.model or not args.condition:
        parser.error("--model and --condition are required unless --report")
    if not args.task and not args.all:
        parser.error("pass --task or --all")

    tasks = (
        [config.TASKS[t] for t in COT_TASKS] if args.all else [config.TASKS[args.task]]
    )
    for task in tasks:
        run(
            task, args.model, args.condition,
            concurrency=args.concurrency, limit=args.limit,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
