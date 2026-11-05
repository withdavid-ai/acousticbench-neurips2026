"""Inference runner with concurrency, checkpointing, and error handling."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from .judge import DEFAULT_EXTRACTOR, extract_choice


def valid_labels_for_question(question: str) -> list[str]:
    """Label set offered to the judge (A/B unless the text lists a C option)."""
    if any(c in question for c in ["C.", "C)", "(C)"]):
        return ["A", "B", "C", "D"]
    return ["A", "B"]


def _run_single(model, item: dict, extractor: str = DEFAULT_EXTRACTOR) -> dict:
    """Run inference on a single item and extract the choice."""
    example_id = item["meta"]["example_id"]
    t0 = time.perf_counter()
    try:
        msg = {
            "audio": item["audio_path"],
            "text": item["question"],
            "messages": item.get("messages"),
            "meta": item["meta"],
        }
        trajectory = model(msg)
        wall_s = time.perf_counter() - t0

        response_text = trajectory.get("response_text", "") or ""
        response_status = trajectory.get("response_status", "ok")

        valid_labels = valid_labels_for_question(item["question"])

        try:
            extraction = extract_choice(
                response_text, item["question"], valid_labels, extractor=extractor,
            )
        except Exception as judge_exc:
            # Keep the model's answer so the row can be re-judged later
            # (scripts/rejudge.py); it is scored as unparsed until then.
            logger.warning(f"Judge failed on {example_id}: {judge_exc}")
            return {
                "example_id": example_id,
                "response_text": response_text,
                "response_status": response_status,
                "judge_error": str(judge_exc),
                "chosen": None,
                "correct": item["answer"],
                "is_correct": False,
                "extraction_source": "judge_error",
                "wall_time_s": round(wall_s, 2),
                "meta": item["meta"],
            }
        chosen = extraction["label"]
        is_correct = chosen == item["answer"] if chosen else False

        return {
            "example_id": example_id,
            "response_text": response_text,
            "response_status": response_status,
            "chosen": chosen,
            "correct": item["answer"],
            "is_correct": is_correct,
            "extraction_source": extraction["source"],
            "extraction_classification": extraction.get("classification"),
            "wall_time_s": round(wall_s, 2),
            "meta": item["meta"],
        }
    except Exception as e:
        wall_s = time.perf_counter() - t0
        logger.warning(f"Error on {example_id}: {e}")
        return {
            "example_id": example_id,
            "response_text": "",
            "response_status": "error",
            "response_error": str(e),
            "chosen": None,
            "correct": item["answer"],
            "is_correct": False,
            "extraction_source": "error",
            "wall_time_s": round(wall_s, 2),
            "meta": item["meta"],
        }


def run_inference(
    model,
    questions: list[dict],
    *,
    concurrency: int = 1,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 20,
    continue_on_error: bool = True,
    extractor: str = DEFAULT_EXTRACTOR,
) -> list[dict]:
    """Run inference on a list of questions.

    Args:
        model: A callable model adapter (BaseModel subclass).
        questions: List of question dicts from loader.
        concurrency: Number of parallel inference threads.
        checkpoint_path: If set, save partial results here.
        checkpoint_every: Save checkpoint every N completed items.
        continue_on_error: If False, raise on first error.
        extractor: Choice extractor passed to ``judge.extract_choice``.

    Returns:
        List of result dicts, one per question.
    """
    results = []
    completed = set()

    if checkpoint_path and checkpoint_path.exists():
        with open(checkpoint_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    results.append(r)
                    completed.add(r["example_id"])
        logger.info(f"Resumed from checkpoint: {len(completed)} already done")

    remaining = [q for q in questions if q["meta"]["example_id"] not in completed]
    logger.info(f"Running inference on {len(remaining)} questions (concurrency={concurrency})")

    if concurrency <= 1:
        for i, item in enumerate(tqdm(remaining, desc="Inference", unit="q")):
            result = _run_single(model, item, extractor)
            results.append(result)

            if not continue_on_error and result["response_status"] == "error":
                raise RuntimeError(f"Inference error: {result.get('response_error')}")

            if checkpoint_path and (i + 1) % checkpoint_every == 0:
                _write_checkpoint(results, checkpoint_path)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_run_single, model, item, extractor): item
                for item in remaining
            }
            pbar = tqdm(total=len(remaining), desc="Inference", unit="q")
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

                if not continue_on_error and result["response_status"] == "error":
                    raise RuntimeError(f"Inference error: {result.get('response_error')}")

                if checkpoint_path and pbar.n % checkpoint_every == 0:
                    _write_checkpoint(results, checkpoint_path)
            pbar.close()

    if checkpoint_path:
        _write_checkpoint(results, checkpoint_path)

    return results


def _write_checkpoint(results: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def save_results(results: list[dict], path: Path):
    """Save results to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    logger.info(f"Saved {len(results)} results to {path}")
