"""Load AcousticBench questions from the canonical JSONL archive.

Reads ``questions/acousticbench_*.jsonl``, resolves audio file paths relative
to an archive root, and optionally verifies SHA256 checksums
(see :func:`verify_archive`).

The output format matches what model adapters expect::

    {
        "index": int,
        "audio_path": str | list[str],
        "question": str,           # user text + formatted options
        "messages": list[dict],    # interleaved text + audio parts (canonical)
        "answer": str,             # correct answer letter
        "subset": str,             # task slice: rad, rsq, ser_classify, ser_intensity
        "meta": {
            "example_id": str,
            "question_set_id": str,
            "answer_key": str,
            "dimension": str,
            "category": str,
            ...
        },
    }
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

TASK_SLICES = {
    "rad": "acousticbench_rad.jsonl",
    "rsq": "acousticbench_rsq.jsonl",
    "ser_classify": "acousticbench_ser_classify.jsonl",
    "ser_classify_text": "acousticbench_ser_classify_text.jsonl",
    "ser_intensity": "acousticbench_ser_intensity.jsonl",
    "all": "acousticbench_all.jsonl",
}

SER_CLASSIFY_TEXT_LIKE_TASKS = {"ser_classify_text"}

# Released row counts per task file (split "all"). ``all`` is the union of the
# five task slices.
EXPECTED_ROW_COUNTS = {
    "rad": 1200,
    "rsq": 1000,
    "ser_classify": 600,
    "ser_classify_text": 600,
    "ser_intensity": 400,
    "all": 3800,
}
SUBTASKS: tuple[str, ...] = ("rad", "rsq", "ser_classify", "ser_classify_text", "ser_intensity")
CHECKSUM_MANIFEST = Path("checksums") / "sha256sum.txt"


def _resolve_audio_path(relative_path: str, archive_root: Path) -> str:
    resolved = archive_root / relative_path
    if not resolved.exists():
        raise FileNotFoundError(f"Audio file not found: {resolved}")
    return str(resolved)


def _sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_checksum(path: str, expected_sha256: str) -> bool:
    return _sha256_file(path) == str(expected_sha256).strip().lower()


def _build_question_text(messages: list[dict]) -> str:
    """Extract and assemble the question text from the messages array.

    The messages array has a user role with interleaved text and audio_file
    parts. We extract only the text parts and join them with newlines.
    """
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if isinstance(content, str):
            return content
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part["text"])
        return "\n".join(text_parts)
    return ""


def _build_canonical_messages(
    raw_messages: list[dict],
    audio_paths: list[str],
) -> list[dict]:
    """Convert raw JSONL messages into canonical interleaved format.

    Replaces audio_file parts with audio parts using audio_index,
    preserving the original interleaving of text and audio.
    """
    audio_order_to_index = {}
    for msg in raw_messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        audio_parts = sorted(
            [p for p in content if isinstance(p, dict) and p.get("type") == "audio_file"],
            key=lambda p: p.get("order", 0),
        )
        for i, part in enumerate(audio_parts):
            audio_order_to_index[part.get("order", i)] = i

    canonical = []
    for msg in raw_messages:
        role = msg.get("role", "user")
        content = msg.get("content", [])
        if isinstance(content, str):
            canonical.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue

        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "audio_file":
                order = part.get("order", 0)
                idx = audio_order_to_index.get(order, 0)
                parts.append({"type": "audio", "audio_index": idx})
            elif part.get("type") == "text":
                parts.append({"type": "text", "text": part["text"]})
        if parts:
            canonical.append({"role": role, "content": parts})

    return canonical


def _extract_audio_paths(
    messages: list[dict],
    archive_root: Path,
    verify_checksums: bool = False,
    allow_missing_hashes: bool = False,
) -> list[str]:
    """Extract audio file paths from the messages array and resolve them.

    With ``verify_checksums`` every audio part must carry a ``sha256`` that
    matches the file; a part without one is an error unless
    ``allow_missing_hashes`` is set.
    """
    paths = []
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        audio_parts = sorted(
            [p for p in content if isinstance(p, dict) and p.get("type") == "audio_file"],
            key=lambda p: p.get("order", 0),
        )
        for part in audio_parts:
            rel_path = part["path"]
            resolved = _resolve_audio_path(rel_path, archive_root)
            if verify_checksums and not part.get("sha256"):
                if not allow_missing_hashes:
                    raise ValueError(
                        f"Audio part {rel_path} has no sha256, so it cannot be verified "
                        f"(pass --allow-missing-hashes to accept unhashed parts)."
                    )
            elif verify_checksums:
                if not _verify_checksum(resolved, part["sha256"]):
                    raise ValueError(
                        f"SHA256 mismatch for {rel_path}: "
                        f"expected {part['sha256']}"
                    )
            paths.append(resolved)
    return paths


VALID_SPLITS: tuple[str, ...] = ("train", "eval", "all")


def _trailing_int(value: Any) -> int | None:
    match = re.search(r"(\d+)$", str(value or ""))
    return int(match.group(1)) if match else None


def _question_order(raw: dict[str, Any]) -> Any:
    question_set = raw.get("question_set")
    if not isinstance(question_set, dict):
        return None
    return question_set.get("question_order")


def _add_split_lookup(
    split_lookup: dict[tuple[str, int], str],
    key: tuple[str, int],
    row_split: str,
) -> None:
    existing = split_lookup.get(key)
    if existing is not None and existing != row_split:
        raise ValueError(
            f"Conflicting split assignments for {key}: {existing!r} vs {row_split!r}"
        )
    split_lookup[key] = row_split


def _load_ser_classify_split_lookup(archive_root: Path) -> dict[tuple[str, int], str]:
    """Build split lookup keys shared by audio and text SER classify rows."""
    source_path = archive_root / "questions" / TASK_SLICES["ser_classify"]
    if not source_path.exists():
        raise FileNotFoundError(
            "SER classify text rows without a split require the paired audio "
            f"question file: {source_path}"
        )

    split_lookup: dict[tuple[str, int], str] = {}
    with source_path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if not line.strip():
                continue
            raw = json.loads(line)
            row_split = raw.get("split")
            if row_split not in VALID_SPLITS:
                continue

            order = _question_order(raw)
            if order is not None:
                _add_split_lookup(
                    split_lookup, ("question_order", int(order)), row_split
                )

            _add_split_lookup(split_lookup, ("line_index", idx), row_split)

            suffix = _trailing_int(raw.get("example_id"))
            if suffix is not None:
                _add_split_lookup(split_lookup, ("example_suffix", suffix), row_split)

    return split_lookup


def _resolve_ser_classify_text_split(
    raw: dict[str, Any],
    idx: int,
    split_lookup: dict[tuple[str, int], str],
) -> str:
    row_split = raw.get("split")
    if row_split in VALID_SPLITS:
        return row_split

    order = _question_order(raw)
    lookup_keys: list[tuple[str, int]] = []
    if order is not None:
        lookup_keys.append(("question_order", int(order)))
    lookup_keys.append(("line_index", idx))

    suffix = _trailing_int(raw.get("example_id"))
    if suffix is not None:
        lookup_keys.append(("example_suffix", suffix))

    for key in lookup_keys:
        matched_split = split_lookup.get(key)
        if matched_split is not None:
            return matched_split

    raise ValueError(
        "Could not infer split for ser_classify_text row "
        f"{raw.get('example_id', idx)!r}; no matching ser_classify question was found."
    )


def load_questions(
    task: str,
    archive_root: str | Path,
    verify_checksums: bool = False,
    split: str = "all",
    allow_missing_hashes: bool = False,
) -> list[dict[str, Any]]:
    """Load questions for a task from the canonical JSONL archive.

    Args:
        task: One of "rad", "rsq", "ser_classify",
            "ser_classify_text", "ser_intensity", or "all".
        archive_root: Path to the archive directory containing questions/ and audio/.
        verify_checksums: If True, verify SHA256 checksums of audio files.
        split: Which row-level split to keep. One of "train", "eval", or
            "all" (default). The split value is read from the per-row
            ``split`` field in the question JSONL. ``ser_classify_text`` rows
            without a split inherit it from the paired ``ser_classify`` row,
            both in their own file and inside ``acousticbench_all.jsonl``.
        allow_missing_hashes: With ``verify_checksums``, accept audio parts
            that carry no ``sha256`` instead of failing.

    Returns:
        List of question dicts in the format model adapters expect.
    """
    archive_root = Path(archive_root)
    if task not in TASK_SLICES:
        raise ValueError(f"Unknown task: {task!r}. Choose from {list(TASK_SLICES)}")
    if split not in VALID_SPLITS:
        raise ValueError(f"Unknown split: {split!r}. Choose from {list(VALID_SPLITS)}")

    jsonl_path = archive_root / "questions" / TASK_SLICES[task]
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Question file not found: {jsonl_path}")

    ser_classify_text_split_lookup: dict[tuple[str, int], str] | None = None
    # In acousticbench_all.jsonl the line index of a text row is not its line
    # index in the text file, so count text rows separately for the fallback.
    text_row_index = 0

    questions = []
    with open(jsonl_path) as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            raw = json.loads(line)

            row_task = task
            if task == "all":
                row_task = (raw.get("question_set") or {}).get("slice")
            if row_task in SER_CLASSIFY_TEXT_LIKE_TASKS:
                text_idx = idx if task != "all" else text_row_index
                text_row_index += 1
                if (
                    raw.get("split") not in VALID_SPLITS
                    and ser_classify_text_split_lookup is None
                ):
                    ser_classify_text_split_lookup = _load_ser_classify_split_lookup(
                        archive_root
                    )
                row_split = _resolve_ser_classify_text_split(
                    raw, text_idx, ser_classify_text_split_lookup or {}
                )
            else:
                row_split = raw.get("split")
            if split != "all" and row_split != split:
                continue

            audio_paths = _extract_audio_paths(
                raw["messages"],
                archive_root,
                verify_checksums=verify_checksums,
                allow_missing_hashes=allow_missing_hashes,
            )
            question_text = _build_question_text(raw["messages"])
            canonical_messages = _build_canonical_messages(raw["messages"], audio_paths)

            question_set = raw["question_set"]
            question_set_name = (
                question_set.get("name")
                or question_set.get("slice")
                or question_set.get("id")
            )

            meta = {
                "example_id": raw["example_id"],
                "question_set_id": question_set["id"],
                "question_set_name": question_set_name,
                "question_order": question_set["question_order"],
                "category": raw["category"],
                "dimension": raw["dimension"],
                "answer_key": raw["answer_key"],
                "split": row_split,
                "task": "MQA",
                "audio_type": "Speech",
                "dataset_name": _dataset_name_for_category(raw["category"]),
                "dataset_series": "AcousticBench",
                "interactive": "Audio-analysis",
            }

            questions.append({
                "index": idx,
                "audio_path": audio_paths[0] if len(audio_paths) == 1 else audio_paths,
                "question": question_text,
                "messages": canonical_messages,
                "answer": raw["answer_key"],
                "subset": question_set["slice"],
                "split": row_split,
                "meta": meta,
            })

    return questions


def load_questions_by_id(
    archive_root: str | Path,
    tasks: tuple[str, ...] = SUBTASKS,
) -> dict[str, dict[str, Any]]:
    """Load every question of ``tasks`` keyed by example_id (missing files are skipped)."""
    by_id: dict[str, dict[str, Any]] = {}
    for task in tasks:
        try:
            for q in load_questions(task, archive_root):
                by_id[q["meta"]["example_id"]] = q
        except FileNotFoundError:
            pass
    return by_id


def _read_checksum_manifest(path: Path) -> dict[str, str]:
    """Parse a ``sha256sum``-style manifest into {relative_path: sha256}."""
    entries: dict[str, str] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            raise ValueError(f"{path}:{line_no}: malformed checksum line: {line!r}")
        digest, rel_path = parts
        entries[rel_path.lstrip("*").strip()] = digest.lower()
    return entries


def verify_archive(
    task: str,
    archive_root: str | Path,
    allow_missing_hashes: bool = False,
) -> dict[str, Any]:
    """Verify a task slice of the archive; raise ``ValueError`` on any failure.

    Checks, in order:
      1. Every audio part of every row carries a ``sha256`` that matches the
         file (unhashed parts fail unless ``allow_missing_hashes``).
      2. The row count of the task file equals ``EXPECTED_ROW_COUNTS``; for
         ``all`` the per-slice counts are checked too.
      3. If ``checksums/sha256sum.txt`` exists, the task's question file(s)
         and every referenced audio file listed there match it (for ``all``,
         every listed file is checked).
    """
    archive_root = Path(archive_root)
    questions = load_questions(
        task,
        archive_root,
        verify_checksums=True,
        allow_missing_hashes=allow_missing_hashes,
    )

    errors: list[str] = []
    expected = EXPECTED_ROW_COUNTS.get(task)
    if expected is not None and len(questions) != expected:
        errors.append(f"{TASK_SLICES[task]} has {len(questions)} rows; expected {expected}")
    slice_counts: dict[str, int] = {}
    for q in questions:
        slice_counts[q["subset"]] = slice_counts.get(q["subset"], 0) + 1
    if task == "all":
        for name in sorted(set(SUBTASKS) | set(slice_counts)):
            want = EXPECTED_ROW_COUNTS.get(name)
            got = slice_counts.get(name, 0)
            if want is None:
                errors.append(f"{TASK_SLICES[task]} has {got} rows with unexpected slice {name!r}")
            elif got != want:
                errors.append(f"{TASK_SLICES[task]} has {got} {name} rows; expected {want}")

    manifest_path = archive_root / CHECKSUM_MANIFEST
    manifest_checked = 0
    if manifest_path.is_file():
        manifest = _read_checksum_manifest(manifest_path)
        if task == "all":
            to_check = sorted(manifest)
        else:
            wanted = {(Path("questions") / TASK_SLICES[task]).as_posix()}
            if task in SER_CLASSIFY_TEXT_LIKE_TASKS:
                wanted.add((Path("questions") / TASK_SLICES["ser_classify"]).as_posix())
            root = archive_root.resolve()
            for q in questions:
                paths = q["audio_path"] if isinstance(q["audio_path"], list) else [q["audio_path"]]
                for audio_path in paths:
                    try:
                        wanted.add(Path(audio_path).resolve().relative_to(root).as_posix())
                    except ValueError:
                        continue
            to_check = sorted(path for path in wanted if path in manifest)
            question_file = (Path("questions") / TASK_SLICES[task]).as_posix()
            if question_file not in manifest:
                errors.append(f"{CHECKSUM_MANIFEST} does not list {question_file}")
        for rel_path in to_check:
            target = archive_root / rel_path
            if not target.is_file():
                errors.append(f"{rel_path} is listed in {CHECKSUM_MANIFEST} but missing")
            elif _sha256_file(target) != manifest[rel_path]:
                errors.append(f"SHA256 mismatch for {rel_path} against {CHECKSUM_MANIFEST}")
            manifest_checked += 1

    if errors:
        shown = "\n  ".join(errors[:20])
        more = f"\n  ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"Archive verification failed for task={task}:\n  {shown}{more}")

    return {
        "task": task,
        "n_questions": len(questions),
        "slice_counts": slice_counts,
        "manifest": str(manifest_path) if manifest_path.is_file() else None,
        "manifest_files_checked": manifest_checked,
    }


def _dataset_name_for_category(category: str) -> str:
    return {
        "acoustic-perception": "AcousticPerceptionBench",
        "emotion-recognition": "EmotionRecognitionBench",
        "speech-quality": "SpeechQualityBench",
    }.get(category, "AcousticBench")
