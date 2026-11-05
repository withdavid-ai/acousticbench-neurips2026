#!/usr/bin/env python3
"""Validate the local AcousticBench repository setup.

This script performs three checks, in order, and prints a concise report:

1. ``.env`` exists next to this file and defines the keys the benchmark
   expects (``GEMINI_API_KEY``, ``OPENAI_API_KEY``, ``HF_TOKEN``). Blank
   values produce a warning, not an error.
2. The benchmark dataset has been downloaded into the archive directory
   (``./archive/`` by default, the same default as ``run.py --archive``) so
   that ``<archive>/questions/`` and ``<archive>/audio/`` exist; every
   per-task ``.jsonl`` has the expected row count; and every audio path
   referenced by ``acousticbench_all.jsonl`` is on disk.
3. With ``--check-adapters``, the Kimi-Audio LoRA adapters (``rad``, ``rsq``, ``ser_classify``,
   ``ser_intensity``, ``simple_audioqa``) are present so that each adapter
   directory contains an ``adapter_model.safetensors`` file. Place user-supplied adapters under ``<archive>/models/<adapter>/``; ``<archive>/<adapter>/``
   and ``<archive>/finetunes/<adapter>/`` are also accepted, matching the
   Docker and Modal LoRA servers.

Run it from the repo root::

    python check_repo_setup.py
    python check_repo_setup.py --archive /path/to/archive   # same as run.py --archive
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent

EXPECTED_ENV_KEYS: tuple[str, ...] = (
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "HF_TOKEN",
)

QUESTION_FILES: dict[str, str] = {
    "rad": "acousticbench_rad.jsonl",
    "rsq": "acousticbench_rsq.jsonl",
    "ser_classify": "acousticbench_ser_classify.jsonl",
    "ser_classify_text": "acousticbench_ser_classify_text.jsonl",
    "ser_intensity": "acousticbench_ser_intensity.jsonl",
    "all": "acousticbench_all.jsonl",
}
EXPECTED_ROW_COUNTS: dict[str, int] = {
    "rad": 1200,
    "rsq": 1000,
    "ser_classify": 600,
    "ser_classify_text": 600,
    "ser_intensity": 400,
    "all": 3800,
}
REQUIRED_AUDIO_DIRS: tuple[str, ...] = ("rad", "rsq", "ser_classify", "ser_intensity")

# Default archive location. Matches the default of ``run.py --archive``.
DEFAULT_ARCHIVE_DIR = "archive"

DATASET_REPO_ID = "davidai-labs/AcousticBench"

# Kimi finetunes (must match harness/modal_serving/kimi_lora_serve.py).
KIMI_ADAPTER_NAMES: tuple[str, ...] = ("rad", "rsq", "ser_classify", "ser_intensity", "simple_audioqa")
KIMI_ADAPTER_SUBDIRS: tuple[str, ...] = ("", "models", "finetunes")
KIMI_ADAPTER_FILE = "adapter_model.safetensors"

MAX_DISPLAYED_ERRORS = 50

EXTRACT_HINT = (
    "\nThe dataset is a gated Hugging Face dataset with manual approval:\n"
    f"  https://huggingface.co/datasets/{DATASET_REPO_ID}\n"
    "Request access on that page, wait for approval, export HF_TOKEN for the\n"
    "approved account, then download into ./archive/:\n"
    "\n"
    "  python -c \"from huggingface_hub import snapshot_download;"
    f" snapshot_download(repo_id='{DATASET_REPO_ID}', repo_type='dataset',"
    " local_dir='archive')\"\n"
    "\n"
    "After that, ./archive/questions/ and ./archive/audio/ should exist."
)

FINETUNES_HINT = (
    "Provide independently obtained Kimi adapters under archive/models/<adapter>/ "
    "with adapter_model.safetensors. No verified public adapter URL is supplied."
)


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #


def _ok(message: str) -> None:
    print(f"OK: {message}")


def _warn(message: str) -> None:
    print(f"WARNING: {message}")


def _eprint(message: str = "") -> None:
    print(message, file=sys.stderr)


def _print_error_block(header: str, items: Iterable[str]) -> None:
    items = list(items)
    _eprint(f"ERROR: {header}")
    for item in items[:MAX_DISPLAYED_ERRORS]:
        _eprint(f"  - {item}")
    extra = len(items) - MAX_DISPLAYED_ERRORS
    if extra > 0:
        _eprint(f"  ... and {extra} more errors")


# --------------------------------------------------------------------------- #
# .env
# --------------------------------------------------------------------------- #


def _parse_env(path: Path) -> dict[str, str]:
    """Best-effort key/value parser for our simple .env files."""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _check_env_file(repo_root: Path) -> int:
    env_file = repo_root / ".env"
    if not env_file.is_file():
        _print_error_block(
            ".env file is missing:",
            [
                f"expected {env_file}",
                "create it with `cp example.env .env` and fill in keys for "
                "the models you plan to run.",
            ],
        )
        return 1

    values = _parse_env(env_file)
    missing = [k for k in EXPECTED_ENV_KEYS if k not in values]
    blank = [k for k in EXPECTED_ENV_KEYS if k in values and not values[k]]

    if missing:
        _print_error_block(
            ".env is missing required keys:",
            [f"{key} (add `{key}=` and a value)" for key in missing],
        )
        return 1

    _ok(".env is present with expected keys.")
    if blank:
        _warn(
            "These .env keys are present but blank; fill them before using "
            f"the corresponding models: {', '.join(blank)}"
        )
    return 0


# --------------------------------------------------------------------------- #
# Extracted benchmark archive
# --------------------------------------------------------------------------- #


def _looks_like_archive_root(path: Path) -> bool:
    return (path / "questions").is_dir() and (path / "audio").is_dir()


def _resolve_archive_root(archive: Path) -> Path:
    """Return ``archive`` if it contains both ``questions/`` and ``audio/``.

    Only the directory that ``run.py --archive`` would use is accepted, so this
    check cannot pass while ``run.py`` fails to find the data.
    """
    candidate = archive.expanduser().resolve()
    if _looks_like_archive_root(candidate):
        return candidate

    raise FileNotFoundError(
        "Could not find the AcousticBench dataset (needs both questions/ and "
        f"audio/) at {candidate}. Download it from Hugging Face into that "
        "directory, or pass the directory you used with --archive."
    )


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}: {exc}"
                ) from exc


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _audio_paths(record: dict[str, Any]) -> Iterator[str]:
    for message in record.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "audio_file":
                continue
            raw_path = str(part.get("path") or "").strip()
            if raw_path:
                yield raw_path


def _check_unpacked(archive: Path) -> int:
    try:
        archive_root = _resolve_archive_root(archive)
    except FileNotFoundError as exc:
        _eprint(f"ERROR: {exc}")
        _eprint(EXTRACT_HINT)
        return 1

    print(f"Checking AcousticBench data at: {archive_root}")
    questions_dir = archive_root / "questions"
    audio_dir = archive_root / "audio"
    errors: list[str] = []

    for task, filename in QUESTION_FILES.items():
        path = questions_dir / filename
        if not path.is_file():
            errors.append(f"Missing question file: {path}")
            continue
        actual = _count_jsonl_rows(path)
        expected = EXPECTED_ROW_COUNTS[task]
        if actual != expected:
            errors.append(f"{path} has {actual} rows, expected {expected}")

    for name in REQUIRED_AUDIO_DIRS:
        path = audio_dir / name
        if not path.is_dir():
            errors.append(f"Missing audio subdirectory: {path}")

    checked_audio = 0
    all_path = questions_dir / QUESTION_FILES["all"]
    if all_path.is_file():
        for record in _iter_jsonl(all_path):
            example_id = record.get("example_id", "<unknown>")
            for relative in _audio_paths(record):
                if (archive_root / relative).is_file():
                    checked_audio += 1
                else:
                    errors.append(
                        f"Missing audio for {example_id}: {relative}"
                    )

    if errors:
        _print_error_block("AcousticBench data layout is invalid:", errors)
        return 1

    _ok(
        "required question files, row counts, audio folders, and "
        "referenced audio files are present."
    )
    print(
        f"Checked {checked_audio:,} audio references from "
        f"{QUESTION_FILES['all']}."
    )
    return 0


# --------------------------------------------------------------------------- #
# Kimi-Audio LoRA adapters (under <archive>/models/ by default)
# --------------------------------------------------------------------------- #


def _adapter_path(archive_root: Path, subdir: str, adapter: str) -> Path:
    return archive_root / subdir / adapter / KIMI_ADAPTER_FILE


def _find_kimi_adapter_root(archive_root: Path) -> Path | None:
    """Return the directory under ``archive_root`` containing all Kimi adapters."""
    for subdir in KIMI_ADAPTER_SUBDIRS:
        root = archive_root / subdir
        if all(
            (root / adapter / KIMI_ADAPTER_FILE).is_file()
            for adapter in KIMI_ADAPTER_NAMES
        ):
            return root
    return None


def _check_finetunes(archive: Path) -> int:
    try:
        archive_root = _resolve_archive_root(archive)
    except FileNotFoundError:
        _eprint(
            "ERROR: cannot check the Kimi-Audio LoRA adapters because the "
            "dataset directory was not found (see previous error)."
        )
        return 1

    adapter_root = _find_kimi_adapter_root(archive_root)
    if adapter_root is None:
        accepted = ", ".join(
            str(archive_root / subdir) for subdir in KIMI_ADAPTER_SUBDIRS
        )
        missing = [
            str(_adapter_path(archive_root, subdir, adapter))
            for subdir in KIMI_ADAPTER_SUBDIRS
            for adapter in KIMI_ADAPTER_NAMES
            if not _adapter_path(archive_root, subdir, adapter).is_file()
        ]
        _print_error_block(
            f"Kimi-Audio LoRA adapters were not found under {archive_root}. "
            f"Expected all adapters at one of: {accepted}.",
            missing,
        )
        _eprint(FINETUNES_HINT)
        return 1

    _ok(
        "Kimi-Audio LoRA adapters found at "
        f"{adapter_root} ({', '.join(KIMI_ADAPTER_NAMES)})."
    )
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check .env, the downloaded dataset, and the Kimi-Audio LoRA adapters.",
    )
    parser.add_argument(
        "--archive",
        type=str,
        default=None,
        help=(
            "Dataset directory, as passed to run.py --archive "
            f"(default: ./{DEFAULT_ARCHIVE_DIR} under the repo root)"
        ),
    )
    parser.add_argument("--check-adapters", action="store_true", help="Also check user-supplied Kimi adapters")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    archive = Path(args.archive) if args.archive else REPO_ROOT / DEFAULT_ARCHIVE_DIR
    status = _check_env_file(REPO_ROOT)
    status |= _check_unpacked(archive)
    if args.check_adapters:
        status |= _check_finetunes(archive)
    if status == 0:
        _ok("repository setup checks passed.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
