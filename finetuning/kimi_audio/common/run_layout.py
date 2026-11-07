"""Shared bundle and run layout helpers for Kimi finetuning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

DATA_MOUNT = "/data"
BUNDLES_ROOT = Path(DATA_MOUNT) / "bundles"
RUNS_ROOT = Path(DATA_MOUNT) / "runs"
DEFAULT_PRETRAINED_PATH = str(Path(DATA_MOUNT) / "output" / "pretrained_hf")
DEFAULT_SERVING_MODEL_PATH = str(Path(DATA_MOUNT) / "serving" / "kimi-audio-ft" / "latest")
DEFAULT_HELDOUT_EVAL_MANIFEST_NAME = "heldout_eval_manifest.jsonl"
LEGACY_HELDOUT_EVAL_MANIFEST_NAMES = ("emotion_heldout_eval_manifest.jsonl",)

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_layout_name(name: str, *, kind: str) -> str:
    value = str(name or "").strip()
    if not value:
        raise ValueError(f"{kind} cannot be empty")
    if not _SAFE_NAME_RE.fullmatch(value):
        raise ValueError(
            f"Invalid {kind} {value!r}. Use only letters, numbers, '.', '-', and '_'."
        )
    return value


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def csv_tags(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.split(",")
    else:
        raw_items = list(value)
    tags: list[str] = []
    seen: set[str] = set()
    for raw in raw_items:
        tag = str(raw or "").strip()
        if not tag or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)
    return tags


def volume_relative(path: str | Path) -> str:
    raw = Path(path)
    try:
        rel = raw.relative_to(DATA_MOUNT)
    except ValueError as exc:
        raise ValueError(f"Expected a path under {DATA_MOUNT}, got {raw}") from exc
    return "/" + rel.as_posix()


@dataclass(frozen=True)
class BundlePaths:
    bundle_name: str
    bundle_root: Path
    export_dir: Path
    audio_dir: Path
    manifest_path: Path
    train_json: Path
    val_json: Path
    train_codes: Path
    val_codes: Path
    combined_codes: Path
    combined_meta: Path
    heldout_eval_manifest: Path


@dataclass(frozen=True)
class RunPaths:
    run_name: str
    run_root: Path
    run_manifest: Path
    run_state: Path
    train_dir: Path
    train_log: Path
    train_history: Path
    train_summary: Path
    eval_root: Path
    checkpoint_metrics: Path
    promotion_dir: Path
    promotion_manifest: Path
    promotion_archives: Path

    def eval_repeat_dir(self, eval_name: str, repeat_index: int) -> Path:
        safe_eval_name = validate_layout_name(eval_name, kind="eval_name")
        return self.eval_root / safe_eval_name / f"repeat-{repeat_index:03d}"

    def promotion_archive_dir(self, checkpoint_slug: str) -> Path:
        safe_slug = validate_layout_name(checkpoint_slug, kind="checkpoint_slug")
        return self.promotion_archives / safe_slug


def heldout_eval_manifest_candidates(export_dir: str | Path) -> list[Path]:
    root = Path(export_dir)
    candidates = [root / DEFAULT_HELDOUT_EVAL_MANIFEST_NAME]
    seen = {DEFAULT_HELDOUT_EVAL_MANIFEST_NAME}
    for name in LEGACY_HELDOUT_EVAL_MANIFEST_NAMES:
        if name in seen:
            continue
        candidates.append(root / name)
        seen.add(name)
    return candidates


def resolve_existing_heldout_eval_manifest(export_dir: str | Path) -> Path:
    candidates = heldout_eval_manifest_candidates(export_dir)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def resolve_bundle_paths(bundle_name: str) -> BundlePaths:
    safe_name = validate_layout_name(bundle_name, kind="bundle_name")
    bundle_root = BUNDLES_ROOT / safe_name
    export_dir = bundle_root / "rad_exports" / "current"
    train_codes = export_dir / "rad_kimi_train_with_semantic_codes.jsonl"
    val_codes = export_dir / "rad_kimi_val_with_semantic_codes.jsonl"
    combined_codes = export_dir / "rad_kimi_trainval_with_semantic_codes.jsonl"
    return BundlePaths(
        bundle_name=safe_name,
        bundle_root=bundle_root,
        export_dir=export_dir,
        audio_dir=bundle_root / "audio",
        manifest_path=export_dir / "manifest.json",
        train_json=export_dir / "rad_kimi_train.jsonl",
        val_json=export_dir / "rad_kimi_val.jsonl",
        train_codes=train_codes,
        val_codes=val_codes,
        combined_codes=combined_codes,
        combined_meta=Path(f"{combined_codes}.meta.json"),
        heldout_eval_manifest=export_dir / DEFAULT_HELDOUT_EVAL_MANIFEST_NAME,
    )


def resolve_run_paths(run_name: str) -> RunPaths:
    safe_name = validate_layout_name(run_name, kind="run_name")
    run_root = RUNS_ROOT / safe_name
    train_dir = run_root / "train"
    eval_root = run_root / "eval"
    promotion_dir = run_root / "promotion"
    return RunPaths(
        run_name=safe_name,
        run_root=run_root,
        run_manifest=run_root / "run_manifest.json",
        run_state=run_root / "run_state.json",
        train_dir=train_dir,
        train_log=train_dir / "train.log",
        train_history=train_dir / "train_history.jsonl",
        train_summary=train_dir / "train_summary.json",
        eval_root=eval_root,
        checkpoint_metrics=eval_root / "checkpoint_metrics.jsonl",
        promotion_dir=promotion_dir,
        promotion_manifest=promotion_dir / "promotion_manifest.json",
        promotion_archives=promotion_dir / "archives",
    )


def prepare_run_directory(
    run_paths: RunPaths,
    *,
    overwrite: bool = False,
    resume: bool = False,
) -> None:
    if overwrite and resume:
        raise ValueError("overwrite and resume cannot both be enabled")
    if run_paths.run_root.exists():
        if overwrite:
            shutil.rmtree(run_paths.run_root)
        elif not resume:
            raise FileExistsError(
                f"Run already exists at {run_paths.run_root}. "
                "Pass overwrite or resume explicitly."
            )
    run_paths.train_dir.mkdir(parents=True, exist_ok=True)
    run_paths.eval_root.mkdir(parents=True, exist_ok=True)
    run_paths.promotion_dir.mkdir(parents=True, exist_ok=True)
    run_paths.promotion_archives.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.is_file():
        return None
    return json.loads(target.read_text(encoding="utf-8"))


def build_initial_run_state(
    *,
    run_paths: RunPaths,
    bundle_name: str = "",
    trainer: str = "",
    tracking_backend: str = "",
    heldout_eval_manifest: str = "",
    heldout_evals: list[dict[str, Any]] | None = None,
    phase: str = "created",
    wandb_run_id: str | None = None,
) -> dict[str, Any]:
    timestamp = now_utc_iso()
    return {
        "run_name": run_paths.run_name,
        "bundle_name": bundle_name or None,
        "trainer": trainer or None,
        "tracking_backend": tracking_backend or None,
        "heldout_eval_manifest": heldout_eval_manifest or None,
        "heldout_evals": heldout_evals or [],
        "phase": phase,
        "wandb_run_id": wandb_run_id,
        "last_scored_checkpoint": None,
        "last_scored_checkpoints": {},
        "selected_best_checkpoint": None,
        "selected_best_checkpoints": {},
        "checkpoint_eval_streams": {},
        "promotion": {
            "status": "pending",
            "source_dir": None,
            "archive_output_dir": None,
            "serving_output_dir": None,
            "completed_at": None,
        },
        "last_error": None,
        "created_at": timestamp,
        "updated_at": timestamp,
    }


def write_run_state(run_paths: RunPaths, payload: dict[str, Any]) -> None:
    state = dict(payload)
    state["run_name"] = run_paths.run_name
    if not state.get("created_at"):
        state["created_at"] = now_utc_iso()
    state["updated_at"] = now_utc_iso()
    write_json(run_paths.run_state, state)


def read_run_state(run_paths: RunPaths) -> dict[str, Any] | None:
    return read_json_if_exists(run_paths.run_state)


def update_run_state(run_paths: RunPaths, **updates: Any) -> dict[str, Any]:
    state = read_run_state(run_paths) or build_initial_run_state(run_paths=run_paths)
    merged = dict(state)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    write_run_state(run_paths, merged)
    return merged


def normalize_trainer_history(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(state.get("log_history") or [], start=1):
        if not isinstance(raw_row, dict):
            continue
        row = {"index": index}
        row.update(raw_row)
        rows.append(row)
    return rows


def build_run_manifest(
    *,
    run_paths: RunPaths,
    task: str,
    bundle_name: str,
    bundle_path: str,
    base_model: str,
    pretrained_path: str,
    hyperparameters: dict[str, Any],
    notes: str = "",
    tags: Iterable[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "run_name": run_paths.run_name,
        "task": task,
        "bundle_name": bundle_name,
        "bundle_path": bundle_path,
        "base_model": base_model,
        "pretrained_path": pretrained_path,
        **hyperparameters,
        "paths": {
            "run_root": str(run_paths.run_root),
            "train_dir": str(run_paths.train_dir),
            "eval_dir": str(run_paths.eval_root),
            "promotion_dir": str(run_paths.promotion_dir),
        },
        "created_at": now_utc_iso(),
        "notes": notes,
        "tags": csv_tags(tags),
    }
    if extra:
        manifest.update(extra)
    return manifest
