"""Kimi-Audio RAD fine-tuning on Modal (acousticbench app namespace).

This app wraps Moonshot's official `finetune_codes` flow against a persistent
Modal volume, keeps a fixed RAD validation split for checkpoint selection, and
supports two training paths:

- default stock Moonshot-style full fine-tuning (recommended)
- optional RAD LoRA / PEFT fine-tuning with the Whisper audio encoder frozen

Run (always use the user-owned acousticbench Modal environment):

  cd /path/to/acousticbench
  modal run --env=acousticbench finetuning/kimi_audio/rad/modal/kimi_rad_finetune.py \
    --help

Expected volume layout:
  /data/rad_exports/current/rad_kimi_train.jsonl
  /data/rad_exports/current/rad_kimi_val.jsonl
  /data/audio/...

Pipeline on Modal:
  0. `upload-bundle` -> upload a staged local RAD bundle into the Modal volume and verify
                        the raw JSONLs at the paths above
  1. `prepare-model` -> materialize `/data/output/pretrained_hf`
  2. `prepare-data`  -> extract semantic codes for train + val, then build a fixed-eval combined JSONL
  3. `train`         -> run stock full fine-tuning (default) or LoRA SFT
  4. `publish`       -> merge the LoRA adapter into the base checkpoint and export an
                        inference-ready model to `/data/serving/kimi-audio-ft/latest`

Legacy aliases are still accepted: `sync-data`, `export`, `codes`, `package`, and `all`.
"""

from __future__ import annotations

from collections import Counter
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import re
from pathlib import Path
from typing import Any

import modal

def _discover_repo_root() -> Path | None:
    """Find a repo root when running locally or from Modal's /root mounts."""
    candidates = [
        Path(__file__).resolve(),
        *Path(__file__).resolve().parents,
        Path.cwd().resolve(),
        *Path.cwd().resolve().parents,
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        root = candidate if candidate.is_dir() else candidate.parent
        if root in seen:
            continue
        seen.add(root)
        if (root / "finetuning").is_dir():
            return root
    return None


REPO_ROOT = _discover_repo_root()
if REPO_ROOT is not None and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import finetuning.kimi_audio as _kimi_audio_pkg

from finetuning.kimi_audio.common.run_layout import (
    DEFAULT_PRETRAINED_PATH as SHARED_DEFAULT_PRETRAINED_PATH,
    DEFAULT_SERVING_MODEL_PATH as SHARED_DEFAULT_SERVING_MODEL_PATH,
    build_run_manifest,
    build_initial_run_state,
    csv_tags,
    normalize_trainer_history,
    now_utc_iso,
    prepare_run_directory,
    resolve_existing_heldout_eval_manifest,
    update_run_state,
    resolve_bundle_paths,
    resolve_run_paths,
    validate_layout_name,
    volume_relative,
    write_json,
    write_run_state,
    write_jsonl,
)
from finetuning.kimi_audio.common.volume_sync import remote_path_exists, volume_rm
from finetuning.kimi_audio.rad.direct_checkpoint_eval import (
    CHECKPOINT_EVAL_INFERENCE_MODE_KIMI_GENERATE,
    CHECKPOINT_EVAL_INFERENCE_MODE_LOCAL_LOOP,
    CHECKPOINT_EVAL_WANDB_STEP_METRIC,
    _heldout_eval_spec,
    evaluate_checkpoint,
    ensure_checkpoint_eval_wandb_metrics_defined,
    finalize_checkpoint_evaluation,
    persist_active_wandb_run,
    run_combined_save_time_checkpoint_eval,
    run_step0_generate_path_comparison,
    run_save_time_checkpoint_eval,
)
from finetuning.kimi_audio.rad.evaluation_common import (
    checkpoint_eval_metric_prefix,
    checkpoint_eval_metric_names,
    simplified_checkpoint_eval_summary,
)

APP_NAME = "acousticbench-kimi-rad-finetune"
VOLUME_NAME = "acousticbench-kimi-rad-data"

app = modal.App(APP_NAME)

KIMI_ROOT = Path("/opt/Kimi-Audio")
# Modal may copy this file to `/root/kimi_rad_finetune.py`; do not derive repo paths from __file__.
VENDORED_KIMI_ROOT = Path(_kimi_audio_pkg.__file__).resolve().parent / "vendor_kimi"
VENDORED_KIMI_FINETUNE_CODES_DIR = VENDORED_KIMI_ROOT / "finetune_codes"
VENDORED_KIMI_MODEL_MODULE = (
    "finetuning.kimi_audio.vendor_kimi.finetune_codes.model"
)
RAD_LORA_ENTRYPOINT_MODULE = "finetuning.kimi_audio.rad.runtime.lora_entrypoint"
RAD_STOCK_FT_ENTRYPOINT_MODULE = (
    "finetuning.kimi_audio.rad.runtime.stock_full_ft_entrypoint"
)
RAD_WANDB_AUTH_PROBE_MODULE = (
    "finetuning.kimi_audio.rad.runtime.wandb_auth_probe"
)
RAD_WANDB_CHECKPOINT_LOG_MODULE = (
    "finetuning.kimi_audio.rad.runtime.wandb_checkpoint_log"
)
RAD_MERGE_LORA_MODULE = "finetuning.kimi_audio.rad.runtime.merge_lora"
DATA_MOUNT = "/data"
HF_CACHE_MOUNT = "/hf-cache"
LEGACY_EXPORT_DIR = f"{DATA_MOUNT}/rad_exports/current"
DEFAULT_EXPORT_DIR = LEGACY_EXPORT_DIR
DEFAULT_TRAIN_JSON = f"{DEFAULT_EXPORT_DIR}/rad_kimi_train.jsonl"
DEFAULT_VAL_JSON = f"{DEFAULT_EXPORT_DIR}/rad_kimi_val.jsonl"
DEFAULT_TRAIN_CODES = f"{DEFAULT_EXPORT_DIR}/rad_kimi_train_with_semantic_codes.jsonl"
DEFAULT_VAL_CODES = f"{DEFAULT_EXPORT_DIR}/rad_kimi_val_with_semantic_codes.jsonl"
DEFAULT_COMBINED_CODES = (
    f"{DEFAULT_EXPORT_DIR}/rad_kimi_trainval_with_semantic_codes.jsonl"
)
DEFAULT_AUDIO_ROOT = f"{DATA_MOUNT}/audio"
DEFAULT_PRETRAINED_PATH = SHARED_DEFAULT_PRETRAINED_PATH
DEFAULT_SERVING_MODEL_PATH = SHARED_DEFAULT_SERVING_MODEL_PATH
DEFAULT_PACKAGE_TMP_ROOT = f"{DATA_MOUNT}/serving/kimi-audio-ft/package_tmp"
DEFAULT_MODAL_ENV = os.environ.get("MODAL_ENVIRONMENT", "acousticbench")
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
DEFAULT_LORA_TARGET_MODULES_CSV = ",".join(DEFAULT_LORA_TARGET_MODULES)
DEFAULT_LORA_GRADIENT_ACCUMULATION_STEPS = 1
DEFAULT_LORA_PER_DEVICE_TRAIN_BATCH_SIZE = 12
DEFAULT_LORA_PER_DEVICE_EVAL_BATCH_SIZE = 16
DEFAULT_WANDB_LOGGING_STEPS = 1
DEFAULT_LLM_EXTRACTION_CONCURRENCY = 64
DEFAULT_JUDGE_RESPONSE_RETRIES = 4
DEFAULT_JUDGE_RESPONSE_RETRY_SECONDS = 2.0
DEFAULT_HELDOUT_EVAL_STAGES = ("step0", "checkpoint")
FINAL_HELDOUT_EVAL_STAGES = ("final",)
STEP_UPLOAD_BUNDLE = "upload-bundle"
STEP_PREPARE_MODEL = "prepare-model"
STEP_PREPARE_DATA = "prepare-data"
STEP_PROFILE = "profile"
STEP_TRAIN = "train"
STEP_PUBLISH = "publish"
STEP_RUN = "run"
STEP_WANDB_CHECKPOINT_EVAL_LOG = "wandb-checkpoint-eval-log"
LEGACY_STEP_ALIASES = {
    "sync-data": STEP_UPLOAD_BUNDLE,
    "export": STEP_PREPARE_MODEL,
    "codes": STEP_PREPARE_DATA,
    "package": STEP_PUBLISH,
    "all": STEP_RUN,
}
TRAINER_LORA = "lora"
TRAINER_STOCK_FULL_FT = "stock-full-ft"
DEFAULT_TRAINER = TRAINER_STOCK_FULL_FT
SUPPORTED_TRAINERS = (TRAINER_LORA, TRAINER_STOCK_FULL_FT)
STOCK_TRANSFORMERS_VERSION = "4.48.2"
STOCK_DEEPSPEED_VERSION = "0.16.9"
STOCK_TORCH_VERSION = "2.6.0"
STOCK_TORCHVISION_VERSION = "0.21.0"
TRAIN_GPU = "H200"
DUAL_TRAIN_GPU = "H200:2"
QUAD_TRAIN_GPU = "H200:4"
STOCK_FULL_FT_PROFILE_SINGLE_SAFE = "single-safe"
STOCK_FULL_FT_PROFILE_DUAL_FAST = "dual-fast"
STOCK_FULL_FT_PROFILE_QUAD_FAST = "quad-fast"
SUPPORTED_STOCK_FULL_FT_PROFILES = (
    STOCK_FULL_FT_PROFILE_SINGLE_SAFE,
    STOCK_FULL_FT_PROFILE_DUAL_FAST,
    STOCK_FULL_FT_PROFILE_QUAD_FAST,
)
DEFAULT_STOCK_FULL_FT_PROFILE = STOCK_FULL_FT_PROFILE_DUAL_FAST
DEFAULT_STOCK_FULL_FT_LEARNING_RATE = 2e-6
DEFAULT_STOCK_FULL_FT_SAVE_STEPS = 50
DEFAULT_STOCK_FULL_FT_EVAL_STEPS = 50
DEFAULT_GRADIENT_CHECKPOINTING_MODE = "auto"
STOCK_FULL_FT_PROFILE_CONFIGS: dict[str, dict[str, Any]] = {
    STOCK_FULL_FT_PROFILE_SINGLE_SAFE: {
        "gpu": TRAIN_GPU,
        "gpu_count": 1,
        "deepspeed_profile": "zero3-cpuoffload",
        "optimizer_offload": True,
        "default_gradient_checkpointing": True,
    },
    STOCK_FULL_FT_PROFILE_DUAL_FAST: {
        "gpu": DUAL_TRAIN_GPU,
        "gpu_count": 2,
        "deepspeed_profile": "zero2-multigpu",
        "optimizer_offload": False,
        "default_gradient_checkpointing": False,
    },
    STOCK_FULL_FT_PROFILE_QUAD_FAST: {
        "gpu": QUAD_TRAIN_GPU,
        "gpu_count": 4,
        "deepspeed_profile": "zero2-multigpu",
        "optimizer_offload": False,
        "default_gradient_checkpointing": False,
    },
}

rad_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name("audiobench-hf-cache", create_if_missing=True)

HF_SECRET_NAME = "huggingface-secret"
OPENAI_SECRET_NAME = "openai-api-key"
WANDB_SECRET_NAME = "acousticbench-wandb"
_wandb_base_url = os.environ.get("WANDB_BASE_URL")
_wandb_secret_values = {
    key: value
    for key in ("WANDB_API_KEY",)
    if (value := os.environ.get(key))
}
_openai_secret_values = {
    key: value
    for key in ("OPENAI_API_KEY",)
    if (value := os.environ.get(key))
}
wandb_training_secrets: list[modal.Secret] = (
    [modal.Secret.from_dict(_wandb_secret_values)]
    if _wandb_secret_values
    else []
    if os.environ.get("AUDIOBENCH_SKIP_WANDB_SECRET") == "1"
    else [modal.Secret.from_name(WANDB_SECRET_NAME, required_keys=["WANDB_API_KEY"])]
)
openai_training_secrets: list[modal.Secret] = (
    [modal.Secret.from_dict(_openai_secret_values)]
    if _openai_secret_values
    else [
        modal.Secret.from_name(
            OPENAI_SECRET_NAME,
            required_keys=["OPENAI_API_KEY"],
        )
    ]
)
if _wandb_base_url:
    wandb_training_secrets.append(
        modal.Secret.from_dict({"WANDB_BASE_URL": _wandb_base_url})
    )
WAND_SECRET_SOURCE = f"modal-secret:{WANDB_SECRET_NAME}"
DEFAULT_WANDB_PROJECT = "acousticbench-kimi-rad"


def _hf_secrets() -> list[modal.Secret]:
    """HF token for the gated Kimi checkpoint.

    Prefer a locally exported token, else the shared Modal secret. Reading only
    the local environment meant a deploy without HF_TOKEN produced a job that
    could not download the model.
    """
    local_values = {
        key: value
        for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
        if (value := os.environ.get(key))
    }
    if local_values:
        return [modal.Secret.from_dict(local_values)]
    return [modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])]


def _openai_secrets() -> list[modal.Secret]:
    return [modal.Secret.from_name(OPENAI_SECRET_NAME, required_keys=["OPENAI_API_KEY"])]


hf_secrets = _hf_secrets()

DEFAULT_KIMI_MODEL_NAME = "moonshotai/Kimi-Audio-7B-Instruct"
_TRAINING_INPUT_TAG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _normalize_training_input_tag(raw_tag: str) -> str:
    value = _TRAINING_INPUT_TAG_RE.sub("-", str(raw_tag or "").strip().lower()).strip("-")
    if not value:
        raise ValueError("training/eval input tag cannot be empty")
    return value


def _safe_output_tag(raw_tag: str) -> str:
    return validate_layout_name(_normalize_training_input_tag(raw_tag), kind="training_input_tag")


def _require_wandb_training_secrets() -> list[modal.Secret]:
    return [*hf_secrets, *wandb_training_secrets]


def _require_training_runtime_secrets() -> list[modal.Secret]:
    return [*hf_secrets, *wandb_training_secrets, *openai_training_secrets]


def _resolve_supported_kimi_model_name(model_name: str) -> str:
    normalized = str(model_name or "").strip().rstrip("/") or DEFAULT_KIMI_MODEL_NAME
    if normalized == "moonshotai/Kimi-Audio-7B":
        raise ValueError(
            "Kimi RAD finetuning no longer supports `moonshotai/Kimi-Audio-7B`; "
            "use `moonshotai/Kimi-Audio-7B-Instruct`."
        )
    return normalized

HF_ENV = {
    "HF_HOME": f"{HF_CACHE_MOUNT}/huggingface",
    "HF_HUB_CACHE": f"{HF_CACHE_MOUNT}/huggingface/hub",
    "HUGGINGFACE_HUB_CACHE": f"{HF_CACHE_MOUNT}/huggingface/hub",
    "TRANSFORMERS_CACHE": f"{HF_CACHE_MOUNT}/huggingface/hub",
    "TOKENIZERS_PARALLELISM": "false",
}

_base_kimi_repo_commands = (
    "git clone --depth 1 https://github.com/MoonshotAI/Kimi-Audio.git /opt/Kimi-Audio",
    "cd /opt/Kimi-Audio && git submodule update --init --recursive",
)

_base_kimi_image_commands = (
    *_base_kimi_repo_commands,
    "cd /opt/Kimi-Audio && pip install -r requirements.txt",
    "cd /opt/Kimi-Audio && pip install -e .",
)

finetune_image = (
    modal.Image.from_registry("moonshotai/kimi-audio:v0.1")
    .apt_install("git")
    .run_commands(
        *_base_kimi_image_commands,
        (
            "pip install "
            "'deepspeed>=0.14' 'accelerate' 'tqdm' 'sentencepiece' 'peft' 'wandb' "
            "'jsonlines' 'pandas' 'aiohttp' 'numpy' 'python-dotenv' 'tiktoken' 'loguru'"
        ),
    )
    # Thin layer: evalkit WER imports `editdistance` and `zhconv`; keep off the heavy clone/pip block.
    .pip_install("editdistance", "zhconv")
    .env(HF_ENV)
    .add_local_python_source("finetuning")
    .add_local_python_source("harness")
)

stock_full_finetune_image = (
    modal.Image.from_registry("moonshotai/kimi-audio:v0.1")
    .apt_install("git")
    .run_commands(
        *_base_kimi_repo_commands,
        "cd /opt/Kimi-Audio && pip install -r requirements.txt",
        (
            "pip install "
            f"'torchvision=={STOCK_TORCHVISION_VERSION}' "
            f"'transformers=={STOCK_TRANSFORMERS_VERSION}' "
            "'wandb' 'jsonlines' 'pandas' 'aiohttp' 'numpy' 'python-dotenv' 'tiktoken' 'loguru'"
        ),
        "pip install 'flash-attn==2.7.4.post1' --no-build-isolation",
    )
    .pip_install("editdistance", "zhconv")
    .env(HF_ENV)
    .add_local_python_source("finetuning")
    .add_local_python_source("harness")
)


def _run_in_kimi_repo(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    stream_output: bool = False,
    log_path: str | Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> str:
    cwd = cwd or KIMI_ROOT
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": os.environ.get("PYTHONUNBUFFERED", "1"),
        **(extra_env or {}),
    }
    log_handle = None
    if log_path is not None:
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_file.open("a", encoding="utf-8")
    if stream_output:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        if proc.stdout is None:
            raise RuntimeError(f"Failed to stream command output: {' '.join(cmd)}")
        print(f"[streaming subprocess] {' '.join(cmd)}", flush=True)
        chunks: list[str] = []
        for line in proc.stdout:
            print(line, end="", flush=True)
            chunks.append(line)
            if log_handle is not None:
                log_handle.write(line)
        proc.wait()
        out = "".join(chunks)
    else:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        if log_handle is not None and out:
            log_handle.write(out)
    if log_handle is not None:
        log_handle.flush()
        log_handle.close()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{out}"
        )
    return out


def _ensure_kimi_repo_on_sys_path() -> None:
    kimi_root = str(KIMI_ROOT)
    if kimi_root not in sys.path:
        sys.path.insert(0, kimi_root)


def _wandb_runtime_env(
    *,
    run_name: str,
    wandb_project: str,
    wandb_entity: str,
    wandb_mode: str,
    tags: list[str],
) -> dict[str, str]:
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "W&B logging requires WANDB_API_KEY on the Modal runner. "
            f"Create the Modal secret `{WANDB_SECRET_NAME}` with WANDB_API_KEY."
        )
    env = {"WANDB_NAME": run_name}
    if wandb_project:
        env["WANDB_PROJECT"] = wandb_project
    if wandb_entity:
        env["WANDB_ENTITY"] = wandb_entity
    if wandb_mode:
        env["WANDB_MODE"] = wandb_mode
    if tags:
        env["WANDB_TAGS"] = ",".join(tags)
    return env


def _resolve_wandb_project_name(*, tracking_backend: str, wandb_project: str) -> str:
    value = str(wandb_project or "").strip()
    if tracking_backend != "wandb":
        return value
    return value or DEFAULT_WANDB_PROJECT


def _resolve_wandb_run_metadata(
    *,
    tracking_backend: str,
    wandb_project: str,
    wandb_auth: dict[str, Any] | None,
    run_state: dict[str, Any] | None,
) -> dict[str, Any]:
    if tracking_backend != "wandb":
        return {
            "wandb_project": None,
            "wandb_entity": None,
            "wandb_run_id": None,
            "wandb_run_url": None,
        }
    auth = wandb_auth if isinstance(wandb_auth, dict) else {}
    resolved_project = str(wandb_project or auth.get("project") or "").strip()
    resolved_entity = str(auth.get("entity") or auth.get("viewer") or "").strip()
    resolved_run_id = str((run_state or {}).get("wandb_run_id") or "").strip()
    base_url = str(auth.get("base_url") or _wandb_base_url or "https://wandb.ai").strip()
    run_url = None
    if base_url and resolved_entity and resolved_project and resolved_run_id:
        run_url = (
            f"{base_url.rstrip('/')}/{resolved_entity}/{resolved_project}/runs/{resolved_run_id}"
        )
    return {
        "wandb_project": resolved_project or None,
        "wandb_entity": resolved_entity or None,
        "wandb_run_id": resolved_run_id or None,
        "wandb_run_url": run_url,
    }


def _maybe_print_wandb_run_url(result: Any) -> None:
    if not isinstance(result, dict):
        return
    wandb_run_url = str(result.get("wandb_run_url") or "").strip()
    if wandb_run_url:
        print(f"[wandb] Run URL: {wandb_run_url}")


def _assert_wandb_remote_auth(
    *,
    wandb_env: dict[str, str],
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    probe_cmd = [
        "python",
        "-m",
        RAD_WANDB_AUTH_PROBE_MODULE,
    ]
    probe_output = _run_in_kimi_repo(probe_cmd, log_path=log_path, extra_env=wandb_env)
    last_line = next((line for line in reversed(probe_output.splitlines()) if line.strip()), "")
    try:
        return json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Failed to parse W&B auth probe output: {last_line!r}"
        ) from exc


def _run_train_remote(
    *,
    trainer: str = DEFAULT_TRAINER,
    tracking_backend: str,
    **kwargs: Any,
) -> dict[str, Any]:
    tracking_backend = _normalize_tracking_backend(tracking_backend)
    resolved_trainer = _normalize_trainer(trainer)
    if resolved_trainer == TRAINER_STOCK_FULL_FT:
        stock_kwargs = dict(kwargs)
        stock_kwargs.pop("lora_r", None)
        stock_kwargs.pop("lora_alpha", None)
        stock_kwargs.pop("lora_dropout", None)
        stock_kwargs.pop("lora_target_modules", None)
        stock_full_ft_profile = _normalize_stock_full_ft_profile(
            stock_kwargs.pop(
                "stock_full_ft_profile",
                DEFAULT_STOCK_FULL_FT_PROFILE,
            )
        )
        stock_runner = _select_stock_full_ft_remote_fn(
            stock_full_ft_profile=stock_full_ft_profile,
            tracking_backend=tracking_backend,
        )
        return stock_runner.remote(
            tracking_backend=tracking_backend,
            stock_full_ft_profile=stock_full_ft_profile,
            **stock_kwargs,
        )
    lora_kwargs = dict(kwargs)
    lora_kwargs.pop("freeze_audio_encoder", None)
    lora_kwargs.pop("stock_full_ft_profile", None)
    lora_kwargs.pop("gradient_checkpointing", None)
    lora_kwargs.pop("step0_compare_generate", None)
    if tracking_backend == "wandb":
        return run_deepspeed_finetune_with_wandb.remote(
            tracking_backend=tracking_backend,
            **lora_kwargs,
        )
    return run_deepspeed_finetune.remote(
        tracking_backend=tracking_backend,
        **lora_kwargs,
    )


def _normalize_trainer(trainer: str) -> str:
    value = str(trainer or DEFAULT_TRAINER).strip() or DEFAULT_TRAINER
    if value not in SUPPORTED_TRAINERS:
        supported = ", ".join(SUPPORTED_TRAINERS)
        raise ValueError(f"Unknown trainer {value!r}. Expected one of: {supported}")
    return value


def _normalize_tracking_backend(tracking_backend: str) -> str:
    value = str(tracking_backend or "wandb").strip().lower() or "wandb"
    if value not in {"wandb", "none"}:
        raise ValueError(
            f"Unknown tracking backend {value!r}. Expected 'wandb' or 'none'."
        )
    return value


def _normalize_step_name(step: str) -> str:
    value = str(step or "").strip() or "help"
    return LEGACY_STEP_ALIASES.get(value, value)


def _normalize_stock_full_ft_profile(stock_full_ft_profile: str) -> str:
    value = (
        str(stock_full_ft_profile or DEFAULT_STOCK_FULL_FT_PROFILE).strip().lower()
        or DEFAULT_STOCK_FULL_FT_PROFILE
    )
    if value not in SUPPORTED_STOCK_FULL_FT_PROFILES:
        supported = ", ".join(SUPPORTED_STOCK_FULL_FT_PROFILES)
        raise ValueError(
            f"Unknown stock full-FT profile {value!r}. Expected one of: {supported}"
        )
    return value


def _resolve_train_launch_defaults(
    *,
    trainer: str,
    save_steps: int | None,
    eval_steps: int | None,
    learning_rate: float | None,
    gradient_accumulation_steps: int | None,
    per_device_train_batch_size: int | None,
    per_device_eval_batch_size: int | None,
) -> dict[str, float | int]:
    """Apply trainer-specific defaults without overriding explicit CLI values."""
    resolved_trainer = _normalize_trainer(trainer)
    if resolved_trainer == TRAINER_STOCK_FULL_FT:
        default_save_steps = DEFAULT_STOCK_FULL_FT_SAVE_STEPS
        default_eval_steps = DEFAULT_STOCK_FULL_FT_EVAL_STEPS
        default_learning_rate = DEFAULT_STOCK_FULL_FT_LEARNING_RATE
        default_gradient_accumulation_steps = 1
        default_per_device_train_batch_size = 1
        default_per_device_eval_batch_size = 1
    else:
        default_save_steps = 25
        default_eval_steps = 25
        default_learning_rate = 5e-5
        default_gradient_accumulation_steps = DEFAULT_LORA_GRADIENT_ACCUMULATION_STEPS
        default_per_device_train_batch_size = DEFAULT_LORA_PER_DEVICE_TRAIN_BATCH_SIZE
        default_per_device_eval_batch_size = DEFAULT_LORA_PER_DEVICE_EVAL_BATCH_SIZE
    return {
        "save_steps": default_save_steps if save_steps is None else int(save_steps),
        "eval_steps": default_eval_steps if eval_steps is None else int(eval_steps),
        "learning_rate": (
            default_learning_rate if learning_rate is None else float(learning_rate)
        ),
        "gradient_accumulation_steps": (
            default_gradient_accumulation_steps
            if gradient_accumulation_steps is None
            else int(gradient_accumulation_steps)
        ),
        "per_device_train_batch_size": (
            default_per_device_train_batch_size
            if per_device_train_batch_size is None
            else int(per_device_train_batch_size)
        ),
        "per_device_eval_batch_size": (
            default_per_device_eval_batch_size
            if per_device_eval_batch_size is None
            else int(per_device_eval_batch_size)
        ),
    }


def _wandb_trace_step0_eval(
    message: str,
    *,
    status: str,
    phase_ix: int,
    eval_tag: str | None = None,
    wandb_metric_prefix: str | None = None,
) -> None:
    """Emit W&B-visible breadcrumbs before long-running step-0 work (model load + eval)."""
    try:
        import wandb
    except Exception:
        print(message, flush=True)
        return
    if getattr(wandb, "run", None) is None:
        print(message, flush=True)
        return
    try:
        metric_names = checkpoint_eval_metric_names(
            eval_tag=eval_tag,
            wandb_metric_prefix=wandb_metric_prefix,
        )
        ensure_checkpoint_eval_wandb_metrics_defined(
            eval_tag=eval_tag,
            wandb_metric_prefix=wandb_metric_prefix,
        )
        wandb.summary["checkpoint_eval_step0_status"] = status
        wandb.log(
            {
                CHECKPOINT_EVAL_WANDB_STEP_METRIC: 0,
                metric_names["checkpoint_step"]: 0,
                metric_names["step0_phase"]: float(phase_ix),
            },
            step=0,
        )
        if hasattr(wandb, "termlog"):
            wandb.termlog(message)
    except Exception:
        pass
    print(message, flush=True)


def _announce_wandb_run_ready_for_step0(
    run: Any,
    *,
    eval_tag: str | None = None,
    wandb_metric_prefix: str | None = None,
) -> None:
    """Log to W&B immediately after init so the UI shows a live run before model load / eval."""
    try:
        import wandb
    except Exception:
        return
    url = ""
    try:
        url = str(run.get_url())  # type: ignore[union-attr]
    except Exception:
        url = str(getattr(run, "url", "") or "")
    if url:
        print(f"[wandb] Step-0 baseline: live run {url}", flush=True)
    try:
        metric_names = checkpoint_eval_metric_names(
            eval_tag=eval_tag,
            wandb_metric_prefix=wandb_metric_prefix,
        )
        ensure_checkpoint_eval_wandb_metrics_defined(
            eval_tag=eval_tag,
            wandb_metric_prefix=wandb_metric_prefix,
        )
        wandb.summary["checkpoint_eval_step0_status"] = "wandb_run_started"
        wandb.log(
            {
                CHECKPOINT_EVAL_WANDB_STEP_METRIC: 0,
                metric_names["checkpoint_step"]: 0,
                metric_names["step0_wandb_ready"]: 1,
            },
            step=0,
        )
        if hasattr(wandb, "termlog"):
            wandb.termlog(
                "W&B is live. Next: load base KimiAudioModel weights, then held-out step-0 eval."
            )
    except Exception:
        pass


def _maybe_run_step0_eval(
    *,
    model_dir: str,
    pretrained_path: str,
    model_name_or_path: str,
    run_name: str,
    heldout_eval_manifest: str,
    step0_compare_generate: bool = False,
    eval_tag: str | None = None,
    wandb_metric_prefix: str | None = None,
) -> dict[str, Any] | None:
    heldout_eval_manifest = str(heldout_eval_manifest or "").strip()
    if not heldout_eval_manifest:
        return None
    checkpoint_dir = str(model_dir).strip()
    if not checkpoint_dir:
        raise ValueError("step-0 eval requires a model directory")
    eval_spec = _heldout_eval_spec(
        heldout_eval_manifest,
        eval_tag=eval_tag,
        wandb_metric_prefix=wandb_metric_prefix,
    )

    _wandb_trace_step0_eval(
        "[checkpoint-eval] step=0: preparing base-model eval (W&B already initialized)",
        status="step0_eval_preparing",
        phase_ix=1,
        eval_tag=eval_spec["eval_tag"],
        wandb_metric_prefix=eval_spec["wandb_metric_prefix"],
    )

    _ensure_kimi_repo_on_sys_path()
    from finetuning.kimi_audio.vendor_kimi.finetune_codes.model import KimiAudioModel

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("step-0 eval requires torch in the training runtime") from exc

    print(
        "[checkpoint-eval] step=0 pre-train baseline requested; starting held-out eval",
        flush=True,
    )
    _wandb_trace_step0_eval(
        "[checkpoint-eval] step=0: loading base checkpoint (from_pretrained; may take several minutes)",
        status="loading_base_model_weights",
        phase_ix=2,
        eval_tag=eval_spec["eval_tag"],
        wandb_metric_prefix=eval_spec["wandb_metric_prefix"],
    )
    model = KimiAudioModel.from_pretrained(
        checkpoint_dir,
        device_map=None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    try:
        if torch.cuda.is_available():
            model.to(torch.cuda.current_device())
        model.eval()
        _wandb_trace_step0_eval(
            "[checkpoint-eval] step=0: running held-out eval (checkpoint metric logging via W&B)",
            status="running_step0_heldout_eval",
            phase_ix=3,
            eval_tag=eval_spec["eval_tag"],
            wandb_metric_prefix=eval_spec["wandb_metric_prefix"],
        )
        result = run_save_time_checkpoint_eval(
            model=model,
            pretrained_path=pretrained_path,
            model_name_or_path=model_name_or_path,
            run_name=run_name,
            checkpoint_step=0,
            output_dir=checkpoint_dir,
            heldout_eval_manifest=heldout_eval_manifest,
            checkpoint_dir_override=checkpoint_dir,
            checkpoint_name_override="base-model",
            eval_tag=eval_spec["eval_tag"],
            wandb_metric_prefix=eval_spec["wandb_metric_prefix"],
        )
        if step0_compare_generate and result is not None:
            comparison_result = run_step0_generate_path_comparison(
                heldout_eval_manifest=heldout_eval_manifest,
                pretrained_path=pretrained_path,
                model_name_or_path=model_name_or_path,
                run_name=run_name,
                checkpoint_dir=checkpoint_dir,
                baseline_metrics_row=result.get("metrics_row") or {},
                eval_tag=eval_spec["eval_tag"],
                wandb_metric_prefix=eval_spec["wandb_metric_prefix"],
            )
            result = {
                **result,
                "step0_generate_compare": {
                    "summary_path": comparison_result.get("summary_path"),
                    "overall_accuracy_pct": (
                        comparison_result.get("metrics_row") or {}
                    ).get("accuracy_pct_total"),
                },
            }
    finally:
        del model
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(
        "[checkpoint-eval] step=0 pre-train baseline finished successfully",
        flush=True,
    )
    return result


def _maybe_run_step0_eval_group(
    *,
    model_dir: str,
    pretrained_path: str,
    model_name_or_path: str,
    run_name: str,
    eval_specs: list[dict[str, Any]],
    step0_compare_generate: bool = False,
) -> dict[str, Any] | None:
    normalized_specs = [
        dict(spec)
        for spec in list(eval_specs or [])
        if str(spec.get("manifest") or "").strip()
    ]
    if not normalized_specs:
        return None
    if len(normalized_specs) == 1:
        spec = normalized_specs[0]
        return _maybe_run_step0_eval(
            model_dir=model_dir,
            pretrained_path=pretrained_path,
            model_name_or_path=model_name_or_path,
            run_name=run_name,
            heldout_eval_manifest=str(spec.get("manifest") or "").strip(),
            step0_compare_generate=step0_compare_generate,
            eval_tag=str(spec.get("tag") or "").strip() or None,
            wandb_metric_prefix=str(spec.get("wandb_metric_prefix") or "").strip() or None,
        )
    if step0_compare_generate:
        results = []
        for spec in normalized_specs:
            results.append(
                _maybe_run_step0_eval(
                    model_dir=model_dir,
                    pretrained_path=pretrained_path,
                    model_name_or_path=model_name_or_path,
                    run_name=run_name,
                    heldout_eval_manifest=str(spec.get("manifest") or "").strip(),
                    step0_compare_generate=True,
                    eval_tag=str(spec.get("tag") or "").strip() or None,
                    wandb_metric_prefix=str(spec.get("wandb_metric_prefix") or "").strip()
                    or None,
                )
            )
        return {"status": "ok", "per_eval_results": results}

    checkpoint_dir = str(model_dir).strip()
    if not checkpoint_dir:
        raise ValueError("step-0 eval requires a model directory")

    _ensure_kimi_repo_on_sys_path()
    from finetuning.kimi_audio.vendor_kimi.finetune_codes.model import KimiAudioModel

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("step-0 eval requires torch in the training runtime") from exc

    print(
        "[checkpoint-eval] step=0 pre-train baseline requested; starting combined held-out eval",
        flush=True,
    )
    for raw_spec in normalized_specs:
        spec = dict(raw_spec)
        eval_tag = str(spec.get("tag") or "").strip() or None
        wandb_metric_prefix = str(spec.get("wandb_metric_prefix") or "").strip() or None
        _wandb_trace_step0_eval(
            "[checkpoint-eval] step=0: preparing combined base-model eval (W&B already initialized)",
            status="step0_eval_preparing",
            phase_ix=1,
            eval_tag=eval_tag,
            wandb_metric_prefix=wandb_metric_prefix,
        )
        _wandb_trace_step0_eval(
            "[checkpoint-eval] step=0: loading base checkpoint once for combined held-out eval",
            status="loading_base_model_weights",
            phase_ix=2,
            eval_tag=eval_tag,
            wandb_metric_prefix=wandb_metric_prefix,
        )

    model = KimiAudioModel.from_pretrained(
        checkpoint_dir,
        device_map=None,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    try:
        if torch.cuda.is_available():
            model.to(torch.cuda.current_device())
        model.eval()
        for raw_spec in normalized_specs:
            spec = dict(raw_spec)
            _wandb_trace_step0_eval(
                "[checkpoint-eval] step=0: running combined held-out eval (checkpoint metric logging via W&B)",
                status="running_step0_heldout_eval",
                phase_ix=3,
                eval_tag=str(spec.get("tag") or "").strip() or None,
                wandb_metric_prefix=str(spec.get("wandb_metric_prefix") or "").strip()
                or None,
            )
        result = run_combined_save_time_checkpoint_eval(
            model=model,
            pretrained_path=pretrained_path,
            model_name_or_path=model_name_or_path,
            run_name=run_name,
            checkpoint_step=0,
            output_dir=checkpoint_dir,
            eval_specs=normalized_specs,
            checkpoint_dir_override=checkpoint_dir,
            checkpoint_name_override="base-model",
        )
    finally:
        del model
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(
        "[checkpoint-eval] step=0 pre-train baseline finished successfully",
        flush=True,
    )
    return result


def _maybe_start_step0_wandb_run(
    *,
    enabled: bool,
    run_name: str,
    wandb_project: str,
    wandb_entity: str,
    wandb_mode: str,
    tags: list[str],
    heldout_evals: list[dict[str, Any]] | None = None,
):
    if not enabled:
        return None
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError("step-0 eval requires wandb in the training runtime") from exc

    init_kwargs: dict[str, Any] = {
        "name": run_name,
        "resume": "allow",
    }
    if wandb_project:
        init_kwargs["project"] = wandb_project
    if wandb_entity:
        init_kwargs["entity"] = wandb_entity
    if wandb_mode:
        init_kwargs["mode"] = wandb_mode
    if tags:
        init_kwargs["tags"] = tags
    run = wandb.init(**init_kwargs)
    persist_active_wandb_run(run_name=run_name)
    if heldout_evals:
        for eval_spec in heldout_evals:
            _announce_wandb_run_ready_for_step0(
                run,
                eval_tag=str(eval_spec.get("tag") or "").strip() or None,
                wandb_metric_prefix=str(eval_spec.get("wandb_metric_prefix") or "").strip()
                or None,
            )
    else:
        _announce_wandb_run_ready_for_step0(run)
    return run


def _resolve_auto_bool_flag(raw_value: Any, *, name: str, default: bool) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    value = str(raw_value or DEFAULT_GRADIENT_CHECKPOINTING_MODE).strip().lower()
    if not value or value == DEFAULT_GRADIENT_CHECKPOINTING_MODE:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"Invalid {name} value {raw_value!r}. Expected auto/true/false."
    )


def _resolve_stock_full_ft_profile_settings(
    *,
    stock_full_ft_profile: str,
    gradient_checkpointing: Any = DEFAULT_GRADIENT_CHECKPOINTING_MODE,
) -> dict[str, Any]:
    resolved_profile = _normalize_stock_full_ft_profile(stock_full_ft_profile)
    base = STOCK_FULL_FT_PROFILE_CONFIGS[resolved_profile]
    resolved_gradient_checkpointing = _resolve_auto_bool_flag(
        gradient_checkpointing,
        name="gradient_checkpointing",
        default=bool(base["default_gradient_checkpointing"]),
    )
    return {
        "stock_full_ft_profile": resolved_profile,
        "gpu": str(base["gpu"]),
        "gpu_count": int(base["gpu_count"]),
        "deepspeed_profile": str(base["deepspeed_profile"]),
        "optimizer_offload": bool(base["optimizer_offload"]),
        "gradient_checkpointing": resolved_gradient_checkpointing,
    }


def _select_stock_full_ft_remote_fn(
    *,
    stock_full_ft_profile: str,
    tracking_backend: str,
) -> Any:
    resolved_profile = _normalize_stock_full_ft_profile(stock_full_ft_profile)
    if tracking_backend == "wandb":
        mapping = {
            STOCK_FULL_FT_PROFILE_SINGLE_SAFE: run_stock_full_finetune_with_wandb,
            STOCK_FULL_FT_PROFILE_DUAL_FAST: run_stock_full_finetune_dual_fast_with_wandb,
            STOCK_FULL_FT_PROFILE_QUAD_FAST: run_stock_full_finetune_quad_fast_with_wandb,
        }
    else:
        mapping = {
            STOCK_FULL_FT_PROFILE_SINGLE_SAFE: run_stock_full_finetune,
            STOCK_FULL_FT_PROFILE_DUAL_FAST: run_stock_full_finetune_dual_fast,
            STOCK_FULL_FT_PROFILE_QUAD_FAST: run_stock_full_finetune_quad_fast,
        }
    return mapping[resolved_profile]


def _combined_meta_path(combined_output_file: str) -> Path:
    return Path(f"{combined_output_file}.meta.json")


def _bundle_paths_or_legacy(bundle_name: str | None) -> dict[str, str]:
    if bundle_name and str(bundle_name).strip():
        bundle = resolve_bundle_paths(bundle_name)
        return {
            "bundle_name": bundle.bundle_name,
            "bundle_root": str(bundle.bundle_root),
            "export_dir": str(bundle.export_dir),
            "audio_root": str(bundle.audio_dir),
            "train_input_file": str(bundle.train_json),
            "val_input_file": str(bundle.val_json),
            "train_output_file": str(bundle.train_codes),
            "val_output_file": str(bundle.val_codes),
            "combined_output_file": str(bundle.combined_codes),
            "combined_meta_file": str(bundle.combined_meta),
        }
    return {
        "bundle_name": "",
        "bundle_root": "",
        "export_dir": DEFAULT_EXPORT_DIR,
        "audio_root": DEFAULT_AUDIO_ROOT,
        "train_input_file": DEFAULT_TRAIN_JSON,
        "val_input_file": DEFAULT_VAL_JSON,
        "train_output_file": DEFAULT_TRAIN_CODES,
        "val_output_file": DEFAULT_VAL_CODES,
        "combined_output_file": DEFAULT_COMBINED_CODES,
        "combined_meta_file": str(_combined_meta_path(DEFAULT_COMBINED_CODES)),
    }


def _default_bundle_tag(bundle_name: str) -> str:
    safe_bundle = validate_layout_name(bundle_name, kind="bundle_name")
    if safe_bundle.startswith("rad-"):
        return "rad"
    if "emotion" in safe_bundle:
        return "emotion"
    return _normalize_training_input_tag(safe_bundle)


def _resolve_bundle_training_input(bundle_name: str) -> dict[str, Any]:
    bundle = resolve_bundle_paths(bundle_name)
    heldout_manifest = str(resolve_existing_heldout_eval_manifest(bundle.export_dir))
    tag = _default_bundle_tag(bundle.bundle_name)
    return {
        "tag": tag,
        "bundle_name": bundle.bundle_name,
        "bundle_root": str(bundle.bundle_root),
        "export_dir": str(bundle.export_dir),
        "audio_root": str(bundle.audio_dir),
        "train_input_file": str(bundle.train_json),
        "val_input_file": str(bundle.val_json),
        "train_output_file": str(bundle.train_codes),
        "val_output_file": str(bundle.val_codes),
        "combined_output_file": str(bundle.combined_codes),
        "combined_meta_file": str(bundle.combined_meta),
        "heldout_eval_manifest": heldout_manifest,
    }


def _split_json_items(raw_value: str) -> list[Any]:
    value = str(raw_value or "").strip()
    if not value:
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError("Expected a JSON list")
    return list(parsed)


def _normalize_heldout_eval_stages(
    raw_value: Any,
    *,
    default: tuple[str, ...] = DEFAULT_HELDOUT_EVAL_STAGES,
) -> list[str]:
    if raw_value in (None, ""):
        return list(default)
    if isinstance(raw_value, str):
        raw_items = [raw_value]
    elif isinstance(raw_value, (list, tuple, set)):
        raw_items = list(raw_value)
    else:
        raw_items = [raw_value]
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        stage = str(raw_item or "").strip().lower().replace("_", "-")
        if not stage:
            continue
        if stage in {"all", "*"}:
            return ["step0", "checkpoint", "final"]
        aliases = {
            "step-0": "step0",
            "baseline": "step0",
            "save": "checkpoint",
            "save-hook": "checkpoint",
            "savehook": "checkpoint",
            "train-end": "final",
            "end": "final",
        }
        stage = aliases.get(stage, stage)
        if stage not in {"step0", "checkpoint", "final"}:
            raise ValueError(
                f"Unsupported held-out eval stage {raw_item!r}. Expected step0/checkpoint/final."
            )
        if stage not in seen:
            normalized.append(stage)
            seen.add(stage)
    return normalized or list(default)


def _heldout_eval_runs_at_stage(eval_spec: dict[str, Any], *, stage: str) -> bool:
    return stage in _normalize_heldout_eval_stages(eval_spec.get("when"))


def _heldout_evals_for_stage(
    heldout_evals: list[dict[str, Any]],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    return [
        dict(eval_spec)
        for eval_spec in heldout_evals
        if _heldout_eval_runs_at_stage(eval_spec, stage=stage)
    ]


def _checkpoint_stage_evals(heldout_evals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _heldout_evals_for_stage(heldout_evals, stage="checkpoint")


def _step0_stage_evals(heldout_evals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _heldout_evals_for_stage(heldout_evals, stage="step0")


def _final_stage_evals(heldout_evals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _heldout_evals_for_stage(heldout_evals, stage="final")


def _full_heldout_eval_tag(tag: str) -> str:
    base_tag = _safe_output_tag(str(tag or "heldout").strip() or "heldout")
    return f"{base_tag}-full"


def _full_heldout_metric_prefix(*, tag: str, sample_prefix: str = "") -> str:
    base_prefix = str(sample_prefix or "").strip().strip("/")
    if base_prefix:
        return f"{base_prefix}/full"
    base_tag = _safe_output_tag(str(tag or "heldout").strip() or "heldout")
    return f"{checkpoint_eval_metric_prefix(eval_tag=base_tag)}/full"


def _default_heldout_eval_specs(
    *,
    tag: str,
    manifest: str,
    wandb_metric_prefix: str = "",
    subset_size: int = 0,
    subset_seed: int | None = None,
) -> list[dict[str, Any]]:
    resolved_tag = _safe_output_tag(str(tag or "heldout").strip() or "heldout")
    resolved_manifest = str(manifest or "").strip()
    if not resolved_manifest:
        return []
    if subset_size > 0:
        sampled_manifest = _materialize_fixed_heldout_eval_subset(
            resolved_manifest,
            subset_size=int(subset_size),
            subset_seed=subset_seed,
        )
        return [
            {
                "tag": resolved_tag,
                "manifest": sampled_manifest,
                "wandb_metric_prefix": str(wandb_metric_prefix or "").strip(),
                "subset_size": int(subset_size),
                "subset_seed": subset_seed,
                "when": list(DEFAULT_HELDOUT_EVAL_STAGES),
            },
            {
                "tag": _full_heldout_eval_tag(resolved_tag),
                "manifest": resolved_manifest,
                "wandb_metric_prefix": _full_heldout_metric_prefix(
                    tag=resolved_tag,
                    sample_prefix=str(wandb_metric_prefix or "").strip(),
                ),
                "subset_size": None,
                "subset_seed": None,
                "when": list(FINAL_HELDOUT_EVAL_STAGES),
            },
        ]
    return [
        {
            "tag": resolved_tag,
            "manifest": resolved_manifest,
            "wandb_metric_prefix": str(wandb_metric_prefix or "").strip(),
            "subset_size": None,
            "subset_seed": subset_seed,
            "when": list(DEFAULT_HELDOUT_EVAL_STAGES),
        }
    ]


def _normalize_training_inputs_config(
    *,
    bundle_name: str = "",
    train_input_file: str = "",
    val_input_file: str = "",
    train_output_file: str = "",
    val_output_file: str = "",
    combined_output_file: str = "",
    training_inputs_json: str = "",
) -> list[dict[str, Any]]:
    explicit_items = _split_json_items(training_inputs_json)
    if explicit_items:
        normalized: list[dict[str, Any]] = []
        for index, raw_item in enumerate(explicit_items, start=1):
            if not isinstance(raw_item, dict):
                raise ValueError(
                    f"training_inputs_json[{index}] must be an object, got {type(raw_item).__name__}"
                )
            tag = _safe_output_tag(raw_item.get("tag") or raw_item.get("slug") or f"input-{index}")
            train_path = str(
                raw_item.get("train_json")
                or raw_item.get("train_input_file")
                or ""
            ).strip()
            if not train_path:
                raise ValueError(f"training input {tag!r} is missing train_json/train_input_file")
            val_path = str(
                raw_item.get("val_json")
                or raw_item.get("val_input_file")
                or ""
            ).strip()
            train_codes = str(
                raw_item.get("train_codes")
                or raw_item.get("train_output_file")
                or ""
            ).strip()
            val_codes = str(
                raw_item.get("val_codes")
                or raw_item.get("val_output_file")
                or ""
            ).strip()
            normalized.append(
                {
                    "tag": tag,
                    "bundle_name": str(raw_item.get("bundle_name") or "").strip() or None,
                    "bundle_root": str(raw_item.get("bundle_root") or "").strip() or "",
                    "export_dir": str(raw_item.get("export_dir") or "").strip() or "",
                    "audio_root": str(raw_item.get("audio_root") or "").strip() or DEFAULT_AUDIO_ROOT,
                    "train_input_file": train_path,
                    "val_input_file": val_path,
                    "train_output_file": train_codes,
                    "val_output_file": val_codes,
                    "heldout_eval_manifest": str(
                        raw_item.get("heldout_eval_manifest") or raw_item.get("manifest") or ""
                    ).strip()
                    or "",
                }
            )
        return normalized

    if bundle_name.strip():
        return [_resolve_bundle_training_input(bundle_name)]

    return [
        {
            "tag": "default",
            "bundle_name": None,
            "bundle_root": "",
            "export_dir": DEFAULT_EXPORT_DIR,
            "audio_root": DEFAULT_AUDIO_ROOT,
            "train_input_file": train_input_file.strip() or DEFAULT_TRAIN_JSON,
            "val_input_file": val_input_file.strip() or DEFAULT_VAL_JSON,
            "train_output_file": train_output_file.strip() or DEFAULT_TRAIN_CODES,
            "val_output_file": val_output_file.strip() or DEFAULT_VAL_CODES,
            "combined_output_file": combined_output_file.strip() or DEFAULT_COMBINED_CODES,
            "heldout_eval_manifest": "",
        }
    ]


def _normalize_heldout_evals_config(
    *,
    training_inputs: list[dict[str, Any]],
    bundle_name: str = "",
    heldout_eval_subset_size: int = 0,
    heldout_evals_json: str = "[]",
) -> list[dict[str, Any]]:
    # An explicit empty list disables in-training held-out eval entirely.
    # Without this, `[]` is indistinguishable from "unset" and falls through to
    # the bundle default below, so there is no way to turn the save-hook eval
    # off. That matters now that W&B is removed: the eval still runs inference
    # and pays for a GPT judge, then fails at finalize ("Expected an active W&B
    # run") and discards the result. Scoring happens separately, after training.
    if str(heldout_evals_json or "").strip() in {"[]", "none", "disabled"}:
        return []

    explicit_items = _split_json_items(heldout_evals_json)
    normalized: list[dict[str, Any]] = []
    if explicit_items:
        for index, raw_item in enumerate(explicit_items, start=1):
            if not isinstance(raw_item, dict):
                raise ValueError(
                    f"heldout_evals_json[{index}] must be an object, got {type(raw_item).__name__}"
                )
            tag = _safe_output_tag(raw_item.get("tag") or raw_item.get("slug") or f"eval-{index}")
            manifest = str(raw_item.get("manifest") or raw_item.get("heldout_eval_manifest") or "").strip()
            if not manifest:
                raise ValueError(f"heldout eval {tag!r} is missing manifest")
            subset_size_raw = raw_item.get("subset_size")
            if subset_size_raw in (None, ""):
                subset_size = int(heldout_eval_subset_size or 0)
            else:
                subset_size = int(subset_size_raw)
            if subset_size < 0:
                raise ValueError(f"heldout eval {tag!r} has negative subset_size={subset_size}")
            subset_seed_raw = raw_item.get("subset_seed")
            subset_seed = None if subset_seed_raw in (None, "") else int(subset_seed_raw)
            if subset_size > 0:
                manifest = _materialize_fixed_heldout_eval_subset(
                    manifest,
                    subset_size=subset_size,
                    subset_seed=subset_seed,
                )
            normalized.append(
                {
                    "tag": tag,
                    "manifest": manifest,
                    "wandb_metric_prefix": str(
                        raw_item.get("wandb_metric_prefix") or f"checkpoint_eval/{tag}"
                    ).strip(),
                    "subset_size": subset_size or None,
                    "subset_seed": subset_seed,
                    "when": _normalize_heldout_eval_stages(
                        raw_item.get("when") or raw_item.get("stages")
                    ),
                }
            )
        return normalized

    if bundle_name.strip():
        bundle_manifest = str(
            resolve_existing_heldout_eval_manifest(resolve_bundle_paths(bundle_name).export_dir)
        )
        return _default_heldout_eval_specs(
            tag=_default_bundle_tag(bundle_name),
            manifest=bundle_manifest,
            wandb_metric_prefix="",
            subset_size=int(heldout_eval_subset_size or 0),
        )

    for item in training_inputs:
        manifest = str(item.get("heldout_eval_manifest") or "").strip()
        if not manifest:
            continue
        normalized.extend(
            _default_heldout_eval_specs(
                tag=_safe_output_tag(item["tag"]),
                manifest=manifest,
                wandb_metric_prefix="",
                subset_size=int(heldout_eval_subset_size or 0),
            )
        )
    return normalized


def _multi_input_combined_defaults(
    training_inputs: list[dict[str, Any]],
    *,
    combined_output_file: str = "",
) -> dict[str, str]:
    explicit_combined = str(combined_output_file or "").strip()
    if explicit_combined:
        return {
            "combined_output_file": explicit_combined,
            "combined_meta_file": str(_combined_meta_path(explicit_combined)),
        }
    if len(training_inputs) == 1 and str(training_inputs[0].get("combined_output_file") or "").strip():
        combined_path = str(training_inputs[0]["combined_output_file"]).strip()
        return {
            "combined_output_file": combined_path,
            "combined_meta_file": str(_combined_meta_path(combined_path)),
        }
    return {
        "combined_output_file": DEFAULT_COMBINED_CODES,
        "combined_meta_file": str(_combined_meta_path(DEFAULT_COMBINED_CODES)),
    }


def _resolve_multi_training_layout(
    *,
    bundle_name: str = "",
    train_input_file: str = "",
    val_input_file: str = "",
    train_output_file: str = "",
    val_output_file: str = "",
    combined_output_file: str = "",
    data_file: str = "",
    heldout_eval_subset_size: int = 0,
    training_inputs_json: str = "",
    heldout_evals_json: str = "[]",
) -> dict[str, Any]:
    if heldout_eval_subset_size > 0 and not (
        bundle_name.strip() or training_inputs_json.strip() or heldout_evals_json.strip()
    ):
        raise ValueError(
            "heldout_eval_subset_size requires a bundle-backed or explicit held-out eval manifest."
        )
    training_inputs = _normalize_training_inputs_config(
        bundle_name=bundle_name,
        train_input_file=train_input_file,
        val_input_file=val_input_file,
        train_output_file=train_output_file,
        val_output_file=val_output_file,
        combined_output_file=combined_output_file,
        training_inputs_json=training_inputs_json,
    )
    combined_defaults = _multi_input_combined_defaults(
        training_inputs,
        combined_output_file=combined_output_file,
    )
    heldout_evals = _normalize_heldout_evals_config(
        training_inputs=training_inputs,
        bundle_name=bundle_name,
        heldout_eval_subset_size=heldout_eval_subset_size,
        heldout_evals_json=heldout_evals_json,
    )
    primary = dict(training_inputs[0]) if training_inputs else {}
    return {
        **primary,
        "bundle_name": (
            validate_layout_name(bundle_name, kind="bundle_name")
            if bundle_name.strip()
            else str(primary.get("bundle_name") or "")
        ),
        "bundle_root": str(primary.get("bundle_root") or ""),
        "export_dir": str(primary.get("export_dir") or DEFAULT_EXPORT_DIR),
        "audio_root": str(primary.get("audio_root") or DEFAULT_AUDIO_ROOT),
        "training_inputs": training_inputs,
        "heldout_evals": heldout_evals,
        "training_inputs_json": json.dumps(training_inputs, sort_keys=True),
        "heldout_evals_json": json.dumps(heldout_evals, sort_keys=True),
        "combined_output_file": combined_defaults["combined_output_file"],
        "combined_meta_file": combined_defaults["combined_meta_file"],
        "data_file": data_file.strip() or combined_defaults["combined_output_file"],
        "heldout_eval_manifest": (
            str(heldout_evals[0]["manifest"]).strip() if len(heldout_evals) == 1 else ""
        ),
        "is_multi_input": len(training_inputs) > 1 or len(heldout_evals) > 1,
    }


def _heldout_eval_subset_manifest_path(
    heldout_eval_manifest: str | Path,
    *,
    subset_size: int,
    subset_seed: int | None = None,
) -> Path:
    source = Path(heldout_eval_manifest)
    suffix = "".join(source.suffixes)
    stem = source.name[: -len(suffix)] if suffix else source.name
    resolved_suffix = suffix or ".jsonl"
    seed_suffix = f"_seed{subset_seed}" if subset_seed is not None else ""
    return source.with_name(f"{stem}_sample{subset_size}{seed_suffix}{resolved_suffix}")


def _materialize_fixed_heldout_eval_subset(
    heldout_eval_manifest: str | Path,
    *,
    subset_size: int,
    subset_seed: int | None = None,
) -> str:
    if subset_size <= 0:
        return str(heldout_eval_manifest)
    source = Path(heldout_eval_manifest).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target = _heldout_eval_subset_manifest_path(
        source,
        subset_size=subset_size,
        subset_seed=subset_seed,
    )
    all_lines: list[str] = []
    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            all_lines.append(raw_line if raw_line.endswith("\n") else raw_line + "\n")
    if len(all_lines) < subset_size:
        raise ValueError(
            f"Held-out eval manifest {source} only has {len(all_lines)} rows; "
            f"cannot materialize fixed subset of {subset_size}."
        )
    rng = random.Random(1 if subset_seed is None else subset_seed)
    kept_indices = sorted(rng.sample(range(len(all_lines)), subset_size))
    kept_lines = [all_lines[index] for index in kept_indices]
    target.write_text("".join(kept_lines), encoding="utf-8")
    return str(target)


def _resolve_training_paths(
    *,
    bundle_name: str = "",
    train_input_file: str = "",
    val_input_file: str = "",
    train_output_file: str = "",
    val_output_file: str = "",
    combined_output_file: str = "",
    data_file: str = "",
    heldout_eval_subset_size: int = 0,
    training_inputs_json: str = "",
    heldout_evals_json: str = "[]",
) -> dict[str, Any]:
    return _resolve_multi_training_layout(
        bundle_name=bundle_name,
        train_input_file=train_input_file,
        val_input_file=val_input_file,
        train_output_file=train_output_file,
        val_output_file=val_output_file,
        combined_output_file=combined_output_file,
        data_file=data_file,
        heldout_eval_subset_size=heldout_eval_subset_size,
        training_inputs_json=training_inputs_json,
        heldout_evals_json=heldout_evals_json,
    )


def _normalize_bundle_cli_path_overrides(
    *,
    bundle_name: str,
    train_input_file: str,
    val_input_file: str,
    train_output_file: str,
    val_output_file: str,
    combined_output_file: str,
    data_file: str,
) -> dict[str, str]:
    """Ignore legacy CLI defaults when a bundle name is provided."""
    if not bundle_name.strip():
        return {
            "train_input_file": train_input_file,
            "val_input_file": val_input_file,
            "train_output_file": train_output_file,
            "val_output_file": val_output_file,
            "combined_output_file": combined_output_file,
            "data_file": data_file,
        }

    def _clear_legacy_default(value: str, legacy_default: str) -> str:
        return "" if value.strip() == legacy_default else value

    return {
        "train_input_file": _clear_legacy_default(train_input_file, DEFAULT_TRAIN_JSON),
        "val_input_file": _clear_legacy_default(val_input_file, DEFAULT_VAL_JSON),
        "train_output_file": _clear_legacy_default(train_output_file, DEFAULT_TRAIN_CODES),
        "val_output_file": _clear_legacy_default(val_output_file, DEFAULT_VAL_CODES),
        "combined_output_file": _clear_legacy_default(
            combined_output_file, DEFAULT_COMBINED_CODES
        ),
        "data_file": _clear_legacy_default(data_file, DEFAULT_COMBINED_CODES),
    }


def _bundle_remote_targets(bundle_name: str) -> tuple[str, str]:
    bundle = resolve_bundle_paths(bundle_name)
    return (
        volume_relative(bundle.bundle_root),
        volume_relative(bundle.audio_dir),
    )


def _ensure_bundle_destination_available(
    bundle_name: str,
    *,
    modal_env: str,
    overwrite_bundle: bool = False,
) -> None:
    bundle = resolve_bundle_paths(bundle_name)
    remote_bundle_root = volume_relative(bundle.bundle_root)
    if remote_path_exists(remote_bundle_root, modal_env=modal_env):
        if overwrite_bundle:
            volume_rm(remote_bundle_root, modal_env=modal_env, recursive=True)
            return
        raise FileExistsError(
            f"Bundle already exists at {remote_bundle_root}. Pass overwrite_bundle to replace it."
        )


def _prepare_fixed_eval_jsonl(
    *,
    train_codes_file: str,
    val_codes_file: str,
    combined_output_file: str,
) -> dict[str, Any]:
    combined_path = Path(combined_output_file)
    combined_path.parent.mkdir(parents=True, exist_ok=True)

    train_count = 0
    val_count = 0
    with combined_path.open("w", encoding="utf-8") as fout:
        # Put validation rows first so Moonshot's eval_ratio split is exact.
        for src_path, counter_name in (
            (val_codes_file, "val"),
            (train_codes_file, "train"),
        ):
            with open(src_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    fout.write(line if line.endswith("\n") else line + "\n")
                    if counter_name == "val":
                        val_count += 1
                    else:
                        train_count += 1

    total_count = train_count + val_count
    if total_count <= 0 or train_count <= 0:
        raise RuntimeError(
            f"Bad fixed-eval layout: train={train_count} val={val_count} total={total_count}"
        )

    meta = {
        "train_count": train_count,
        "val_count": val_count,
        "combined_count": total_count,
        "eval_ratio": val_count / total_count,
        "train_codes_file": train_codes_file,
        "val_codes_file": val_codes_file,
        "combined_output_file": combined_output_file,
    }
    _combined_meta_path(combined_output_file).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def _tag_semantic_code_rows(*, input_path: str, output_path: str, source_tag: str) -> int:
    count = 0
    source_value = _safe_output_tag(source_tag)
    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta = dict(row.get("_meta") or {})
            meta["source_tag"] = source_value
            row["_meta"] = meta
            fout.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def _merge_semantic_code_groups(
    *,
    val_code_files: list[dict[str, Any]],
    train_code_files: list[dict[str, Any]],
    combined_output_file: str,
) -> dict[str, Any]:
    combined_path = Path(combined_output_file)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    train_count = 0
    val_count = 0
    by_source: dict[str, dict[str, int]] = {}
    with combined_path.open("w", encoding="utf-8") as fout:
        for group_name, items in (("val", val_code_files), ("train", train_code_files)):
            for item in items:
                tag = _safe_output_tag(item["tag"])
                source_counts = by_source.setdefault(tag, {"train_count": 0, "val_count": 0})
                with open(item["path"], "r", encoding="utf-8") as fin:
                    for raw_line in fin:
                        if not raw_line.strip():
                            continue
                        fout.write(raw_line if raw_line.endswith("\n") else raw_line + "\n")
                        if group_name == "val":
                            val_count += 1
                            source_counts["val_count"] += 1
                        else:
                            train_count += 1
                            source_counts["train_count"] += 1
    total_count = train_count + val_count
    if total_count <= 0 or train_count <= 0:
        raise RuntimeError(
            f"Bad fixed-eval layout: train={train_count} val={val_count} total={total_count}"
        )
    meta = {
        "train_count": train_count,
        "val_count": val_count,
        "combined_count": total_count,
        "eval_ratio": val_count / total_count,
        "combined_output_file": combined_output_file,
        "training_inputs": [
            {
                "tag": tag,
                "train_count": counts["train_count"],
                "val_count": counts["val_count"],
            }
            for tag, counts in by_source.items()
        ],
    }
    _combined_meta_path(combined_output_file).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def _trainer_summary(output_dir: str) -> dict[str, Any]:
    out_dir = Path(output_dir)
    summary: dict[str, Any] = {
        "output_dir": output_dir,
        "checkpoint_dirs": sorted(
            str(path) for path in out_dir.glob("checkpoint-*") if path.is_dir()
        ),
    }
    state_path = out_dir / "trainer_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        summary["global_step"] = state.get("global_step")
        summary["best_metric"] = state.get("best_metric")
        summary["best_model_checkpoint"] = state.get("best_model_checkpoint")
        summary["log_history_tail"] = state.get("log_history", [])[-10:]
    return summary


def _looks_like_lora_adapter_dir(path: str) -> bool:
    candidate = Path(path)
    return any(
        (candidate / filename).is_file()
        for filename in (
            "adapter_config.json",
            "adapter_model.bin",
            "adapter_model.safetensors",
        )
    )


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_local_command(cmd: list[str], *, cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
        env={**os.environ},
        check=False,
    )
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{out}"
        )
    return out


def _deploy_and_validate_kimi_ft_serving(
    *,
    modal_env: str,
    model_path_override: str,
) -> dict[str, Any]:
    raise RuntimeError("Use the public checkpoint serving apps in harness/modal_serving and the evaluation harness; legacy serving validation is not included.")


def _conversation_audio_paths(row: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for message in row.get("conversation") or []:
        if not isinstance(message, dict):
            continue
        if str(message.get("message_type") or "") != "audio":
            continue
        paths.append(str(message.get("content") or ""))
    return paths


def _row_audio_paths(row: dict[str, Any]) -> list[str]:
    conversation_paths = _conversation_audio_paths(row)
    if conversation_paths:
        return conversation_paths
    audio_paths = row.get("audio_path")
    if isinstance(audio_paths, list):
        return [str(path or "") for path in audio_paths if str(path or "").strip()]
    if isinstance(audio_paths, str) and audio_paths.strip():
        return [audio_paths]
    return []


def _row_local_audio_paths(row: dict[str, Any]) -> list[str]:
    for meta_key in ("_meta", "meta"):
        metadata = row.get(meta_key)
        if not isinstance(metadata, dict):
            continue
        local_audio_paths = metadata.get("local_audio_paths")
        if isinstance(local_audio_paths, list):
            return [str(path or "") for path in local_audio_paths if str(path or "").strip()]
    return []


def _jsonl_row_count(path: str | Path) -> int:
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    with candidate.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _summarize_export_jsonl(
    path: Path,
    *,
    expected_audio_root: str,
    require_volume_audio: bool,
    local_audio_dir: Path | None = None,
    allow_empty: bool = False,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_prefix = expected_audio_root.rstrip("/") + "/"
    local_audio_root = local_audio_dir.resolve() if local_audio_dir is not None else None

    line_count = 0
    unique_audio_paths: set[str] = set()
    unique_local_audio_paths: set[str] = set()
    audio_message_count_histogram: dict[int, int] = {}
    invalid_records: list[dict[str, Any]] = []
    missing_volume_audio: list[dict[str, Any]] = []
    missing_local_audio: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            audio_paths = _row_audio_paths(row)
            audio_count = len(audio_paths)
            audio_message_count_histogram[audio_count] = (
                audio_message_count_histogram.get(audio_count, 0) + 1
            )
            if audio_count <= 0:
                invalid_records.append(
                    {
                        "line": line_no,
                        "reason": f"expected at least 1 audio message, got {audio_count}",
                    }
                )
                continue

            for audio_path in audio_paths:
                unique_audio_paths.add(audio_path)
                if not audio_path.startswith(expected_prefix):
                    invalid_records.append(
                        {
                            "line": line_no,
                            "reason": "audio path outside expected Modal root",
                            "audio_path": audio_path,
                        }
                    )
                    continue
                if require_volume_audio and not Path(audio_path).is_file():
                    missing_volume_audio.append(
                        {
                            "line": line_no,
                            "audio_path": audio_path,
                        }
                    )

            if local_audio_root is not None:
                local_audio_paths = _row_local_audio_paths(row)
                if len(local_audio_paths) != audio_count:
                    invalid_records.append(
                        {
                            "line": line_no,
                            "reason": (
                                "expected _meta.local_audio_paths to match audio message count "
                                f"({audio_count}), got {len(local_audio_paths)}"
                            ),
                        }
                    )
                else:
                    for local_audio_path in local_audio_paths:
                        resolved_local = Path(str(local_audio_path)).resolve()
                        unique_local_audio_paths.add(str(resolved_local))
                        if not resolved_local.is_file():
                            missing_local_audio.append(
                                {
                                    "line": line_no,
                                    "audio_path": str(resolved_local),
                                }
                            )
                            continue
                        if not resolved_local.is_relative_to(local_audio_root):
                            invalid_records.append(
                                {
                                    "line": line_no,
                                    "reason": "local audio path escaped staged audio dir",
                                    "audio_path": str(resolved_local),
                                }
                            )

            line_count += 1

    if line_count <= 0 and allow_empty:
        summary: dict[str, Any] = {
            "path": str(path),
            "row_count": 0,
            "unique_modal_audio_paths": 0,
            "expected_audio_root": expected_audio_root,
            "audio_message_count_histogram": {},
        }
        if local_audio_root is not None:
            summary["unique_local_audio_paths"] = 0
            summary["local_audio_dir"] = str(local_audio_root)
        return summary
    if line_count <= 0:
        raise RuntimeError(f"No rows found in {path}")
    if invalid_records:
        raise RuntimeError(
            f"Invalid staged export rows in {path}: {json.dumps(invalid_records[:10], indent=2)}"
        )
    if missing_volume_audio:
        raise RuntimeError(
            f"Missing Modal audio files referenced by {path}: "
            f"{json.dumps(missing_volume_audio[:10], indent=2)}"
        )
    if missing_local_audio:
        raise RuntimeError(
            f"Missing staged local audio files referenced by {path}: "
            f"{json.dumps(missing_local_audio[:10], indent=2)}"
        )

    summary: dict[str, Any] = {
        "path": str(path),
        "row_count": line_count,
        "unique_modal_audio_paths": len(unique_audio_paths),
        "expected_audio_root": expected_audio_root,
        "audio_message_count_histogram": dict(
            sorted(audio_message_count_histogram.items())
        ),
    }
    if local_audio_root is not None:
        summary["unique_local_audio_paths"] = len(unique_local_audio_paths)
        summary["local_audio_dir"] = str(local_audio_root)
    return summary


def _resolve_staged_bundle(staged_bundle_dir: str) -> tuple[Path, Path, Path]:
    bundle_root = Path(staged_bundle_dir).expanduser().resolve()
    export_dir = bundle_root / "rad_exports" / "current"
    audio_dir = bundle_root / "audio"
    for required_dir in (bundle_root / "rad_exports", export_dir, audio_dir):
        if not required_dir.is_dir():
            raise FileNotFoundError(required_dir)
    for required_file in (
        export_dir / "rad_kimi_train.jsonl",
        export_dir / "rad_kimi_val.jsonl",
        resolve_existing_heldout_eval_manifest(export_dir),
        export_dir / "manifest.json",
    ):
        if not required_file.is_file():
            raise FileNotFoundError(required_file)
    return bundle_root, export_dir, audio_dir


def _rewrite_staged_bundle_audio_paths(
    export_dir: Path,
    *,
    src_audio_root: str,
    dst_audio_root: str,
) -> tuple[Path, Path, Path, Path]:
    upload_root = export_dir.parent.parent / ".modal_upload_bundle"
    rewritten_dir = upload_root / "rad_exports" / "current"
    if upload_root.exists():
        shutil.rmtree(upload_root)
    if rewritten_dir.exists():
        shutil.rmtree(rewritten_dir)
    rewritten_dir.mkdir(parents=True, exist_ok=True)
    src_prefix = src_audio_root.rstrip("/") + "/"
    dst_prefix = dst_audio_root.rstrip("/") + "/"
    outputs: list[Path] = []
    for input_path in (
        export_dir / "rad_kimi_train.jsonl",
        export_dir / "rad_kimi_val.jsonl",
    ):
        output_path = rewritten_dir / input_path.name
        with input_path.open("r", encoding="utf-8") as fin, output_path.open(
            "w",
            encoding="utf-8",
        ) as fout:
            for raw_line in fin:
                line = raw_line.strip()
                if not line:
                    continue
                row = json.loads(line)
                for message in row.get("conversation") or []:
                    if not isinstance(message, dict):
                        continue
                    if str(message.get("message_type") or "") != "audio":
                        continue
                    content = str(message.get("content") or "")
                    if not content.startswith(src_prefix):
                        raise RuntimeError(
                            f"Expected staged audio path under {src_audio_root}, got {content}"
                        )
                    message["content"] = dst_prefix + content[len(src_prefix) :]
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        outputs.append(output_path)
    heldout_input_path = resolve_existing_heldout_eval_manifest(export_dir)
    heldout_output_path = rewritten_dir / heldout_input_path.name
    with heldout_input_path.open("r", encoding="utf-8") as fin, heldout_output_path.open(
        "w",
        encoding="utf-8",
    ) as fout:
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            audio_paths = row.get("audio_path")
            rewritten_audio_paths: list[str] = []
            if isinstance(audio_paths, list):
                path_iter = audio_paths
            else:
                path_iter = [audio_paths]
            for raw_audio_path in path_iter:
                content = str(raw_audio_path or "")
                if not content.startswith(src_prefix):
                    raise RuntimeError(
                        f"Expected staged held-out audio path under {src_audio_root}, got {content}"
                    )
                rewritten_audio_paths.append(dst_prefix + content[len(src_prefix) :])
            row["audio_path"] = (
                rewritten_audio_paths
                if isinstance(audio_paths, list)
                else rewritten_audio_paths[0]
            )
            for message in row.get("conversation") or []:
                if not isinstance(message, dict):
                    continue
                if str(message.get("message_type") or "") != "audio":
                    continue
                content = str(message.get("content") or "")
                if not content.startswith(src_prefix):
                    raise RuntimeError(
                        "Expected staged held-out conversation audio path under "
                        f"{src_audio_root}, got {content}"
                    )
                message["content"] = dst_prefix + content[len(src_prefix) :]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    shutil.copy2(export_dir / "manifest.json", rewritten_dir / "manifest.json")
    return upload_root / "rad_exports", outputs[0], outputs[1], heldout_output_path


def _sync_staged_bundle_to_volume(
    *,
    staged_bundle_dir: str,
    modal_env: str,
    bundle_name: str,
    train_input_file: str,
    val_input_file: str,
    audio_root: str,
    overwrite_bundle: bool = False,
) -> dict[str, Any]:
    safe_bundle_name = validate_layout_name(bundle_name, kind="bundle_name")
    bundle_root, export_dir, audio_dir = _resolve_staged_bundle(staged_bundle_dir)
    _ensure_bundle_destination_available(
        safe_bundle_name,
        modal_env=modal_env,
        overwrite_bundle=overwrite_bundle,
    )
    remote_bundle_root, remote_audio_root = _bundle_remote_targets(safe_bundle_name)
    mounted_audio_root = str(resolve_bundle_paths(safe_bundle_name).audio_dir)
    (
        upload_rad_exports_dir,
        rewritten_train_json,
        rewritten_val_json,
        rewritten_heldout_manifest,
    ) = _rewrite_staged_bundle_audio_paths(
        export_dir,
        src_audio_root=audio_root,
        dst_audio_root=mounted_audio_root,
    )
    train_summary = _summarize_export_jsonl(
        rewritten_train_json,
        expected_audio_root=mounted_audio_root,
        require_volume_audio=False,
        local_audio_dir=audio_dir,
    )
    val_summary = _summarize_export_jsonl(
        rewritten_val_json,
        expected_audio_root=mounted_audio_root,
        require_volume_audio=False,
        local_audio_dir=audio_dir,
        allow_empty=True,
    )
    heldout_summary = _summarize_export_jsonl(
        rewritten_heldout_manifest,
        expected_audio_root=mounted_audio_root,
        require_volume_audio=False,
        local_audio_dir=audio_dir,
        allow_empty=False,
    )

    put_outputs = {
        "rad_exports": _run_local_command(
            [
                "modal",
                "volume",
                "put",
                "--env",
                modal_env,
                "--force",
                VOLUME_NAME,
                str(upload_rad_exports_dir),
                f"{remote_bundle_root}/rad_exports",
            ]
        )[-4000:],
        "audio": _run_local_command(
            [
                "modal",
                "volume",
                "put",
                "--env",
                modal_env,
                "--force",
                VOLUME_NAME,
                str(bundle_root / "audio"),
                remote_audio_root,
            ]
        )[-4000:],
    }
    remote_summary = verify_volume_training_inputs.remote(
        train_input_file=train_input_file,
        val_input_file=val_input_file,
        heldout_eval_manifest=str(
            resolve_existing_heldout_eval_manifest(
                resolve_bundle_paths(safe_bundle_name).export_dir
            )
        ),
        audio_root=mounted_audio_root,
    )
    return {
        "status": "ok",
        "modal_env": modal_env,
        "volume_name": VOLUME_NAME,
        "bundle_name": safe_bundle_name,
        "staged_bundle_dir": str(bundle_root),
        "remote_bundle_root": remote_bundle_root,
        "local_train_summary": train_summary,
        "local_val_summary": val_summary,
        "local_heldout_summary": heldout_summary,
        "remote_summary": remote_summary,
        "put_output_tail": put_outputs,
    }


def _copy_kimi_runtime_assets(model_name: str, output_dir: str) -> list[str]:
    """Copy tokenizer/runtime assets directly from the HF snapshot.

    Kimi's tokenizer implementation does not reliably support `save_pretrained`,
    so packaging copies the required runtime files straight from the source repo.
    """
    from huggingface_hub import snapshot_download

    model_path = Path(model_name)
    snapshot_dir = model_path if model_path.exists() else Path(snapshot_download(model_name))
    output_path = Path(output_dir)
    copied: list[str] = []
    for rel_path in (
        "tokenization_kimia.py",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tiktoken.model",
        "generation_config.json",
        "audio_detokenizer",
        "vocoder",
    ):
        src = snapshot_dir / rel_path
        dest = output_path / rel_path
        if src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            copied.append(rel_path)
        elif src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(rel_path)
    return copied


def _write_lora_single_gpu_deepspeed_config(output_dir: str) -> str:
    """Single-GPU DeepSpeed config for LoRA-only RAD SFT."""
    config = {
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "bf16": {"enabled": "auto"},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            },
        },
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": False,
            "contiguous_gradients": True,
            "reduce_bucket_size": "auto",
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 100,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }
    config_path = Path(output_dir) / "ds_config_lora_zero2.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return str(config_path)


def _write_stock_single_gpu_deepspeed_config(output_dir: str) -> str:
    """Single-GPU stock full-FT config with CPU optimizer offload.

    Upstream's ZeRO-3 config assumes distributed sharding. On a single GPU, the
    full optimizer state lands on device at the first optimizer step and can OOM
    even on H200. CPU optimizer offload keeps the stock training path intact
    while making the diagnostic runnable on one accelerator.
    """
    config = {
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "bf16": {"enabled": "auto"},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            },
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True,
            },
            "offload_param": {
                "device": "none",
                "pin_memory": True,
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 100,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }
    config_path = Path(output_dir) / "ds_config_stock_zero3_cpuopt.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return str(config_path)


def _write_stock_multigpu_deepspeed_config(output_dir: str) -> str:
    """Multi-GPU stock full-FT config without CPU optimizer offload.

    Kimi's upstream training path can exercise different parameter access orders
    across ranks for variable-media samples. ZeRO-3 shards model parameters and
    inserts per-parameter all-gathers, which can deadlock when those orders
    diverge. ZeRO-2 still shards optimizer state/gradients across GPUs while
    keeping full parameters replicated on each rank, which is substantially
    more robust for this workload on dual/quad H200.
    """
    config = {
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "bf16": {"enabled": "auto"},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": "auto",
                "eps": "auto",
                "weight_decay": "auto",
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": "auto",
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            },
        },
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {
                "device": "none",
                "pin_memory": False,
            },
            "allgather_partitions": True,
            "allgather_bucket_size": 200000000,
            "overlap_comm": True,
            "reduce_scatter": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 200000000,
        },
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "steps_per_print": 100,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False,
    }
    config_path = Path(output_dir) / "ds_config_stock_zero2_multigpu.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return str(config_path)


def _normalized_pkg_version(version: str) -> str:
    """Strip local build suffixes like `+cu124` before strict pin checks."""
    return str(version).split("+", 1)[0]


def _stock_full_ft_runtime_versions() -> dict[str, str]:
    import deepspeed
    import torch
    import torchaudio
    import torchvision
    import transformers

    versions = {
        "torch": str(torch.__version__),
        "torchaudio": str(torchaudio.__version__),
        "torchvision": str(torchvision.__version__),
        "transformers": str(transformers.__version__),
        "deepspeed": str(deepspeed.__version__),
    }
    if _normalized_pkg_version(versions["torch"]) != STOCK_TORCH_VERSION:
        raise RuntimeError(
            "Unexpected stock full-FT torch version: "
            f"{versions['torch']} != {STOCK_TORCH_VERSION}"
        )
    if _normalized_pkg_version(versions["torchaudio"]) != STOCK_TORCH_VERSION:
        raise RuntimeError(
            "Unexpected stock full-FT torchaudio version: "
            f"{versions['torchaudio']} != {STOCK_TORCH_VERSION}"
        )
    if _normalized_pkg_version(versions["torchvision"]) != STOCK_TORCHVISION_VERSION:
        raise RuntimeError(
            "Unexpected stock full-FT torchvision version: "
            f"{versions['torchvision']} != {STOCK_TORCHVISION_VERSION}"
        )
    if versions["transformers"] != STOCK_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Unexpected stock full-FT transformers version: "
            f"{versions['transformers']} != {STOCK_TRANSFORMERS_VERSION}"
        )
    if versions["deepspeed"] != STOCK_DEEPSPEED_VERSION:
        raise RuntimeError(
            "Unexpected stock full-FT deepspeed version: "
            f"{versions['deepspeed']} != {STOCK_DEEPSPEED_VERSION}"
        )
    return versions


def _training_eval_strategy_flag() -> str:
    import transformers

    fields = getattr(transformers.TrainingArguments, "__dataclass_fields__", {})
    if "eval_strategy" in fields:
        return "--eval_strategy"
    if "evaluation_strategy" in fields:
        return "--evaluation_strategy"
    raise RuntimeError("Could not resolve TrainingArguments eval strategy flag")


def _checkpoint_eval_runtime_env(run_paths) -> dict[str, str]:
    cache_dir = run_paths.eval_root / "choice_extraction_cache"
    return {
        "AUDIOBENCH_LLM_EXTRACTION_CONCURRENCY": str(
            DEFAULT_LLM_EXTRACTION_CONCURRENCY
        ),
        "AUDIOBENCH_LLM_EXTRACTION_CACHE_DIR": str(cache_dir),
        "AUDIOBENCH_LLM_EXTRACTION_FAIL_OPEN": "1",
        "AUDIOBENCH_JUDGE_RESPONSE_RETRIES": str(
            DEFAULT_JUDGE_RESPONSE_RETRIES
        ),
        "AUDIOBENCH_JUDGE_RESPONSE_RETRY_SECONDS": str(
            DEFAULT_JUDGE_RESPONSE_RETRY_SECONDS
        ),
        "AUDIOBENCH_CHECKPOINT_EVAL_NONFATAL_FINALIZE_ERRORS": "1",
    }


@app.function(
    image=stock_full_finetune_image,
    timeout=3600,
    retries=0,
)
def validate_stock_full_finetune_env() -> dict[str, Any]:
    versions = _stock_full_ft_runtime_versions()
    help_text = _run_in_kimi_repo(["python", "finetune.py", "--help"])
    return {
        "status": "ok",
        "versions": versions,
        "eval_strategy_flag": _training_eval_strategy_flag(),
        "help_tail": help_text[-4000:],
    }


@app.function(
    image=finetune_image,
    timeout=300,
    retries=0,
)
def validate_lora_finetune_eval_imports() -> dict[str, str]:
    """Smoke: same metric imports as checkpoint-eval finalize (pip_install editdistance/zhconv layer)."""
    import editdistance  # noqa: F401
    import zhconv  # noqa: F401
    import harness.judge  # noqa: F401

    return {"status": "ok", "image": "finetune_image"}


@app.function(
    image=finetune_image,
    gpu=TRAIN_GPU,
    volumes={
        DATA_MOUNT: rad_volume,
        HF_CACHE_MOUNT: hf_cache,
    },
    secrets=[*hf_secrets, *openai_training_secrets],
    timeout=86400,
    retries=0,
)
def evaluate_local_checkpoint_path(
    bundle_name: str = "",
    heldout_eval_manifest: str = "",
    model_dir: str = DEFAULT_PRETRAINED_PATH,
    pretrained_path: str = DEFAULT_PRETRAINED_PATH,
    model_name_or_path: str = DEFAULT_KIMI_MODEL_NAME,
    run_name: str = "",
    inference_mode_override: str = "",
    direct_mqa_scoring: bool = False,
) -> dict[str, Any]:
    """Run the local checkpoint-eval path on a full heldout manifest."""

    manifest = str(heldout_eval_manifest or "").strip()
    if not manifest:
        manifest = _resolve_training_paths(
            bundle_name=str(bundle_name or "").strip(),
            heldout_eval_subset_size=0,
        ).get("heldout_eval_manifest", "")
    manifest = str(manifest or "").strip()
    if not manifest:
        raise ValueError(
            "evaluate_local_checkpoint_path requires --heldout-eval-manifest or "
            "--bundle-name with a heldout_eval_manifest.jsonl export."
        )
    manifest_path = Path(manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    run_slug = str(run_name or "").strip() or f"heldout-local-{time.strftime('%Y%m%d-%H%M%S')}"
    validate_layout_name(run_slug, kind="run_name")
    work_root = Path(DATA_MOUNT) / "tmp" / "heldout-local-evals" / run_slug
    local_eval_dir = work_root / "local-loop"
    partial_root = local_eval_dir / "_partials"
    partial_root.mkdir(parents=True, exist_ok=True)

    configured_model_dir = str(model_dir or "").strip()
    model_dir_path = Path(configured_model_dir) if configured_model_dir else None
    checkpoint_dir_path = model_dir_path if model_dir_path and model_dir_path.is_dir() else (work_root / "checkpoint-source")
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    model_load_source = configured_model_dir
    if not model_load_source or not Path(model_load_source).exists():
        model_load_source = str(model_name_or_path or "").strip() or DEFAULT_KIMI_MODEL_NAME
    tokenizer_source = str(pretrained_path or "").strip() or model_load_source

    _ensure_kimi_repo_on_sys_path()
    from finetuning.kimi_audio.vendor_kimi.finetune_codes.model import KimiAudioModel

    import torch

    requested_inference_mode = str(inference_mode_override or "").strip()
    supported_inference_modes = {
        CHECKPOINT_EVAL_INFERENCE_MODE_KIMI_GENERATE,
        CHECKPOINT_EVAL_INFERENCE_MODE_LOCAL_LOOP,
    }
    if requested_inference_mode and requested_inference_mode not in supported_inference_modes:
        raise ValueError(
            f"Unsupported inference_mode_override={requested_inference_mode!r}; "
            f"expected one of {sorted(supported_inference_modes)}"
        )
    packaged_checkpoint = (checkpoint_dir_path / "modeling_moonshot_kimia.py").is_file()
    inference_mode = requested_inference_mode or (
        CHECKPOINT_EVAL_INFERENCE_MODE_KIMI_GENERATE
        if packaged_checkpoint
        else CHECKPOINT_EVAL_INFERENCE_MODE_LOCAL_LOOP
    )
    checkpoint_eval_name = (
        "base-model-kimi-generate"
        if inference_mode == CHECKPOINT_EVAL_INFERENCE_MODE_KIMI_GENERATE
        else "base-model-local-loop"
    )
    print(
        f"[checkpoint-eval] local-loop heldout eval starting manifest={manifest} "
        f"model_load_source={model_load_source} inference_mode={inference_mode}",
        flush=True,
    )
    model = None
    try:
        if inference_mode == CHECKPOINT_EVAL_INFERENCE_MODE_LOCAL_LOOP:
            model = KimiAudioModel.from_pretrained(
                model_load_source,
                device_map=None,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            if torch.cuda.is_available():
                model.to(torch.cuda.current_device())
            model.eval()
        evaluate_checkpoint(
            heldout_eval_manifest=manifest,
            model=model,
            pretrained_path=tokenizer_source,
            model_name_or_path=model_name_or_path,
            checkpoint_dir=str(checkpoint_dir_path),
            output_root=str(partial_root),
            checkpoint_name=checkpoint_eval_name,
            checkpoint_step=0,
            run_name=run_slug,
            world_size=1,
            rank=0,
            inference_mode=inference_mode,
        )
        if direct_mqa_scoring:
            partial_paths = sorted(partial_root.glob("rank-*.jsonl"))
            direct_rows: list[dict[str, Any]] = []
            for partial_path in partial_paths:
                with partial_path.open(encoding="utf-8") as handle:
                    direct_rows.extend(
                        json.loads(line) for line in handle if line.strip()
                    )
            direct_rows.sort(key=lambda row: int(row.get("index", 0)))
            unparseable: list[dict[str, Any]] = []
            correct = 0
            choice_counts: Counter[str] = Counter()
            for row in direct_rows:
                prediction = str(row.get("prediction") or "")
                match = re.match(
                    r"^\s*\(?([A-Z])\)?(?:[.)]|\b)",
                    prediction,
                    flags=re.IGNORECASE,
                )
                chosen = match.group(1).upper() if match else None
                answer_key = str((row.get("meta") or {}).get("answer_key") or "")
                if chosen is None:
                    unparseable.append(
                        {
                            "example_id": row.get("example_id"),
                            "prediction": prediction,
                        }
                    )
                else:
                    choice_counts[chosen] += 1
                is_correct = chosen is not None and chosen == answer_key
                correct += is_correct
                row["chosen"] = chosen
                row["correct"] = answer_key
                row["is_correct"] = is_correct
            direct_result_path = local_eval_dir / "direct_mqa_results.jsonl"
            direct_summary_path = local_eval_dir / "direct_mqa_summary.json"
            with direct_result_path.open("w", encoding="utf-8") as handle:
                for row in direct_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            direct_summary = {
                "status": "ok",
                "run_name": run_slug,
                "heldout_eval_manifest": manifest,
                "model_load_source": model_load_source,
                "inference_mode": inference_mode,
                "total": len(direct_rows),
                "correct": correct,
                "accuracy_pct": (
                    100.0 * correct / len(direct_rows) if direct_rows else None
                ),
                "unparseable": len(unparseable),
                "unparseable_examples": unparseable[:10],
                "choice_counts": dict(sorted(choice_counts.items())),
                "result_file": str(direct_result_path),
                "summary_path": str(direct_summary_path),
            }
            direct_summary_path.write_text(
                json.dumps(direct_summary, indent=2) + "\n",
                encoding="utf-8",
            )
            rad_volume.commit()
            return direct_summary
        result = finalize_checkpoint_evaluation(
            heldout_eval_manifest=manifest,
            output_root=str(partial_root),
            checkpoint_dir=str(checkpoint_dir_path),
            checkpoint_name=checkpoint_eval_name,
            checkpoint_step=0,
            run_name=run_slug,
            checkpoint_metrics_path=str(local_eval_dir / "checkpoint_metrics.jsonl"),
            eval_repeat_dir=str(local_eval_dir),
        )
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_row = dict(result.get("metrics_row") or {})
    return {
        "status": "ok",
        "run_name": run_slug,
        "heldout_eval_manifest": manifest,
        "model_load_source": model_load_source,
        "inference_mode": inference_mode,
        "checkpoint_dir": str(checkpoint_dir_path),
        "summary_path": result.get("summary_path"),
        "combined_summary_path": result.get("combined_summary_path"),
        "result_file": result.get("result_file"),
        "performance_file": result.get("performance_file"),
        "judge_file": result.get("judge_file"),
        "overall_accuracy_pct": metrics_row.get("overall_accuracy_pct"),
        "accuracy_on_valid_pct": metrics_row.get("accuracy_on_valid_pct"),
        "unparseable_or_unsure_pct": metrics_row.get("unparseable_or_unsure_pct"),
    }


@app.function(
    image=stock_full_finetune_image,
    timeout=300,
    retries=0,
)
def validate_stock_finetune_eval_imports() -> dict[str, str]:
    """Same as validate_lora_finetune_eval_imports for stock_full_finetune_image (no finetune.py import)."""
    import editdistance  # noqa: F401
    import zhconv  # noqa: F401
    import harness.judge  # noqa: F401

    return {"status": "ok", "image": "stock_full_finetune_image"}


@app.function(
    image=finetune_image,
    gpu=TRAIN_GPU,
    volumes={
        DATA_MOUNT: rad_volume,
        HF_CACHE_MOUNT: hf_cache,
    },
    secrets=hf_secrets,
    timeout=86400,
    retries=0,
)
def export_pretrained_hf(
    model_name: str = DEFAULT_KIMI_MODEL_NAME,
    output_dir: str = DEFAULT_PRETRAINED_PATH,
) -> dict[str, Any]:
    """Run the vendored Kimi model module to materialize the HF training layout."""
    model_name = _resolve_supported_kimi_model_name(model_name)
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "python",
        "-m",
        VENDORED_KIMI_MODEL_MODULE,
        "--model_name",
        model_name,
        "--output_dir",
        output_dir,
    ]
    log = _run_in_kimi_repo(cmd)
    rad_volume.commit()
    return {"status": "ok", "output_dir": output_dir, "log_tail": log[-4000:]}


@app.function(
    image=finetune_image,
    gpu=TRAIN_GPU,
    volumes={
        DATA_MOUNT: rad_volume,
        HF_CACHE_MOUNT: hf_cache,
    },
    secrets=hf_secrets,
    timeout=86400,
    retries=0,
)
def prepare_semantic_codes(
    bundle_name: str = "",
    train_input_file: str = DEFAULT_TRAIN_JSON,
    val_input_file: str = DEFAULT_VAL_JSON,
    train_output_file: str = DEFAULT_TRAIN_CODES,
    val_output_file: str = DEFAULT_VAL_CODES,
    combined_output_file: str = DEFAULT_COMBINED_CODES,
    model_name_or_path: str = DEFAULT_KIMI_MODEL_NAME,
    training_inputs_json: str = "",
    heldout_evals_json: str = "[]",
) -> dict[str, Any]:
    """Extract semantic codes for train + optional val, then build the combined file."""
    model_name_or_path = _resolve_supported_kimi_model_name(model_name_or_path)
    resolved = _resolve_training_paths(
        bundle_name=bundle_name,
        train_input_file=train_input_file,
        val_input_file=val_input_file,
        train_output_file=train_output_file,
        val_output_file=val_output_file,
        combined_output_file=combined_output_file,
        training_inputs_json=training_inputs_json,
        heldout_evals_json=heldout_evals_json,
    )
    combined_output_file = resolved["combined_output_file"]

    log_chunks: list[str] = []
    per_input_results: list[dict[str, Any]] = []
    tagged_train_files: list[dict[str, Any]] = []
    tagged_val_files: list[dict[str, Any]] = []
    for item in resolved["training_inputs"]:
        tag = _safe_output_tag(item["tag"])
        input_train_path = str(item["train_input_file"]).strip()
        input_val_path = str(item.get("val_input_file") or "").strip()
        if not input_train_path:
            raise RuntimeError(f"Training input {tag!r} is missing train_input_file")
        train_row_count = _jsonl_row_count(input_train_path)
        val_row_count = _jsonl_row_count(input_val_path) if input_val_path else 0
        if train_row_count <= 0:
            raise RuntimeError(f"No rows found in {input_train_path}")

        train_codes_path = (
            str(item.get("train_output_file") or "").strip()
            or str(
                Path(combined_output_file).with_name(f"{tag}_train_with_semantic_codes.jsonl")
            )
        )
        val_codes_path = (
            str(item.get("val_output_file") or "").strip()
            or str(
                Path(combined_output_file).with_name(f"{tag}_val_with_semantic_codes.jsonl")
            )
        )
        raw_train_codes_path = (
            str(Path(train_codes_path).with_name(f"{Path(train_codes_path).stem}.raw.jsonl"))
        )
        raw_val_codes_path = (
            str(Path(val_codes_path).with_name(f"{Path(val_codes_path).stem}.raw.jsonl"))
        )

        for input_file, raw_output_file, row_count in (
            (input_train_path, raw_train_codes_path, train_row_count),
            (input_val_path, raw_val_codes_path, val_row_count),
        ):
            if not input_file:
                continue
            os.makedirs(os.path.dirname(raw_output_file) or ".", exist_ok=True)
            if row_count <= 0:
                Path(raw_output_file).write_text("", encoding="utf-8")
                log_chunks.append(
                    f"Skipped semantic code extraction for empty input: {input_file}"
                )
                continue
            cmd = [
                "python",
                "finetune_codes/extract_semantic_codes.py",
                "--model_name_or_path",
                model_name_or_path,
                "--input_file",
                input_file,
                "--output_file",
                raw_output_file,
            ]
            log_chunks.append(_run_in_kimi_repo(cmd))

        os.makedirs(os.path.dirname(train_codes_path) or ".", exist_ok=True)
        tagged_train_count = _tag_semantic_code_rows(
            input_path=raw_train_codes_path,
            output_path=train_codes_path,
            source_tag=tag,
        )
        tagged_train_files.append({"tag": tag, "path": train_codes_path})
        os.remove(raw_train_codes_path)

        tagged_val_count = 0
        if input_val_path:
            os.makedirs(os.path.dirname(val_codes_path) or ".", exist_ok=True)
            tagged_val_count = _tag_semantic_code_rows(
                input_path=raw_val_codes_path,
                output_path=val_codes_path,
                source_tag=tag,
            )
            tagged_val_files.append({"tag": tag, "path": val_codes_path})
            os.remove(raw_val_codes_path)
        else:
            Path(val_codes_path).write_text("", encoding="utf-8")

        per_input_results.append(
            {
                "tag": tag,
                "bundle_name": item.get("bundle_name"),
                "train_input_file": input_train_path,
                "val_input_file": input_val_path or None,
                "train_output_file": train_codes_path,
                "val_output_file": val_codes_path,
                "train_count": tagged_train_count,
                "val_count": tagged_val_count,
            }
        )

    combined_meta = _merge_semantic_code_groups(
        train_code_files=tagged_train_files,
        val_code_files=tagged_val_files,
        combined_output_file=combined_output_file,
    )
    rad_volume.commit()
    return {
        "status": "ok",
        "bundle_name": resolved["bundle_name"],
        "training_inputs": per_input_results,
        "train_output_file": per_input_results[0]["train_output_file"] if len(per_input_results) == 1 else None,
        "val_output_file": per_input_results[0]["val_output_file"] if len(per_input_results) == 1 else None,
        "combined_output_file": combined_output_file,
        "heldout_evals": resolved["heldout_evals"],
        **combined_meta,
        "log_tail": "\n".join(log_chunks)[-8000:],
    }


def _run_deepspeed_finetune_impl(
    pretrained_path: str = DEFAULT_PRETRAINED_PATH,
    data_file: str = DEFAULT_COMBINED_CODES,
    model_name: str = DEFAULT_KIMI_MODEL_NAME,
    bundle_name: str = "",
    run_name: str = "",
    output_dir: str = "",
    notes: str = "",
    tags: str = "",
    overwrite_run: bool = False,
    resume_run: bool = False,
    tracking_backend: str = "wandb",
    wandb_project: str = "",
    wandb_entity: str = "",
    wandb_mode: str = "",
    seed: int = 1,
    num_train_epochs: float = 1.0,
    max_steps: int = 100,
    save_steps: int = 25,
    eval_steps: int = 25,
    learning_rate: float = 5e-5,
    gradient_accumulation_steps: int = DEFAULT_LORA_GRADIENT_ACCUMULATION_STEPS,
    per_device_train_batch_size: int = DEFAULT_LORA_PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size: int = DEFAULT_LORA_PER_DEVICE_EVAL_BATCH_SIZE,
    logging_steps: int = DEFAULT_WANDB_LOGGING_STEPS,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.1,
    model_max_length: int = 512,
    memory_profile_steps: int = 0,
    profile_only: bool = False,
    run_step0_eval: bool = False,
    heldout_eval_subset_size: int = 0,
    training_inputs_json: str = "",
    heldout_evals_json: str = "[]",
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    lora_target_modules: str = DEFAULT_LORA_TARGET_MODULES_CSV,
    lora_include_audio_encoder: bool = False,
) -> dict[str, Any]:
    """Run RAD-only DeepSpeed LoRA / PEFT SFT with frozen Whisper audio encoder."""
    model_name = _resolve_supported_kimi_model_name(model_name)
    safe_run_name = validate_layout_name(run_name, kind="run_name")
    wandb_project = _resolve_wandb_project_name(
        tracking_backend=tracking_backend,
        wandb_project=wandb_project,
    )
    if heldout_eval_subset_size < 0:
        raise ValueError("heldout_eval_subset_size must be non-negative")
    resolved = _resolve_training_paths(
        bundle_name=bundle_name,
        data_file=data_file,
        heldout_eval_subset_size=heldout_eval_subset_size,
        training_inputs_json=training_inputs_json,
        heldout_evals_json=heldout_evals_json,
    )
    bundle_name = resolved["bundle_name"]
    data_file = resolved["data_file"]
    heldout_eval_manifest = str(resolved.get("heldout_eval_manifest") or "").strip()
    heldout_evals = list(resolved.get("heldout_evals") or [])
    heldout_eval_specs_json = json.dumps(heldout_evals, sort_keys=True)
    run_paths = resolve_run_paths(safe_run_name)
    prepare_run_directory(
        run_paths,
        overwrite=overwrite_run,
        resume=resume_run,
    )
    if not os.path.isdir(pretrained_path):
        raise FileNotFoundError(pretrained_path)
    if not os.path.isfile(data_file):
        raise FileNotFoundError(data_file)

    meta_path = _combined_meta_path(data_file)
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    eval_ratio = float(meta["eval_ratio"])
    if eval_ratio < 0.0 or eval_ratio >= 1.0:
        raise ValueError(f"Bad eval_ratio {eval_ratio} from {meta_path}")
    has_eval = eval_ratio > 0.0
    if save_steps <= 0:
        raise ValueError("save_steps must be positive")
    if has_eval and eval_steps <= 0:
        raise ValueError("eval_steps must be positive when eval data is present")
    if has_eval and save_steps % eval_steps != 0:
        raise ValueError("save_steps must be a multiple of eval_steps")
    if logging_steps <= 0:
        raise ValueError("logging_steps must be positive")

    train_output_dir = output_dir.strip() or str(run_paths.train_dir)
    os.makedirs(train_output_dir, exist_ok=True)
    deepspeed_config_path = _write_lora_single_gpu_deepspeed_config(train_output_dir)
    resolved_tags = csv_tags(tags)

    run_manifest = build_run_manifest(
        run_paths=run_paths,
        task="rad",
        bundle_name=bundle_name or "legacy",
        bundle_path=resolved["bundle_root"] or DEFAULT_EXPORT_DIR,
        base_model=model_name,
        pretrained_path=pretrained_path,
        hyperparameters={
            "learning_rate": learning_rate,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_train_epochs": num_train_epochs,
            "max_steps": max_steps,
            "save_steps": save_steps,
            "eval_steps": eval_steps,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "logging_steps": logging_steps,
            "model_max_length": model_max_length,
            "seed": seed,
            "memory_profile_steps": memory_profile_steps,
            "profile_only": profile_only,
            "run_step0_eval": run_step0_eval,
            "heldout_eval_subset_size": heldout_eval_subset_size,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "lora_target_modules": [
                module.strip() for module in lora_target_modules.split(",") if module.strip()
            ],
        },
        notes=notes,
        tags=resolved_tags,
        extra={
            "trainer": TRAINER_LORA,
            "tracking_backend": tracking_backend,
            "wandb_project": wandb_project or None,
            "wandb_entity": wandb_entity or None,
            "wandb_mode": wandb_mode or None,
            "wandb_secret_source": WAND_SECRET_SOURCE if tracking_backend == "wandb" else None,
            "data_file": data_file,
            "heldout_eval_subset_size": heldout_eval_subset_size or None,
            "heldout_eval_manifest": heldout_eval_manifest or None,
            "training_inputs": resolved.get("training_inputs") or [],
            "heldout_evals": heldout_evals,
            "heldout_eval_specs_json": heldout_eval_specs_json or None,
        },
    )
    write_json(run_paths.run_manifest, run_manifest)
    write_run_state(
        run_paths,
        build_initial_run_state(
            run_paths=run_paths,
            bundle_name=bundle_name or "legacy",
            trainer=TRAINER_LORA,
            tracking_backend=tracking_backend,
            heldout_eval_manifest=heldout_eval_manifest,
            heldout_evals=heldout_evals,
            phase="starting",
        ),
    )

    cmd = [
        "torchrun",
        "--nproc_per_node",
        "1",
        "--nnodes",
        "1",
        "--node_rank",
        "0",
        "--master_addr",
        "localhost",
        "--master_port",
        "6001",
        "-m",
        RAD_LORA_ENTRYPOINT_MODULE,
        "--model_name_or_path",
        model_name,
        "--model_path",
        pretrained_path,
        "--pretrained_path",
        pretrained_path,
        "--model_name_or_path_for_eval",
        model_name,
        "--data_path",
        data_file,
        "--run_name",
        safe_run_name,
        "--heldout_eval_manifest",
        heldout_eval_manifest,
        "--heldout_eval_specs_json",
        heldout_eval_specs_json,
        "--eval_ratio",
        f"{eval_ratio:.12f}",
        "--bf16",
        "True",
        "--output_dir",
        train_output_dir,
        "--num_train_epochs",
        str(num_train_epochs),
        "--per_device_train_batch_size",
        str(per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(per_device_eval_batch_size),
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--save_strategy",
        "steps",
        "--save_steps",
        str(save_steps),
        "--learning_rate",
        str(learning_rate),
        "--weight_decay",
        str(weight_decay),
        "--adam_beta2",
        "0.95",
        "--warmup_ratio",
        str(warmup_ratio),
        "--lr_scheduler_type",
        "cosine",
        "--logging_steps",
        str(logging_steps),
        "--disable_tqdm",
        "True",
        "--report_to",
        tracking_backend,
        "--model_max_length",
        str(model_max_length),
        "--memory_profile_steps",
        str(memory_profile_steps),
        "--profile_only",
        "True" if profile_only else "False",
        "--gradient_checkpointing",
        "True",
        "--lazy_preprocess",
        "True",
        "--deepspeed",
        deepspeed_config_path,
        "--lora_r",
        str(lora_r),
        "--lora_alpha",
        str(lora_alpha),
        "--lora_dropout",
        str(lora_dropout),
        *(
            ["--lora_include_audio_encoder", "True"]
            if lora_include_audio_encoder
            else []
        ),
        "--lora_target_modules",
        lora_target_modules,
    ]
    if has_eval:
        cmd.extend(
            [
                "--eval_strategy",
                "steps",
                "--eval_steps",
                str(eval_steps),
                "--load_best_model_at_end",
                "True",
                "--metric_for_best_model",
                "eval_loss",
                "--greater_is_better",
                "False",
            ]
        )
    else:
        cmd.extend(["--eval_strategy", "no"])
    if max_steps > 0:
        cmd.extend(["--max_steps", str(max_steps)])

    wandb_auth = None
    wandb_env: dict[str, str] = {}
    if tracking_backend == "wandb":
        wandb_env = _wandb_runtime_env(
            run_name=safe_run_name,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_mode=wandb_mode,
            tags=resolved_tags,
        )
        wandb_auth = _assert_wandb_remote_auth(
            wandb_env=wandb_env,
            log_path=run_paths.train_log,
        )
    step0_eval_specs = _step0_stage_evals(heldout_evals)
    baseline_wandb_run = _maybe_start_step0_wandb_run(
        enabled=bool(
            run_step0_eval
            and tracking_backend == "wandb"
            and not profile_only
            and step0_eval_specs
        ),
        run_name=safe_run_name,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        tags=resolved_tags,
        heldout_evals=step0_eval_specs,
    )
    if baseline_wandb_run is not None:
        wandb_env["WANDB_RUN_ID"] = str(baseline_wandb_run.id)
        wandb_env["WANDB_RESUME"] = "allow"
        try:
            _maybe_run_step0_eval_group(
                model_dir=pretrained_path,
                pretrained_path=pretrained_path,
                model_name_or_path=model_name,
                run_name=safe_run_name,
                eval_specs=step0_eval_specs,
            )
        finally:
            baseline_wandb_run.finish()
            rad_volume.commit()

    # Stream trainer stdout/stderr into Modal logs for live monitoring.
    log = _run_in_kimi_repo(
        cmd,
        stream_output=True,
        log_path=run_paths.train_log,
        extra_env={
            **wandb_env,
            **_checkpoint_eval_runtime_env(run_paths),
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
            "AUDIOBENCH_TRAINING_MODE": "lora-peft-frozen-audio",
            "AUDIOBENCH_MODEL_NAME_OR_PATH": model_name,
            "AUDIOBENCH_HELDOUT_EVAL_SPECS_JSON": heldout_eval_specs_json,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    )
    update_run_state(
        run_paths,
        phase="completed",
    )
    rad_volume.commit()
    hf_cache.commit()
    lora_summary = _read_json_if_exists(Path(train_output_dir) / "rad_lora_summary.json")
    memory_profile = _read_json_if_exists(Path(train_output_dir) / "rad_memory_profile.json")
    trainer_summary = _trainer_summary(train_output_dir)
    state_path = Path(train_output_dir) / "trainer_state.json"
    if state_path.is_file():
        trainer_state = json.loads(state_path.read_text(encoding="utf-8"))
        write_jsonl(run_paths.train_history, normalize_trainer_history(trainer_state))
    wandb_metadata = _resolve_wandb_run_metadata(
        tracking_backend=tracking_backend,
        wandb_project=wandb_project,
        wandb_auth=wandb_auth,
        run_state=_read_json_if_exists(run_paths.run_state),
    )
    train_summary = {
        **trainer_summary,
        "run_name": safe_run_name,
        "bundle_name": bundle_name,
        "trainer": TRAINER_LORA,
        "tracking_backend": tracking_backend,
        "created_at": now_utc_iso(),
        "wandb_auth": wandb_auth,
        **wandb_metadata,
        "wandb_secret_source": WAND_SECRET_SOURCE if tracking_backend == "wandb" else None,
        "lora_summary": lora_summary,
        "memory_profile": memory_profile,
        "run_step0_eval": run_step0_eval,
        "heldout_eval_subset_size": heldout_eval_subset_size,
    }
    write_json(run_paths.train_summary, train_summary)
    return {
        "status": "profiled" if profile_only else "ok",
        "run_name": safe_run_name,
        "bundle_name": bundle_name,
        "run_manifest": str(run_paths.run_manifest),
        "train_log": str(run_paths.train_log),
        "train_history": str(run_paths.train_history),
        "train_summary": str(run_paths.train_summary),
        "trainer": TRAINER_LORA,
        "tracking_backend": tracking_backend,
        "wandb_auth": wandb_auth,
        **wandb_metadata,
        "wandb_secret_source": WAND_SECRET_SOURCE if tracking_backend == "wandb" else None,
        "training_mode": "lora-peft-frozen-audio",
        "audio_encoder_frozen": True,
        "gpu": TRAIN_GPU,
        "pretrained_path": pretrained_path,
        "data_file": data_file,
        "eval_ratio": eval_ratio,
        "has_eval": has_eval,
        "profile_only": profile_only,
        "memory_profile_steps": memory_profile_steps,
        "run_step0_eval": run_step0_eval,
        "heldout_eval_subset_size": heldout_eval_subset_size,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": [
            module.strip() for module in lora_target_modules.split(",") if module.strip()
        ],
        **trainer_summary,
        "lora_summary": lora_summary,
        "memory_profile": memory_profile,
        "log_tail": log[-12000:],
    }


def _run_stock_full_finetune_impl(
    pretrained_path: str = DEFAULT_PRETRAINED_PATH,
    data_file: str = DEFAULT_COMBINED_CODES,
    model_name: str = DEFAULT_KIMI_MODEL_NAME,
    bundle_name: str = "",
    run_name: str = "",
    output_dir: str = "",
    notes: str = "",
    tags: str = "",
    overwrite_run: bool = False,
    resume_run: bool = False,
    tracking_backend: str = "wandb",
    wandb_project: str = "",
    wandb_entity: str = "",
    wandb_mode: str = "",
    seed: int = 1,
    num_train_epochs: float = 1.0,
    max_steps: int = 100,
    save_steps: int = DEFAULT_STOCK_FULL_FT_SAVE_STEPS,
    eval_steps: int = DEFAULT_STOCK_FULL_FT_EVAL_STEPS,
    learning_rate: float = DEFAULT_STOCK_FULL_FT_LEARNING_RATE,
    gradient_accumulation_steps: int = 1,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    logging_steps: int = DEFAULT_WANDB_LOGGING_STEPS,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.1,
    model_max_length: int = 512,
    memory_profile_steps: int = 0,
    profile_only: bool = False,
    run_step0_eval: bool = False,
    heldout_eval_subset_size: int = 0,
    training_inputs_json: str = "",
    heldout_evals_json: str = "[]",
    step0_compare_generate: bool = False,
    freeze_audio_encoder: bool = False,
    stock_full_ft_profile: str = DEFAULT_STOCK_FULL_FT_PROFILE,
    gradient_checkpointing: Any = DEFAULT_GRADIENT_CHECKPOINTING_MODE,
) -> dict[str, Any]:
    """Run stock Moonshot-style full fine-tuning as a diagnostic control."""
    model_name = _resolve_supported_kimi_model_name(model_name)
    if memory_profile_steps > 0 or profile_only:
        raise ValueError("stock-full-ft does not support memory profiling")
    if per_device_train_batch_size != 1 or per_device_eval_batch_size != 1:
        raise ValueError(
            "stock-full-ft follows Moonshot's upstream microbatch-size-1 path; "
            "set both train and eval batch size to 1"
        )

    wandb_project = _resolve_wandb_project_name(
        tracking_backend=tracking_backend,
        wandb_project=wandb_project,
    )
    runtime_versions = _stock_full_ft_runtime_versions()
    profile_settings = _resolve_stock_full_ft_profile_settings(
        stock_full_ft_profile=stock_full_ft_profile,
        gradient_checkpointing=gradient_checkpointing,
    )
    stock_full_ft_profile = str(profile_settings["stock_full_ft_profile"])
    gpu_count = int(profile_settings["gpu_count"])
    deepspeed_profile = str(profile_settings["deepspeed_profile"])
    optimizer_offload = bool(profile_settings["optimizer_offload"])
    gradient_checkpointing_enabled = bool(profile_settings["gradient_checkpointing"])
    safe_run_name = validate_layout_name(run_name, kind="run_name")
    resolved = _resolve_training_paths(
        bundle_name=bundle_name,
        data_file=data_file,
        heldout_eval_subset_size=heldout_eval_subset_size,
        training_inputs_json=training_inputs_json,
        heldout_evals_json=heldout_evals_json,
    )
    bundle_name = resolved["bundle_name"]
    data_file = resolved["data_file"]
    heldout_eval_manifest = str(resolved.get("heldout_eval_manifest") or "").strip()
    heldout_evals = list(resolved.get("heldout_evals") or [])
    heldout_eval_specs_json = json.dumps(heldout_evals, sort_keys=True)
    run_paths = resolve_run_paths(safe_run_name)
    prepare_run_directory(
        run_paths,
        overwrite=overwrite_run,
        resume=resume_run,
    )
    if not os.path.isdir(pretrained_path):
        raise FileNotFoundError(pretrained_path)
    if not os.path.isfile(data_file):
        raise FileNotFoundError(data_file)

    meta_path = _combined_meta_path(data_file)
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    eval_ratio = float(meta["eval_ratio"])
    if eval_ratio < 0.0 or eval_ratio >= 1.0:
        raise ValueError(f"Bad eval_ratio {eval_ratio} from {meta_path}")
    has_eval = eval_ratio > 0.0
    if save_steps <= 0:
        raise ValueError("save_steps must be positive")
    if has_eval and eval_steps <= 0:
        raise ValueError("eval_steps must be positive when eval data is present")
    if has_eval and save_steps % eval_steps != 0:
        raise ValueError("save_steps must be a multiple of eval_steps")
    if logging_steps <= 0:
        raise ValueError("logging_steps must be positive")

    train_output_dir = output_dir.strip() or str(run_paths.train_dir)
    os.makedirs(train_output_dir, exist_ok=True)
    resolved_tags = csv_tags(tags)

    run_manifest = build_run_manifest(
        run_paths=run_paths,
        task="rad",
        bundle_name=bundle_name or "legacy",
        bundle_path=resolved["bundle_root"] or DEFAULT_EXPORT_DIR,
        base_model=model_name,
        pretrained_path=pretrained_path,
        hyperparameters={
            "learning_rate": learning_rate,
            "per_device_train_batch_size": per_device_train_batch_size,
            "per_device_eval_batch_size": per_device_eval_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "num_train_epochs": num_train_epochs,
            "max_steps": max_steps,
            "save_steps": save_steps,
            "eval_steps": eval_steps,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "logging_steps": logging_steps,
            "model_max_length": model_max_length,
            "seed": seed,
            "freeze_audio_encoder": freeze_audio_encoder,
            "stock_full_ft_profile": stock_full_ft_profile,
            "gpu_count": gpu_count,
            "deepspeed_profile": deepspeed_profile,
            "optimizer_offload": optimizer_offload,
            "gradient_checkpointing": gradient_checkpointing_enabled,
            "run_step0_eval": run_step0_eval,
            "step0_compare_generate": step0_compare_generate,
        },
        notes=notes,
        tags=resolved_tags,
        extra={
            "trainer": TRAINER_STOCK_FULL_FT,
            "tracking_backend": tracking_backend,
            "wandb_project": wandb_project or None,
            "wandb_entity": wandb_entity or None,
            "wandb_mode": wandb_mode or None,
            "wandb_secret_source": WAND_SECRET_SOURCE if tracking_backend == "wandb" else None,
            "data_file": data_file,
            "heldout_eval_manifest": heldout_eval_manifest or None,
            "training_inputs": resolved.get("training_inputs") or [],
            "heldout_evals": heldout_evals,
            "heldout_eval_specs_json": heldout_eval_specs_json or None,
            "audio_encoder_frozen": freeze_audio_encoder,
            "stock_full_ft_profile": stock_full_ft_profile,
            "gpu_count": gpu_count,
            "deepspeed_profile": deepspeed_profile,
            "optimizer_offload": optimizer_offload,
            "gradient_checkpointing": gradient_checkpointing_enabled,
            "runtime_versions": runtime_versions,
        },
    )
    write_json(run_paths.run_manifest, run_manifest)
    write_run_state(
        run_paths,
        build_initial_run_state(
            run_paths=run_paths,
            bundle_name=bundle_name or "legacy",
            trainer=TRAINER_STOCK_FULL_FT,
            tracking_backend=tracking_backend,
            heldout_eval_manifest=heldout_eval_manifest,
            heldout_evals=heldout_evals,
            phase="starting",
        ),
    )
    if deepspeed_profile == "zero3-cpuoffload":
        deepspeed_config_path = _write_stock_single_gpu_deepspeed_config(train_output_dir)
    elif deepspeed_profile == "zero2-multigpu":
        deepspeed_config_path = _write_stock_multigpu_deepspeed_config(train_output_dir)
    else:
        raise ValueError(f"Unknown DeepSpeed profile {deepspeed_profile!r}")
    training_mode = (
        "stock-full-ft-frozen-audio" if freeze_audio_encoder else TRAINER_STOCK_FULL_FT
    )

    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(gpu_count),
        "--nnodes",
        "1",
        "--node_rank",
        "0",
        "--master_addr",
        "localhost",
        "--master_port",
        "6001",
        "-m",
        RAD_STOCK_FT_ENTRYPOINT_MODULE,
        "--model_name_or_path",
        model_name,
        "--model_path",
        pretrained_path,
        "--data_path",
        data_file,
        "--eval_ratio",
        f"{eval_ratio:.12f}",
        "--bf16",
        "True",
        "--output_dir",
        train_output_dir,
        "--num_train_epochs",
        str(num_train_epochs),
        "--per_device_train_batch_size",
        str(per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(per_device_eval_batch_size),
        "--gradient_accumulation_steps",
        str(gradient_accumulation_steps),
        "--save_strategy",
        "steps",
        "--save_steps",
        str(save_steps),
        "--learning_rate",
        str(learning_rate),
        "--weight_decay",
        str(weight_decay),
        "--adam_beta2",
        "0.95",
        "--warmup_ratio",
        str(warmup_ratio),
        "--lr_scheduler_type",
        "cosine",
        "--logging_steps",
        str(logging_steps),
        "--disable_tqdm",
        "True",
        "--report_to",
        tracking_backend,
        "--model_max_length",
        str(model_max_length),
        "--gradient_checkpointing",
        "True" if gradient_checkpointing_enabled else "False",
        "--lazy_preprocess",
        "True",
        "--seed",
        str(seed),
        "--deepspeed",
        deepspeed_config_path,
    ]
    if freeze_audio_encoder:
        cmd.append("--freeze-audio-encoder")
    eval_strategy_flag = _training_eval_strategy_flag()
    if has_eval:
        cmd.extend(
            [
                eval_strategy_flag,
                "steps",
                "--eval_steps",
                str(eval_steps),
                "--load_best_model_at_end",
                "True",
                "--metric_for_best_model",
                "eval_loss",
                "--greater_is_better",
                "False",
            ]
        )
    else:
        cmd.extend([eval_strategy_flag, "no"])
    if max_steps > 0:
        cmd.extend(["--max_steps", str(max_steps)])

    wandb_auth = None
    wandb_env: dict[str, str] = {}
    if tracking_backend == "wandb":
        wandb_env = _wandb_runtime_env(
            run_name=safe_run_name,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_mode=wandb_mode,
            tags=resolved_tags,
        )
        wandb_auth = _assert_wandb_remote_auth(
            wandb_env=wandb_env,
            log_path=run_paths.train_log,
        )
    step0_eval_specs = _step0_stage_evals(heldout_evals)
    baseline_wandb_run = _maybe_start_step0_wandb_run(
        enabled=bool(run_step0_eval and tracking_backend == "wandb" and step0_eval_specs),
        run_name=safe_run_name,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        tags=resolved_tags,
        heldout_evals=step0_eval_specs,
    )
    if baseline_wandb_run is not None:
        wandb_env["WANDB_RUN_ID"] = str(baseline_wandb_run.id)
        wandb_env["WANDB_RESUME"] = "allow"
        try:
            _maybe_run_step0_eval_group(
                model_dir=pretrained_path,
                pretrained_path=pretrained_path,
                model_name_or_path=model_name,
                run_name=safe_run_name,
                eval_specs=step0_eval_specs,
                step0_compare_generate=step0_compare_generate,
            )
        finally:
            baseline_wandb_run.finish()
            rad_volume.commit()

    log = _run_in_kimi_repo(
        cmd,
        stream_output=True,
        log_path=run_paths.train_log,
        extra_env={
            **wandb_env,
            **_checkpoint_eval_runtime_env(run_paths),
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
            "AUDIOBENCH_RUN_NAME": safe_run_name,
            "AUDIOBENCH_TRAINING_MODE": training_mode,
            "AUDIOBENCH_MODEL_NAME_OR_PATH": model_name,
            "AUDIOBENCH_PRETRAINED_PATH": pretrained_path,
            "AUDIOBENCH_HELDOUT_EVAL_MANIFEST": heldout_eval_manifest,
            "AUDIOBENCH_HELDOUT_EVAL_SPECS_JSON": heldout_eval_specs_json,
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    )
    update_run_state(
        run_paths,
        phase="completed",
    )
    rad_volume.commit()
    hf_cache.commit()
    trainer_summary = _trainer_summary(train_output_dir)
    state_path = Path(train_output_dir) / "trainer_state.json"
    if state_path.is_file():
        trainer_state = json.loads(state_path.read_text(encoding="utf-8"))
        write_jsonl(run_paths.train_history, normalize_trainer_history(trainer_state))
    wandb_metadata = _resolve_wandb_run_metadata(
        tracking_backend=tracking_backend,
        wandb_project=wandb_project,
        wandb_auth=wandb_auth,
        run_state=_read_json_if_exists(run_paths.run_state),
    )
    train_summary = {
        **trainer_summary,
        "run_name": safe_run_name,
        "bundle_name": bundle_name,
        "trainer": TRAINER_STOCK_FULL_FT,
        "training_mode": training_mode,
        "audio_encoder_frozen": freeze_audio_encoder,
        "stock_full_ft_profile": stock_full_ft_profile,
        "gpu_count": gpu_count,
        "deepspeed_profile": deepspeed_profile,
        "optimizer_offload": optimizer_offload,
        "gradient_checkpointing": gradient_checkpointing_enabled,
        "tracking_backend": tracking_backend,
        "created_at": now_utc_iso(),
        "wandb_auth": wandb_auth,
        **wandb_metadata,
        "wandb_secret_source": WAND_SECRET_SOURCE if tracking_backend == "wandb" else None,
        "runtime_versions": runtime_versions,
        "run_step0_eval": run_step0_eval,
    }
    write_json(run_paths.train_summary, train_summary)
    return {
        "status": "ok",
        "run_name": safe_run_name,
        "bundle_name": bundle_name,
        "run_manifest": str(run_paths.run_manifest),
        "train_log": str(run_paths.train_log),
        "train_history": str(run_paths.train_history),
        "train_summary": str(run_paths.train_summary),
        "trainer": TRAINER_STOCK_FULL_FT,
        "tracking_backend": tracking_backend,
        "wandb_auth": wandb_auth,
        **wandb_metadata,
        "wandb_secret_source": WAND_SECRET_SOURCE if tracking_backend == "wandb" else None,
        "training_mode": training_mode,
        "audio_encoder_frozen": freeze_audio_encoder,
        "stock_full_ft_profile": stock_full_ft_profile,
        "gpu": profile_settings["gpu"],
        "gpu_count": gpu_count,
        "deepspeed_profile": deepspeed_profile,
        "optimizer_offload": optimizer_offload,
        "gradient_checkpointing": gradient_checkpointing_enabled,
        "pretrained_path": pretrained_path,
        "data_file": data_file,
        "eval_ratio": eval_ratio,
        "has_eval": has_eval,
        "profile_only": False,
        "memory_profile_steps": 0,
        "run_step0_eval": run_step0_eval,
        "runtime_versions": runtime_versions,
        **trainer_summary,
        "log_tail": log[-12000:],
    }


@app.function(
    image=finetune_image,
    gpu=TRAIN_GPU,
    volumes={
        DATA_MOUNT: rad_volume,
        HF_CACHE_MOUNT: hf_cache,
    },
    secrets=_require_training_runtime_secrets(),
    timeout=86400,
    retries=0,
)
def run_deepspeed_finetune(
    pretrained_path: str = DEFAULT_PRETRAINED_PATH,
    data_file: str = DEFAULT_COMBINED_CODES,
    model_name: str = DEFAULT_KIMI_MODEL_NAME,
    bundle_name: str = "",
    run_name: str = "",
    output_dir: str = "",
    notes: str = "",
    tags: str = "",
    overwrite_run: bool = False,
    resume_run: bool = False,
    tracking_backend: str = "wandb",
    wandb_project: str = "",
    wandb_entity: str = "",
    wandb_mode: str = "",
    seed: int = 1,
    num_train_epochs: float = 1.0,
    max_steps: int = 100,
    save_steps: int = 25,
    eval_steps: int = 25,
    learning_rate: float = 5e-5,
    gradient_accumulation_steps: int = DEFAULT_LORA_GRADIENT_ACCUMULATION_STEPS,
    per_device_train_batch_size: int = DEFAULT_LORA_PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size: int = DEFAULT_LORA_PER_DEVICE_EVAL_BATCH_SIZE,
    logging_steps: int = DEFAULT_WANDB_LOGGING_STEPS,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.1,
    model_max_length: int = 512,
    memory_profile_steps: int = 0,
    profile_only: bool = False,
    run_step0_eval: bool = False,
    heldout_eval_subset_size: int = 0,
    training_inputs_json: str = "",
    heldout_evals_json: str = "[]",
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    lora_target_modules: str = DEFAULT_LORA_TARGET_MODULES_CSV,
    lora_include_audio_encoder: bool = False,
) -> dict[str, Any]:
    return _run_deepspeed_finetune_impl(
        pretrained_path=pretrained_path,
        data_file=data_file,
        model_name=model_name,
        bundle_name=bundle_name,
        run_name=run_name,
        output_dir=output_dir,
        notes=notes,
        tags=tags,
        overwrite_run=overwrite_run,
        resume_run=resume_run,
        tracking_backend=tracking_backend,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        seed=seed,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        logging_steps=logging_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        model_max_length=model_max_length,
        memory_profile_steps=memory_profile_steps,
        profile_only=profile_only,
        run_step0_eval=run_step0_eval,
        heldout_eval_subset_size=heldout_eval_subset_size,
        training_inputs_json=training_inputs_json,
        heldout_evals_json=heldout_evals_json,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_include_audio_encoder=lora_include_audio_encoder,
        lora_target_modules=lora_target_modules,
    )

@app.function(
    image=finetune_image,
    gpu=TRAIN_GPU,
    volumes={
        DATA_MOUNT: rad_volume,
        HF_CACHE_MOUNT: hf_cache,
    },
    secrets=_require_training_runtime_secrets(),
    timeout=86400,
    retries=0,
)
def run_deepspeed_finetune_with_wandb(
    pretrained_path: str = DEFAULT_PRETRAINED_PATH,
    data_file: str = DEFAULT_COMBINED_CODES,
    model_name: str = DEFAULT_KIMI_MODEL_NAME,
    bundle_name: str = "",
    run_name: str = "",
    output_dir: str = "",
    notes: str = "",
    tags: str = "",
    overwrite_run: bool = False,
    resume_run: bool = False,
    tracking_backend: str = "wandb",
    wandb_project: str = "",
    wandb_entity: str = "",
    wandb_mode: str = "",
    seed: int = 1,
    num_train_epochs: float = 1.0,
    max_steps: int = 100,
    save_steps: int = 25,
    eval_steps: int = 25,
    learning_rate: float = 5e-5,
    gradient_accumulation_steps: int = DEFAULT_LORA_GRADIENT_ACCUMULATION_STEPS,
    per_device_train_batch_size: int = DEFAULT_LORA_PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size: int = DEFAULT_LORA_PER_DEVICE_EVAL_BATCH_SIZE,
    logging_steps: int = DEFAULT_WANDB_LOGGING_STEPS,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.1,
    model_max_length: int = 512,
    memory_profile_steps: int = 0,
    profile_only: bool = False,
    run_step0_eval: bool = False,
    heldout_eval_subset_size: int = 0,
    training_inputs_json: str = "",
    heldout_evals_json: str = "[]",
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    lora_target_modules: str = DEFAULT_LORA_TARGET_MODULES_CSV,
    lora_include_audio_encoder: bool = False,
) -> dict[str, Any]:
    return _run_deepspeed_finetune_impl(
        pretrained_path=pretrained_path,
        data_file=data_file,
        model_name=model_name,
        bundle_name=bundle_name,
        run_name=run_name,
        output_dir=output_dir,
        notes=notes,
        tags=tags,
        overwrite_run=overwrite_run,
        resume_run=resume_run,
        tracking_backend=tracking_backend,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        seed=seed,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        logging_steps=logging_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        model_max_length=model_max_length,
        memory_profile_steps=memory_profile_steps,
        profile_only=profile_only,
        run_step0_eval=run_step0_eval,
        heldout_eval_subset_size=heldout_eval_subset_size,
        training_inputs_json=training_inputs_json,
        heldout_evals_json=heldout_evals_json,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_include_audio_encoder=lora_include_audio_encoder,
        lora_target_modules=lora_target_modules,
    )


@app.function(
    image=stock_full_finetune_image,
    gpu=TRAIN_GPU,
    volumes={
        DATA_MOUNT: rad_volume,
        HF_CACHE_MOUNT: hf_cache,
    },
    secrets=_require_training_runtime_secrets(),
    timeout=86400,
    retries=0,
)
def run_stock_full_finetune(
    pretrained_path: str = DEFAULT_PRETRAINED_PATH,
    data_file: str = DEFAULT_COMBINED_CODES,
    model_name: str = DEFAULT_KIMI_MODEL_NAME,
    bundle_name: str = "",
    run_name: str = "",
    output_dir: str = "",
    notes: str = "",
    tags: str = "",
    overwrite_run: bool = False,
    resume_run: bool = False,
    tracking_backend: str = "wandb",
    wandb_project: str = "",
    wandb_entity: str = "",
    wandb_mode: str = "",
    seed: int = 1,
    num_train_epochs: float = 1.0,
    max_steps: int = 100,
    save_steps: int = DEFAULT_STOCK_FULL_FT_SAVE_STEPS,
    eval_steps: int = DEFAULT_STOCK_FULL_FT_EVAL_STEPS,
    learning_rate: float = DEFAULT_STOCK_FULL_FT_LEARNING_RATE,
    gradient_accumulation_steps: int = 1,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    logging_steps: int = DEFAULT_WANDB_LOGGING_STEPS,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.1,
    model_max_length: int = 512,
    memory_profile_steps: int = 0,
    profile_only: bool = False,
    run_step0_eval: bool = False,
    heldout_eval_subset_size: int = 0,
    training_inputs_json: str = "",
    heldout_evals_json: str = "[]",
    step0_compare_generate: bool = False,
    freeze_audio_encoder: bool = False,
    stock_full_ft_profile: str = STOCK_FULL_FT_PROFILE_SINGLE_SAFE,
    gradient_checkpointing: Any = DEFAULT_GRADIENT_CHECKPOINTING_MODE,
) -> dict[str, Any]:
    return _run_stock_full_finetune_impl(
        pretrained_path=pretrained_path,
        data_file=data_file,
        model_name=model_name,
        bundle_name=bundle_name,
        run_name=run_name,
        output_dir=output_dir,
        notes=notes,
        tags=tags,
        overwrite_run=overwrite_run,
        resume_run=resume_run,
        tracking_backend=tracking_backend,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        seed=seed,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        logging_steps=logging_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        model_max_length=model_max_length,
        memory_profile_steps=memory_profile_steps,
        profile_only=profile_only,
        run_step0_eval=run_step0_eval,
        heldout_eval_subset_size=heldout_eval_subset_size,
        training_inputs_json=training_inputs_json,
        heldout_evals_json=heldout_evals_json,
        step0_compare_generate=step0_compare_generate,
        freeze_audio_encoder=freeze_audio_encoder,
        stock_full_ft_profile=stock_full_ft_profile,
        gradient_checkpointing=gradient_checkpointing,
    )


@app.function(
    image=stock_full_finetune_image,
    gpu=TRAIN_GPU,
    volumes={
        DATA_MOUNT: rad_volume,
        HF_CACHE_MOUNT: hf_cache,
    },
    secrets=_require_training_runtime_secrets(),
    timeout=86400,
    retries=0,
)
def run_stock_full_finetune_with_wandb(
    pretrained_path: str = DEFAULT_PRETRAINED_PATH,
    data_file: str = DEFAULT_COMBINED_CODES,
    model_name: str = DEFAULT_KIMI_MODEL_NAME,
    bundle_name: str = "",
    run_name: str = "",
    output_dir: str = "",
    notes: str = "",
    tags: str = "",
    overwrite_run: bool = False,
    resume_run: bool = False,
    tracking_backend: str = "wandb",
    wandb_project: str = "",
    wandb_entity: str = "",
    wandb_mode: str = "",
    seed: int = 1,
    num_train_epochs: float = 1.0,
    max_steps: int = 100,
    save_steps: int = DEFAULT_STOCK_FULL_FT_SAVE_STEPS,
    eval_steps: int = DEFAULT_STOCK_FULL_FT_EVAL_STEPS,
    learning_rate: float = DEFAULT_STOCK_FULL_FT_LEARNING_RATE,
    gradient_accumulation_steps: int = 1,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    logging_steps: int = DEFAULT_WANDB_LOGGING_STEPS,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.1,
    model_max_length: int = 512,
    memory_profile_steps: int = 0,
    profile_only: bool = False,
    run_step0_eval: bool = False,
    heldout_eval_subset_size: int = 0,
    training_inputs_json: str = "",
    heldout_evals_json: str = "[]",
    step0_compare_generate: bool = False,
    freeze_audio_encoder: bool = False,
    stock_full_ft_profile: str = STOCK_FULL_FT_PROFILE_SINGLE_SAFE,
    gradient_checkpointing: Any = DEFAULT_GRADIENT_CHECKPOINTING_MODE,
) -> dict[str, Any]:
    return _run_stock_full_finetune_impl(
        pretrained_path=pretrained_path,
        data_file=data_file,
        model_name=model_name,
        bundle_name=bundle_name,
        run_name=run_name,
        output_dir=output_dir,
        notes=notes,
        tags=tags,
        overwrite_run=overwrite_run,
        resume_run=resume_run,
        tracking_backend=tracking_backend,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        seed=seed,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        logging_steps=logging_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        model_max_length=model_max_length,
        memory_profile_steps=memory_profile_steps,
        profile_only=profile_only,
        run_step0_eval=run_step0_eval,
        heldout_eval_subset_size=heldout_eval_subset_size,
        training_inputs_json=training_inputs_json,
        heldout_evals_json=heldout_evals_json,
        step0_compare_generate=step0_compare_generate,
        freeze_audio_encoder=freeze_audio_encoder,
        stock_full_ft_profile=stock_full_ft_profile,
        gradient_checkpointing=gradient_checkpointing,
    )


@app.function(
    image=stock_full_finetune_image,
    gpu=DUAL_TRAIN_GPU,
    volumes={
        DATA_MOUNT: rad_volume,
        HF_CACHE_MOUNT: hf_cache,
    },
    secrets=_require_training_runtime_secrets(),
    timeout=86400,
    retries=0,
)
def run_stock_full_finetune_dual_fast(
    pretrained_path: str = DEFAULT_PRETRAINED_PATH,
    data_file: str = DEFAULT_COMBINED_CODES,
    model_name: str = DEFAULT_KIMI_MODEL_NAME,
    bundle_name: str = "",
    run_name: str = "",
    output_dir: str = "",
    notes: str = "",
    tags: str = "",
    overwrite_run: bool = False,
    resume_run: bool = False,
    tracking_backend: str = "wandb",
    wandb_project: str = "",
    wandb_entity: str = "",
    wandb_mode: str = "",
    seed: int = 1,
    num_train_epochs: float = 1.0,
    max_steps: int = 100,
    save_steps: int = DEFAULT_STOCK_FULL_FT_SAVE_STEPS,
    eval_steps: int = DEFAULT_STOCK_FULL_FT_EVAL_STEPS,
    learning_rate: float = DEFAULT_STOCK_FULL_FT_LEARNING_RATE,
    gradient_accumulation_steps: int = 1,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    logging_steps: int = DEFAULT_WANDB_LOGGING_STEPS,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.1,
    model_max_length: int = 512,
    memory_profile_steps: int = 0,
    profile_only: bool = False,
    run_step0_eval: bool = False,
    heldout_eval_subset_size: int = 0,
    training_inputs_json: str = "",
    heldout_evals_json: str = "[]",
    step0_compare_generate: bool = False,
    freeze_audio_encoder: bool = False,
    stock_full_ft_profile: str = STOCK_FULL_FT_PROFILE_DUAL_FAST,
    gradient_checkpointing: Any = DEFAULT_GRADIENT_CHECKPOINTING_MODE,
) -> dict[str, Any]:
    return _run_stock_full_finetune_impl(
        pretrained_path=pretrained_path,
        data_file=data_file,
        model_name=model_name,
        bundle_name=bundle_name,
        run_name=run_name,
        output_dir=output_dir,
        notes=notes,
        tags=tags,
        overwrite_run=overwrite_run,
        resume_run=resume_run,
        tracking_backend=tracking_backend,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        seed=seed,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        logging_steps=logging_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        model_max_length=model_max_length,
        memory_profile_steps=memory_profile_steps,
        profile_only=profile_only,
        run_step0_eval=run_step0_eval,
        heldout_eval_subset_size=heldout_eval_subset_size,
        training_inputs_json=training_inputs_json,
        heldout_evals_json=heldout_evals_json,
        step0_compare_generate=step0_compare_generate,
        freeze_audio_encoder=freeze_audio_encoder,
        stock_full_ft_profile=stock_full_ft_profile,
        gradient_checkpointing=gradient_checkpointing,
    )


@app.function(
    image=stock_full_finetune_image,
    gpu=DUAL_TRAIN_GPU,
    volumes={
        DATA_MOUNT: rad_volume,
        HF_CACHE_MOUNT: hf_cache,
    },
    secrets=_require_training_runtime_secrets(),
    timeout=86400,
    retries=0,
)
def run_stock_full_finetune_dual_fast_with_wandb(
    pretrained_path: str = DEFAULT_PRETRAINED_PATH,
    data_file: str = DEFAULT_COMBINED_CODES,
    model_name: str = DEFAULT_KIMI_MODEL_NAME,
    bundle_name: str = "",
    run_name: str = "",
    output_dir: str = "",
    notes: str = "",
    tags: str = "",
    overwrite_run: bool = False,
    resume_run: bool = False,
    tracking_backend: str = "wandb",
    wandb_project: str = "",
    wandb_entity: str = "",
    wandb_mode: str = "",
    seed: int = 1,
    num_train_epochs: float = 1.0,
    max_steps: int = 100,
    save_steps: int = DEFAULT_STOCK_FULL_FT_SAVE_STEPS,
    eval_steps: int = DEFAULT_STOCK_FULL_FT_EVAL_STEPS,
    learning_rate: float = DEFAULT_STOCK_FULL_FT_LEARNING_RATE,
    gradient_accumulation_steps: int = 1,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    logging_steps: int = DEFAULT_WANDB_LOGGING_STEPS,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.1,
    model_max_length: int = 512,
    memory_profile_steps: int = 0,
    profile_only: bool = False,
    run_step0_eval: bool = False,
    heldout_eval_subset_size: int = 0,
    training_inputs_json: str = "",
    heldout_evals_json: str = "[]",
    step0_compare_generate: bool = False,
    freeze_audio_encoder: bool = False,
    stock_full_ft_profile: str = STOCK_FULL_FT_PROFILE_DUAL_FAST,
    gradient_checkpointing: Any = DEFAULT_GRADIENT_CHECKPOINTING_MODE,
) -> dict[str, Any]:
    return run_stock_full_finetune_dual_fast.local(
        pretrained_path=pretrained_path,
        data_file=data_file,
        model_name=model_name,
        bundle_name=bundle_name,
        run_name=run_name,
        output_dir=output_dir,
        notes=notes,
        tags=tags,
        overwrite_run=overwrite_run,
        resume_run=resume_run,
        tracking_backend=tracking_backend,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        seed=seed,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        logging_steps=logging_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        model_max_length=model_max_length,
        memory_profile_steps=memory_profile_steps,
        profile_only=profile_only,
        run_step0_eval=run_step0_eval,
        heldout_eval_subset_size=heldout_eval_subset_size,
        training_inputs_json=training_inputs_json,
        heldout_evals_json=heldout_evals_json,
        step0_compare_generate=step0_compare_generate,
        freeze_audio_encoder=freeze_audio_encoder,
        stock_full_ft_profile=stock_full_ft_profile,
        gradient_checkpointing=gradient_checkpointing,
    )


@app.function(
    image=stock_full_finetune_image,
    gpu=QUAD_TRAIN_GPU,
    volumes={
        DATA_MOUNT: rad_volume,
        HF_CACHE_MOUNT: hf_cache,
    },
    secrets=_require_training_runtime_secrets(),
    timeout=86400,
    retries=0,
)
def run_stock_full_finetune_quad_fast(
    pretrained_path: str = DEFAULT_PRETRAINED_PATH,
    data_file: str = DEFAULT_COMBINED_CODES,
    model_name: str = DEFAULT_KIMI_MODEL_NAME,
    bundle_name: str = "",
    run_name: str = "",
    output_dir: str = "",
    notes: str = "",
    tags: str = "",
    overwrite_run: bool = False,
    resume_run: bool = False,
    tracking_backend: str = "wandb",
    wandb_project: str = "",
    wandb_entity: str = "",
    wandb_mode: str = "",
    seed: int = 1,
    num_train_epochs: float = 1.0,
    max_steps: int = 100,
    save_steps: int = DEFAULT_STOCK_FULL_FT_SAVE_STEPS,
    eval_steps: int = DEFAULT_STOCK_FULL_FT_EVAL_STEPS,
    learning_rate: float = DEFAULT_STOCK_FULL_FT_LEARNING_RATE,
    gradient_accumulation_steps: int = 1,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    logging_steps: int = DEFAULT_WANDB_LOGGING_STEPS,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.1,
    model_max_length: int = 512,
    memory_profile_steps: int = 0,
    profile_only: bool = False,
    run_step0_eval: bool = False,
    heldout_eval_subset_size: int = 0,
    training_inputs_json: str = "",
    heldout_evals_json: str = "[]",
    step0_compare_generate: bool = False,
    freeze_audio_encoder: bool = False,
    stock_full_ft_profile: str = STOCK_FULL_FT_PROFILE_QUAD_FAST,
    gradient_checkpointing: Any = DEFAULT_GRADIENT_CHECKPOINTING_MODE,
) -> dict[str, Any]:
    return _run_stock_full_finetune_impl(
        pretrained_path=pretrained_path,
        data_file=data_file,
        model_name=model_name,
        bundle_name=bundle_name,
        run_name=run_name,
        output_dir=output_dir,
        notes=notes,
        tags=tags,
        overwrite_run=overwrite_run,
        resume_run=resume_run,
        tracking_backend=tracking_backend,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        seed=seed,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        logging_steps=logging_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        model_max_length=model_max_length,
        memory_profile_steps=memory_profile_steps,
        profile_only=profile_only,
        run_step0_eval=run_step0_eval,
        heldout_eval_subset_size=heldout_eval_subset_size,
        training_inputs_json=training_inputs_json,
        heldout_evals_json=heldout_evals_json,
        step0_compare_generate=step0_compare_generate,
        freeze_audio_encoder=freeze_audio_encoder,
        stock_full_ft_profile=stock_full_ft_profile,
        gradient_checkpointing=gradient_checkpointing,
    )


@app.function(
    image=stock_full_finetune_image,
    gpu=QUAD_TRAIN_GPU,
    volumes={
        DATA_MOUNT: rad_volume,
        HF_CACHE_MOUNT: hf_cache,
    },
    secrets=_require_training_runtime_secrets(),
    timeout=86400,
    retries=0,
)
def run_stock_full_finetune_quad_fast_with_wandb(
    pretrained_path: str = DEFAULT_PRETRAINED_PATH,
    data_file: str = DEFAULT_COMBINED_CODES,
    model_name: str = DEFAULT_KIMI_MODEL_NAME,
    bundle_name: str = "",
    run_name: str = "",
    output_dir: str = "",
    notes: str = "",
    tags: str = "",
    overwrite_run: bool = False,
    resume_run: bool = False,
    tracking_backend: str = "wandb",
    wandb_project: str = "",
    wandb_entity: str = "",
    wandb_mode: str = "",
    seed: int = 1,
    num_train_epochs: float = 1.0,
    max_steps: int = 100,
    save_steps: int = DEFAULT_STOCK_FULL_FT_SAVE_STEPS,
    eval_steps: int = DEFAULT_STOCK_FULL_FT_EVAL_STEPS,
    learning_rate: float = DEFAULT_STOCK_FULL_FT_LEARNING_RATE,
    gradient_accumulation_steps: int = 1,
    per_device_train_batch_size: int = 1,
    per_device_eval_batch_size: int = 1,
    logging_steps: int = DEFAULT_WANDB_LOGGING_STEPS,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.1,
    model_max_length: int = 512,
    memory_profile_steps: int = 0,
    profile_only: bool = False,
    run_step0_eval: bool = False,
    heldout_eval_subset_size: int = 0,
    training_inputs_json: str = "",
    heldout_evals_json: str = "[]",
    step0_compare_generate: bool = False,
    freeze_audio_encoder: bool = False,
    stock_full_ft_profile: str = STOCK_FULL_FT_PROFILE_QUAD_FAST,
    gradient_checkpointing: Any = DEFAULT_GRADIENT_CHECKPOINTING_MODE,
) -> dict[str, Any]:
    return run_stock_full_finetune_quad_fast.local(
        pretrained_path=pretrained_path,
        data_file=data_file,
        model_name=model_name,
        bundle_name=bundle_name,
        run_name=run_name,
        output_dir=output_dir,
        notes=notes,
        tags=tags,
        overwrite_run=overwrite_run,
        resume_run=resume_run,
        tracking_backend=tracking_backend,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        seed=seed,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        logging_steps=logging_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        model_max_length=model_max_length,
        memory_profile_steps=memory_profile_steps,
        profile_only=profile_only,
        run_step0_eval=run_step0_eval,
        heldout_eval_subset_size=heldout_eval_subset_size,
        training_inputs_json=training_inputs_json,
        heldout_evals_json=heldout_evals_json,
        step0_compare_generate=step0_compare_generate,
        freeze_audio_encoder=freeze_audio_encoder,
        stock_full_ft_profile=stock_full_ft_profile,
        gradient_checkpointing=gradient_checkpointing,
    )


@app.function(
    image=finetune_image,
    gpu="A100-80GB",
    volumes={
        DATA_MOUNT: rad_volume,
        HF_CACHE_MOUNT: hf_cache,
    },
    secrets=hf_secrets,
    timeout=86400,
    retries=0,
)
def export_inference_checkpoint(
    input_dir: str,
    output_dir: str = DEFAULT_SERVING_MODEL_PATH,
    archive_output_dir: str = "",
    model_name: str = DEFAULT_KIMI_MODEL_NAME,
    base_model_path: str = DEFAULT_PRETRAINED_PATH,
) -> dict[str, Any]:
    """Merge a RAD LoRA adapter if needed, then export an inference-ready checkpoint."""
    model_name = _resolve_supported_kimi_model_name(model_name)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(input_dir)
    requested_output_dir = str(output_dir or "").strip()
    requested_archive_output_dir = str(archive_output_dir or "").strip()
    # When the caller provides a checkpoint archive path, export straight there
    # instead of writing the full package twice and copying it afterward.
    effective_output_dir = requested_archive_output_dir or requested_output_dir
    if not effective_output_dir:
        raise ValueError("export_inference_checkpoint requires output_dir or archive_output_dir")
    os.makedirs(os.path.dirname(effective_output_dir) or ".", exist_ok=True)
    if os.path.isdir(effective_output_dir):
        shutil.rmtree(effective_output_dir)
    merged_from_lora = False
    log_chunks: list[str] = []

    if _looks_like_lora_adapter_dir(input_dir):
        if not os.path.isdir(base_model_path):
            raise FileNotFoundError(
                f"LoRA adapter packaging requires base_model_path, missing: {base_model_path}"
            )
        package_tmp_root = Path(DEFAULT_PACKAGE_TMP_ROOT)
        package_tmp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=str(package_tmp_root), prefix="kimi-rad-merge-"
        ) as tmp_dir:
            merged_input_dir = Path(tmp_dir) / "merged_preexport"
            merge_cmd = [
                "python",
                "-m",
                RAD_MERGE_LORA_MODULE,
                "--base-model-path",
                base_model_path,
                "--adapter-dir",
                input_dir,
                "--output-dir",
                str(merged_input_dir),
            ]
            log_chunks.append(_run_in_kimi_repo(merge_cmd))
            export_cmd = [
                "python",
                "-m",
                VENDORED_KIMI_MODEL_MODULE,
                "--model_name",
                model_name,
                "--action",
                "export_model",
                "--input_dir",
                str(merged_input_dir),
                "--output_dir",
                effective_output_dir,
            ]
            log_chunks.append(_run_in_kimi_repo(export_cmd))
        merged_from_lora = True
    else:
        export_cmd = [
            "python",
            "-m",
            VENDORED_KIMI_MODEL_MODULE,
            "--model_name",
            model_name,
            "--action",
            "export_model",
            "--input_dir",
            input_dir,
            "--output_dir",
            effective_output_dir,
        ]
        log_chunks.append(_run_in_kimi_repo(export_cmd))

    # Re-copy the vendored modeling files explicitly so packaged inference keeps
    # the same multi-audio behavior as the training/export container.
    shutil.copy2(
        VENDORED_KIMI_FINETUNE_CODES_DIR / "modeling_kimia.py",
        Path(effective_output_dir) / "modeling_moonshot_kimia.py",
    )
    shutil.copy2(
        VENDORED_KIMI_FINETUNE_CODES_DIR / "configuration_moonshot_kimia.py",
        Path(effective_output_dir) / "configuration_moonshot_kimia.py",
    )
    copied_runtime_assets = _copy_kimi_runtime_assets(
        model_name=model_name,
        output_dir=effective_output_dir,
    )

    rad_volume.commit()
    hf_cache.commit()
    return {
        "status": "ok",
        "input_dir": input_dir,
        "requested_output_dir": requested_output_dir or None,
        "output_dir": effective_output_dir,
        "archive_output_dir": requested_archive_output_dir or None,
        "single_write_export": bool(requested_archive_output_dir),
        "training_mode": "lora-peft-frozen-audio"
        if merged_from_lora
        else "full-checkpoint-export",
        "merged_from_lora_adapter": merged_from_lora,
        "base_model_path": base_model_path if merged_from_lora else None,
        "copied_runtime_assets": copied_runtime_assets,
        "log_tail": "\n".join(log_chunks)[-8000:],
    }


@app.function(
    image=finetune_image,
    volumes={
        DATA_MOUNT: rad_volume,
    },
    timeout=3600,
    retries=0,
)
def verify_volume_training_inputs(
    train_input_file: str = DEFAULT_TRAIN_JSON,
    val_input_file: str = DEFAULT_VAL_JSON,
    heldout_eval_manifest: str = "",
    audio_root: str = DEFAULT_AUDIO_ROOT,
) -> dict[str, Any]:
    train_summary = _summarize_export_jsonl(
        Path(train_input_file),
        expected_audio_root=audio_root,
        require_volume_audio=True,
    )
    val_summary = _summarize_export_jsonl(
        Path(val_input_file),
        expected_audio_root=audio_root,
        require_volume_audio=True,
        allow_empty=True,
    )
    heldout_summary = None
    if heldout_eval_manifest.strip():
        heldout_summary = _summarize_export_jsonl(
            Path(heldout_eval_manifest),
            expected_audio_root=audio_root,
            require_volume_audio=True,
        )
    return {
        "status": "ok",
        "train": train_summary,
        "val": val_summary,
        "heldout": heldout_summary,
        "train_input_file": train_input_file,
        "val_input_file": val_input_file,
        "heldout_eval_manifest": heldout_eval_manifest or None,
        "audio_root": audio_root,
    }


@app.function(
    image=stock_full_finetune_image,
    secrets=_require_wandb_training_secrets(),
    volumes={DATA_MOUNT: rad_volume},
    timeout=900,
    retries=0,
)
def log_checkpoint_eval_to_wandb(
    run_name: str,
    wandb_run_id: str,
    checkpoint_step: int,
    accuracy_pct_total: float | None = None,
    accuracy_pct_parsed: float | None = None,
    correct: int | None = None,
    valid_predictions: int | None = None,
    total_examples: int | None = None,
    choice_extraction_llm_total: int | None = None,
    choice_extraction_unsure_total: int | None = None,
    choice_extraction_refusal_total: int | None = None,
    eval_tag: str = "",
    wandb_metric_prefix: str = "",
) -> dict[str, Any]:
    run_manifest = _read_json_if_exists(resolve_run_paths(run_name).run_manifest)
    if not run_manifest:
        raise FileNotFoundError(f"Missing run manifest for {run_name!r}")
    if str(run_manifest.get("tracking_backend") or "").strip().lower() != "wandb":
        raise RuntimeError(f"Run {run_name!r} is not configured for W&B tracking")

    wandb_project = str(run_manifest.get("wandb_project") or "").strip()
    wandb_entity = str(run_manifest.get("wandb_entity") or "").strip()
    wandb_mode = str(run_manifest.get("wandb_mode") or "").strip()
    tags = csv_tags(run_manifest.get("tags") or [])
    wandb_env = _wandb_runtime_env(
        run_name=run_name,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
        tags=tags,
    )
    simplified_metrics = simplified_checkpoint_eval_summary(
        {
            "accuracy_pct_total": accuracy_pct_total,
            "accuracy_pct_parsed": accuracy_pct_parsed,
            "valid_predictions": valid_predictions,
            "total_examples": total_examples,
            "choice_extraction_llm_total": choice_extraction_llm_total,
            "choice_extraction_unsure_total": choice_extraction_unsure_total,
            "choice_extraction_refusal_total": choice_extraction_refusal_total,
        }
    )
    metric_names = checkpoint_eval_metric_names(
        eval_tag=eval_tag or None,
        wandb_metric_prefix=wandb_metric_prefix or None,
    )
    metrics = {
        metric_names["checkpoint_step"]: int(checkpoint_step),
        metric_names["overall_accuracy_pct"]: simplified_metrics.get(
            "overall_accuracy_pct"
        ),
        metric_names["accuracy_on_valid_pct"]: simplified_metrics.get(
            "accuracy_on_valid_pct"
        ),
        metric_names["unparseable_or_unsure_pct"]: simplified_metrics.get(
            "unparseable_or_unsure_pct"
        ),
    }
    metrics = {key: value for key, value in metrics.items() if value is not None}
    payload = {
        "status": "ok",
        "run_name": run_name,
        "wandb_run_id": wandb_run_id,
        "checkpoint_step": checkpoint_step,
        "metrics": metrics,
    }
    metrics_json = json.dumps(metrics)
    payload_json = json.dumps(payload)
    log_cmd = [
        "python",
        "-m",
        RAD_WANDB_CHECKPOINT_LOG_MODULE,
        "--metrics-json",
        metrics_json,
        "--project",
        wandb_project,
        "--entity",
        wandb_entity,
        "--run-id",
        wandb_run_id,
        "--payload-json",
        payload_json,
    ]
    output = _run_in_kimi_repo(log_cmd, extra_env=wandb_env)
    last_line = next((line for line in reversed(output.splitlines()) if line.strip()), "")
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return payload


@app.local_entrypoint()
def main(
    step: str = "help",
    modal_env: str = DEFAULT_MODAL_ENV,
    staged_bundle_dir: str = "",
    bundle_name: str = "",
    run_name: str = "",
    model_name: str = DEFAULT_KIMI_MODEL_NAME,
    train_input_file: str = DEFAULT_TRAIN_JSON,
    val_input_file: str = DEFAULT_VAL_JSON,
    train_output_file: str = DEFAULT_TRAIN_CODES,
    val_output_file: str = DEFAULT_VAL_CODES,
    combined_output_file: str = DEFAULT_COMBINED_CODES,
    training_inputs_json: str = "",
    heldout_evals_json: str = "[]",
    pretrained_path: str = DEFAULT_PRETRAINED_PATH,
    data_file: str = DEFAULT_COMBINED_CODES,
    output_dir: str = "",
    package_input_dir: str = "",
    package_output_dir: str = DEFAULT_SERVING_MODEL_PATH,
    archive_output_dir: str = "",
    overwrite_bundle: bool = False,
    overwrite_run: bool = False,
    resume_run: bool = False,
    notes: str = "",
    tags: str = "",
    trainer: str = DEFAULT_TRAINER,
    tracking_backend: str = "wandb",
    wandb_project: str = "",
    wandb_entity: str = "",
    wandb_mode: str = "",
    seed: int = 1,
    num_train_epochs: float = 1.0,
    max_steps: int = 100,
    save_steps: int | None = None,
    eval_steps: int | None = None,
    learning_rate: float | None = None,
    gradient_accumulation_steps: int | None = None,
    per_device_train_batch_size: int | None = None,
    per_device_eval_batch_size: int | None = None,
    logging_steps: int = DEFAULT_WANDB_LOGGING_STEPS,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.1,
    model_max_length: int = 512,
    memory_profile_steps: int = 0,
    profile_only: bool = False,
    run_step0_eval: bool = False,
    heldout_eval_subset_size: int = 0,
    step0_compare_generate: bool = False,
    freeze_audio_encoder: bool = False,
    stock_full_ft_profile: str = DEFAULT_STOCK_FULL_FT_PROFILE,
    gradient_checkpointing: str = DEFAULT_GRADIENT_CHECKPOINTING_MODE,
    lora_r: int = DEFAULT_LORA_R,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    lora_target_modules: str = DEFAULT_LORA_TARGET_MODULES_CSV,
    lora_include_audio_encoder: bool = False,
    checkpoint_step: int = 0,
    accuracy_pct_total: float = -1.0,
    accuracy_pct_parsed: float = -1.0,
    correct: int = -1,
    valid_predictions: int = -1,
    total_examples: int = -1,
    choice_extraction_llm_total: int = -1,
    choice_extraction_unsure_total: int = -1,
    choice_extraction_refusal_total: int = -1,
    eval_tag: str = "",
    wandb_metric_prefix: str = "",
    wandb_run_id: str = "",
) -> None:
    """Dispatch one pipeline step remotely.

    Steps: `upload-bundle` | `prepare-model` | `prepare-data` | `profile` | `train` | `publish` | `run`
    """
    if step in ("help", "-h", "--help"):
        print(
            __doc__
            + "\nExamples:\n"
            "  modal run ...::main --step upload-bundle --staged-bundle-dir /tmp/rad_bundle\n"
            "  modal run ...::main --step prepare-model\n"
            "  modal run ...::main --step prepare-data\n"
            "  modal run ...::main --step profile --per-device-train-batch-size 2 --memory-profile-steps 2\n"
            "  modal run ...::main --step train --max-steps 500 --run-step0-eval\n"
            "  modal run ...::main --step train --max-steps 500 --run-step0-eval --step0-compare-generate\n"
            "  modal run ...::main --step train --max-steps 25\n"
            "  modal run ...::main --step train --max-steps 50 --freeze-audio-encoder\n"
            "  modal run ...::main --step train --max-steps 25 --trainer lora --lora-r 16\n"
            "  modal run ...::main --step train --stock-full-ft-profile dual-fast --gradient-checkpointing false\n"
            "  modal run ...::main --step publish\n"
            "  modal run ...::main --step wandb-checkpoint-eval-log --run-name demo --wandb-run-id abc123 --checkpoint-step 50 --accuracy-pct-total 55.2\n"
            "  modal run ...::main --step run\n"
            "\nLegacy aliases still work: sync-data, export, codes, package, all.\n"
            "\nDefault training mode: stock-full-ft. Use --trainer lora only if you need the LoRA path.\n"
            f"W&B tracking is required for training runs via Modal secret `{WANDB_SECRET_NAME}`.\n"
        )
        return

    step = _normalize_step_name(step)
    trainer = _normalize_trainer(trainer)
    model_name = _resolve_supported_kimi_model_name(model_name)
    train_launch_defaults = _resolve_train_launch_defaults(
        trainer=trainer,
        save_steps=save_steps,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
    )
    save_steps = int(train_launch_defaults["save_steps"])
    eval_steps = int(train_launch_defaults["eval_steps"])
    learning_rate = float(train_launch_defaults["learning_rate"])
    gradient_accumulation_steps = int(train_launch_defaults["gradient_accumulation_steps"])
    per_device_train_batch_size = int(train_launch_defaults["per_device_train_batch_size"])
    per_device_eval_batch_size = int(train_launch_defaults["per_device_eval_batch_size"])

    cli_path_overrides = _normalize_bundle_cli_path_overrides(
        bundle_name=bundle_name,
        train_input_file=train_input_file,
        val_input_file=val_input_file,
        train_output_file=train_output_file,
        val_output_file=val_output_file,
        combined_output_file=combined_output_file,
        data_file=data_file,
    )
    train_input_file = cli_path_overrides["train_input_file"]
    val_input_file = cli_path_overrides["val_input_file"]
    train_output_file = cli_path_overrides["train_output_file"]
    val_output_file = cli_path_overrides["val_output_file"]
    combined_output_file = cli_path_overrides["combined_output_file"]
    data_file = cli_path_overrides["data_file"]

    if step == STEP_UPLOAD_BUNDLE:
        if not staged_bundle_dir.strip():
            raise SystemExit("--staged-bundle-dir is required for --step upload-bundle")
        if not bundle_name.strip():
            raise SystemExit("--bundle-name is required for --step upload-bundle")
        resolved = _resolve_training_paths(bundle_name=bundle_name)
        print(
            _sync_staged_bundle_to_volume(
                staged_bundle_dir=staged_bundle_dir,
                modal_env=modal_env,
                bundle_name=bundle_name,
                train_input_file=resolved["train_input_file"],
                val_input_file=resolved["val_input_file"],
                audio_root=DEFAULT_AUDIO_ROOT,
                overwrite_bundle=overwrite_bundle,
            )
        )
    elif step == STEP_PREPARE_MODEL:
        print(
            export_pretrained_hf.remote(
                model_name=model_name,
                output_dir=pretrained_path,
            )
        )
    elif step == STEP_PREPARE_DATA:
        print(
            prepare_semantic_codes.remote(
                bundle_name=bundle_name,
                train_input_file=train_input_file,
                val_input_file=val_input_file,
                train_output_file=train_output_file,
                val_output_file=val_output_file,
                combined_output_file=combined_output_file,
                training_inputs_json=training_inputs_json,
                heldout_evals_json=heldout_evals_json,
                model_name_or_path=model_name,
            )
        )
    elif step == STEP_TRAIN:
        if not run_name.strip():
            raise SystemExit("--run-name is required for --step train")
        train_result = _run_train_remote(
            trainer=trainer,
            pretrained_path=pretrained_path,
            data_file=data_file,
            model_name=model_name,
            bundle_name=bundle_name,
            run_name=run_name,
            output_dir=output_dir,
            notes=notes,
            tags=tags,
            overwrite_run=overwrite_run,
            resume_run=resume_run,
            tracking_backend=tracking_backend,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_mode=wandb_mode,
            seed=seed,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            logging_steps=logging_steps,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            model_max_length=model_max_length,
            memory_profile_steps=memory_profile_steps,
            profile_only=profile_only,
            run_step0_eval=run_step0_eval,
            heldout_eval_subset_size=heldout_eval_subset_size,
            training_inputs_json=training_inputs_json,
            heldout_evals_json=heldout_evals_json,
            step0_compare_generate=step0_compare_generate,
            freeze_audio_encoder=freeze_audio_encoder,
            stock_full_ft_profile=stock_full_ft_profile,
            gradient_checkpointing=gradient_checkpointing,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        lora_include_audio_encoder=lora_include_audio_encoder,
            lora_target_modules=lora_target_modules,
        )
        print(train_result)
        _maybe_print_wandb_run_url(train_result)
    elif step == STEP_PROFILE:
        if not run_name.strip():
            raise SystemExit("--run-name is required for --step profile")
        if trainer != TRAINER_LORA:
            raise SystemExit(
                "--step profile only supports LoRA memory profiling; pass --trainer lora "
                "(default trainer is stock-full-ft)"
            )
        profile_result = _run_train_remote(
            trainer=trainer,
            pretrained_path=pretrained_path,
            data_file=data_file,
            model_name=model_name,
            bundle_name=bundle_name,
            run_name=run_name,
            output_dir=output_dir,
            notes=notes,
            tags=tags,
            overwrite_run=overwrite_run,
            resume_run=resume_run,
            tracking_backend=tracking_backend,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_mode=wandb_mode,
            seed=seed,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            logging_steps=logging_steps,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            model_max_length=model_max_length,
            memory_profile_steps=max(1, memory_profile_steps),
            profile_only=True,
            run_step0_eval=run_step0_eval,
            heldout_eval_subset_size=heldout_eval_subset_size,
            training_inputs_json=training_inputs_json,
            heldout_evals_json=heldout_evals_json,
            stock_full_ft_profile=stock_full_ft_profile,
            gradient_checkpointing=gradient_checkpointing,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        lora_include_audio_encoder=lora_include_audio_encoder,
            lora_target_modules=lora_target_modules,
        )
        print(profile_result)
        _maybe_print_wandb_run_url(profile_result)
    elif step == STEP_PUBLISH:
        source_dir = package_input_dir.strip() or output_dir.strip()
        if not source_dir and run_name.strip():
            source_dir = str(resolve_run_paths(run_name).train_dir)
        if not archive_output_dir.strip() and run_name.strip():
            source_dir_for_slug = source_dir.rstrip("/") if source_dir else ""
            if source_dir_for_slug:
                slug = Path(source_dir_for_slug).name.replace("_", "-")
                archive_output_dir = str(resolve_run_paths(run_name).promotion_archive_dir(slug))
        if not source_dir:
            raise SystemExit(
                "--package-input-dir or --output-dir is required for --step publish"
            )
        publish_result = export_inference_checkpoint.remote(
            input_dir=source_dir,
            output_dir=package_output_dir,
            archive_output_dir=archive_output_dir,
            model_name=model_name,
            base_model_path=pretrained_path,
        )
        print(publish_result)
        print(
            _deploy_and_validate_kimi_ft_serving(
                modal_env=modal_env,
                model_path_override=archive_output_dir.strip() or package_output_dir,
            )
        )
    elif step == STEP_WANDB_CHECKPOINT_EVAL_LOG:
        if not run_name.strip():
            raise SystemExit("--run-name is required for --step wandb-checkpoint-eval-log")
        if not wandb_run_id.strip():
            raise SystemExit("--wandb-run-id is required for --step wandb-checkpoint-eval-log")
        if checkpoint_step <= 0:
            raise SystemExit("--checkpoint-step must be positive for --step wandb-checkpoint-eval-log")
        print(
            log_checkpoint_eval_to_wandb.remote(
                run_name=run_name,
                wandb_run_id=wandb_run_id,
                checkpoint_step=checkpoint_step,
                accuracy_pct_total=None if accuracy_pct_total < 0 else accuracy_pct_total,
                accuracy_pct_parsed=None if accuracy_pct_parsed < 0 else accuracy_pct_parsed,
                correct=None if correct < 0 else correct,
                valid_predictions=None if valid_predictions < 0 else valid_predictions,
                total_examples=None if total_examples < 0 else total_examples,
                choice_extraction_llm_total=None
                if choice_extraction_llm_total < 0
                else choice_extraction_llm_total,
                choice_extraction_unsure_total=None
                if choice_extraction_unsure_total < 0
                else choice_extraction_unsure_total,
                choice_extraction_refusal_total=None
                if choice_extraction_refusal_total < 0
                else choice_extraction_refusal_total,
                eval_tag=eval_tag,
                wandb_metric_prefix=wandb_metric_prefix,
            )
        )
    elif step == STEP_RUN:
        if not bundle_name.strip():
            raise SystemExit("--bundle-name is required for --step run")
        if not run_name.strip():
            raise SystemExit("--run-name is required for --step run")
        export_result = export_pretrained_hf.remote(
            model_name=model_name,
            output_dir=pretrained_path,
        )
        print(export_result)
        codes_result = prepare_semantic_codes.remote(
            bundle_name=bundle_name,
            train_input_file=train_input_file,
            val_input_file=val_input_file,
            train_output_file=train_output_file,
            val_output_file=val_output_file,
            combined_output_file=combined_output_file,
            training_inputs_json=training_inputs_json,
            heldout_evals_json=heldout_evals_json,
            model_name_or_path=model_name,
        )
        print(codes_result)
        train_result = _run_train_remote(
            trainer=trainer,
            pretrained_path=pretrained_path,
            data_file=data_file,
            model_name=model_name,
            bundle_name=bundle_name,
            run_name=run_name,
            output_dir=output_dir,
            notes=notes,
            tags=tags,
            overwrite_run=overwrite_run,
            resume_run=resume_run,
            tracking_backend=tracking_backend,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_mode=wandb_mode,
            seed=seed,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            learning_rate=learning_rate,
            gradient_accumulation_steps=gradient_accumulation_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            logging_steps=logging_steps,
            warmup_ratio=warmup_ratio,
            weight_decay=weight_decay,
            model_max_length=model_max_length,
            heldout_eval_subset_size=heldout_eval_subset_size,
            training_inputs_json=training_inputs_json,
            heldout_evals_json=heldout_evals_json,
            freeze_audio_encoder=freeze_audio_encoder,
            run_step0_eval=run_step0_eval,
            stock_full_ft_profile=stock_full_ft_profile,
            gradient_checkpointing=gradient_checkpointing,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        lora_include_audio_encoder=lora_include_audio_encoder,
            lora_target_modules=lora_target_modules,
        )
        print(train_result)
        _maybe_print_wandb_run_url(train_result)
        resolved_archive_output_dir = archive_output_dir
        if not resolved_archive_output_dir.strip():
            source_dir = package_input_dir.strip() or train_result["output_dir"]
            slug = Path(source_dir.rstrip("/")).name.replace("_", "-")
            resolved_archive_output_dir = str(
                resolve_run_paths(run_name).promotion_archive_dir(slug)
            )
        print(
            export_inference_checkpoint.remote(
                input_dir=package_input_dir.strip() or train_result["output_dir"],
                output_dir=package_output_dir,
                archive_output_dir=resolved_archive_output_dir,
                model_name=model_name,
                base_model_path=pretrained_path,
            )
        )
    else:
        raise SystemExit(f"Unknown --step {step!r}")
