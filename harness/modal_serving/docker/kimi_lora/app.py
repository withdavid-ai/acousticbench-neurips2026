"""FastAPI shim mirroring KimiLoRAServer in kimi_lora_serve.py.

Modal -> Docker mapping:

  * @modal.enter() + safe_enter(load_model) -> FastAPI lifespan + _Handle.load()
  * @modal.method() KimiLoRAServer.generate(item) -> POST /generate
  * @app.function() prefetch() -> POST /prefetch (re-exports the merged ckpt)
  * volumes={CACHE_PATH: hf_cache} -> -v <host>:/hf-cache at docker run
  * volumes={DATA_VOLUME_PATH: adapter_volume} -> -v <archive>:/adapters at docker run

Adapter selection is driven by the ADAPTER_NAME env var (one container per
LoRA), to match how the Modal class is invoked with adapter_name="rad" etc.
"""
from __future__ import annotations

import logging
import json
import os
import shutil
import tempfile
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from serving_utils import (
    apply_benchmark_sampling,
    decode_messages_payload,
    flatten_messages_to_text,
    hf_revision,
    materialize_audio_files,
    render_messages,
)

logger = logging.getLogger("kimi_lora_serve")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

BASE_MODEL_ID = "moonshotai/Kimi-Audio-7B-Instruct"
BASE_MODEL_REVISION = hf_revision(BASE_MODEL_ID)
ADAPTER_NAME = os.environ.get("ADAPTER_NAME", "rad").strip()
ADAPTER_ROOT = Path(os.environ.get("ADAPTER_ROOT", "/adapters"))
PRETRAINED_ROOT = Path(os.environ.get("PRETRAINED_ROOT", "/var/lib/kimi-lora-exports/pretrained_hf"))
EXPORT_ROOT = Path(os.environ.get("EXPORT_ROOT", "/var/lib/kimi-lora-exports/exported"))

VALID_ADAPTERS = ("rad", "rsq", "ser_classify", "ser_intensity", "simple_audioqa")
ADAPTER_SUBDIRS = ("", "models", "finetunes")
EXPORT_PACKAGING_VERSION = "acousticbench-vendored-kimi-lora-v1"


def _to_kimi_native_messages(messages, audio_paths):
    native_messages = []
    for message in render_messages(
        messages,
        audio_paths,
        build_text_part=lambda text, _role: (
            {"message_type": "text", "content": text} if str(text).strip() else None
        ),
        build_audio_part=lambda audio_path, _audio_index, _role: {
            "message_type": "audio",
            "content": audio_path,
        },
    ):
        role = "assistant" if message["role"] == "assistant" else "user"
        if message["role"] == "system":
            logger.warning(
                "Kimi-Audio remapping system role to user role for provider compatibility."
            )
        native_messages.extend(
            {"role": role, **part}
            for part in message["content"]
        )
    return native_messages


def _generate_text_only(model, native_messages, sampling_params):
    import torch

    history = model.prompt_manager.get_prompt(native_messages, output_type="text")
    audio_input_ids, text_input_ids, is_continuous_mask, _, _ = history.to_tensor()
    max_new_tokens = int(sampling_params.get("max_new_tokens", -1))
    if max_new_tokens == -1:
        max_new_tokens = 7500 - audio_input_ids.shape[1]

    audio_input_ids = audio_input_ids.to(torch.cuda.current_device())
    text_input_ids = text_input_ids.to(torch.cuda.current_device())
    is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())

    audio_features = None
    if history.continuous_feature:
        audio_features = [
            feature.to(torch.cuda.current_device())
            for feature in history.continuous_feature
        ]

    _, generated_text_tokens = model._generate_loop(
        audio_input_ids=audio_input_ids,
        text_input_ids=text_input_ids,
        max_new_tokens=max_new_tokens,
        audio_temperature=sampling_params["audio_temperature"],
        audio_top_k=sampling_params["audio_top_k"],
        audio_repetition_penalty=sampling_params["audio_repetition_penalty"],
        audio_repetition_window_size=sampling_params["audio_repetition_window_size"],
        text_top_k=sampling_params["text_top_k"],
        text_temperature=sampling_params["text_temperature"],
        text_repetition_penalty=sampling_params["text_repetition_penalty"],
        text_repetition_window_size=sampling_params["text_repetition_window_size"],
        is_continuous_mask=is_continuous_mask,
        continous_feature=audio_features,
        output_type="text",
    )
    generated_text_tokens = [
        token for token in generated_text_tokens if token < model.kimia_token_offset
    ]
    return model.detokenize_text(generated_text_tokens)


def _resolve_adapter_dir(adapter_name: str) -> Path:
    """Find ``<root>/<adapter>`` under ADAPTER_ROOT or its known subdirs.

    Mirrors ``KIMI_ADAPTER_SUBDIRS`` from the harness so the unzipped archive
    layout (``archive/{rad,rsq,...}/`` or ``archive/models/<adapter>/``)
    works with no extra plumbing.
    """
    if adapter_name not in VALID_ADAPTERS:
        raise ValueError(
            f"Unknown Kimi LoRA adapter {adapter_name!r}; "
            f"expected one of {list(VALID_ADAPTERS)}"
        )
    candidates: list[Path] = []
    for subdir in ADAPTER_SUBDIRS:
        candidate = ADAPTER_ROOT / subdir / adapter_name if subdir else ADAPTER_ROOT / adapter_name
        candidates.append(candidate)
        if (candidate / "adapter_model.safetensors").is_file():
            return candidate
    raise FileNotFoundError(
        f"Missing adapter weights for {adapter_name!r}. Looked at:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nMount the unzipped archive (with archive/models/<adapter>/) "
          "at /adapters when running the container."
    )


def _copy_kimi_runtime_assets(
    model_name: str, output_dir: Path, revision: str | None = None,
) -> list[str]:
    from huggingface_hub import snapshot_download

    snapshot_dir = Path(model_name)
    if not snapshot_dir.exists():
        snapshot_dir = Path(snapshot_download(model_name, revision=revision))

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
        dest = output_dir / rel_path
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


def _ensure_pretrained_hf() -> Path:
    if (PRETRAINED_ROOT / "config.json").is_file():
        return PRETRAINED_ROOT

    from kimi_vendor.finetune_codes.model import KimiAudioModel

    t0 = time.perf_counter()
    tmp_root = Path(tempfile.mkdtemp(prefix="kimi-lora-pretrained-", dir="/tmp"))
    try:
        tmp_output = tmp_root / "pretrained_hf"
        logger.info("materializing base pretrained_hf model=%s", BASE_MODEL_ID)
        from huggingface_hub import snapshot_download

        base_snapshot = snapshot_download(BASE_MODEL_ID, revision=BASE_MODEL_REVISION)
        model = KimiAudioModel.init_from_pretrained(base_snapshot, model_load_kwargs={})
        tmp_output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(tmp_output)
        if PRETRAINED_ROOT.exists():
            shutil.rmtree(PRETRAINED_ROOT)
        PRETRAINED_ROOT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_output, PRETRAINED_ROOT)
        logger.info(
            "materialized base pretrained_hf path=%s elapsed_s=%.2f",
            PRETRAINED_ROOT,
            time.perf_counter() - t0,
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return PRETRAINED_ROOT


def _export_lora_adapter(adapter_name: str, adapter_dir: Path) -> Path:
    """Merge LoRA into base weights and save a Kimi-Audio-compatible export.

    Caches the merged checkpoint under ``EXPORT_ROOT/<adapter>``; on subsequent
    boots the function returns immediately if the export already exists.
    """
    import torch

    from kimi_vendor.finetune_codes.model import KimiAudioModel
    from peft import PeftModel

    export_dir = EXPORT_ROOT / adapter_name
    manifest_path = export_dir / "acousticbench_lora_export_manifest.json"
    if (export_dir / "modeling_moonshot_kimia.py").is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        if manifest.get("packaging_version") == EXPORT_PACKAGING_VERSION:
            logger.info(
                "reusing cached LoRA export adapter=%s export_dir=%s",
                adapter_name, export_dir,
            )
            return export_dir
        shutil.rmtree(export_dir)

    if export_dir.exists() and not manifest_path.is_file():
        shutil.rmtree(export_dir)

    pretrained_dir = _ensure_pretrained_hf()
    t0 = time.perf_counter()
    tmp_root = Path(tempfile.mkdtemp(prefix=f"kimi-lora-{adapter_name}-", dir="/tmp"))
    merged_dir = tmp_root / "merged_preexport"
    try:
        logger.info("loading pretrained_hf for adapter=%s path=%s", adapter_name, pretrained_dir)
        base_model = KimiAudioModel.from_pretrained(
            pretrained_dir,
            device_map=None,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        logger.info("loading LoRA adapter adapter=%s dir=%s", adapter_name, adapter_dir)
        adapter_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        merged_model = adapter_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "exporting merged checkpoint adapter=%s export_dir=%s",
            adapter_name, export_dir,
        )
        KimiAudioModel.export_model(str(merged_dir), str(export_dir))
        vendored_dir = Path(__file__).resolve().parent / "kimi_vendor" / "finetune_codes"
        shutil.copy2(vendored_dir / "modeling_kimia.py", export_dir / "modeling_moonshot_kimia.py")
        shutil.copy2(
            vendored_dir / "configuration_moonshot_kimia.py",
            export_dir / "configuration_moonshot_kimia.py",
        )
        copied_assets = _copy_kimi_runtime_assets(
            BASE_MODEL_ID, export_dir, revision=BASE_MODEL_REVISION,
        )
        logger.info("copied Kimi runtime assets adapter=%s assets=%s", adapter_name, copied_assets)
        manifest_path.write_text(
            json.dumps(
                {
                    "packaging_version": EXPORT_PACKAGING_VERSION,
                    "adapter_name": adapter_name,
                    "base_model_id": BASE_MODEL_ID,
                    "base_model_revision": BASE_MODEL_REVISION,
                    "pretrained_root": str(pretrained_dir),
                    "export_dir": str(export_dir),
                    "copied_runtime_assets": copied_assets,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    logger.info(
        "exported LoRA adapter adapter=%s elapsed_s=%.2f",
        adapter_name, time.perf_counter() - t0,
    )
    return export_dir


class _Handle:
    """Mirrors safe_enter semantics: capture init errors instead of crashing."""

    def __init__(self) -> None:
        self.adapter_name: str = ADAPTER_NAME
        self.model: Any = None
        self.model_path: str | None = None
        self.sampling_params: dict = {}
        self.init_error: str | None = None
        self.init_traceback: str | None = None
        self.ready: bool = False
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()

    def load(self) -> None:
        with self._load_lock:
            if self.ready or self.init_error:
                return
            try:
                from kimia_infer.api.kimia import KimiAudio

                adapter_dir = _resolve_adapter_dir(self.adapter_name)
                model_path = _export_lora_adapter(self.adapter_name, adapter_dir)
                self.model = KimiAudio(
                    model_path=str(model_path),
                    load_detokenizer=False,
                )
                self.model_path = str(model_path)
                self.sampling_params = {
                    "audio_temperature": 0.8,
                    "audio_top_k": 10,
                    "text_temperature": 0.0,
                    "text_top_k": 5,
                    "audio_repetition_penalty": 1.0,
                    "audio_repetition_window_size": 64,
                    "text_repetition_penalty": 1.0,
                    "text_repetition_window_size": 16,
                    "max_new_tokens": -1,
                }
                self.ready = True
                logger.info(
                    "Kimi-Audio LoRA loaded: adapter=%s model_path=%s",
                    self.adapter_name, self.model_path,
                )
            except Exception as exc:
                self.init_error = f"{type(exc).__name__}: {exc}"
                self.init_traceback = traceback.format_exc()
                logger.error("Kimi-Audio LoRA init failed:\n%s", self.init_traceback)


HANDLE = _Handle()


def _init_error_response(prompt: str, include_traceback: bool) -> dict[str, Any] | None:
    if not HANDLE.init_error:
        return None
    response: dict[str, Any] = {
        "prompt": prompt,
        "response": f"[INIT_ERROR] {HANDLE.init_error}",
        "submission_payload": {
            "provider": "kimi-audio-lora-init",
            "transport": "docker-startup-error",
            "adapter_name": HANDLE.adapter_name,
            "prompt_text": prompt,
            "error": HANDLE.init_error,
        },
        "submission_trace": {
            "exact_submission_available": True,
            "builder": "KimiLoRAServer.init_error_response",
        },
    }
    if include_traceback:
        response["traceback"] = HANDLE.init_traceback
    return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    HANDLE.load()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    if HANDLE.init_error:
        return JSONResponse(
            status_code=200,
            content={
                "status": "init_error",
                "adapter_name": HANDLE.adapter_name,
                "error": HANDLE.init_error,
            },
        )
    if not HANDLE.ready:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "adapter_name": HANDLE.adapter_name},
        )
    return {
        "status": "ready",
        "adapter_name": HANDLE.adapter_name,
        "model_path": HANDLE.model_path,
        "base_model_id": BASE_MODEL_ID,
    }


@app.post("/prefetch")
def prefetch():
    """Force the merged-checkpoint export for this container's adapter.

    Useful for warming a cold ``/var/lib/kimi-lora-exports`` volume before
    sending traffic.
    """
    adapter_dir = _resolve_adapter_dir(HANDLE.adapter_name)
    export_dir = _export_lora_adapter(HANDLE.adapter_name, adapter_dir)
    return {
        "status": "ok",
        "adapter_name": HANDLE.adapter_name,
        "model_path": str(export_dir),
    }


@app.post("/generate")
def generate(item: dict):
    messages, audio_items, meta = decode_messages_payload(item)
    prompt = flatten_messages_to_text(messages, include_roles=True)

    err = _init_error_response(prompt, include_traceback=bool(meta.get("_debug")))
    if err is not None:
        return err

    if not HANDLE.ready:
        HANDLE.load()
        err = _init_error_response(prompt, include_traceback=bool(meta.get("_debug")))
        if err is not None:
            return err

    sampling_params = apply_benchmark_sampling(
        meta,
        dict(HANDLE.sampling_params),
        do_sample_key=None,
        temperature_key="text_temperature",
        top_p_key=None,
        benchmark_temperature_default=1.0,
    )
    if str(meta.get("task") or "").strip().upper() == "MQA":
        current = int(sampling_params.get("max_new_tokens", 16))
        sampling_params["max_new_tokens"] = min(16, current if current > 0 else 16)

    with materialize_audio_files(audio_items) as audio_paths:
        native_messages = _to_kimi_native_messages(messages, audio_paths)
        with HANDLE._generate_lock:
            if audio_paths:
                _, response = HANDLE.model.generate(
                    native_messages, **sampling_params, output_type="text",
                )
            else:
                response = _generate_text_only(HANDLE.model, native_messages, sampling_params)

    return {
        "prompt": prompt,
        "response": response,
        "submission_payload": {
            "provider": "kimi-audio-lora",
            "transport": "moonshot-kimia-python",
            "adapter_name": HANDLE.adapter_name,
            "model_path": HANDLE.model_path,
            "messages": native_messages,
            "sampling_params": sampling_params,
            "output_type": "text",
        },
        "submission_trace": {
            "exact_submission_available": True,
            "builder": "KimiLoRAServer.generate",
        },
    }
