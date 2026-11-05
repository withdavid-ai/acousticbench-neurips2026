"""FastAPI shim mirroring Qwen3OmniServer in qwen3_omni_serve.py.

Modal -> Docker mapping:

  * @modal.enter() + safe_enter(load_model) -> FastAPI lifespan + _Handle.load()
  * @modal.method() Qwen3OmniServer.generate(item) -> POST /generate
  * @app.function() prefetch() -> POST /prefetch
  * volumes={CACHE_PATH: hf_cache} -> -v <host>:/hf-cache at docker run

Runtime device + dtype overrides:

  * AUDIOBENCH_TORCH_DTYPE     - bfloat16 (default), float16, float32
  * AUDIOBENCH_DEVICE_MAP      - "auto" (default), "cpu", "cuda"
  * AUDIOBENCH_MAX_NEW_TOKENS  - clamps the generation length (default 512)
"""
from __future__ import annotations

import logging
import os
import threading
import traceback
from contextlib import asynccontextmanager
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

logger = logging.getLogger("qwen3_omni_serve")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_REVISION = hf_revision(MODEL_ID)
MODEL_REPOS = [MODEL_ID]
HF_HOME = os.environ.get("HF_HOME", "/hf-cache/huggingface")

DEFAULT_DEVICE_MAP = os.environ.get("AUDIOBENCH_DEVICE_MAP", "auto")
DEFAULT_DTYPE_NAME = os.environ.get("AUDIOBENCH_TORCH_DTYPE", "bfloat16")


def _resolve_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }[name.lower()]


class _Handle:
    """Mirrors safe_enter semantics: capture init errors instead of crashing."""

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.dtype: Any = None
        self.device: str | None = None
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
                import torch
                from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")

                self.dtype = _resolve_dtype(DEFAULT_DTYPE_NAME)
                self.processor = Qwen3OmniMoeProcessor.from_pretrained(
                    MODEL_ID, revision=MODEL_REVISION,
                )
                self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                    MODEL_ID,
                    revision=MODEL_REVISION,
                    device_map=DEFAULT_DEVICE_MAP,
                    torch_dtype=self.dtype,
                    attn_implementation="sdpa",
                ).eval()
                self.device = str(getattr(self.model, "device", "unknown"))
                self.ready = True
                logger.info(
                    "Qwen3-Omni loaded on device=%s dtype=%s",
                    self.device,
                    DEFAULT_DTYPE_NAME,
                )
            except Exception as exc:
                self.init_error = f"{type(exc).__name__}: {exc}"
                self.init_traceback = traceback.format_exc()
                logger.error("Qwen3-Omni init failed:\n%s", self.init_traceback)


HANDLE = _Handle()


def _init_error_response(prompt: str, include_traceback: bool) -> dict[str, Any] | None:
    if not HANDLE.init_error:
        return None
    response: dict[str, Any] = {
        "prompt": prompt,
        "response": f"[INIT_ERROR] {HANDLE.init_error}",
        "submission_payload": {
            "provider": "modal-init",
            "transport": "modal-startup-error",
            "prompt_text": prompt,
            "error": HANDLE.init_error,
        },
        "submission_trace": {
            "exact_submission_available": True,
            "builder": "Qwen3OmniServer.init_error_response",
        },
    }
    if include_traceback:
        response["traceback"] = HANDLE.init_traceback
    return response


def _generation_kwargs(meta: dict) -> dict:
    meta_override = meta.get("_thinker_max_new_tokens")
    if meta_override is not None:
        try:
            thinker_tokens = int(meta_override)
        except (TypeError, ValueError):
            thinker_tokens = None
    else:
        thinker_tokens = None

    return apply_benchmark_sampling(meta, {
        "thinker_max_new_tokens": thinker_tokens if thinker_tokens is not None else 256,
        "thinker_do_sample": False,
    }, do_sample_key="thinker_do_sample", temperature_key=None, top_p_key=None)


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
            content={"status": "init_error", "error": HANDLE.init_error},
        )
    if not HANDLE.ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {
        "status": "ready",
        "device": HANDLE.device,
        "dtype": DEFAULT_DTYPE_NAME,
        "model_id": MODEL_ID,
    }


@app.post("/prefetch")
def prefetch():
    from huggingface_hub import snapshot_download

    downloaded = []
    for repo_id in MODEL_REPOS:
        path = snapshot_download(
            repo_id=repo_id, revision=hf_revision(repo_id), cache_dir=HF_HOME,
        )
        downloaded.append({"repo": repo_id, "path": path})
    return {"status": "ok", "downloaded": downloaded}


@app.post("/generate")
def generate(item: dict):
    import torch
    from qwen_omni_utils import process_mm_info

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

    with materialize_audio_files(audio_items) as audio_paths:
        native_messages = render_messages(
            messages,
            audio_paths,
            build_text_part=lambda text, _role: (
                {"type": "text", "text": text} if str(text).strip() else None
            ),
            build_audio_part=lambda audio_path, _audio_index, _role: {
                "type": "audio",
                "audio": audio_path,
            },
        )
        chat_text = HANDLE.processor.apply_chat_template(
            native_messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(native_messages, use_audio_in_video=True)

        inputs = HANDLE.processor(
            text=chat_text, audio=audios, images=images, videos=videos,
            return_tensors="pt", padding=True, use_audio_in_video=True,
        ).to(HANDLE.model.device)

        dtype = HANDLE.model.dtype
        for key, value in list(inputs.items()):
            if torch.is_tensor(value) and torch.is_floating_point(value):
                inputs[key] = value.to(dtype=dtype)

        gen_kwargs = _generation_kwargs(meta)
        with HANDLE._generate_lock:
            with torch.inference_mode():
                generated_ids = HANDLE.model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    return_audio=False,
                    **gen_kwargs,
                )
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        response = HANDLE.processor.batch_decode(
            generated_ids, skip_special_tokens=True,
        )[0]

    return {
        "prompt": prompt,
        "response": response,
        "submission_payload": {
            "provider": "qwen3-omni-native",
            "transport": "transformers-local-generate",
            "native_messages": native_messages,
            "chat_text": chat_text,
            "generation_kwargs": gen_kwargs,
        },
        "submission_trace": {
            "exact_submission_available": True,
            "builder": "Qwen3OmniServer.generate",
        },
    }
