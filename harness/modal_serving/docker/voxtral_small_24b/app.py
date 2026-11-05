"""FastAPI shim mirroring VoxtralSmall24BServer in voxtral_serve.py.

Modal -> Docker mapping:

  * @modal.enter() + safe_enter(load_model) -> FastAPI lifespan + _Handle.load()
  * @modal.method() VoxtralSmall24BServer.generate(item) -> POST /generate
  * @app.function() prefetch() -> POST /prefetch
  * volumes={CACHE_PATH: hf_cache} -> -v <host>:/hf-cache at docker run

Runtime device + dtype overrides:

  * AUDIOBENCH_DEVICE_MAP      - "auto" (default), "cpu", "cuda"
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

logger = logging.getLogger("voxtral_small_24b_serve")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

MODEL_ID = "mistralai/Voxtral-Small-24B-2507"
MODEL_REVISION = hf_revision(MODEL_ID)
MODEL_REPOS = [MODEL_ID]
HF_HOME = os.environ.get("HF_HOME", "/hf-cache/huggingface")

DEFAULT_DEVICE_MAP = os.environ.get("AUDIOBENCH_DEVICE_MAP", "auto")


class _Handle:
    """Mirrors safe_enter semantics: capture init errors instead of crashing."""

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
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
                from transformers import AutoProcessor, VoxtralForConditionalGeneration

                self.processor = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
                self.model = VoxtralForConditionalGeneration.from_pretrained(
                    MODEL_ID,
                    revision=MODEL_REVISION,
                    device_map=DEFAULT_DEVICE_MAP,
                    torch_dtype=torch.bfloat16,
                ).eval()
                self.ready = True
                logger.info("Voxtral-Small-24B loaded on device=%s", DEFAULT_DEVICE_MAP)
            except Exception as exc:
                self.init_error = f"{type(exc).__name__}: {exc}"
                self.init_traceback = traceback.format_exc()
                logger.error("Voxtral-Small-24B init failed:\n%s", self.init_traceback)


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
            "builder": "VoxtralSmall24BServer.init_error_response",
        },
    }
    if include_traceback:
        response["traceback"] = HANDLE.init_traceback
    return response


def _build_generation_kwargs(meta: dict) -> dict:
    return apply_benchmark_sampling(
        meta,
        {"max_new_tokens": 512, "do_sample": False},
    )


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
        "device_map": DEFAULT_DEVICE_MAP,
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
                "path": audio_path,
            },
        )

        inputs = HANDLE.processor.apply_chat_template(native_messages)
        inputs = inputs.to(HANDLE.model.device, dtype=torch.bfloat16)
        input_ids = inputs.input_ids if hasattr(inputs, "input_ids") else inputs["input_ids"]
        gen_kwargs = _build_generation_kwargs(meta)

        with HANDLE._generate_lock:
            outputs = HANDLE.model.generate(**inputs, **gen_kwargs)
        response = HANDLE.processor.batch_decode(
            outputs[:, input_ids.shape[1]:], skip_special_tokens=True,
        )[0]

    return {
        "prompt": prompt,
        "response": response,
        "submission_payload": {
            "provider": "voxtral-native",
            "transport": "transformers-local-generate",
            "native_messages": native_messages,
            "generation_kwargs": _build_generation_kwargs(meta),
        },
        "submission_trace": {
            "exact_submission_available": True,
            "builder": "VoxtralSmall24BServer.generate",
        },
    }
