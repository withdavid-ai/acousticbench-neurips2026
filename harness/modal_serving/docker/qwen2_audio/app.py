"""FastAPI shim mirroring Qwen2AudioChatServer in qwen2_audio_serve.py.

Modal -> Docker mapping:

  * @modal.enter() + safe_enter(load_model) -> FastAPI lifespan + _Handle.load()
  * @modal.method() Qwen2AudioChatServer.generate_chat(item) -> POST /generate
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
    audio_items_to_ndarrays,
    decode_messages_payload,
    flatten_messages_to_text,
    hf_revision,
    render_messages,
)

logger = logging.getLogger("qwen2_audio_serve")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
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
                from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

                self.processor = AutoProcessor.from_pretrained(
                    MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    MODEL_ID, revision=MODEL_REVISION, device_map=DEFAULT_DEVICE_MAP,
                ).eval()
                self.ready = True
                logger.info("Qwen2-Audio loaded: %s", MODEL_ID)
            except Exception as exc:
                self.init_error = f"{type(exc).__name__}: {exc}"
                self.init_traceback = traceback.format_exc()
                logger.error("Qwen2-Audio init failed:\n%s", self.init_traceback)


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
            "builder": "Qwen2AudioChatServer.init_error_response",
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
            content={"status": "init_error", "error": HANDLE.init_error},
        )
    if not HANDLE.ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {
        "status": "ready",
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

    sr = HANDLE.processor.feature_extractor.sampling_rate
    audio_datas = audio_items_to_ndarrays(audio_items, sr=sr) if audio_items else []
    conversation = render_messages(
        messages,
        audio_datas,
        build_text_part=lambda text, _role: (
            {"type": "text", "text": text} if str(text).strip() else None
        ),
        build_audio_part=lambda _audio, _audio_index, _role: {
            "type": "audio",
            "audio_url": "placeholder",
        },
    )
    for message in conversation:
        if message["role"] == "system":
            message["content"] = flatten_messages_to_text([message], include_roles=False)

    chat_text = HANDLE.processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False)

    processor_kwargs = {
        "text": chat_text, "return_tensors": "pt",
        "padding": True, "sampling_rate": sr,
    }
    if audio_datas:
        processor_kwargs["audios"] = audio_datas
    inputs = HANDLE.processor(**processor_kwargs).to(HANDLE.model.device)

    gen_kwargs = apply_benchmark_sampling(
        meta,
        {"max_new_tokens": 256, "do_sample": False},
    )
    with HANDLE._generate_lock:
        with torch.inference_mode():
            generated_ids = HANDLE.model.generate(**inputs, **gen_kwargs)
    generated_ids = generated_ids[:, inputs.input_ids.size(1):]
    response = HANDLE.processor.batch_decode(
        generated_ids, skip_special_tokens=True,
    )[0]

    return {
        "prompt": prompt,
        "response": response,
        "submission_payload": {
            "provider": "qwen2-audio-native",
            "transport": "transformers-local-generate",
            "conversation": conversation,
            "chat_text": chat_text,
            "generation_kwargs": gen_kwargs,
            "processor_inputs": {
                "input_keys": sorted(inputs.keys()),
                "input_shapes": {
                    key: list(value.shape)
                    for key, value in inputs.items()
                    if hasattr(value, "shape")
                },
            },
        },
        "submission_trace": {
            "exact_submission_available": True,
            "builder": "Qwen2AudioChatServer.generate_chat",
        },
    }
