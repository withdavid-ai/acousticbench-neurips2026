"""FastAPI shim mirroring Phi4MMServer in phi4mm_serve.py.

Modal -> Docker mapping:

  * @modal.enter() + safe_enter(load_model) -> FastAPI lifespan + _Handle.load()
  * @modal.method() Phi4MMServer.generate(item) -> POST /generate
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
    coerce_text_response,
    decode_messages_payload,
    flatten_messages_to_text,
    hf_revision,
    materialize_audio_files,
)

logger = logging.getLogger("phi4mm_serve")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
MODEL_REVISION = hf_revision(MODEL_ID)  # head of refs/pr/78
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
                from transformers import AutoModelForCausalLM, AutoProcessor

                self.processor = AutoProcessor.from_pretrained(
                    MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    MODEL_ID,
                    revision=MODEL_REVISION,
                    device_map=DEFAULT_DEVICE_MAP,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    _attn_implementation="sdpa",
                ).eval()
                self.ready = True
                logger.info("Phi-4-Multimodal loaded on device=%s", DEFAULT_DEVICE_MAP)
            except Exception as exc:
                self.init_error = f"{type(exc).__name__}: {exc}"
                self.init_traceback = traceback.format_exc()
                logger.error("Phi-4-Multimodal init failed:\n%s", self.init_traceback)


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
            "builder": "Phi4MMServer.init_error_response",
        },
    }
    if include_traceback:
        response["traceback"] = HANDLE.init_traceback
    return response


def _build_prompt(rendered_text: str) -> str:
    """Build a Phi-4 prompt with chat markers around the rendered conversation."""
    user_prompt = "<|user|>"
    assistant_prompt = "<|assistant|>"
    prompt_suffix = "<|end|>"
    if rendered_text:
        return f"{user_prompt}\n{rendered_text}{prompt_suffix}\n{assistant_prompt}\n"
    return f"{user_prompt}\n{prompt_suffix}\n{assistant_prompt}\n"


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
        path = snapshot_download(repo_id=repo_id, revision=MODEL_REVISION, cache_dir=HF_HOME)
        downloaded.append({"repo": repo_id, "path": path})
    return {"status": "ok", "downloaded": downloaded}


@app.post("/generate")
def generate(item: dict):
    import soundfile as sf

    messages, audio_items, meta = decode_messages_payload(item)
    rendered_text = flatten_messages_to_text(
        messages,
        audio_placeholder=lambda idx: f"<|audio_{idx + 1}|>",
        include_roles=True,
    )
    prompt = _build_prompt(rendered_text)

    err = _init_error_response(prompt, include_traceback=bool(meta.get("_debug")))
    if err is not None:
        return err

    if not HANDLE.ready:
        HANDLE.load()
        err = _init_error_response(prompt, include_traceback=bool(meta.get("_debug")))
        if err is not None:
            return err

    with materialize_audio_files(audio_items) as audio_paths:
        audios = []
        for path in audio_paths:
            audio_data, sr = sf.read(path)
            audios.append((audio_data, sr))

        processor_kwargs = {"text": prompt, "return_tensors": "pt"}
        if audios:
            processor_kwargs["audios"] = audios
        inputs = HANDLE.processor(**processor_kwargs).to(HANDLE.model.device)

        gen_kwargs = apply_benchmark_sampling(
            meta,
            {"max_new_tokens": 64, "do_sample": False},
        )
        with HANDLE._generate_lock:
            generated_ids = HANDLE.model.generate(**inputs, **gen_kwargs)
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        response = HANDLE.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

    return {
        "prompt": prompt,
        "response": coerce_text_response(response),
        "submission_payload": {
            "provider": "phi4mm-native",
            "transport": "transformers-local-generate",
            "prompt_text": prompt,
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
            "builder": "Phi4MMServer.generate",
        },
    }
