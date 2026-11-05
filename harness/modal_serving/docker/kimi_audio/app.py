"""FastAPI shim mirroring KimiAudioServer in kimi_audio_serve.py.

Modal -> Docker mapping:

  * @modal.enter() + safe_enter(load_model) -> FastAPI lifespan + _Handle.load()
  * @modal.method() KimiAudioServer.generate(item) -> POST /generate
  * @app.function() prefetch() -> POST /prefetch
  * volumes={CACHE_PATH: hf_cache} -> -v <host>:/hf-cache at docker run
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

logger = logging.getLogger("kimi_audio_serve")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

MODEL_ID = "moonshotai/Kimi-Audio-7B-Instruct"
MODEL_REVISION = hf_revision(MODEL_ID)
MODEL_REPOS = [MODEL_ID]
HF_HOME = os.environ.get("HF_HOME", "/hf-cache/huggingface")


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
            logger.warning("Kimi-Audio remapping system role to user role for provider compatibility.")
        native_messages.extend(
            {
                "role": role,
                **part,
            }
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

    # Upstream Kimi-Audio uses [] for text-only prompts, then indexes [0][0]
    # in the Whisper branch. None correctly means "no continuous audio features."
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


class _Handle:
    """Mirrors safe_enter semantics: capture init errors instead of crashing."""

    def __init__(self) -> None:
        self.model: Any = None
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
                from huggingface_hub import snapshot_download
                from kimia_infer.api.kimia import KimiAudio

                # KimiAudio downloads repo ids without a revision, so resolve
                # the pinned snapshot first and pass the local path.
                self.model = KimiAudio(
                    model_path=snapshot_download(MODEL_ID, revision=MODEL_REVISION),
                    load_detokenizer=False,
                )
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
                logger.info("Kimi-Audio loaded: %s", MODEL_ID)
            except Exception as exc:
                self.init_error = f"{type(exc).__name__}: {exc}"
                self.init_traceback = traceback.format_exc()
                logger.error("Kimi-Audio init failed:\n%s", self.init_traceback)


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
            "builder": "KimiAudioServer.init_error_response",
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

    with materialize_audio_files(audio_items) as audio_paths:
        native_messages = _to_kimi_native_messages(messages, audio_paths)
        with HANDLE._generate_lock:
            if audio_paths:
                _, response = HANDLE.model.generate(native_messages, **sampling_params, output_type="text")
            else:
                response = _generate_text_only(HANDLE.model, native_messages, sampling_params)

    return {
        "prompt": prompt,
        "response": response,
        "submission_payload": {
            "provider": "kimi-audio-native",
            "transport": "moonshot-kimia-python",
            "messages": native_messages,
            "sampling_params": sampling_params,
            "output_type": "text",
            "text_only_empty_audio_features_patched": not bool(audio_items),
        },
        "submission_trace": {
            "exact_submission_available": True,
            "builder": "KimiAudioServer.generate",
        },
    }
