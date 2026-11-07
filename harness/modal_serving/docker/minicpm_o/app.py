"""FastAPI shim mirroring MiniCPMo45Server in minicpm_o_serve.py.

Modal -> Docker mapping:

  * @modal.enter() + safe_enter(load_model) -> FastAPI lifespan + _Handle.load()
  * @modal.method() MiniCPMo45Server.generate(item) -> POST /generate
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
import types
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
)

logger = logging.getLogger("minicpm_o_serve")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

MODEL_ID = "openbmb/MiniCPM-o-4_5"
MODEL_REVISION = hf_revision(MODEL_ID)
MODEL_REPOS = [MODEL_ID]
HF_HOME = os.environ.get("HF_HOME", "/hf-cache/huggingface")

DEFAULT_DEVICE_MAP = os.environ.get("AUDIOBENCH_DEVICE_MAP", "auto")


# ---------------------------------------------------------------------------
# Audio embedding alignment patch (ported from minicpm_o_serve.py)
# ---------------------------------------------------------------------------

def _bound_index(value) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def _align_audio_segment(embs, target_len: int):
    import torch
    import torch.nn.functional as F

    current_len = int(embs.shape[0])
    if current_len == target_len:
        return embs

    if target_len <= 0:
        return embs[:0]

    if current_len == 0:
        return torch.zeros((target_len, embs.shape[1]), dtype=embs.dtype, device=embs.device)

    delta = target_len - current_len
    if abs(delta) <= 2:
        if delta < 0:
            return embs[:target_len]
        pad = embs[-1:, :].expand(delta, -1)
        return torch.cat([embs, pad], dim=0)

    resized = F.interpolate(
        embs.transpose(0, 1).unsqueeze(0),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).squeeze(0).transpose(0, 1)
    print(
        f"[MiniCPM patch] resampled audio embeddings from {current_len} to {target_len} tokens",
        flush=True,
    )
    return resized


def _patch_get_omni_embedding(model):
    original_get_omni_embedding = model.get_omni_embedding

    def patched_get_omni_embedding(self, data, input_embeddings, chunk_length=-1, stream_input=False):
        if stream_input:
            return original_get_omni_embedding(
                data,
                input_embeddings,
                chunk_length=chunk_length,
                stream_input=stream_input,
            )

        audio_embeddings = self.get_audio_embedding(data, chunk_length)
        bs = len(input_embeddings)
        if len(data.get("audio_features", [])) > 0:
            assert len(audio_embeddings) == len(input_embeddings)

            if len(audio_embeddings) > 0:
                audio_bounds = data["audio_bounds"]
                for batch_idx in range(bs):
                    for embs, bound in zip(audio_embeddings[batch_idx], audio_bounds[batch_idx]):
                        start = _bound_index(bound[0])
                        end = _bound_index(bound[1])
                        fixed_embs = _align_audio_segment(embs, end - start)
                        if fixed_embs.shape[0] != embs.shape[0]:
                            print(
                                f"[MiniCPM patch] adjusted audio embeddings from "
                                f"{embs.shape[0]} to {fixed_embs.shape[0]} tokens",
                                flush=True,
                            )
                        input_embeddings[batch_idx, start:end] = fixed_embs.to(
                            device=input_embeddings.device,
                            dtype=input_embeddings.dtype,
                        )
        elif self.training:
            for _ in range(bs):
                input_embeddings += audio_embeddings[0].mean() * 0

        return input_embeddings

    model.get_omni_embedding = types.MethodType(patched_get_omni_embedding, model)
    model.config.stream_input = False


# ---------------------------------------------------------------------------
# Tail-padding helpers (ported from MiniCPMo45Server)
# ---------------------------------------------------------------------------

def _pad_audio_tail(audio_data, sr: int = 16000, pad_s: float = 0.0):
    import numpy as np

    if pad_s <= 0:
        return audio_data
    pad_len = int(sr * pad_s)
    if pad_len <= 0:
        return audio_data
    return np.concatenate([audio_data, np.zeros(pad_len, dtype=audio_data.dtype)])


def _is_shape_mismatch(exc: Exception) -> bool:
    message = str(exc)
    markers = (
        "expanded size of the tensor",
        "must match the existing size",
        "Sizes of tensors must match",
        "audio/token mismatch",
    )
    return any(marker in message for marker in markers)


# ---------------------------------------------------------------------------
# Handle / app
# ---------------------------------------------------------------------------

class _Handle:
    """Mirrors safe_enter semantics: capture init errors instead of crashing."""

    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
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
                from transformers import AutoModel, AutoTokenizer

                self.model = AutoModel.from_pretrained(
                    MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
                    torch_dtype=torch.bfloat16, device_map=DEFAULT_DEVICE_MAP,
                ).eval()
                _patch_get_omni_embedding(self.model)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
                self.ready = True
                logger.info("MiniCPM-o-4.5 loaded: %s", MODEL_ID)
            except Exception as exc:
                self.init_error = f"{type(exc).__name__}: {exc}"
                self.init_traceback = traceback.format_exc()
                logger.error("MiniCPM-o-4.5 init failed:\n%s", self.init_traceback)


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
            "builder": "MiniCPMo45Server.init_error_response",
        },
    }
    if include_traceback:
        response["traceback"] = HANDLE.init_traceback
    return response


def _chat_once(prompt: str, audio_data, chat_kwargs: dict):
    if hasattr(HANDLE.model, "reset_session"):
        HANDLE.model.reset_session(reset_token2wav_cache=False)
    msgs = [{"role": "user", "content": [prompt, audio_data]}]
    return HANDLE.model.chat(
        msgs=msgs,
        tokenizer=HANDLE.tokenizer,
        **chat_kwargs,
    )


def _chat_messages_once(messages, chat_kwargs: dict):
    if hasattr(HANDLE.model, "reset_session"):
        HANDLE.model.reset_session(reset_token2wav_cache=False)
    return HANDLE.model.chat(
        msgs=messages,
        tokenizer=HANDLE.tokenizer,
        **chat_kwargs,
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
    import numpy as np

    messages, audio_items, meta = decode_messages_payload(item)
    prompt = flatten_messages_to_text(
        messages,
        audio_placeholder=lambda idx: f"[Audio {idx + 1}]",
        include_roles=True,
    )

    err = _init_error_response(prompt, include_traceback=bool(meta.get("_debug")))
    if err is not None:
        return err

    if not HANDLE.ready:
        HANDLE.load()
        err = _init_error_response(prompt, include_traceback=bool(meta.get("_debug")))
        if err is not None:
            return err

    chat_kwargs = apply_benchmark_sampling(
        meta,
        {
            "max_new_tokens": 512,
            "do_sample": False,
            "generate_audio": False,
        },
    )

    if not audio_items:
        native_messages = [{"role": "user", "content": prompt}]
        with HANDLE._generate_lock:
            response = _chat_messages_once(native_messages, chat_kwargs)
        return {
            "prompt": prompt,
            "response": response,
            "submission_payload": {
                "provider": "minicpm-o",
                "transport": "transformers-model-chat",
                "prompt_text": prompt,
                "native_messages": native_messages,
                "generation_kwargs": chat_kwargs,
                "text_only_empty_audio_features_patched": True,
            },
            "submission_trace": {
                "exact_submission_available": True,
                "builder": "MiniCPMo45Server.generate.text_only",
            },
        }

    if len(audio_items) != 1:
        raise ValueError(
            "MiniCPM-o-4.5 serving expects exactly one audio clip. "
            "Use the MiniCPM input adapter to merge multi-clip examples before serving."
        )

    combined_audio = np.ascontiguousarray(
        audio_items_to_ndarrays(audio_items, sr=16000)[0],
        dtype=np.float32,
    )

    last_error = None
    with HANDLE._generate_lock:
        for pad_s in (0.0, 0.10, 0.25, 0.50):
            candidate_audio = _pad_audio_tail(combined_audio, sr=16000, pad_s=pad_s)
            try:
                response = _chat_once(prompt, candidate_audio, chat_kwargs)
                return {
                    "prompt": prompt,
                    "response": response,
                    "submission_payload": {
                        "provider": "minicpm-o",
                        "transport": "transformers-model-chat",
                        "prompt_text": prompt,
                        "messages": [{"role": "user", "content": [prompt, {"audio_ndarray_shape": list(candidate_audio.shape)}]}],
                        "generation_kwargs": chat_kwargs,
                        "audio_shape": list(candidate_audio.shape),
                        "audio_dtype": str(candidate_audio.dtype),
                        "tail_pad_seconds": pad_s,
                    },
                    "submission_trace": {
                        "exact_submission_available": True,
                        "builder": "MiniCPMo45Server.generate",
                    },
                }
            except Exception as exc:
                last_error = exc
                if not _is_shape_mismatch(exc) or pad_s >= 0.50:
                    raise
                print(
                    f"[MiniCPM retry] shape mismatch with tail pad {pad_s:.2f}s: {exc}",
                    flush=True,
                )

    raise last_error
