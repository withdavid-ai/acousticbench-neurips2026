"""FastAPI shim mirroring MiMoAudioServer in mimo_audio_serve.py.

Modal -> Docker mapping:

  * @modal.enter() + safe_enter(load_model) -> FastAPI lifespan + _Handle.load()
  * @modal.method() MiMoAudioServer.generate(item) -> POST /generate
  * @app.function() prefetch() -> POST /prefetch
  * volumes={CACHE_PATH: hf_cache} -> -v <host>:/hf-cache at docker run

Runtime device + dtype overrides:

  * AUDIOBENCH_DEVICE_MAP      - "auto" (default), "cpu", "cuda"
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import traceback
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from serving_utils import (
    decode_messages_payload,
    flatten_messages_to_text,
    hf_revision,
    materialize_audio_files,
)

logger = logging.getLogger("mimo_audio_serve")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

MODEL_REPO = "XiaomiMiMo/MiMo-Audio-7B-Instruct"
TOKENIZER_REPO = "XiaomiMiMo/MiMo-Audio-Tokenizer"
MODEL_REPOS = [MODEL_REPO, TOKENIZER_REPO]
HF_HOME = os.environ.get("HF_HOME", "/hf-cache/huggingface")

CONCAT_GAP_S = 1.0
SAMPLE_RATE = 16000


def _concatenate_audio_files(paths: list[str], gap_s: float = CONCAT_GAP_S) -> str:
    """Concatenate multiple audio files with silence gaps into a single temp file."""
    import numpy as np
    import soundfile as sf

    arrays = []
    for path in paths:
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if sr != SAMPLE_RATE:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=SAMPLE_RATE)
        if data.ndim > 1:
            data = data.mean(axis=1)
        arrays.append(data)

    silence = np.zeros(int(gap_s * SAMPLE_RATE), dtype=np.float32)
    combined = []
    for i, arr in enumerate(arrays):
        if i > 0:
            combined.append(silence)
        combined.append(arr)

    merged = np.concatenate(combined)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, merged, SAMPLE_RATE)
    return tmp.name


class _Handle:
    """Mirrors safe_enter semantics: capture init errors instead of crashing."""

    def __init__(self) -> None:
        self.model: Any = None
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
                from src.mimo_audio.mimo_audio import MimoAudio

                self.model = MimoAudio(
                    snapshot_download(MODEL_REPO, revision=hf_revision(MODEL_REPO)),
                    snapshot_download(TOKENIZER_REPO, revision=hf_revision(TOKENIZER_REPO)),
                )
                # MiMo defaults to do_sample=True (temperature=0.3).
                # Override to greedy for deterministic reproducibility.
                from src.mimo_audio.modeling_mimo_audio import MiMoSampler
                greedy = MiMoSampler(do_sample=False, temperature=1.0, top_p=1.0)
                for task_name in self.model.task_sampler_configs:
                    self.model.task_sampler_configs[task_name] = {
                        "global": greedy,
                        "local": greedy,
                    }
                self.ready = True
                logger.info("MiMo-Audio loaded (greedy decoding): %s", MODEL_REPO)
            except Exception as exc:
                self.init_error = f"{type(exc).__name__}: {exc}"
                self.init_traceback = traceback.format_exc()
                logger.error("MiMo-Audio init failed:\n%s", self.init_traceback)


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
            "builder": "MiMoAudioServer.init_error_response",
        },
    }
    if include_traceback:
        response["traceback"] = HANDLE.init_traceback
    return response


def _generate_impl(item: dict, *, thinking: bool, builder_name: str):
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

    system_text_parts = []
    user_text_parts = []
    for message in messages:
        role = message.get("role", "")
        for part in message.get("content", []):
            if part.get("type") == "text":
                text = str(part.get("text") or "").strip()
                if text:
                    if role == "system":
                        system_text_parts.append(text)
                    elif role == "user":
                        user_text_parts.append(text)

    choice_slugs = meta.get("choice_slugs")
    if isinstance(choice_slugs, list) and choice_slugs:
        labels = [chr(ord("A") + i) for i in range(len(choice_slugs))]
        options_lines = [
            f"({label}) {s.replace('_', ' ').title()}" for label, s in zip(labels, choice_slugs)
        ]
        user_text_parts.append(
            "Options:\n" + "\n".join(options_lines)
            + "\nAnswer only with the correct option letter."
        )

    if system_text_parts:
        user_text_parts.insert(0, "\n".join(system_text_parts))

    question_text = "\n".join(user_text_parts)

    concat_path = None
    with materialize_audio_files(audio_items) as audio_paths:
        if not audio_paths:
            import numpy as np
            import soundfile as sf
            silence = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, silence, SAMPLE_RATE)
            audio_paths = [tmp.name]
            concat_path = tmp.name

        multi_audio = len(audio_paths) > 1
        if multi_audio:
            concat_path = _concatenate_audio_files(audio_paths)
            audio_path = concat_path
            clip_labels = ", ".join(
                f"Clip {chr(ord('A') + i)}" for i in range(len(audio_paths))
            )
            order_hint = (
                f"The provided audio contains {clip_labels} in sequence, "
                f"separated by short silences."
            )
            question_text = f"{order_hint}\n{question_text}"
        else:
            audio_path = audio_paths[0]

        try:
            with HANDLE._generate_lock:
                response = HANDLE.model.audio_understanding_sft(
                    audio_path, question_text, thinking=thinking,
                )
        finally:
            if concat_path and os.path.exists(concat_path):
                os.unlink(concat_path)

    return {
        "prompt": prompt,
        "response": response,
        "submission_payload": {
            "provider": "mimo-audio-native",
            "transport": "mimo-audio-python",
            "question_text": question_text,
            "audio_path_count": len(audio_items),
            "multi_audio_concat": multi_audio,
            "thinking": thinking,
        },
        "submission_trace": {
            "exact_submission_available": True,
            "builder": builder_name,
        },
    }


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
        "model_id": MODEL_REPO,
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
    return _generate_impl(
        item, thinking=False, builder_name="MiMoAudioServer.generate",
    )
