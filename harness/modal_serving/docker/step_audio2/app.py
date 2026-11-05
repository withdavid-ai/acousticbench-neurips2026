"""FastAPI shim mirroring StepAudio2MiniServer / StepAudio2MiniBaseServer
in step_audio2_serve.py.

Modal -> Docker mapping:

  * @modal.enter() + safe_enter(load_model) -> FastAPI lifespan + _Handle.load()
  * @modal.method() StepAudio2MiniServer.generate(item) -> POST /generate
  * @modal.method() StepAudio2MiniBaseServer.generate(item) -> POST /generate-base
  * @app.function() prefetch() -> POST /prefetch
  * volumes={CACHE_PATH: hf_cache} -> -v <host>:/hf-cache at docker run

Runtime device + dtype overrides:

  * AUDIOBENCH_DEVICE_MAP      - "auto" (default), "cpu", "cuda"

IMPORTANT: Audio placeholder tokens use NO spaces between them:
  <audio_start><audio_patch>...<audio_end>   (correct)
  <audio_start> <audio_patch> ... <audio_end> (WRONG - was a critical bug)
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
    coerce_text_response,
    decode_messages_payload,
    flatten_messages_to_text,
    hf_revision,
)

logger = logging.getLogger("step_audio2_serve")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

MODEL_REPOS = [
    "stepfun-ai/Step-Audio-2-mini",
    "stepfun-ai/Step-Audio-2-mini-Base",
]
HF_HOME = os.environ.get("HF_HOME", "/hf-cache/huggingface")


def _log_mel_spectrogram(audio, n_mels=128, padding=479):
    """128-bin log-mel matching upstream Step-Audio-2."""
    import librosa
    import torch
    import torch.nn.functional as F

    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(400).to(audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels))
    log_spec = torch.clamp(filters @ magnitudes, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def _compute_token_num(max_feature_len):
    """How many LLM placeholder tokens one mel chunk occupies."""
    max_feature_len -= 2
    encoder_out = (max_feature_len + 1) // 2 // 2
    return (encoder_out + 2 * 1 - 3) // 2 + 1


class _MiniHandle:
    """Mirrors safe_enter semantics for Step-Audio-2-mini (instruct)."""

    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.eos_token_id: int | None = None
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
                from transformers import AutoModelForCausalLM, AutoTokenizer

                model_path = "stepfun-ai/Step-Audio-2-mini"
                revision = hf_revision(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, revision=revision, trust_remote_code=True, padding_side="right")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, revision=revision, trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                ).cuda().eval()
                self.tokenizer.eos_token = "<|EOT|>"
                self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|EOT|>")
                self.model.config.eos_token_id = self.eos_token_id
                self.ready = True
                logger.info("Step-Audio-2-mini loaded")
            except Exception as exc:
                self.init_error = f"{type(exc).__name__}: {exc}"
                self.init_traceback = traceback.format_exc()
                logger.error("Step-Audio-2-mini init failed:\n%s", self.init_traceback)

    def build_inputs(self, audio_datas, messages):
        """Build tokenized input_ids + mel tensors following upstream chat template."""
        import torch
        from torch.nn.utils.rnn import pad_sequence

        mels = []
        audio_placeholders: list[str] = []
        for clip_idx, audio_np in enumerate(audio_datas):
            placeholder = ""
            audio_tensor = torch.from_numpy(audio_np).float()
            for i in range(0, audio_tensor.shape[0], 16000 * 25):
                chunk = audio_tensor[i:i + 16000 * 25]
                mel = _log_mel_spectrogram(chunk)
                mels.append(mel)
                audio_tokens = "<audio_patch>" * _compute_token_num(mel.shape[1])
                placeholder += f"<audio_start>{audio_tokens}<audio_end>"
            audio_placeholders.append(placeholder)

        rendered_prompt = flatten_messages_to_text(
            messages,
            audio_placeholder=lambda idx: audio_placeholders[idx],
            include_roles=True,
        )

        parts = [
            f"<|BOT|>human\n{rendered_prompt}<|EOT|>",
            "<|BOT|>assistant\n",
        ]
        token_parts = [
            self.tokenizer(p, return_tensors="pt", padding=True)["input_ids"]
            for p in parts
        ]
        input_ids = torch.cat(token_parts, dim=-1).cuda()
        attention_mask = torch.ones_like(input_ids)

        if mels:
            lengths = torch.tensor(
                [m.shape[1] - 2 for m in mels], dtype=torch.int32)
            padded = pad_sequence(
                [m.t() for m in mels], batch_first=True, padding_value=0
            ).transpose(1, 2).cuda()
            return input_ids, attention_mask, padded, lengths.cuda()
        return input_ids, attention_mask, None, None


class _BaseHandle:
    """Mirrors safe_enter semantics for Step-Audio-2-mini-Base."""

    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.eos_token_id: int | None = None
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
                from transformers import AutoModelForCausalLM, AutoTokenizer

                model_path = "stepfun-ai/Step-Audio-2-mini-Base"
                revision = hf_revision(model_path)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, revision=revision, trust_remote_code=True, padding_side="right")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path, revision=revision, trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                ).cuda().eval()
                self.eos_token_id = self.tokenizer.eos_token_id
                self.ready = True
                logger.info("Step-Audio-2-mini-Base loaded")
            except Exception as exc:
                self.init_error = f"{type(exc).__name__}: {exc}"
                self.init_traceback = traceback.format_exc()
                logger.error("Step-Audio-2-mini-Base init failed:\n%s", self.init_traceback)

    def build_inputs(self, audio_datas, messages):
        """Base model: flat template with no role markers."""
        import torch
        from torch.nn.utils.rnn import pad_sequence

        mels = []
        audio_placeholders: list[str] = []
        for clip_idx, audio_np in enumerate(audio_datas):
            placeholder = ""
            audio_tensor = torch.from_numpy(audio_np).float()
            for i in range(0, audio_tensor.shape[0], 16000 * 25):
                chunk = audio_tensor[i:i + 16000 * 25]
                mel = _log_mel_spectrogram(chunk)
                mels.append(mel)
                audio_tokens = "<audio_patch>" * _compute_token_num(mel.shape[1])
                placeholder += f"<audio_start>{audio_tokens}<audio_end>"
            audio_placeholders.append(placeholder)

        text_input = flatten_messages_to_text(
            messages,
            audio_placeholder=lambda idx: audio_placeholders[idx],
            include_roles=True,
        )
        input_ids = self.tokenizer(
            text_input, return_tensors="pt", padding=True)["input_ids"].cuda()
        attention_mask = torch.ones_like(input_ids)

        if mels:
            lengths = torch.tensor(
                [m.shape[1] - 2 for m in mels], dtype=torch.int32)
            padded = pad_sequence(
                [m.t() for m in mels], batch_first=True, padding_value=0
            ).transpose(1, 2).cuda()
            return input_ids, attention_mask, padded, lengths.cuda()
        return input_ids, attention_mask, None, None


MINI_HANDLE = _MiniHandle()
BASE_HANDLE = _BaseHandle()


def _init_error_response(handle, prompt: str, include_traceback: bool, builder: str) -> dict[str, Any] | None:
    if not handle.init_error:
        return None
    response: dict[str, Any] = {
        "prompt": prompt,
        "response": f"[INIT_ERROR] {handle.init_error}",
        "submission_payload": {
            "provider": "modal-init",
            "transport": "modal-startup-error",
            "prompt_text": prompt,
            "error": handle.init_error,
        },
        "submission_trace": {
            "exact_submission_available": True,
            "builder": f"{builder}.init_error_response",
        },
    }
    if include_traceback:
        response["traceback"] = handle.init_traceback
    return response


def _generate_for_handle(handle, item: dict, *, builder_name: str):
    from transformers import GenerationConfig

    messages, audio_items, meta = decode_messages_payload(item)
    audio_datas = audio_items_to_ndarrays(audio_items, sr=16000) if audio_items else []
    prompt = flatten_messages_to_text(messages, include_roles=True)

    err = _init_error_response(handle, prompt, include_traceback=bool(meta.get("_debug")), builder=builder_name)
    if err is not None:
        return err

    if not handle.ready:
        handle.load()
        err = _init_error_response(handle, prompt, include_traceback=bool(meta.get("_debug")), builder=builder_name)
        if err is not None:
            return err

    input_ids, attention_mask, wavs, wav_lens = handle.build_inputs(audio_datas, messages)

    gen_kwargs = apply_benchmark_sampling(
        meta,
        {
            "max_new_tokens": 512,
            "pad_token_id": handle.tokenizer.pad_token_id,
            "eos_token_id": handle.eos_token_id,
            "do_sample": False,
        },
    )
    gen_cfg = GenerationConfig(**gen_kwargs)
    with handle._generate_lock:
        outputs = handle.model.generate(
            input_ids=input_ids,
            wavs=wavs,
            wav_lens=wav_lens,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
            tokenizer=handle.tokenizer,
        )
    output_tokens = outputs[0, input_ids.shape[-1]:-1].tolist()
    text_tokens = [t for t in output_tokens if t < 151688]
    response = handle.tokenizer.decode(text_tokens)

    return {
        "prompt": prompt,
        "response": coerce_text_response(response),
        "submission_payload": {
            "provider": "step-audio2-native",
            "transport": "transformers-local-generate",
            "prompt_text": prompt,
            "audio_clip_count": len(audio_datas),
            "generation_kwargs": gen_kwargs,
        },
        "submission_trace": {
            "exact_submission_available": True,
            "builder": f"{builder_name}.generate",
        },
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Only load mini by default; base loaded on first request
    MINI_HANDLE.load()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    if MINI_HANDLE.init_error:
        return JSONResponse(
            status_code=200,
            content={"status": "init_error", "error": MINI_HANDLE.init_error},
        )
    if not MINI_HANDLE.ready:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {
        "status": "ready",
        "model_id": "stepfun-ai/Step-Audio-2-mini",
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
    return _generate_for_handle(
        MINI_HANDLE, item, builder_name="StepAudio2MiniServer",
    )


@app.post("/generate-base")
def generate_base(item: dict):
    if not BASE_HANDLE.ready and not BASE_HANDLE.init_error:
        BASE_HANDLE.load()
    return _generate_for_handle(
        BASE_HANDLE, item, builder_name="StepAudio2MiniBaseServer",
    )
