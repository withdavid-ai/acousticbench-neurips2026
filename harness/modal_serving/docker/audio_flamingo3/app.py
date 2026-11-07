"""FastAPI shim mirroring AudioFlamingo3Server in audio_flamingo3_serve.py.

Modal -> Docker mapping:

  * @modal.enter() + safe_enter(load_model) -> FastAPI lifespan + _Handle.load()
  * @modal.method() _AudioFlamingo3ServerBase.generate(item) -> POST /generate
  * @app.function() prefetch() -> POST /prefetch
  * volumes={CACHE_PATH: hf_cache} -> -v <host>:/hf-cache at docker run

Runtime env overrides:

  * AUDIO_FLAMINGO3_VARIANT - "chat" (default) or "base"
"""
from __future__ import annotations

import json
import logging
import os
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
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

logger = logging.getLogger("audio_flamingo3_serve")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

AF3_AUDIO_TOKEN_ID = 151669
AF3_BOS_TOKEN_ID = 151670
AF3_EOS_TOKEN_ID = 151645
AF3_PAD_TOKEN_ID = 151671
AF3_MAX_AUDIO_SECONDS = 600
AF3_HF_ALLOW_PATTERNS = [
    "config.json",
    "llm/*",
    "sound_tower/*",
    "sound_mm_projector/*",
]
AF3_CHAT_TEMPLATE = """{% if messages[0]['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}{% for m in messages if m['content'] is not none %}<|im_start|>{{ m['role'] }}
{% if m['content'] is string %}{{ m['content'] }}{% else %}{% set audio = namespace(found=False) %}{% set text_buf = namespace(v='') %}{% for c in m['content'] %}{% if c.get('type') == 'audio' or 'audio' in c %}{% set audio.found = True %}{% elif c.get('type') == 'text' or 'text' in c %}{% set text_buf.v = text_buf.v + c['text'] %}{% endif %}{% endfor %}{% if audio.found %}{{ ' ' }}{% endif %}{{ text_buf.v }}{% endif %}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""

DEFAULT_VARIANT = os.environ.get("AUDIO_FLAMINGO3_VARIANT", "chat").strip().lower()
MODEL_VARIANTS = {
    "base": {
        "model_repo": "nvidia/audio-flamingo-3",
    },
    "chat": {
        "model_repo": "nvidia/audio-flamingo-3-chat",
    },
}
if DEFAULT_VARIANT not in MODEL_VARIANTS:
    raise ValueError(
        f"Unknown AUDIO_FLAMINGO3_VARIANT={DEFAULT_VARIANT!r}; "
        f"expected one of {sorted(MODEL_VARIANTS)}"
    )
MODEL_REPO = MODEL_VARIANTS[DEFAULT_VARIANT]["model_repo"]
MODEL_REPOS = [
    {
        "repo_id": variant["model_repo"],
        "revision": hf_revision(variant["model_repo"]),
        "allow_patterns": AF3_HF_ALLOW_PATTERNS,
    }
    for variant in MODEL_VARIANTS.values()
]
HF_HOME = os.environ.get("HF_HOME", "/hf-cache/huggingface")


def _af3_prompts_for_messages(messages, placeholder_counts: list[int], audio_token: str) -> tuple[str, str]:
    if not placeholder_counts:
        raise ValueError("AF3 requires at least one audio clip")
    return (
        flatten_messages_to_text(
            messages,
            audio_placeholder=lambda idx: f"<sound-{idx + 1}>",
            include_roles=True,
        ),
        flatten_messages_to_text(
            messages,
            audio_placeholder=lambda idx: audio_token * placeholder_counts[idx],
            include_roles=True,
        ),
    )


def _generate_text_only(
    model,
    tokenizer,
    messages,
    meta: dict,
    *,
    model_repo: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    text_model = getattr(model, "language_model", None)
    if text_model is None:
        raise ValueError("AF3 text-only fallback requires model.language_model")

    rendered_prompt = flatten_messages_to_text(
        messages,
        audio_placeholder="",
        include_roles=True,
    )
    model_prompt = rendered_prompt
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": model_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    text_inputs = tokenizer(chat_text, return_tensors="pt")

    generation_kwargs = apply_benchmark_sampling(
        meta,
        {"max_new_tokens": max_new_tokens, "do_sample": False},
    )
    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(text_model.parameters()).device
    with torch.inference_mode():
        outputs = text_model.generate(
            input_ids=text_inputs.input_ids.to(model_device),
            attention_mask=text_inputs.attention_mask.to(model_device),
            **generation_kwargs,
        )
    response = tokenizer.batch_decode(
        outputs[:, text_inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    return {
        "prompt": rendered_prompt,
        "response": response,
        "submission_payload": {
            "provider": "audio-flamingo3-native",
            "transport": "transformers-local-generate",
            "model_repo": model_repo,
            "text_only_fallback": True,
            "rendered_prompt": rendered_prompt,
            "model_prompt": model_prompt,
            "chat_text": chat_text,
            "generation_kwargs": generation_kwargs,
            "audio_placeholder_counts": [],
            "processor_inputs": {
                "input_ids_shape": list(text_inputs.input_ids.shape),
                "attention_mask_shape": list(text_inputs.attention_mask.shape),
            },
        },
        "submission_trace": {
            "exact_submission_available": True,
            "builder": "AudioFlamingo3Server.generate_text_only",
        },
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_audio_features(feature_extractor, audio_datas):
    import torch

    all_features = []
    all_masks = []
    placeholder_counts = []
    chunk_size = int(feature_extractor.sampling_rate * feature_extractor.chunk_length)
    max_windows = int(AF3_MAX_AUDIO_SECONDS // feature_extractor.chunk_length)

    for audio_data in audio_datas:
        n_samples = int(audio_data.shape[0])
        n_windows = max(1, (n_samples + chunk_size - 1) // chunk_size)
        n_windows = min(n_windows, max_windows)
        chunks = []
        time_cap = min(n_samples, n_windows * chunk_size)
        for idx in range(n_windows):
            start = idx * chunk_size
            end = min((idx + 1) * chunk_size, time_cap)
            chunks.append(audio_data[start:end])

        features = feature_extractor(
            chunks,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
        )
        input_features = features.input_features
        input_features_mask = features.attention_mask
        input_lengths = input_features_mask.sum(-1).to(torch.long)
        conv2_lengths = (input_lengths - 1) // 2 + 1
        post_pool_lengths = (conv2_lengths - 2) // 2 + 1

        all_features.append(input_features)
        all_masks.append(input_features_mask)
        placeholder_counts.append(int(post_pool_lengths.sum().item()))

    return (
        torch.cat(all_features, dim=0),
        torch.cat(all_masks, dim=0),
        placeholder_counts,
    )


def _build_native_audio_config(sound_tower_config: dict) -> dict:
    return {
        "model_type": "audioflamingo3_encoder",
        "hidden_size": int(sound_tower_config["d_model"]),
        "num_hidden_layers": int(sound_tower_config["encoder_layers"]),
        "num_attention_heads": int(sound_tower_config["encoder_attention_heads"]),
        "intermediate_size": int(sound_tower_config["encoder_ffn_dim"]),
        "max_source_positions": int(sound_tower_config["max_source_positions"]),
        "num_mel_bins": int(sound_tower_config["num_mel_bins"]),
        "activation_function": str(sound_tower_config.get("activation_function", "gelu")),
        "dropout": float(sound_tower_config.get("dropout", 0.0)),
        "attention_dropout": float(sound_tower_config.get("attention_dropout", 0.0)),
        "activation_dropout": float(sound_tower_config.get("activation_dropout", 0.0)),
        "initializer_range": float(sound_tower_config.get("init_std", 0.02)),
        "scale_embedding": bool(sound_tower_config.get("scale_embedding", False)),
    }


def _iter_llm_weight_paths(llm_dir: Path) -> list[Path]:
    sharded = sorted(llm_dir.glob("model-*.safetensors"))
    if sharded:
        return sharded
    single = llm_dir / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"No LLM safetensor weights found under {llm_dir}")


def _copy_safetensor_weights(model, model_path: Path):
    import torch
    from safetensors import safe_open

    state = model.state_dict()
    loaded_keys = set()
    unexpected_keys = []

    def copy_file(path: Path, rename):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for source_key in handle.keys():
                target_key = rename(source_key)
                target_tensor = state.get(target_key)
                if target_tensor is None:
                    unexpected_keys.append(target_key)
                    continue
                source_tensor = handle.get_tensor(source_key)
                if source_tensor.shape != target_tensor.shape:
                    raise ValueError(
                        f"AF3 weight shape mismatch for {target_key}: "
                        f"expected {tuple(target_tensor.shape)}, got {tuple(source_tensor.shape)}"
                    )
                with torch.no_grad():
                    target_tensor.copy_(source_tensor.to(dtype=target_tensor.dtype))
                loaded_keys.add(target_key)

    copy_file(model_path / "sound_tower" / "model.safetensors", lambda key: f"audio_tower.{key}")
    projector_key_map = {
        "layers.0.weight": "multi_modal_projector.linear_1.weight",
        "layers.0.bias": "multi_modal_projector.linear_1.bias",
        "layers.2.weight": "multi_modal_projector.linear_2.weight",
        "layers.2.bias": "multi_modal_projector.linear_2.bias",
    }
    copy_file(
        model_path / "sound_mm_projector" / "model.safetensors",
        lambda key: projector_key_map[key],
    )
    for llm_path in _iter_llm_weight_paths(model_path / "llm"):
        copy_file(llm_path, lambda key: f"language_model.{key}")

    missing_keys = sorted(set(state) - loaded_keys)
    if missing_keys or unexpected_keys:
        raise ValueError(
            "AF3 native load produced key mismatch: "
            f"missing={missing_keys[:10]} unexpected={unexpected_keys[:10]}"
        )


def _load_native_components(model_path: Path):
    import torch
    from transformers import (
        AudioFlamingo3Config,
        AudioFlamingo3ForConditionalGeneration,
        AutoConfig,
        AutoTokenizer,
        WhisperFeatureExtractor,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path / "llm"))
    tokenizer.chat_template = AF3_CHAT_TEMPLATE
    tokenizer.padding_side = "left"
    audio_token = tokenizer.convert_ids_to_tokens(AF3_AUDIO_TOKEN_ID)
    if not isinstance(audio_token, str):
        raise ValueError(f"Could not resolve AF3 audio token {AF3_AUDIO_TOKEN_ID}")

    text_config = AutoConfig.from_pretrained(str(model_path / "llm"))
    text_config.bos_token_id = tokenizer.bos_token_id or AF3_BOS_TOKEN_ID
    text_config.eos_token_id = tokenizer.eos_token_id or AF3_EOS_TOKEN_ID
    text_config.pad_token_id = tokenizer.pad_token_id or AF3_PAD_TOKEN_ID

    sound_tower_config = _read_json(model_path / "sound_tower" / "config.json")
    config = AudioFlamingo3Config(
        audio_config=_build_native_audio_config(sound_tower_config),
        text_config=text_config,
        audio_token_id=AF3_AUDIO_TOKEN_ID,
    )

    model = AudioFlamingo3ForConditionalGeneration(config)
    model = model.to(dtype=torch.bfloat16)
    _copy_safetensor_weights(model, model_path)
    model = model.to("cuda:0").eval()

    feature_extractor = WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=16000,
        chunk_length=30,
        hop_length=160,
        n_fft=400,
        padding_value=0.0,
        return_attention_mask=True,
    )
    return model, tokenizer, feature_extractor, audio_token


class _Handle:
    """Mirrors safe_enter semantics: capture init errors instead of crashing."""

    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.feature_extractor: Any = None
        self.audio_token: str | None = None
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

                model_path = Path(snapshot_download(
                    MODEL_REPO,
                    revision=hf_revision(MODEL_REPO),
                    allow_patterns=AF3_HF_ALLOW_PATTERNS,
                ))
                self.model, self.tokenizer, self.feature_extractor, self.audio_token = (
                    _load_native_components(model_path)
                )
                self.ready = True
                logger.info("Audio-Flamingo-3 loaded: %s (variant=%s)", MODEL_REPO, DEFAULT_VARIANT)
            except Exception as exc:
                self.init_error = f"{type(exc).__name__}: {exc}"
                self.init_traceback = traceback.format_exc()
                logger.error("Audio-Flamingo-3 init failed:\n%s", self.init_traceback)


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
            "builder": "AudioFlamingo3Server.init_error_response",
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
        "model_repo": MODEL_REPO,
        "variant": DEFAULT_VARIANT,
    }


@app.post("/prefetch")
def prefetch():
    from huggingface_hub import snapshot_download

    downloaded = []
    for spec in MODEL_REPOS:
        path = snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            allow_patterns=spec["allow_patterns"],
            cache_dir=HF_HOME,
        )
        downloaded.append({"repo": spec["repo_id"], "path": path})
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

    if not audio_items:
        with HANDLE._generate_lock:
            return _generate_text_only(
                HANDLE.model,
                HANDLE.tokenizer,
                messages,
                meta,
                model_repo=MODEL_REPO,
                max_new_tokens=512,
            )

    audio_datas = audio_items_to_ndarrays(audio_items, sr=HANDLE.feature_extractor.sampling_rate)
    input_features, input_features_mask, placeholder_counts = _build_audio_features(
        HANDLE.feature_extractor,
        audio_datas,
    )
    rendered_prompt, model_prompt = _af3_prompts_for_messages(
        messages,
        placeholder_counts,
        HANDLE.audio_token,
    )
    chat_text = HANDLE.tokenizer.apply_chat_template(
        [{"role": "user", "content": model_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    text_inputs = HANDLE.tokenizer(chat_text, return_tensors="pt")

    generation_kwargs = apply_benchmark_sampling(
        meta,
        {"max_new_tokens": 512, "do_sample": False},
    )
    with HANDLE._generate_lock:
        with torch.inference_mode():
            outputs = HANDLE.model.generate(
                input_ids=text_inputs.input_ids.to(HANDLE.model.device),
                attention_mask=text_inputs.attention_mask.to(HANDLE.model.device),
                input_features=input_features.to(HANDLE.model.device, dtype=HANDLE.model.dtype),
                input_features_mask=input_features_mask.to(HANDLE.model.device),
                **generation_kwargs,
            )
    response = HANDLE.tokenizer.batch_decode(
        outputs[:, text_inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    return {
        "prompt": rendered_prompt,
        "response": response,
        "submission_payload": {
            "provider": "audio-flamingo3-native",
            "transport": "transformers-local-generate",
            "model_repo": MODEL_REPO,
            "rendered_prompt": rendered_prompt,
            "model_prompt": model_prompt,
            "chat_text": chat_text,
            "generation_kwargs": generation_kwargs,
            "audio_placeholder_counts": placeholder_counts,
            "processor_inputs": {
                "input_ids_shape": list(text_inputs.input_ids.shape),
                "attention_mask_shape": list(text_inputs.attention_mask.shape),
                "input_features_shape": list(input_features.shape),
                "input_features_mask_shape": list(input_features_mask.shape),
            },
        },
        "submission_trace": {
            "exact_submission_available": True,
            "builder": "AudioFlamingo3Server.generate",
        },
    }
