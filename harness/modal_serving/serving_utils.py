"""Transport-agnostic utilities for AcousticBench model servers.

This module contains all the pure utility functions extracted from common.py,
with zero dependency on the Modal SDK. Docker-based serving apps import from
here directly; Modal-based serving apps get these re-exported via common.py.
"""

from __future__ import annotations

import functools
import os
import traceback
from contextlib import contextmanager
from typing import Any, Callable


CACHE_PATH = "/hf-cache"
HF_HOME = os.environ.get("HF_HOME", f"{CACHE_PATH}/huggingface")
HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE", f"{HF_HOME}/hub")
WHISPER_CACHE_PATH = f"{CACHE_PATH}/whisper"

HF_RUNTIME_ENV = {
    "HF_HOME": HF_HOME,
    "HF_HUB_CACHE": HF_HUB_CACHE,
    "HUGGINGFACE_HUB_CACHE": HF_HUB_CACHE,
    "TRANSFORMERS_CACHE": HF_HUB_CACHE,
    "HF_XET_HIGH_PERFORMANCE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

COMMON_PIP_PACKAGES = (
    "accelerate==1.13.0",
    "librosa==0.11.0",
    "soundfile==0.13.1",
    # Pinned to the latest PyPI release on or before 2026-05-07, the last day
    # of the reference runs (these were previously unpinned).
    "einops==0.8.2",
    "pillow==12.2.0",
    "sentencepiece==0.2.1",
)

# Hugging Face commit pins for every checkpoint the servers load. Each value is
# the latest commit on the listed ref on or before 2026-05-07 (the reference
# runs took place 2026-05-01 to 2026-05-07); the commit date is in the comment.
HF_REVISIONS = {
    "Qwen/Qwen3-Omni-30B-A3B-Instruct": "26291f793822fb6be9555850f06dfe95f2d7e695",  # main, 2025-09-22
    "moonshotai/Kimi-Audio-7B-Instruct": "9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b",  # main, 2025-05-29
    "openbmb/MiniCPM-o-4_5": "44151b35f1b232a280bda5a87ea1a7575d5433fc",  # main, 2026-03-07
    # Phi-4 uses the head of refs/pr/78, which adds the
    # prepare_inputs_for_generation placeholder needed by peft.
    "microsoft/Phi-4-multimodal-instruct": "0cb22ab20b10ac01c49ecd8b7138dcd98bc00548",  # refs/pr/78, 2025-07-14
    "nvidia/audio-flamingo-3-chat": "f5101f44de138aed8527791212be2e070224fb86",  # main, 2025-07-14
    "nvidia/audio-flamingo-3": "ee26c58423988d7d7cda7b85dd3ce5d97ee8753d",  # main, 2025-11-28
    "XiaomiMiMo/MiMo-Audio-7B-Instruct": "6fd52f6aa692f46f5091b12d538b18cf3cb0327c",  # main, 2025-09-23
    "XiaomiMiMo/MiMo-Audio-Tokenizer": "c00d7b05062f6eda60890a03b9dfcb7da09b1d29",  # main, 2025-09-19
    "stepfun-ai/Step-Audio-2-mini": "e36fdd5d71e0ea22f09dd94bbab9bfc544ca1e36",  # main, 2026-02-14
    "stepfun-ai/Step-Audio-2-mini-Base": "83717d15455d2f831a35c8e6cbc7f998dd48c0e3",  # main, 2025-09-01
    "Qwen/Qwen2-Audio-7B-Instruct": "0a095220c30b7b31434169c3086508ef3ea5bf0a",  # main, 2025-01-12
    "mistralai/Voxtral-Small-24B-2507": "da5b42409f279fdd92febee0511a6c32828569c1",  # main, 2025-12-20
    "mistralai/Voxtral-Mini-3B-2507": "3060fe34b35ba5d44202ce9ff3c097642914f8f3",  # main, 2025-07-28
}


def hf_revision(repo_id: str) -> str:
    """Return the pinned commit for ``repo_id``; unpinned repos are an error."""
    try:
        return HF_REVISIONS[repo_id]
    except KeyError:
        raise KeyError(
            f"No pinned Hugging Face revision for {repo_id!r}; add it to HF_REVISIONS."
        ) from None


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def prefetch_repo(repo_id: str, revision: str | None = None, allow_patterns=None):
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": repo_id,
        "cache_dir": HF_HOME,
    }
    token = hf_token()
    if token:
        kwargs["token"] = token
    if revision is None:
        revision = HF_REVISIONS.get(repo_id)
    if revision:
        kwargs["revision"] = revision
    if allow_patterns:
        kwargs["allow_patterns"] = allow_patterns
    return snapshot_download(**kwargs)


def safe_enter(fn):
    """Capture startup failures so deployed apps do not crash-loop forever."""

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        self._init_error = None
        self._init_traceback = None
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:
            self._init_error = f"{type(e).__name__}: {e}"
            self._init_traceback = traceback.format_exc()
            print(self._init_traceback, flush=True)
            return None

    return wrapper


def init_error_response(instance, prompt: str = "", include_traceback: bool = False):
    err = getattr(instance, "_init_error", None)
    if not err:
        return None
    response = {
        "prompt": prompt,
        "response": f"[INIT_ERROR] {err}",
        "submission_payload": {
            "provider": "modal-init",
            "transport": "modal-startup-error",
            "prompt_text": prompt,
            "error": err,
        },
        "submission_trace": {
            "exact_submission_available": True,
            "builder": f"{instance.__class__.__name__}.init_error_response",
        },
    }
    if include_traceback:
        response["traceback"] = getattr(instance, "_init_traceback", None)
    return response


def coerce_text_response(response) -> str:
    """Normalize common model chat/generate return shapes into plain text."""
    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        for key in ("response", "text", "answer", "content", "output"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value
        raise TypeError(f"Unsupported response dict keys: {sorted(response)}")

    if isinstance(response, (list, tuple)):
        for value in response:
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, dict):
                try:
                    return coerce_text_response(value)
                except TypeError:
                    continue
        raise TypeError("No text content found in response tuple/list")

    raise TypeError(f"Unsupported response type: {type(response).__name__}")


def audio_format_from_path(audio_path: str) -> str:
    ext = audio_path.rsplit(".", 1)[-1].lower()
    return {"mp3": "mp3", "flac": "flac", "ogg": "ogg"}.get(ext, "wav")


def normalize_message_content(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        text = str(content).strip()
        return [{"type": "text", "text": text}] if text else []
    if not isinstance(content, list):
        return []

    normalized: list[dict[str, Any]] = []
    next_audio_index = 0
    for part in content:
        if isinstance(part, str):
            text = str(part).strip()
            if text:
                normalized.append({"type": "text", "text": text})
            continue
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "").strip()
        if part_type in {"audio", "audio_file"}:
            audio_index = part.get("audio_index")
            if audio_index is None:
                audio_index = next_audio_index
            audio_index = int(audio_index)
            next_audio_index = max(next_audio_index, audio_index + 1)
            normalized.append({"type": "audio", "audio_index": audio_index})
            continue
        text = str(part.get("text") or "").strip()
        if text:
            normalized.append({"type": "text", "text": text})
    return normalized


def normalize_messages(messages: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(messages, list):
        return normalized
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").strip() or "user"
        normalized.append({"role": role, "content": normalize_message_content(message.get("content"))})
    return normalized


def _legacy_messages(prompt: str, audio_count: int) -> list[dict[str, Any]]:
    content = [{"type": "audio", "audio_index": idx} for idx in range(max(0, int(audio_count)))]
    text = str(prompt or "").strip()
    if text:
        content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def flatten_messages_to_text(
    messages: Any,
    *,
    audio_placeholder: str | Callable[[int], str] = "[audio]",
    include_roles: bool = False,
) -> str:
    rendered_messages: list[str] = []
    for message in normalize_messages(messages):
        rendered_parts: list[str] = []
        for part in message["content"]:
            if part.get("type") == "audio":
                placeholder = (
                    audio_placeholder(int(part["audio_index"]))
                    if callable(audio_placeholder)
                    else str(audio_placeholder)
                )
                placeholder = str(placeholder or "").strip()
                if placeholder:
                    rendered_parts.append(placeholder)
                continue
            text = str(part.get("text") or "").strip()
            if text:
                rendered_parts.append(text)
        rendered = "\n".join(rendered_parts).strip()
        if not rendered:
            continue
        if include_roles:
            rendered_messages.append(f"{message['role'].upper()}:\n{rendered}")
        else:
            rendered_messages.append(rendered)
    return "\n\n".join(rendered_messages).strip()


def last_user_text(messages: Any, *, audio_placeholder: str | Callable[[int], str] = "[audio]") -> str:
    normalized = normalize_messages(messages)
    for message in reversed(normalized):
        if message["role"] != "user":
            continue
        return flatten_messages_to_text(
            [message],
            audio_placeholder=audio_placeholder,
            include_roles=False,
        )
    return ""


def render_messages(
    messages: Any,
    audio_values: list[Any],
    *,
    build_text_part: Callable[[str, str], Any],
    build_audio_part: Callable[[Any, int, str], Any],
    drop_empty_messages: bool = True,
) -> list[dict[str, Any]]:
    rendered_messages: list[dict[str, Any]] = []
    for message in normalize_messages(messages):
        role = message["role"]
        rendered_content = render_audio_placeholder(
            message["content"],
            audio_values,
            build_text_part=lambda text: build_text_part(text, role),
            build_audio_part=lambda audio_value, audio_index: build_audio_part(audio_value, audio_index, role),
        )
        if drop_empty_messages and not rendered_content:
            continue
        rendered_messages.append({"role": role, "content": rendered_content})
    return rendered_messages


def render_audio_placeholder(
    content: Any,
    audio_values: list[Any],
    *,
    build_text_part: Callable[[str], Any],
    build_audio_part: Callable[[Any, int], Any],
) -> list[Any]:
    rendered: list[Any] = []
    for part in normalize_message_content(content):
        if part.get("type") == "audio":
            built = build_audio_part(audio_values[int(part["audio_index"])], int(part["audio_index"]))
        else:
            built = build_text_part(str(part.get("text") or ""))
        if built is None:
            continue
        if isinstance(built, list):
            rendered.extend(item for item in built if item is not None)
        else:
            rendered.append(built)
    return rendered


@contextmanager
def materialize_audio_files(audio_items):
    import tempfile

    paths = []
    try:
        for audio_item in audio_items:
            suffix = f".{audio_item.get('format', 'wav')}"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                f.write(audio_item["bytes"])
                paths.append(f.name)
        yield paths
    finally:
        for path in paths:
            if os.path.exists(path):
                os.unlink(path)


def decode_messages_payload(item: dict) -> tuple:
    """Decode the standard audio payload from the adapter.

    Returns (messages, audio_items, meta).
    """
    import base64

    encoded_audios = item.get("audios")
    if encoded_audios is None:
        legacy_audio_b64 = item.get("audio_b64", "")
        encoded_audios = (
            [{"data": legacy_audio_b64, "format": item.get("audio_format", "wav")}]
            if legacy_audio_b64
            else []
        )

    audio_items = []
    for encoded_audio in encoded_audios:
        if not encoded_audio.get("data"):
            continue
        audio_items.append({
            "bytes": base64.b64decode(encoded_audio["data"]),
            "format": encoded_audio.get("format", "wav"),
        })
    messages = normalize_messages(item.get("messages"))
    if not messages:
        messages = _legacy_messages(item.get("text", ""), len(audio_items))
    meta = item.get("meta", {})
    return messages, audio_items, meta


def decode_audio_payload(item: dict) -> tuple:
    messages, audio_items, meta = decode_messages_payload(item)
    return audio_items, last_user_text(messages), meta


def benchmark_sampling_enabled(meta: dict) -> bool:
    return str(meta.get("_sampling_mode", "adapter-default")) == "benchmark-default"


def _meta_optional_float(meta: dict, key: str) -> float | None:
    value = meta.get(key)
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def apply_benchmark_sampling(
    meta: dict,
    kwargs: dict,
    *,
    do_sample_key: str | None = "do_sample",
    temperature_key: str | None = "temperature",
    top_p_key: str | None = "top_p",
    benchmark_temperature_default: float | None = None,
) -> dict:
    updated = dict(kwargs)
    if not benchmark_sampling_enabled(meta):
        return updated

    if do_sample_key:
        updated[do_sample_key] = True

    if temperature_key:
        temperature = _meta_optional_float(meta, "_benchmark_temperature")
        if temperature is None:
            temperature = benchmark_temperature_default
        if temperature is not None:
            updated[temperature_key] = temperature
        elif updated.get(temperature_key) == 0.0:
            updated.pop(temperature_key, None)

    if top_p_key:
        top_p = _meta_optional_float(meta, "_benchmark_top_p")
        if top_p is not None:
            updated[top_p_key] = top_p

    return updated


def build_smoke_test_item(
    audio_path: str = "",
    text: str = "",
    task: str = "ASR",
    interactive: str = "",
    audio_type: str = "",
    dataset_name: str = "",
    lang: str = "",
    meta_json: str = "",
    audio_paths: list[str] | None = None,
) -> dict:
    import base64
    import json
    from pathlib import Path

    meta = {}
    if meta_json:
        meta = json.loads(meta_json)
        if not isinstance(meta, dict):
            raise ValueError("meta_json must decode to a JSON object")

    for key, value in {
        "task": task,
        "interactive": interactive,
        "audio_type": audio_type,
        "dataset_name": dataset_name,
        "lang": lang,
    }.items():
        if value:
            meta[key] = value

    meta["_debug"] = True
    if audio_paths is None:
        if not audio_path:
            raise ValueError("audio_path or audio_paths must be provided")
        audio_paths = [audio_path]

    audios = []
    for path in audio_paths:
        audio_bytes = Path(path).read_bytes()
        audios.append({
            "data": base64.b64encode(audio_bytes).decode("utf-8"),
            "format": audio_format_from_path(path),
        })
    messages = _legacy_messages(text, len(audios))
    return {
        "audio_b64": audios[0]["data"],
        "audio_format": audios[0]["format"],
        "audios": audios,
        "messages": messages,
        "text": text,
        "meta": meta,
    }


def audio_bytes_to_ndarray(audio_bytes: bytes, sr: int = 16000):
    """Convert raw audio bytes to a numpy array via librosa."""
    import io
    import librosa

    audio_data, actual_sr = librosa.load(io.BytesIO(audio_bytes), sr=sr)
    return audio_data, actual_sr


def audio_items_to_ndarrays(audio_items, sr: int = 16000):
    return [
        audio_bytes_to_ndarray(audio_item["bytes"], sr=sr)[0]
        for audio_item in audio_items
    ]
