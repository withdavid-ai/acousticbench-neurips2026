"""Shared Modal infrastructure for all AcousticBench model servers.

Usage from a serving file:

    from common import CACHE_PATH, base_image, hf_cache, hf_secrets, prefetch_repos
    app = modal.App("audiobench-<model>")
    image = base_image().pip_install("extra-dep==1.0")

All transport-agnostic utilities (message decoding, audio handling, benchmark
sampling, etc.) live in serving_utils.py and are re-exported here so existing
Modal serve scripts continue to work unchanged.
"""

import os

import modal

# Re-export everything from serving_utils so `from common import X` still works
# for all the pure utility functions.
from serving_utils import (  # noqa: F401
    CACHE_PATH,
    COMMON_PIP_PACKAGES,
    HF_HOME,
    HF_HUB_CACHE,
    HF_REVISIONS,
    HF_RUNTIME_ENV,
    WHISPER_CACHE_PATH,
    _legacy_messages,
    _meta_optional_float,
    apply_benchmark_sampling,
    audio_bytes_to_ndarray,
    audio_format_from_path,
    audio_items_to_ndarrays,
    benchmark_sampling_enabled,
    build_smoke_test_item,
    coerce_text_response,
    decode_audio_payload,
    decode_messages_payload,
    flatten_messages_to_text,
    hf_revision,
    hf_token,
    init_error_response,
    last_user_text,
    materialize_audio_files,
    normalize_message_content,
    normalize_messages,
    prefetch_repo,
    render_audio_placeholder,
    render_messages,
    safe_enter,
)

# ---------------------------------------------------------------------------
# Modal-specific objects (not in serving_utils)
# ---------------------------------------------------------------------------

hf_cache = modal.Volume.from_name("audiobench-hf-cache", create_if_missing=True)

_hf_secret_values = {
    key: value
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
    if (value := os.environ.get(key))
}
hf_secrets = [modal.Secret.from_dict(_hf_secret_values)] if _hf_secret_values else []


def _runtime_image(
    *,
    python_version: str,
    torch_version: str,
    transformers_version: str,
):
    return (
        modal.Image.debian_slim(python_version=python_version)
        .apt_install("ffmpeg", "git", "libsndfile1")
        .pip_install(
            *COMMON_PIP_PACKAGES,
            f"torch=={torch_version}",
            f"torchaudio=={torch_version}",
            f"transformers=={transformers_version}",
        )
        .env(HF_RUNTIME_ENV)
        .add_local_python_source("common", "serving_utils", copy=True)
    )


def base_image(python_version: str = "3.11"):
    """Conservative stack for trust-remote-code audio models."""
    return _runtime_image(
        python_version=python_version,
        torch_version="2.4.1",
        transformers_version="4.52.4",
    )


def modern_image(python_version: str = "3.11"):
    """Newer stack for models that require post-4.x Transformers support."""
    return _runtime_image(
        python_version=python_version,
        torch_version="2.11.0",
        transformers_version="5.3.0",
    )


def audio_flamingo_image(python_version: str = "3.11"):
    """AF3 stack pinned to NVIDIA's 4.46-era runtime expectations."""
    return _runtime_image(
        python_version=python_version,
        torch_version="2.4.1",
        transformers_version="4.46.0",
    ).pip_install(
        "torchvision==0.19.1",
        "hydra-core==1.3.2",
        "kaldiio==2.18.1",
        "loguru==0.7.3",
        "opencv-python-headless==4.8.0.76",
        "pydub==0.25.1",
        "pytorchvideo==0.1.5",
    )


def prefetch_repos(repo_specs):
    for spec in repo_specs:
        if isinstance(spec, str):
            prefetch_repo(spec)
        else:
            prefetch_repo(**spec)
    hf_cache.commit()


def run_smoke_test(
    server_cls,
    method_name: str,
    audio_path: str,
    text: str = "",
    task: str = "ASR",
    interactive: str = "",
    audio_type: str = "",
    dataset_name: str = "",
    lang: str = "",
    meta_json: str = "",
    prefetch_fn=None,
    audio_path_2: str = "",
):
    import json

    if prefetch_fn is not None:
        prefetch_fn.remote()

    item_kw = dict(
        text=text,
        task=task,
        interactive=interactive,
        audio_type=audio_type,
        dataset_name=dataset_name,
        lang=lang,
        meta_json=meta_json,
    )
    if audio_path_2:
        item = build_smoke_test_item(
            audio_path="",
            audio_paths=[audio_path, audio_path_2],
            **item_kw,
        )
    else:
        item = build_smoke_test_item(audio_path=audio_path, **item_kw)
    result = getattr(server_cls(), method_name).remote(item)
    response = result.get("response")
    if isinstance(response, str) and response.startswith("[INIT_ERROR]"):
        traceback_text = result.get("traceback")
        if traceback_text:
            raise RuntimeError(f"{response}\n\n{traceback_text}")
        raise RuntimeError(response)

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result
