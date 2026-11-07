"""Modal serving app for AcousticBench Kimi-Audio LoRA adapters.

Deploy:  modal deploy [--env <modal-env>] modal_serving/kimi_lora_serve.py

Before deploying, extract Kimi finetunes into the repository
archive directory. This file uses the same layout as check_repo_setup.py:
adapter folders named `rad`, `rsq`, `ser_classify`, and `ser_intensity` may
live at `archive/`, `archive/models/`, or `archive/finetunes/`.
"""

import logging
import json
import shutil
import tempfile
import time
from pathlib import Path

import modal
from common import (
    HF_RUNTIME_ENV,
    apply_benchmark_sampling,
    decode_messages_payload,
    flatten_messages_to_text,
    hf_cache,
    hf_revision,
    hf_secrets,
    init_error_response,
    materialize_audio_files,
    render_messages,
    run_smoke_test,
    safe_enter,
)

app = modal.App("audiobench-kimi-lora")
logger = logging.getLogger(__name__)

MOONSHOT_CACHE_PATH = "/hf-cache"
MOONSHOT_HF_HOME = f"{MOONSHOT_CACHE_PATH}/huggingface"
KIMI_AUDIO_REPO_DIR = "/opt/Kimi-Audio"
MOONSHOT_HF_ENV = {
    **HF_RUNTIME_ENV,
    "HF_HOME": MOONSHOT_HF_HOME,
    "HF_HUB_CACHE": f"{MOONSHOT_HF_HOME}/hub",
    "HUGGINGFACE_HUB_CACHE": f"{MOONSHOT_HF_HOME}/hub",
    "TRANSFORMERS_CACHE": f"{MOONSHOT_HF_HOME}/hub",
}

BASE_MODEL_ID = "moonshotai/Kimi-Audio-7B-Instruct"
BASE_MODEL_REVISION = hf_revision(BASE_MODEL_ID)
# Upstream Kimi-Audio at the last commit on or before 2026-05-07 (2025-06-03).
KIMI_AUDIO_GIT_SHA = "349251e1d8f4f98d58fda59246381faecd7392e0"
DATA_VOLUME_PATH = "/kimi-lora-data"
ADAPTER_ROOT = Path(DATA_VOLUME_PATH) / "adapters"
PRETRAINED_ROOT = Path(DATA_VOLUME_PATH) / "pretrained_hf"
EXPORT_ROOT = Path(DATA_VOLUME_PATH) / "exported"
IMAGE_ADAPTER_SOURCE_ROOT = Path("/root/kimi-lora-adapters")
VALID_ADAPTERS = {"rad", "rsq", "ser_classify", "ser_intensity", "simple_audioqa"}
KIMI_ADAPTER_SUBDIRS = ("", "models", "finetunes")
EXPORT_PACKAGING_VERSION = "acousticbench-vendored-kimi-lora-v2-2026-05-06"

adapter_volume = modal.Volume.from_name("audiobench-kimi-lora-adapters", create_if_missing=True)


def _repo_root_from_file() -> Path:
    path = Path(__file__).resolve()
    if len(path.parents) > 2:
        return path.parents[2]
    return Path.cwd()


REPO_ROOT = _repo_root_from_file()
LOCAL_ARCHIVE_ROOT = REPO_ROOT / "archive"


def _find_adapter_root(root: Path) -> Path:
    """Find extracted Kimi LoRA adapters using check_repo_setup.py's layout."""
    candidates = [
        root / subdir if subdir else root
        for subdir in KIMI_ADAPTER_SUBDIRS
    ]
    for candidate in candidates:
        if all((candidate / adapter / "adapter_model.safetensors").is_file() for adapter in VALID_ADAPTERS):
            return candidate
    raise FileNotFoundError(
        "Could not find Kimi-Audio LoRA adapters. Expected adapter folders "
        "`rad`, `rsq`, `ser_classify`, `ser_intensity`, and `simple_audioqa`, each containing "
        "`adapter_model.safetensors`, under one of: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


LOCAL_ADAPTER_SOURCE_ROOT = (
    _find_adapter_root(LOCAL_ARCHIVE_ROOT)
    if LOCAL_ARCHIVE_ROOT.exists()
    else None
)

image = (
    modal.Image.from_registry("moonshotai/kimi-audio:v0.1")
    .apt_install("git")
    .run_commands(
        # Match the main AcousticBench finetuning image: clone the upstream
        # repo with submodules, install Kimi requirements, then install the
        # package editable. The LoRA merge/export code below uses our vendored
        # patched finetune module, but `kimia_infer` still comes from here.
        f"git clone https://github.com/MoonshotAI/Kimi-Audio.git {KIMI_AUDIO_REPO_DIR}",
        f"cd {KIMI_AUDIO_REPO_DIR} && git checkout {KIMI_AUDIO_GIT_SHA}",
        f"cd {KIMI_AUDIO_REPO_DIR} && git submodule update --init --recursive",
        f"cd {KIMI_AUDIO_REPO_DIR} && pip install -r requirements.txt",
        f"cd {KIMI_AUDIO_REPO_DIR} && pip install -e .",
    )
    # Latest peft on or before 2026-05-07 (released 2026-04-16).
    .pip_install("peft==0.19.1")
    .env(MOONSHOT_HF_ENV)
    .add_local_python_source("common", copy=True)
    .add_local_python_source("serving_utils", copy=True)
    .add_local_python_source("kimi_vendor", copy=True)
)
if LOCAL_ADAPTER_SOURCE_ROOT is not None:
    image = image.add_local_dir(
        str(LOCAL_ADAPTER_SOURCE_ROOT),
        remote_path=str(IMAGE_ADAPTER_SOURCE_ROOT),
        copy=True,
    )


def _log(message: str) -> None:
    print(f"[KimiLoRAServer] {message}", flush=True)


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


def _sync_adapters_from_archive(source_root: Path = IMAGE_ADAPTER_SOURCE_ROOT) -> None:
    source_root = _find_adapter_root(source_root)
    if ADAPTER_ROOT.exists():
        shutil.rmtree(ADAPTER_ROOT)
    ADAPTER_ROOT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_root, ADAPTER_ROOT)
    # Adapter weights changed; merged exports under EXPORT_ROOT were merged
    # against the previous weights so they are now stale. Wipe them so the
    # next request triggers a fresh merge against the new adapter weights.
    if EXPORT_ROOT.exists():
        _log(f"clearing stale merged exports at {EXPORT_ROOT}")
        shutil.rmtree(EXPORT_ROOT)
    adapter_volume.commit()


def _adapter_dir(adapter_name: str) -> Path:
    adapter_name = str(adapter_name or "").strip()
    if adapter_name not in VALID_ADAPTERS:
        raise ValueError(f"Unknown Kimi LoRA adapter {adapter_name!r}; expected one of {sorted(VALID_ADAPTERS)}")
    return ADAPTER_ROOT / adapter_name


def _ensure_adapter_present(adapter_name: str) -> Path:
    adapter_volume.reload()
    adapter_dir = _adapter_dir(adapter_name)
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        _log(
            f"adapter {adapter_name} missing from volume; "
            f"syncing from bundled archive adapters at {IMAGE_ADAPTER_SOURCE_ROOT}"
        )
        _sync_adapters_from_archive()
        adapter_volume.reload()
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Missing adapter weights: {adapter_dir}")
    return adapter_dir


def _copy_kimi_runtime_assets(
    model_name: str, output_dir: Path, revision: str | None = None,
) -> list[str]:
    """Copy tokenizer/runtime assets from the HF snapshot, matching main repo.

    Kimi's tokenizer implementation does not reliably support
    `save_pretrained`, so the finetuning pipeline copies the required runtime
    files straight from the source snapshot after export.
    """
    from huggingface_hub import snapshot_download

    snapshot_dir = Path(model_name)
    if not snapshot_dir.exists():
        snapshot_dir = Path(snapshot_download(model_name, revision=revision))

    copied: list[str] = []
    for rel_path in (
        "tokenization_kimia.py",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tiktoken.model",
        "generation_config.json",
        "audio_detokenizer",
        "vocoder",
    ):
        src = snapshot_dir / rel_path
        dest = output_dir / rel_path
        if src.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            copied.append(rel_path)
        elif src.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(rel_path)
    return copied


def _ensure_pretrained_hf() -> Path:
    """Materialize the base Kimi checkpoint in the main finetuning layout."""
    if (PRETRAINED_ROOT / "config.json").is_file():
        return PRETRAINED_ROOT

    from kimi_vendor.finetune_codes.model import KimiAudioModel

    t0 = time.perf_counter()
    tmp_root = Path(tempfile.mkdtemp(prefix="kimi-lora-pretrained-", dir="/tmp"))
    try:
        tmp_output = tmp_root / "pretrained_hf"
        _log(f"materializing base pretrained_hf model={BASE_MODEL_ID}")
        from huggingface_hub import snapshot_download

        base_snapshot = snapshot_download(BASE_MODEL_ID, revision=BASE_MODEL_REVISION)
        model = KimiAudioModel.init_from_pretrained(base_snapshot, model_load_kwargs={})
        tmp_output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(tmp_output)
        if PRETRAINED_ROOT.exists():
            shutil.rmtree(PRETRAINED_ROOT)
        PRETRAINED_ROOT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_output, PRETRAINED_ROOT)
        adapter_volume.commit()
        _log(
            "materialized base pretrained_hf "
            f"path={PRETRAINED_ROOT} elapsed_s={time.perf_counter() - t0:.2f}"
        )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    return PRETRAINED_ROOT


def _export_lora_adapter(adapter_name: str, adapter_dir: Path) -> Path:
    import torch

    from kimi_vendor.finetune_codes.model import KimiAudioModel
    from peft import PeftModel

    export_dir = EXPORT_ROOT / adapter_name
    manifest_path = export_dir / "acousticbench_lora_export_manifest.json"
    if (export_dir / "modeling_moonshot_kimia.py").is_file() and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
        if manifest.get("packaging_version") == EXPORT_PACKAGING_VERSION:
            return export_dir
        shutil.rmtree(export_dir)

    if export_dir.exists() and not manifest_path.is_file():
        shutil.rmtree(export_dir)

    t0 = time.perf_counter()
    pretrained_dir = _ensure_pretrained_hf()
    package_tmp_root = Path(DATA_VOLUME_PATH) / "package_tmp"
    package_tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix=f"kimi-lora-{adapter_name}-", dir=str(package_tmp_root)))
    merged_dir = tmp_root / "merged_preexport"
    try:
        _log(f"loading pretrained_hf for adapter={adapter_name} path={pretrained_dir}")
        base_model = KimiAudioModel.from_pretrained(
            pretrained_dir,
            device_map=None,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        _log(f"loading LoRA adapter adapter={adapter_name} dir={adapter_dir}")
        adapter_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        merged_model = adapter_model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.parent.mkdir(parents=True, exist_ok=True)
        _log(f"exporting merged checkpoint adapter={adapter_name} export_dir={export_dir}")
        KimiAudioModel.export_model(str(merged_dir), str(export_dir))
        vendored_dir = Path(__file__).resolve().parent / "kimi_vendor" / "finetune_codes"
        shutil.copy2(vendored_dir / "modeling_kimia.py", export_dir / "modeling_moonshot_kimia.py")
        shutil.copy2(vendored_dir / "configuration_moonshot_kimia.py", export_dir / "configuration_moonshot_kimia.py")
        copied_assets = _copy_kimi_runtime_assets(
            BASE_MODEL_ID, export_dir, revision=BASE_MODEL_REVISION,
        )
        _log(
            "copied Kimi runtime assets "
            f"adapter={adapter_name} assets={copied_assets}"
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "packaging_version": EXPORT_PACKAGING_VERSION,
                    "adapter_name": adapter_name,
                    "base_model_id": BASE_MODEL_ID,
                    "base_model_revision": BASE_MODEL_REVISION,
                    "pretrained_root": str(pretrained_dir),
                    "export_dir": str(export_dir),
                    "copied_runtime_assets": copied_assets,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        adapter_volume.commit()
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    _log(
        "exported LoRA adapter "
        f"adapter={adapter_name} elapsed_s={time.perf_counter() - t0:.2f}"
    )
    return export_dir


@app.function(
    image=image,
    volumes={
        MOONSHOT_CACHE_PATH: hf_cache,
        DATA_VOLUME_PATH: adapter_volume,
    },
    secrets=hf_secrets,
    timeout=7200,
    retries=0,
)
def sync_adapters():
    _sync_adapters_from_archive()
    return {"status": "ok", "adapter_root": str(ADAPTER_ROOT)}


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={
        MOONSHOT_CACHE_PATH: hf_cache,
        DATA_VOLUME_PATH: adapter_volume,
    },
    secrets=hf_secrets,
    timeout=7200,
    retries=0,
)
def prefetch(adapter_name: str = "rad"):
    adapter_dir = _ensure_adapter_present(adapter_name)
    export_dir = _export_lora_adapter(adapter_name, adapter_dir)
    return {"status": "ok", "adapter_name": adapter_name, "model_path": str(export_dir)}


@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={
        MOONSHOT_CACHE_PATH: hf_cache,
        DATA_VOLUME_PATH: adapter_volume,
    },
    secrets=hf_secrets,
    timeout=1200,
    scaledown_window=300,
    retries=0,
)
class KimiLoRAServer:
    adapter_name: str = modal.parameter(default="rad")

    @modal.enter()
    @safe_enter
    def load_model(self):
        from kimia_infer.api.kimia import KimiAudio

        t0 = time.perf_counter()
        self.adapter_name = str(self.adapter_name or "rad").strip()
        self._generate_count = 0
        self._load_count = int(getattr(self, "_load_count", 0) or 0) + 1
        self.model = None
        self.loaded_model_path = ""
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
        adapter_dir = _ensure_adapter_present(self.adapter_name)
        model_path = _export_lora_adapter(self.adapter_name, adapter_dir)
        self.model = KimiAudio(
            model_path=str(model_path),
            load_detokenizer=False,
        )
        self.loaded_model_path = str(model_path)
        self.loaded_at_monotonic_s = time.monotonic()
        _log(
            "loaded Kimi LoRA "
            f"adapter={self.adapter_name} load_count={self._load_count} "
            f"model_path={self.loaded_model_path} "
            f"elapsed_s={time.perf_counter() - t0:.2f}"
        )

    @modal.method()
    def generate(self, item: dict):
        request_t0 = time.perf_counter()
        messages, audio_items, meta = decode_messages_payload(item)
        example_id = str(meta.get("example_id") or "unknown")
        prompt = flatten_messages_to_text(messages, include_roles=True)
        err = init_error_response(self, prompt, include_traceback=bool(meta.get("_debug")))
        if err:
            return err
        if self.model is None or not self.loaded_model_path:
            raise RuntimeError(
                "KimiLoRAServer.generate called before @modal.enter completed "
                f"for adapter={self.adapter_name!r}"
            )

        self._generate_count = int(getattr(self, "_generate_count", 0) or 0) + 1
        _log(
            "generate start "
            f"adapter={self.adapter_name} generate_count={self._generate_count} "
            f"load_count={getattr(self, '_load_count', 'unknown')} "
            f"example_id={example_id} model_path={self.loaded_model_path}"
        )

        sampling_params = apply_benchmark_sampling(
            meta,
            dict(self.sampling_params),
            do_sample_key=None,
            temperature_key="text_temperature",
            top_p_key=None,
            benchmark_temperature_default=1.0,
        )
        if str(meta.get("task") or "").strip().upper() == "MQA":
            current = int(sampling_params.get("max_new_tokens", 16))
            sampling_params["max_new_tokens"] = min(16, current if current > 0 else 16)

        with materialize_audio_files(audio_items) as audio_paths:
            native_messages = _to_kimi_native_messages(messages, audio_paths)
            if audio_paths:
                _, response = self.model.generate(native_messages, **sampling_params, output_type="text")
            else:
                response = _generate_text_only(self.model, native_messages, sampling_params)
        _log(
            "generate done "
            f"adapter={self.adapter_name} generate_count={self._generate_count} "
            f"example_id={example_id} elapsed_s={time.perf_counter() - request_t0:.2f} "
            f"response={str(response)[:80]!r}"
        )
        return {
            "prompt": prompt,
            "response": response,
            "submission_payload": {
                "provider": "kimi-audio-lora",
                "transport": "moonshot-kimia-python",
                "adapter_name": self.adapter_name,
                "model_path": self.loaded_model_path,
                "messages": native_messages,
                "sampling_params": sampling_params,
                "output_type": "text",
            },
            "submission_trace": {
                "exact_submission_available": True,
                "builder": "KimiLoRAServer.generate",
            },
        }


@app.local_entrypoint()
def smoke_test(
    audio_path: str,
    text: str = "",
    adapter_name: str = "rad",
    task: str = "ASR",
    interactive: str = "",
    audio_type: str = "",
    dataset_name: str = "",
    lang: str = "",
    meta_json: str = "",
    prefetch_first: bool = False,
    audio_path_2: str = "",
):
    if prefetch_first:
        prefetch.remote(adapter_name=adapter_name)
    run_smoke_test(
        KimiLoRAServer(adapter_name=adapter_name),
        "generate",
        audio_path=audio_path,
        text=text,
        task=task,
        interactive=interactive,
        audio_type=audio_type,
        dataset_name=dataset_name,
        lang=lang,
        meta_json=meta_json,
        audio_path_2=audio_path_2,
    )
