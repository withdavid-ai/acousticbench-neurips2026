"""Modal serving app for the fine-tuned Kimi-Audio checkpoint.

Deploy:  modal deploy --env=acoustic-bench modal_serving/kimi_audio_ft_serve.py
"""

import logging
import time
from pathlib import Path

import modal
from common import (
    HF_RUNTIME_ENV,
    apply_benchmark_sampling,
    decode_messages_payload,
    flatten_messages_to_text,
    hf_cache,
    hf_secrets,
    init_error_response,
    materialize_audio_files,
    render_messages,
    run_smoke_test,
    safe_enter,
)

app = modal.App("audiobench-kimi-audio-ft")
logger = logging.getLogger(__name__)

MOONSHOT_CACHE_PATH = "/hf-cache"
MOONSHOT_HF_HOME = f"{MOONSHOT_CACHE_PATH}/huggingface"
MOONSHOT_HF_ENV = {
    **HF_RUNTIME_ENV,
    "HF_HOME": MOONSHOT_HF_HOME,
    "HF_HUB_CACHE": f"{MOONSHOT_HF_HOME}/hub",
    "HUGGINGFACE_HUB_CACHE": f"{MOONSHOT_HF_HOME}/hub",
    "TRANSFORMERS_CACHE": f"{MOONSHOT_HF_HOME}/hub",
}
MODEL_VOLUME_PATH = "/data"
MODEL_PATH = f"{MODEL_VOLUME_PATH}/serving/kimi-audio-ft/latest"

hf_model_volume = modal.Volume.from_name(
    "acousticbench-kimi-rad-data",
    create_if_missing=True,
)


def _log(message: str) -> None:
    print(f"[KimiAudioFtServer] {message}", flush=True)

image = (
    modal.Image.from_registry("moonshotai/kimi-audio:v0.1")
    .run_commands("python -m pip install --no-deps git+https://github.com/MoonshotAI/Kimi-Audio.git")
    .env(MOONSHOT_HF_ENV)
    .add_local_python_source("common", copy=True)
)


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


@app.function(
    image=image,
    volumes={
        MOONSHOT_CACHE_PATH: hf_cache,
        MODEL_VOLUME_PATH: hf_model_volume,
    },
    secrets=hf_secrets,
    timeout=7200,
    retries=0,
)
def prefetch(model_path: str = MODEL_PATH):
    model_path = Path(str(model_path or MODEL_PATH).strip())
    if not model_path.exists():
        raise FileNotFoundError(
            f"Expected packaged checkpoint at {model_path}. Run the package step first."
        )
    return {"status": "ok", "model_path": str(model_path)}


@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={
        MOONSHOT_CACHE_PATH: hf_cache,
        MODEL_VOLUME_PATH: hf_model_volume,
    },
    secrets=hf_secrets,
    timeout=900,
    scaledown_window=300,
    retries=0,
)
class KimiAudioFtServer:
    model_path: str = modal.parameter(default=MODEL_PATH)

    def _resolve_model_path(self) -> Path:
        t0 = time.perf_counter()
        requested_path = str(self.model_path or MODEL_PATH).strip()
        model_path = Path(requested_path or MODEL_PATH)
        _log(f"resolve_model_path start model_path={model_path}")
        if str(model_path).startswith(f"{MODEL_VOLUME_PATH}/"):
            _log(f"volume_reload start model_path={model_path}")
            hf_model_volume.reload()
            _log(
                "volume_reload done "
                f"elapsed_s={time.perf_counter() - t0:.2f} model_path={model_path}"
            )
        if not model_path.exists():
            raise FileNotFoundError(
                f"Expected packaged checkpoint at {model_path}. Run the package step first."
            )
        _log(
            "resolve_model_path done "
            f"elapsed_s={time.perf_counter() - t0:.2f} model_path={model_path}"
        )
        return model_path

    def _load_model_at_path(self, model_path: Path) -> None:
        t0 = time.perf_counter()
        _log(f"load_model import start model_path={model_path}")
        from kimia_infer.api.kimia import KimiAudio

        _log(
            "load_model constructor start "
            f"elapsed_s={time.perf_counter() - t0:.2f} model_path={model_path}"
        )
        self.model = KimiAudio(
            model_path=str(model_path),
            load_detokenizer=False,
        )
        self.loaded_model_path = str(model_path)
        self._init_error = None
        self._init_traceback = None
        _log(
            "load_model done "
            f"elapsed_s={time.perf_counter() - t0:.2f} model_path={model_path}"
        )

    @modal.enter()
    @safe_enter
    def load_model(self):
        t0 = time.perf_counter()
        _log(f"enter start requested_model_path={self.model_path}")
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
        model_path = self._resolve_model_path()
        self._load_model_at_path(model_path)
        _log(
            "enter done "
            f"elapsed_s={time.perf_counter() - t0:.2f} model_path={model_path}"
        )

    @modal.method()
    def generate(self, item: dict):
        request_t0 = time.perf_counter()
        messages, audio_items, meta = decode_messages_payload(item)
        eval_index = meta.get("_eval_index", "unknown")
        example_id = meta.get("example_id", "unknown")
        prompt = flatten_messages_to_text(messages, include_roles=True)
        err = init_error_response(self, prompt, include_traceback=bool(meta.get("_debug")))
        if err:
            _log(
                "generate init_error "
                f"eval_index={eval_index} example_id={example_id} error={err.get('response')}"
            )
            return err
        resolved_model_path = Path(getattr(self, "loaded_model_path", "") or self.model_path)
        _log(
            "generate start "
            f"eval_index={eval_index} example_id={example_id} "
            f"audio_count={len(audio_items)} model_path={resolved_model_path}"
        )
        legacy_override = str(meta.get("_model_path_override") or "").strip()
        if legacy_override and legacy_override != str(resolved_model_path):
            raise ValueError(
                "KimiAudioFtServer model_path is frozen at class initialization. "
                "Instantiate KimiAudioFtServer(model_path=...) instead of sending "
                "_model_path_override in request metadata."
            )

        sampling_params = apply_benchmark_sampling(
            meta,
            dict(self.sampling_params),
            do_sample_key=None,
            temperature_key="text_temperature",
            top_p_key=None,
            benchmark_temperature_default=1.0,
        )
        # Pairwise benchmark tasks should terminate after a very short text answer.
        # The default `-1` can hang on some exported checkpoints during validation.
        if str(meta.get("task") or "").strip().upper() == "MQA":
            sampling_params["max_new_tokens"] = min(
                16,
                int(sampling_params.get("max_new_tokens", 16))
                if int(sampling_params.get("max_new_tokens", 16)) > 0
                else 16,
            )

        with materialize_audio_files(audio_items) as audio_paths:
            _log(
                "materialize_audio done "
                f"eval_index={eval_index} elapsed_s={time.perf_counter() - request_t0:.2f}"
            )
            native_messages = _to_kimi_native_messages(messages, audio_paths)
            _, response = self.model.generate(native_messages, **sampling_params, output_type="text")
        _log(
            "generate done "
            f"eval_index={eval_index} elapsed_s={time.perf_counter() - request_t0:.2f} "
            f"response={str(response)[:80]!r}"
        )
        return {
            "prompt": prompt,
            "response": response,
            "submission_payload": {
                "provider": "kimi-audio-native",
                "transport": "moonshot-kimia-python",
                "model_path": str(resolved_model_path),
                "messages": native_messages,
                "sampling_params": sampling_params,
                "output_type": "text",
            },
            "submission_trace": {
                "exact_submission_available": True,
                "builder": "KimiAudioFtServer.generate",
            },
        }


@app.local_entrypoint()
def smoke_test(
    audio_path: str,
    text: str = "",
    task: str = "ASR",
    interactive: str = "",
    audio_type: str = "",
    dataset_name: str = "",
    lang: str = "",
    meta_json: str = "",
    prefetch_first: bool = False,
    audio_path_2: str = "",
):
    run_smoke_test(
        KimiAudioFtServer,
        "generate",
        audio_path=audio_path,
        text=text,
        task=task,
        interactive=interactive,
        audio_type=audio_type,
        dataset_name=dataset_name,
        lang=lang,
        meta_json=meta_json,
        prefetch_fn=prefetch if prefetch_first else None,
        audio_path_2=audio_path_2,
    )
