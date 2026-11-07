"""Modal serving app for Kimi-Audio using Moonshot's Docker image.

Deploy:  modal deploy [--env <modal-env>] modal_serving/kimi_audio_serve.py
"""

import logging

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

app = modal.App("audiobench-kimi-audio")
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
MODEL_ID = "moonshotai/Kimi-Audio-7B-Instruct"
MODEL_REVISION = hf_revision(MODEL_ID)
MODEL_REPOS = [MODEL_ID]
# Upstream Kimi-Audio at the last commit on or before 2026-05-07 (2025-06-03).
KIMI_AUDIO_GIT_SHA = "349251e1d8f4f98d58fda59246381faecd7392e0"

image = (
    modal.Image.from_registry("moonshotai/kimi-audio:v0.1")
    .run_commands(
        "python -m pip install --no-deps "
        f"git+https://github.com/MoonshotAI/Kimi-Audio.git@{KIMI_AUDIO_GIT_SHA}"
    )
    .env(MOONSHOT_HF_ENV)
    .add_local_python_source("common", copy=True)
    .add_local_python_source("serving_utils", copy=True)
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


@app.function(
    image=image,
    volumes={MOONSHOT_CACHE_PATH: hf_cache},
    secrets=hf_secrets,
    timeout=7200,
    retries=0,
)
def prefetch():
    from huggingface_hub import snapshot_download

    for repo in MODEL_REPOS:
        snapshot_download(repo, revision=hf_revision(repo))
    hf_cache.commit()


@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={MOONSHOT_CACHE_PATH: hf_cache},
    secrets=hf_secrets,
    timeout=600,
    scaledown_window=300,
    retries=0,
)
class KimiAudioServer:

    @modal.enter()
    @safe_enter
    def load_model(self):
        from huggingface_hub import snapshot_download
        from kimia_infer.api.kimia import KimiAudio

        # KimiAudio downloads repo ids without a revision, so resolve the
        # pinned snapshot first and pass the local path.
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

    @modal.method()
    def generate(self, item: dict):
        messages, audio_items, meta = decode_messages_payload(item)
        prompt = flatten_messages_to_text(messages, include_roles=True)
        err = init_error_response(self, prompt, include_traceback=bool(meta.get("_debug")))
        if err:
            return err

        sampling_params = apply_benchmark_sampling(
            meta,
            dict(self.sampling_params),
            do_sample_key=None,
            temperature_key="text_temperature",
            top_p_key=None,
            benchmark_temperature_default=1.0,
        )

        with materialize_audio_files(audio_items) as audio_paths:
            native_messages = _to_kimi_native_messages(messages, audio_paths)
            if audio_paths:
                _, response = self.model.generate(native_messages, **sampling_params, output_type="text")
            else:
                response = _generate_text_only(self.model, native_messages, sampling_params)
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
):
    run_smoke_test(
        KimiAudioServer,
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
    )
