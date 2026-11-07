"""Modal serving app for MiMo-Audio-7B-Instruct.

Supports both single-audio (audio_understanding_sft) and multi-audio
(concatenated waveform with labeled silence gaps) question formats.

Deploy:  modal deploy [--env <modal-env>] modal_serving/mimo_audio_serve.py
Prefetch: modal run [--env <modal-env>] modal_serving/mimo_audio_serve.py::prefetch
Smoke test: modal run [--env <modal-env>] modal_serving/mimo_audio_serve.py
"""

import logging

import modal
from common import (
    decode_messages_payload,
    flatten_messages_to_text,
    hf_cache,
    hf_revision,
    hf_secrets,
    init_error_response,
    materialize_audio_files,
    run_smoke_test,
    safe_enter,
)

app = modal.App("audiobench-mimo-audio")
logger = logging.getLogger(__name__)

HF_CACHE_PATH = "/hf-cache"
MODEL_REPO = "XiaomiMiMo/MiMo-Audio-7B-Instruct"
TOKENIZER_REPO = "XiaomiMiMo/MiMo-Audio-Tokenizer"
MODEL_REPOS = [MODEL_REPO, TOKENIZER_REPO]
# Upstream MiMo-Audio at the last commit on or before 2026-05-07 (2025-09-20).
MIMO_AUDIO_GIT_SHA = "62d956b4a1a45419bee5e41f477078c3684dbbcc"

CONCAT_GAP_S = 1.0
SAMPLE_RATE = 16000

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
    )
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        # Previously floating requirements are pinned to the latest release on
        # or before 2026-05-07.
        "accelerate==1.13.0",
        "torchaudio==2.6.0",
        "transformers==4.49.0",
        "librosa==0.11.0",
        "scipy==1.17.1",
        "soundfile==0.13.1",
        "einops==0.8.2",
        "sentencepiece==0.2.1",
        "packaging==26.2",
        "ninja==1.13.0",
        "wheel==0.47.0",
        "setuptools==82.0.1",
    )
    .run_commands(
        "pip install flash-attn==2.7.4.post1 --no-build-isolation",
        gpu="A100-80GB",
    )
    .run_commands(
        "git clone https://github.com/XiaomiMiMo/MiMo-Audio.git /app/MiMo-Audio",
        f"cd /app/MiMo-Audio && git checkout {MIMO_AUDIO_GIT_SHA}",
    )
    .env({
        "PYTHONPATH": "/app/MiMo-Audio",
        "HF_HOME": f"{HF_CACHE_PATH}/huggingface",
        "HF_HUB_CACHE": f"{HF_CACHE_PATH}/huggingface/hub",
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/huggingface/hub",
    })
    .add_local_python_source("common", copy=True)
    .add_local_python_source("serving_utils", copy=True)
)


def _concatenate_audio_files(paths: list[str], gap_s: float = CONCAT_GAP_S) -> str:
    """Concatenate multiple audio files with silence gaps into a single temp file."""
    import tempfile

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


@app.function(
    image=image,
    volumes={HF_CACHE_PATH: hf_cache},
    secrets=hf_secrets,
    timeout=7200,
    retries=0,
)
def prefetch():
    from huggingface_hub import snapshot_download

    for repo in MODEL_REPOS:
        snapshot_download(repo, revision=hf_revision(repo))
    hf_cache.commit()


def _generate_impl(model, item: dict, *, thinking: bool, builder_name: str):
    import os

    messages, audio_items, meta = decode_messages_payload(item)
    prompt = flatten_messages_to_text(messages, include_roles=True)

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
            import tempfile
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
            response = model.audio_understanding_sft(
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


def _patch_greedy(model):
    from src.mimo_audio.modeling_mimo_audio import MiMoSampler
    greedy = MiMoSampler(do_sample=False, temperature=1.0, top_p=1.0)
    for task_name in model.task_sampler_configs:
        model.task_sampler_configs[task_name] = {"global": greedy, "local": greedy}


@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={HF_CACHE_PATH: hf_cache},
    secrets=hf_secrets,
    timeout=600,
    scaledown_window=300,
    retries=0,
)
class MiMoAudioServer:

    @modal.enter()
    @safe_enter
    def load_model(self):
        from huggingface_hub import snapshot_download
        from src.mimo_audio.mimo_audio import MimoAudio

        self.model = MimoAudio(
            snapshot_download(MODEL_REPO, revision=hf_revision(MODEL_REPO)),
            snapshot_download(TOKENIZER_REPO, revision=hf_revision(TOKENIZER_REPO)),
        )
        _patch_greedy(self.model)

    @modal.method()
    def generate(self, item: dict):
        err = init_error_response(self, "", include_traceback=bool(item.get("meta", {}).get("_debug")))
        if err:
            return err
        return _generate_impl(self.model, item, thinking=False, builder_name="MiMoAudioServer.generate")


@app.local_entrypoint()
def smoke_test(
    audio_path: str = "",
    text: str = "Describe the emotion in this audio.",
):
    run_smoke_test(
        MiMoAudioServer,
        audio_path=audio_path,
        text=text,
        task="AudioAnalysis",
    )
