"""Modal serving app for Qwen2-Audio-7B-Instruct.

Deploy:
    modal deploy [--env <modal-env>] modal_serving/qwen2_audio_serve.py

Provides one endpoint:
    /generate_chat  — Qwen2-Audio-7B-Instruct (chat)
"""

import modal
from common import (
    apply_benchmark_sampling,
    audio_items_to_ndarrays,
    base_image,
    decode_messages_payload,
    flatten_messages_to_text,
    hf_cache,
    hf_revision,
    hf_secrets,
    init_error_response,
    prefetch_repos,
    render_messages,
    run_smoke_test,
    safe_enter,
)

HF_CACHE_PATH = "/hf-cache"

app = modal.App("audiobench-qwen2-audio")
image = (
    base_image()
    .pip_install("torch==2.5.1", "torchaudio==2.5.1")
    .env({
        "HF_HOME": f"{HF_CACHE_PATH}/huggingface",
        "HF_HUB_CACHE": f"{HF_CACHE_PATH}/huggingface/hub",
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/huggingface/hub",
    })
    .add_local_python_source("serving_utils", copy=True)
)

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
MODEL_REVISION = hf_revision(MODEL_ID)
MODEL_REPOS = [MODEL_ID]


@app.function(image=image, volumes={HF_CACHE_PATH: hf_cache}, secrets=hf_secrets, timeout=7200, retries=0)
def prefetch():
    prefetch_repos(MODEL_REPOS)


@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={HF_CACHE_PATH: hf_cache},
    secrets=hf_secrets,
    timeout=600,
    scaledown_window=300,
    retries=0,
)
class Qwen2AudioChatServer:

    @modal.enter()
    @safe_enter
    def load_model(self):
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, device_map="auto",
        ).eval()
        hf_cache.commit()

    @modal.method()
    def generate_chat(self, item: dict):

        messages, audio_items, meta = decode_messages_payload(item)
        prompt = flatten_messages_to_text(messages, include_roles=True)
        err = init_error_response(self, prompt, include_traceback=bool(meta.get("_debug")))
        if err:
            return err

        sr = self.processor.feature_extractor.sampling_rate
        audio_datas = audio_items_to_ndarrays(audio_items, sr=sr) if audio_items else []
        conversation = render_messages(
            messages,
            audio_datas,
            build_text_part=lambda text, _role: (
                {"type": "text", "text": text} if str(text).strip() else None
            ),
            build_audio_part=lambda _audio, _audio_index, _role: {
                "type": "audio",
                "audio_url": "placeholder",
            },
        )
        for message in conversation:
            if message["role"] == "system":
                message["content"] = flatten_messages_to_text([message], include_roles=False)

        chat_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)

        processor_kwargs = {
            "text": chat_text, "return_tensors": "pt",
            "padding": True, "sampling_rate": sr,
        }
        if audio_datas:
            processor_kwargs["audios"] = audio_datas
        inputs = self.processor(**processor_kwargs).to(self.model.device)

        gen_kwargs = apply_benchmark_sampling(
            meta,
            {"max_new_tokens": 256, "do_sample": False},
        )
        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True,
        )[0]
        return {
            "prompt": prompt,
            "response": response,
            "submission_payload": {
                "provider": "qwen2-audio-native",
                "transport": "transformers-local-generate",
                "conversation": conversation,
                "chat_text": chat_text,
                "generation_kwargs": gen_kwargs,
                "processor_inputs": {
                    "input_keys": sorted(inputs.keys()),
                    "input_shapes": {
                        key: list(value.shape)
                        for key, value in inputs.items()
                        if hasattr(value, "shape")
                    },
                },
            },
            "submission_trace": {
                "exact_submission_available": True,
                "builder": "Qwen2AudioChatServer.generate_chat",
            },
        }


SMOKE_TEST_VARIANTS = {
    "chat": (Qwen2AudioChatServer, "generate_chat"),
}


@app.local_entrypoint()
def smoke_test(
    audio_path: str,
    variant: str = "chat",
    text: str = "",
    task: str = "ASR",
    interactive: str = "",
    audio_type: str = "",
    dataset_name: str = "",
    lang: str = "",
    meta_json: str = "",
    prefetch_first: bool = False,
):
    if variant not in SMOKE_TEST_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {sorted(SMOKE_TEST_VARIANTS)}")

    server_cls, method_name = SMOKE_TEST_VARIANTS[variant]
    run_smoke_test(
        server_cls,
        method_name,
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
