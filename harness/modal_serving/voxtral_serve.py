"""Modal serving app for Voxtral models (Small-24B and Mini-3B).

Deploy:
    modal deploy [--env <modal-env>] modal_serving/voxtral_serve.py
"""

import modal
from common import (
    CACHE_PATH,
    apply_benchmark_sampling,
    decode_messages_payload,
    flatten_messages_to_text,
    hf_cache,
    hf_revision,
    hf_secrets,
    init_error_response,
    modern_image,
    materialize_audio_files,
    prefetch_repos,
    render_messages,
    run_smoke_test,
    safe_enter,
)

app = modal.App("audiobench-voxtral")
image = modern_image().pip_install("mistral_common[audio]==1.10.0")

SMALL_MODEL_ID = "mistralai/Voxtral-Small-24B-2507"
MINI_MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"
MODEL_REPOS = [SMALL_MODEL_ID, MINI_MODEL_ID]


@app.function(image=image, volumes={CACHE_PATH: hf_cache}, secrets=hf_secrets, timeout=7200, retries=0)
def prefetch():
    prefetch_repos(MODEL_REPOS)


def _build_generation_kwargs(meta: dict) -> dict:
    return apply_benchmark_sampling(
        meta,
        {"max_new_tokens": 512, "do_sample": False},
    )


def _run_inference(model, processor, native_messages: list[dict], meta: dict) -> str:
    import torch

    inputs = processor.apply_chat_template(native_messages)
    inputs = inputs.to(model.device, dtype=torch.bfloat16)
    input_ids = inputs.input_ids if hasattr(inputs, "input_ids") else inputs["input_ids"]
    gen_kwargs = _build_generation_kwargs(meta)
    outputs = model.generate(**inputs, **gen_kwargs)
    return processor.batch_decode(
        outputs[:, input_ids.shape[1]:], skip_special_tokens=True,
    )[0]


@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={CACHE_PATH: hf_cache},
    secrets=hf_secrets,
    timeout=600,
    scaledown_window=300,
    retries=0,
)
class VoxtralSmall24BServer:

    @modal.enter()
    @safe_enter
    def load_model(self):
        import torch
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        revision = hf_revision(SMALL_MODEL_ID)
        self.processor = AutoProcessor.from_pretrained(SMALL_MODEL_ID, revision=revision)
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            SMALL_MODEL_ID,
            revision=revision,
            device_map="auto", torch_dtype=torch.bfloat16,
        ).eval()
        hf_cache.commit()

    @modal.method()
    def generate(self, item: dict):
        messages, audio_items, meta = decode_messages_payload(item)
        prompt = flatten_messages_to_text(messages, include_roles=True)
        err = init_error_response(self, prompt, include_traceback=bool(meta.get("_debug")))
        if err:
            return err
        with materialize_audio_files(audio_items) as audio_paths:
            native_messages = render_messages(
                messages,
                audio_paths,
                build_text_part=lambda text, _role: (
                    {"type": "text", "text": text} if str(text).strip() else None
                ),
                build_audio_part=lambda audio_path, _audio_index, _role: {
                    "type": "audio",
                    "path": audio_path,
                },
            )
            response = _run_inference(self.model, self.processor, native_messages, meta)
        return {
            "prompt": prompt,
            "response": response,
            "submission_payload": {
                "provider": "voxtral-native",
                "transport": "transformers-local-generate",
                "native_messages": native_messages,
                "generation_kwargs": _build_generation_kwargs(meta),
            },
            "submission_trace": {
                "exact_submission_available": True,
                "builder": "VoxtralSmall24BServer.generate",
            },
        }


@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={CACHE_PATH: hf_cache},
    secrets=hf_secrets,
    timeout=600,
    scaledown_window=300,
    retries=0,
)
class VoxtralMini3BServer:

    @modal.enter()
    @safe_enter
    def load_model(self):
        import torch
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        revision = hf_revision(MINI_MODEL_ID)
        self.processor = AutoProcessor.from_pretrained(MINI_MODEL_ID, revision=revision)
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            MINI_MODEL_ID,
            revision=revision,
            device_map="auto", torch_dtype=torch.bfloat16,
        ).eval()
        hf_cache.commit()

    @modal.method()
    def generate(self, item: dict):
        messages, audio_items, meta = decode_messages_payload(item)
        prompt = flatten_messages_to_text(messages, include_roles=True)
        err = init_error_response(self, prompt, include_traceback=bool(meta.get("_debug")))
        if err:
            return err
        with materialize_audio_files(audio_items) as audio_paths:
            native_messages = render_messages(
                messages,
                audio_paths,
                build_text_part=lambda text, _role: (
                    {"type": "text", "text": text} if str(text).strip() else None
                ),
                build_audio_part=lambda audio_path, _audio_index, _role: {
                    "type": "audio",
                    "path": audio_path,
                },
            )
            response = _run_inference(self.model, self.processor, native_messages, meta)
        return {
            "prompt": prompt,
            "response": response,
            "submission_payload": {
                "provider": "voxtral-native",
                "transport": "transformers-local-generate",
                "native_messages": native_messages,
                "generation_kwargs": _build_generation_kwargs(meta),
            },
            "submission_trace": {
                "exact_submission_available": True,
                "builder": "VoxtralMini3BServer.generate",
            },
        }


SMOKE_TEST_VARIANTS = {
    "small-24b": (VoxtralSmall24BServer, "generate"),
    "mini-3b": (VoxtralMini3BServer, "generate"),
}


@app.local_entrypoint()
def smoke_test(
    audio_path: str,
    variant: str = "mini-3b",
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
