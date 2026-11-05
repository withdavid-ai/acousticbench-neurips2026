"""Modal serving app for Qwen3-Omni (30B-A3B MoE).

Deploy:  modal deploy [--env <modal-env>] modal_serving/qwen3_omni_serve.py

Performance notes
-----------------
- `Qwen3OmniMoeForConditionalGeneration.generate` routes length limits through
  **thinker_max_new_tokens** (not bare ``max_new_tokens``). Passing only the latter
  leaves the thinker at its default **1024** decode steps — a major slowdown.
- We use ``attn_implementation=\"sdpa\"`` (PyTorch fused attention; no flash-attn
  compile in the image). For more speed + lower memory, the Qwen README suggests
  ``flash_attention_2`` after ``pip install flash-attn``.
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

app = modal.App("audiobench-qwen3-omni")
image = modern_image().pip_install(
    "qwen-omni-utils[decord]==0.0.9",
    "torchvision==0.26.0",
)

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_REVISION = hf_revision(MODEL_ID)
MODEL_REPOS = [MODEL_ID]


@app.function(image=image, volumes={CACHE_PATH: hf_cache}, secrets=hf_secrets, timeout=7200, retries=0)
def prefetch():
    prefetch_repos(MODEL_REPOS)


@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={CACHE_PATH: hf_cache},
    secrets=hf_secrets,
    timeout=1800,
    scaledown_window=300,
    retries=0,
)
class Qwen3OmniServer:

    @modal.enter()
    @safe_enter
    def load_model(self):
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
        )
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).eval()
        hf_cache.commit()

    def _generation_kwargs(self, meta: dict) -> dict:
        # Must use thinker_* keys: bare max_new_tokens is NOT applied to the thinker
        # (generate() seeds thinker_kwargs with max_new_tokens=thinker_max_new_tokens
        # default 1024, and shared kwargs skip keys already present).
        #
        # Per-call override: set _thinker_max_new_tokens in meta (passed from model adapter
        # via AUDIOBENCH_QWEN3_OMNI_THINKER_MAX_TOKENS env var).
        meta_override = meta.get("_thinker_max_new_tokens")
        if meta_override is not None:
            try:
                thinker_tokens = int(meta_override)
            except (TypeError, ValueError):
                thinker_tokens = None
        else:
            thinker_tokens = None

        return apply_benchmark_sampling(meta, {
            "thinker_max_new_tokens": thinker_tokens if thinker_tokens is not None else 256,
            "thinker_do_sample": False,
        }, do_sample_key="thinker_do_sample", temperature_key=None, top_p_key=None)

    @modal.method()
    def generate(self, item: dict):
        import torch
        from qwen_omni_utils import process_mm_info

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
                    "audio": audio_path,
                },
            )
            chat_text = self.processor.apply_chat_template(
                native_messages, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(native_messages, use_audio_in_video=True)

            inputs = self.processor(
                text=chat_text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=True,
            ).to(self.model.device)

            # Cast floating inputs once (avoid second .to(dtype) walk on BatchEncoding).
            dtype = self.model.dtype
            for key, value in list(inputs.items()):
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    inputs[key] = value.to(dtype=dtype)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    return_audio=False,
                    **self._generation_kwargs(meta),
                )
            generated_ids = generated_ids[:, inputs.input_ids.size(1):]
            response = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True,
            )[0]
            generation_kwargs = self._generation_kwargs(meta)
            return {
                "prompt": prompt,
                "response": response,
                "submission_payload": {
                    "provider": "qwen3-omni-native",
                    "transport": "transformers-local-generate",
                    "native_messages": native_messages,
                    "chat_text": chat_text,
                    "generation_kwargs": generation_kwargs,
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
                    "builder": "Qwen3OmniServer.generate",
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
        Qwen3OmniServer,
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
