"""Modal serving app for Microsoft Phi-4-multimodal-instruct (5.6B).

Deploy:  modal deploy [--env <modal-env>] modal_serving/phi4mm_serve.py

Uses the head commit of refs/pr/78 of the model repo (pinned in HF_REVISIONS) which includes the
``prepare_inputs_for_generation`` placeholder needed by peft.
Audio is fed to the processor as ``(ndarray, sample_rate)`` tuples.
"""


import modal
from common import (
    CACHE_PATH,
    apply_benchmark_sampling,
    base_image,
    coerce_text_response,
    decode_messages_payload,
    flatten_messages_to_text,
    hf_cache,
    hf_revision,
    hf_secrets,
    init_error_response,
    materialize_audio_files,
    prefetch_repo,
    run_smoke_test,
    safe_enter,
)

app = modal.App("audiobench-phi4mm")
image = base_image().pip_install("backoff==2.2.1", "peft==0.19.1", "torchvision==0.19.1")

MODEL_ID = "microsoft/Phi-4-multimodal-instruct"
MODEL_REVISION = hf_revision(MODEL_ID)  # head of refs/pr/78


@app.function(image=image, volumes={CACHE_PATH: hf_cache}, secrets=hf_secrets, timeout=7200, retries=0)
def prefetch():
    prefetch_repo(MODEL_ID, revision=MODEL_REVISION)
    hf_cache.commit()


@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={CACHE_PATH: hf_cache},
    secrets=hf_secrets,
    timeout=600,
    scaledown_window=300,
    retries=0,
)
class Phi4MMServer:

    @modal.enter()
    @safe_enter
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation="sdpa",
        ).eval()
        hf_cache.commit()

    def _build_prompt(self, rendered_text: str) -> str:
        """Build a Phi-4 prompt with chat markers around the rendered conversation."""
        user_prompt = "<|user|>"
        assistant_prompt = "<|assistant|>"
        prompt_suffix = "<|end|>"
        if rendered_text:
            return f"{user_prompt}\n{rendered_text}{prompt_suffix}\n{assistant_prompt}\n"
        return f"{user_prompt}\n{prompt_suffix}\n{assistant_prompt}\n"

    @modal.method()
    def generate(self, item: dict):
        import soundfile as sf

        messages, audio_items, meta = decode_messages_payload(item)
        rendered_text = flatten_messages_to_text(
            messages,
            audio_placeholder=lambda idx: f"<|audio_{idx + 1}|>",
            include_roles=True,
        )
        prompt = self._build_prompt(rendered_text)
        err = init_error_response(self, prompt, include_traceback=bool(meta.get("_debug")))
        if err:
            return err

        with materialize_audio_files(audio_items) as audio_paths:
            audios = []
            for path in audio_paths:
                audio_data, sr = sf.read(path)
                audios.append((audio_data, sr))

            gen_kwargs = apply_benchmark_sampling(
                meta,
                {"max_new_tokens": 64, "do_sample": False},
            )

            processor_kwargs = {"text": prompt, "return_tensors": "pt"}
            if audios:
                processor_kwargs["audios"] = audios
            inputs = self.processor(**processor_kwargs).to(self.model.device)

            generated_ids = self.model.generate(**inputs, **gen_kwargs)
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
            response = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0]

        return {
            "prompt": prompt,
            "response": coerce_text_response(response),
            "submission_payload": {
                "provider": "phi4mm-native",
                "transport": "transformers-local-generate",
                "prompt_text": prompt,
                "generation_kwargs": gen_kwargs,
                "text_only_empty_audio_features_patched": not bool(audios),
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
                "builder": "Phi4MMServer.generate",
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
        Phi4MMServer,
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
