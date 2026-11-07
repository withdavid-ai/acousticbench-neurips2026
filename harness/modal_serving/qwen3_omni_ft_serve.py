"""Modal serving app for LoRA-finetuned Qwen3-Omni checkpoints.

Deploy:  modal deploy --env=acousticbench harness/modal_serving/qwen3_omni_ft_serve.py

Serves base weights + a PEFT adapter, rather than a merged checkpoint.

The Kimi side merges each adapter into a standalone checkpoint because its
serving class expects one. Doing the same here would mean materialising roughly
60GB per checkpoint; at two arms x four tasks that is ~480GB of volume to write,
store and read back, to encode ~50MB of actual difference. Loading the base once
and attaching an adapter keeps each checkpoint at adapter size and makes
switching between them cheap.

`adapter_path` is a `modal.parameter`, so each distinct value gets its own
instance and the caller picks the checkpoint per request batch -- the same
contract `KimiAudioFtServer` exposes through `model_path`.

Everything about prompt handling is deliberately identical to
`qwen3_omni_serve.py`: same `render_messages` call, same interleaving, same
generation kwargs. The only difference between STOCK and a trained arm must be
the weights.
"""

import modal
from common import (
    HF_RUNTIME_ENV,
    apply_benchmark_sampling,
    decode_messages_payload,
    flatten_messages_to_text,
    hf_cache,
    hf_secrets,
    init_error_response,
    modern_image,
    materialize_audio_files,
    render_messages,
    safe_enter,
)

app = modal.App("audiobench-qwen3-omni-ft")

# See qwen3_omni_serve.py: mounting the HF cache on /root/.cache fails because
# Modal's pip_install runs uv and leaves /root/.cache/uv behind.
QWEN_CACHE_PATH = "/hf-cache"
QWEN_HF_HOME = f"{QWEN_CACHE_PATH}/huggingface"
QWEN_HF_ENV = {
    **HF_RUNTIME_ENV,
    "HF_HOME": QWEN_HF_HOME,
    "HF_HUB_CACHE": f"{QWEN_HF_HOME}/hub",
    "HUGGINGFACE_HUB_CACHE": f"{QWEN_HF_HOME}/hub",
    "TRANSFORMERS_CACHE": f"{QWEN_HF_HOME}/hub",
}

DATA_MOUNT = "/data"
MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

image = (
    modern_image()
    .pip_install(
        "qwen-omni-utils[decord]==0.0.9",
        "torchvision==0.26.0",
        "peft",
    )
    .env(QWEN_HF_ENV)
)

data_volume = modal.Volume.from_name(
    "acousticbench-swift-data", create_if_missing=True
)


@app.cls(
    gpu="H200",
    image=image,
    volumes={QWEN_CACHE_PATH: hf_cache, DATA_MOUNT: data_volume},
    secrets=hf_secrets,
    timeout=1800,
    scaledown_window=300,
    retries=0,
)
class Qwen3OmniFtServer:
    # Path to an ms-swift LoRA adapter directory on the data volume, e.g.
    # /data/runs/rad-baseline-seed1-qwen3omni/checkpoint-1000
    adapter_path: str = modal.parameter(default="")

    @staticmethod
    def _resolve_adapter(requested: str) -> str:
        """Find the adapter directory, tolerating ms-swift's version subdirectory.

        `swift sft --output_dir /data/runs/<run>` does not write checkpoints
        directly there; it inserts a timestamped version directory, e.g.
        `/data/runs/<run>/v0-20260726-104521/checkpoint-1000`. The timestamp is
        unknowable to the caller, so accept the logical path and resolve it here.
        """
        from pathlib import Path

        candidate = Path(requested)
        if candidate.is_dir():
            return str(candidate)

        # <run>/checkpoint-N -> <run>/*/checkpoint-N
        run_dir, _, leaf = requested.rpartition("/")
        matches = sorted(Path(run_dir).glob(f"*/{leaf}")) if run_dir else []
        if len(matches) == 1:
            return str(matches[0])
        if len(matches) > 1:
            # Several version directories: take the newest by name, which sorts
            # by the timestamp swift embeds.
            return str(matches[-1])
        raise FileNotFoundError(
            f"adapter not found on volume: {requested} "
            f"(also tried {run_dir}/*/{leaf})"
        )

    @modal.enter()
    @safe_enter
    def load_model(self):
        from pathlib import Path

        import torch
        from peft import PeftModel
        from transformers import (
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
        )

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID)
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).eval()

        # Text-output MCQA never uses the Talker; dropping it frees ~10GB and
        # keeps it out of the device_map.
        if hasattr(model, "disable_talker"):
            model.disable_talker()

        adapter = str(self.adapter_path or "").strip()
        if not adapter:
            raise ValueError(
                "Qwen3OmniFtServer requires adapter_path; instantiate as "
                "Qwen3OmniFtServer(adapter_path='/data/runs/<run>/checkpoint-N')"
            )
        data_volume.reload()
        adapter = self._resolve_adapter(adapter)

        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        model.eval()
        self.model = model
        self.loaded_adapter = adapter
        print(f"[Qwen3OmniFtServer] loaded adapter {adapter}", flush=True)
        hf_cache.commit()

    def _generation_kwargs(self, meta: dict) -> dict:
        # thinker_* keys, not bare max_new_tokens -- see qwen3_omni_serve.py.
        meta_override = meta.get("_thinker_max_new_tokens")
        try:
            thinker_tokens = int(meta_override) if meta_override is not None else None
        except (TypeError, ValueError):
            thinker_tokens = None

        return apply_benchmark_sampling(
            meta,
            {
                "thinker_max_new_tokens": (
                    thinker_tokens if thinker_tokens is not None else 256
                ),
                "thinker_do_sample": False,
            },
            do_sample_key="thinker_do_sample",
            temperature_key=None,
            top_p_key=None,
        )

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
                native_messages, tokenize=False, add_generation_prompt=True
            )
            audios, images, videos = process_mm_info(
                native_messages, use_audio_in_video=True
            )

            inputs = self.processor(
                text=chat_text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True, use_audio_in_video=True,
            ).to(self.model.device)

            dtype = self.model.dtype
            for key, value in list(inputs.items()):
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    inputs[key] = value.to(dtype=dtype)

            generation_kwargs = self._generation_kwargs(meta)
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    use_audio_in_video=True,
                    return_audio=False,
                    **generation_kwargs,
                )
            generated_ids = generated_ids[:, inputs.input_ids.size(1):]
            response = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]

            return {
                "prompt": prompt,
                "response": response,
                "submission_payload": {
                    "provider": "qwen3-omni-native-ft",
                    "transport": "transformers-local-generate",
                    "adapter_path": self.loaded_adapter,
                    "native_messages": native_messages,
                    "chat_text": chat_text,
                    "generation_kwargs": generation_kwargs,
                },
                "submission_trace": {
                    "exact_submission_available": True,
                    "builder": "Qwen3OmniFtServer.generate",
                },
            }
