"""Modal serving app for Step-Audio-2-mini.

Deploy:  modal deploy [--env <modal-env>] modal_serving/step_audio2_serve.py

Inference follows the official StepAudio2 wrapper from
https://github.com/stepfun-ai/Step-Audio2/blob/main/stepaudio2.py:
  1. Audio → 128-bin log-mel spectrogram (16 kHz, 25 s chunks)
  2. Chat template with <|BOT|>role markers, audio placeholder tokens
  3. model.generate(input_ids=..., wavs=mels, wav_lens=..., ...)
  4. Decode text-only tokens (< 151688)
"""

import modal
from common import (
    apply_benchmark_sampling,
    audio_items_to_ndarrays,
    base_image,
    coerce_text_response,
    decode_messages_payload,
    flatten_messages_to_text,
    hf_cache,
    hf_revision,
    hf_secrets,
    init_error_response,
    prefetch_repos,
    run_smoke_test,
    safe_enter,
)

app = modal.App("audiobench-step-audio2")
HF_CACHE_PATH = "/hf-cache"

image = (
    base_image()
    .pip_install(
        "torch==2.5.1",
        "torchaudio==2.5.1",
        "transformers==4.49.0",
    )
    .env({
        "HF_HOME": f"{HF_CACHE_PATH}/huggingface",
        "HF_HUB_CACHE": f"{HF_CACHE_PATH}/huggingface/hub",
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/huggingface/hub",
    })
    .add_local_python_source("serving_utils", copy=True)
)

MODEL_REPOS = [
    "stepfun-ai/Step-Audio-2-mini",
    "stepfun-ai/Step-Audio-2-mini-Base",
]


@app.function(image=image, volumes={HF_CACHE_PATH: hf_cache}, secrets=hf_secrets, timeout=7200, retries=0)
def prefetch():
    prefetch_repos(MODEL_REPOS)


def _log_mel_spectrogram(audio, n_mels=128, padding=479):
    """128-bin log-mel matching upstream Step-Audio-2."""
    import librosa
    import torch
    import torch.nn.functional as F

    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(400).to(audio.device)
    stft = torch.stft(audio, 400, 160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2
    filters = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels))
    log_spec = torch.clamp(filters @ magnitudes, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    return (log_spec + 4.0) / 4.0


def _compute_token_num(max_feature_len):
    """How many LLM placeholder tokens one mel chunk occupies."""
    max_feature_len -= 2
    encoder_out = (max_feature_len + 1) // 2 // 2
    return (encoder_out + 2 * 1 - 3) // 2 + 1


@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={HF_CACHE_PATH: hf_cache},
    secrets=hf_secrets,
    timeout=600,
    scaledown_window=300,
    retries=0,
)
class StepAudio2MiniServer:

    @modal.enter()
    @safe_enter
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = "stepfun-ai/Step-Audio-2-mini"
        revision = hf_revision(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, revision=revision, trust_remote_code=True, padding_side="right")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, revision=revision, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).cuda().eval()
        self.tokenizer.eos_token = "<|EOT|>"
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids("<|EOT|>")
        self.model.config.eos_token_id = self.eos_token_id
        hf_cache.commit()

    def _build_inputs(self, audio_datas, messages):
        """Build tokenized input_ids + mel tensors following upstream chat template."""
        import torch
        from torch.nn.utils.rnn import pad_sequence

        mels = []
        audio_placeholders: list[str] = []
        for clip_idx, audio_np in enumerate(audio_datas):
            placeholder = ""
            audio_tensor = torch.from_numpy(audio_np).float()
            for i in range(0, audio_tensor.shape[0], 16000 * 25):
                chunk = audio_tensor[i:i + 16000 * 25]
                mel = _log_mel_spectrogram(chunk)
                mels.append(mel)
                audio_tokens = "<audio_patch>" * _compute_token_num(mel.shape[1])
                placeholder += f"<audio_start>{audio_tokens}<audio_end>"
            audio_placeholders.append(placeholder)

        rendered_prompt = flatten_messages_to_text(
            messages,
            audio_placeholder=lambda idx: audio_placeholders[idx],
            include_roles=True,
        )

        parts = [
            f"<|BOT|>human\n{rendered_prompt}<|EOT|>",
            "<|BOT|>assistant\n",
        ]
        token_parts = [
            self.tokenizer(p, return_tensors="pt", padding=True)["input_ids"]
            for p in parts
        ]
        input_ids = torch.cat(token_parts, dim=-1).cuda()
        attention_mask = torch.ones_like(input_ids)

        if mels:
            lengths = torch.tensor(
                [m.shape[1] - 2 for m in mels], dtype=torch.int32)
            padded = pad_sequence(
                [m.t() for m in mels], batch_first=True, padding_value=0
            ).transpose(1, 2).cuda()
            return input_ids, attention_mask, padded, lengths.cuda()
        return input_ids, attention_mask, None, None

    @modal.method()
    def generate(self, item: dict):
        from transformers import GenerationConfig

        messages, audio_items, meta = decode_messages_payload(item)
        audio_datas = audio_items_to_ndarrays(audio_items, sr=16000) if audio_items else []
        prompt = flatten_messages_to_text(messages, include_roles=True)
        err = init_error_response(self, prompt, include_traceback=bool(meta.get("_debug")))
        if err:
            return err

        input_ids, attention_mask, wavs, wav_lens = self._build_inputs(audio_datas, messages)

        gen_kwargs = apply_benchmark_sampling(
            meta,
            {
                "max_new_tokens": 512,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.eos_token_id,
                "do_sample": False,
            },
        )
        gen_cfg = GenerationConfig(**gen_kwargs)
        outputs = self.model.generate(
            input_ids=input_ids,
            wavs=wavs,
            wav_lens=wav_lens,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
            tokenizer=self.tokenizer,
        )
        output_tokens = outputs[0, input_ids.shape[-1]:-1].tolist()
        text_tokens = [t for t in output_tokens if t < 151688]
        response = self.tokenizer.decode(text_tokens)

        return {
            "prompt": prompt,
            "response": coerce_text_response(response),
            "submission_payload": {
                "provider": "step-audio2-native",
                "transport": "transformers-local-generate",
                "prompt_text": prompt,
                "audio_clip_count": len(audio_datas),
                "generation_kwargs": gen_kwargs,
            },
            "submission_trace": {
                "exact_submission_available": True,
                "builder": "StepAudio2MiniServer.generate",
            },
        }


# ---------------------------------------------------------------------------
# Step-Audio-2-mini-Base  (no role markers, default EOS)
# ---------------------------------------------------------------------------
@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={HF_CACHE_PATH: hf_cache},
    secrets=hf_secrets,
    timeout=600,
    scaledown_window=300,
    retries=0,
)
class StepAudio2MiniBaseServer:

    @modal.enter()
    @safe_enter
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = "stepfun-ai/Step-Audio-2-mini-Base"
        revision = hf_revision(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, revision=revision, trust_remote_code=True, padding_side="right")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, revision=revision, trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).cuda().eval()
        self.eos_token_id = self.tokenizer.eos_token_id
        hf_cache.commit()

    def _build_inputs(self, audio_datas, messages):
        """Base model: flat template with no role markers."""
        import torch
        from torch.nn.utils.rnn import pad_sequence

        mels = []
        audio_placeholders: list[str] = []
        for clip_idx, audio_np in enumerate(audio_datas):
            placeholder = ""
            audio_tensor = torch.from_numpy(audio_np).float()
            for i in range(0, audio_tensor.shape[0], 16000 * 25):
                chunk = audio_tensor[i:i + 16000 * 25]
                mel = _log_mel_spectrogram(chunk)
                mels.append(mel)
                audio_tokens = "<audio_patch>" * _compute_token_num(mel.shape[1])
                placeholder += f"<audio_start>{audio_tokens}<audio_end>"
            audio_placeholders.append(placeholder)

        text_input = flatten_messages_to_text(
            messages,
            audio_placeholder=lambda idx: audio_placeholders[idx],
            include_roles=True,
        )
        input_ids = self.tokenizer(
            text_input, return_tensors="pt", padding=True)["input_ids"].cuda()
        attention_mask = torch.ones_like(input_ids)

        if mels:
            lengths = torch.tensor(
                [m.shape[1] - 2 for m in mels], dtype=torch.int32)
            padded = pad_sequence(
                [m.t() for m in mels], batch_first=True, padding_value=0
            ).transpose(1, 2).cuda()
            return input_ids, attention_mask, padded, lengths.cuda()
        return input_ids, attention_mask, None, None

    @modal.method()
    def generate(self, item: dict):
        from transformers import GenerationConfig

        messages, audio_items, meta = decode_messages_payload(item)
        audio_datas = audio_items_to_ndarrays(audio_items, sr=16000) if audio_items else []
        prompt = flatten_messages_to_text(messages, include_roles=True)
        err = init_error_response(self, prompt, include_traceback=bool(meta.get("_debug")))
        if err:
            return err

        input_ids, attention_mask, wavs, wav_lens = self._build_inputs(audio_datas, messages)

        gen_kwargs = apply_benchmark_sampling(
            meta,
            {
                "max_new_tokens": 512,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.eos_token_id,
                "do_sample": False,
            },
        )
        gen_cfg = GenerationConfig(**gen_kwargs)
        outputs = self.model.generate(
            input_ids=input_ids,
            wavs=wavs,
            wav_lens=wav_lens,
            attention_mask=attention_mask,
            generation_config=gen_cfg,
            tokenizer=self.tokenizer,
        )
        output_tokens = outputs[0, input_ids.shape[-1]:-1].tolist()
        text_tokens = [t for t in output_tokens if t < 151688]
        response = self.tokenizer.decode(text_tokens)

        return {
            "prompt": prompt,
            "response": coerce_text_response(response),
            "submission_payload": {
                "provider": "step-audio2-native",
                "transport": "transformers-local-generate",
                "prompt_text": prompt,
                "audio_clip_count": len(audio_datas),
                "generation_kwargs": gen_kwargs,
            },
            "submission_trace": {
                "exact_submission_available": True,
                "builder": "StepAudio2MiniBaseServer.generate",
            },
        }


SMOKE_TEST_VARIANTS = {
    "mini": (StepAudio2MiniServer, "generate"),
    "base": (StepAudio2MiniBaseServer, "generate"),
}


@app.local_entrypoint()
def smoke_test(
    audio_path: str,
    variant: str = "mini",
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
