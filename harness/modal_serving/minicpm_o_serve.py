"""Modal serving app for MiniCPM-o 4.5.

Deploy:  modal deploy [--env <modal-env>] modal_serving/minicpm_o_serve.py
"""

import threading
import types

import modal
from common import (
    CACHE_PATH,
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
    run_smoke_test,
    safe_enter,
)

app = modal.App("audiobench-minicpm-o")
image = base_image().pip_install(
    # minicpmo-utils 1.0.6 pins Pillow==10.4.0, so that is what this layer
    # resolved to while Pillow was unpinned.
    "Pillow==10.4.0",
    "torch==2.8.0",
    "torchaudio==2.8.0",
    "transformers==4.51.0",
    "minicpmo-utils[all]==1.0.6",
)

MODEL_ID = "openbmb/MiniCPM-o-4_5"
MODEL_REVISION = hf_revision(MODEL_ID)
MODEL_REPOS = [MODEL_ID]


def _bound_index(value) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def _align_audio_segment(embs, target_len: int):
    import torch
    import torch.nn.functional as F

    current_len = int(embs.shape[0])
    if current_len == target_len:
        return embs

    if target_len <= 0:
        return embs[:0]

    if current_len == 0:
        return torch.zeros((target_len, embs.shape[1]), dtype=embs.dtype, device=embs.device)

    delta = target_len - current_len
    if abs(delta) <= 2:
        if delta < 0:
            return embs[:target_len]
        pad = embs[-1:, :].expand(delta, -1)
        return torch.cat([embs, pad], dim=0)

    resized = F.interpolate(
        embs.transpose(0, 1).unsqueeze(0),
        size=target_len,
        mode="linear",
        align_corners=False,
    ).squeeze(0).transpose(0, 1)
    print(
        f"[MiniCPM patch] resampled audio embeddings from {current_len} to {target_len} tokens",
        flush=True,
    )
    return resized


def _patch_get_omni_embedding(model):
    original_get_omni_embedding = model.get_omni_embedding

    def patched_get_omni_embedding(self, data, input_embeddings, chunk_length=-1, stream_input=False):
        if stream_input:
            return original_get_omni_embedding(
                data,
                input_embeddings,
                chunk_length=chunk_length,
                stream_input=stream_input,
            )

        audio_embeddings = self.get_audio_embedding(data, chunk_length)
        bs = len(input_embeddings)
        if len(data.get("audio_features", [])) > 0:
            assert len(audio_embeddings) == len(input_embeddings)

            if len(audio_embeddings) > 0:
                audio_bounds = data["audio_bounds"]
                for batch_idx in range(bs):
                    for embs, bound in zip(audio_embeddings[batch_idx], audio_bounds[batch_idx]):
                        start = _bound_index(bound[0])
                        end = _bound_index(bound[1])
                        fixed_embs = _align_audio_segment(embs, end - start)
                        if fixed_embs.shape[0] != embs.shape[0]:
                            print(
                                f"[MiniCPM patch] adjusted audio embeddings from "
                                f"{embs.shape[0]} to {fixed_embs.shape[0]} tokens",
                                flush=True,
                            )
                        input_embeddings[batch_idx, start:end] = fixed_embs.to(
                            device=input_embeddings.device,
                            dtype=input_embeddings.dtype,
                        )
        elif self.training:
            for _ in range(bs):
                input_embeddings += audio_embeddings[0].mean() * 0

        return input_embeddings

    model.get_omni_embedding = types.MethodType(patched_get_omni_embedding, model)
    model.config.stream_input = False


@app.function(image=image, volumes={CACHE_PATH: hf_cache}, secrets=hf_secrets, timeout=7200, retries=0)
def prefetch():
    prefetch_repos(MODEL_REPOS)


@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={CACHE_PATH: hf_cache},
    secrets=hf_secrets,
    timeout=600,
    scaledown_window=300,
    retries=0,
)
class MiniCPMo45Server:

    @modal.enter()
    @safe_enter
    def load_model(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model = AutoModel.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map="auto",
        ).eval()
        _patch_get_omni_embedding(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
        # MiniCPM chat maintains mutable KV cache/session state on the model.
        # Serialize calls through one model instance to avoid cross-request bleed.
        self._generate_lock = threading.Lock()
        hf_cache.commit()

    @staticmethod
    def _pad_audio_tail(audio_data, sr: int = 16000, pad_s: float = 0.0):
        import numpy as np

        if pad_s <= 0:
            return audio_data
        pad_len = int(sr * pad_s)
        if pad_len <= 0:
            return audio_data
        return np.concatenate([audio_data, np.zeros(pad_len, dtype=audio_data.dtype)])

    @staticmethod
    def _is_shape_mismatch(exc: Exception) -> bool:
        message = str(exc)
        markers = (
            "expanded size of the tensor",
            "must match the existing size",
            "Sizes of tensors must match",
            "audio/token mismatch",
        )
        return any(marker in message for marker in markers)

    def _chat_once(self, prompt: str, audio_data, chat_kwargs: dict):
        if hasattr(self.model, "reset_session"):
            self.model.reset_session(reset_token2wav_cache=False)
        msgs = [{"role": "user", "content": [prompt, audio_data]}]
        return self.model.chat(
            msgs=msgs,
            tokenizer=self.tokenizer,
            **chat_kwargs,
        )

    def _chat_messages_once(self, messages, chat_kwargs: dict):
        if hasattr(self.model, "reset_session"):
            self.model.reset_session(reset_token2wav_cache=False)
        return self.model.chat(
            msgs=messages,
            tokenizer=self.tokenizer,
            **chat_kwargs,
        )

    @modal.method()
    def generate(self, item: dict):
        import numpy as np

        messages, audio_items, meta = decode_messages_payload(item)
        prompt = flatten_messages_to_text(
            messages,
            audio_placeholder=lambda idx: f"[Audio {idx + 1}]",
            include_roles=True,
        )
        err = init_error_response(self, prompt, include_traceback=bool(meta.get("_debug")))
        if err:
            return err

        # API uses do_sample (not sampling); wrong kwarg was ignored → stochastic decoding.
        chat_kwargs = apply_benchmark_sampling(
            meta,
            {
                "max_new_tokens": 512,
                "do_sample": False,
                "generate_audio": False,
            },
        )

        if not audio_items:
            native_messages = [{"role": "user", "content": prompt}]
            with self._generate_lock:
                response = self._chat_messages_once(native_messages, chat_kwargs)
            return {
                "prompt": prompt,
                "response": response,
                "submission_payload": {
                    "provider": "minicpm-o",
                    "transport": "transformers-model-chat",
                    "prompt_text": prompt,
                    "native_messages": native_messages,
                    "generation_kwargs": chat_kwargs,
                    "text_only_empty_audio_features_patched": True,
                },
                "submission_trace": {
                    "exact_submission_available": True,
                    "builder": "MiniCPMo45Server.generate.text_only",
                },
            }

        if len(audio_items) != 1:
            raise ValueError(
                "MiniCPM-o-4.5 serving expects exactly one audio clip. "
                "Use the MiniCPM input adapter to merge multi-clip examples before serving."
            )

        combined_audio = np.ascontiguousarray(
            audio_items_to_ndarrays(audio_items, sr=16000)[0],
            dtype=np.float32,
        )

        last_error = None
        with self._generate_lock:
            for pad_s in (0.0, 0.10, 0.25, 0.50):
                candidate_audio = self._pad_audio_tail(combined_audio, sr=16000, pad_s=pad_s)
                try:
                    response = self._chat_once(prompt, candidate_audio, chat_kwargs)
                    return {
                        "prompt": prompt,
                        "response": response,
                        "submission_payload": {
                            "provider": "minicpm-o",
                            "transport": "transformers-model-chat",
                            "prompt_text": prompt,
                            "messages": [{"role": "user", "content": [prompt, {"audio_ndarray_shape": list(candidate_audio.shape)}]}],
                            "generation_kwargs": chat_kwargs,
                            "audio_shape": list(candidate_audio.shape),
                            "audio_dtype": str(candidate_audio.dtype),
                            "tail_pad_seconds": pad_s,
                        },
                        "submission_trace": {
                            "exact_submission_available": True,
                            "builder": "MiniCPMo45Server.generate",
                        },
                    }
                except Exception as exc:
                    last_error = exc
                    if not self._is_shape_mismatch(exc) or pad_s >= 0.50:
                        raise
                    print(
                        f"[MiniCPM retry] shape mismatch with tail pad {pad_s:.2f}s: {exc}",
                        flush=True,
                    )

        raise last_error


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
        MiniCPMo45Server,
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
