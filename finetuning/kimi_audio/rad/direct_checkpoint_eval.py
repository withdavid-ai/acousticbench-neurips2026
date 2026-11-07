from __future__ import annotations

import contextlib
import io
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

import jsonlines
import numpy as np
import torch
from finetuning.kimi_audio.common.run_layout import (
    append_jsonl,
    now_utc_iso,
    read_run_state,
    resolve_run_paths,
    update_run_state,
    write_json,
)
from finetuning.kimi_audio.rad.evaluation_common import (
    build_checkpoint_metrics_row,
    checkpoint_eval_metric_names,
    checkpoint_eval_metric_prefix,
    simplified_checkpoint_eval_summary,
    summarize_performance_payload,
)

DEFAULT_BENCHMARK_NAME = "EmotionRecognitionBench"
BENCHMARK_CATEGORY_BY_NAME = {
    "AcousticPerceptionBench": "acoustic-perception",
    "EmotionRecognitionBench": "emotion-recognition",
    "LanguageIdentificationBench": "language-identification",
    "SpeechQualityBench": "speech-quality",
}
EVAL_METHOD = "gpt-5.4-mini"
DEFAULT_EVAL_NAME = "emotion-heldout-gpt54mini"
MODEL_NAME = "Kimi-Audio-FT"
DEFAULT_KIMI_TOKENIZER_SOURCE = "moonshotai/Kimi-Audio-7B-Instruct"
CHECKPOINT_EVAL_PROGRESS_EVERY = 50
CHECKPOINT_EVAL_INFERENCE_MODE_LOCAL_LOOP = "local-loop"
CHECKPOINT_EVAL_INFERENCE_MODE_KIMI_GENERATE = "kimi-generate"
GENERATE_COMPARE_EVAL_SUFFIX = "-generate-compare"
CHECKPOINT_EVAL_WANDB_STEP_METRIC = "checkpoint_eval/checkpoint_step"
NONFATAL_CHECKPOINT_FINALIZE_ERRORS_ENV = (
    "AUDIOBENCH_CHECKPOINT_EVAL_NONFATAL_FINALIZE_ERRORS"
)
COMBINED_CHECKPOINT_EVAL_BENCHMARK_NAME = "AcousticBenchCheckpointEval"
COMBINED_CHECKPOINT_EVAL_CATEGORY = "checkpoint-eval"
_DEFINED_CHECKPOINT_EVAL_WANDB_RUN_IDS: set[str] = set()


def _log_checkpoint_eval(message: str, *, rank: int | None = None) -> None:
    prefix = "[checkpoint-eval]"
    if rank is not None:
        prefix += f"[rank{int(rank)}]"
    print(f"{prefix} {message}", flush=True)


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _nonfatal_checkpoint_finalize_errors() -> bool:
    return _truthy_env(NONFATAL_CHECKPOINT_FINALIZE_ERRORS_ENV)


def _has_kimi_tokenizer_assets(path: Path) -> bool:
    return (path / "tokenization_kimia.py").is_file() and (
        path / "tokenizer_config.json"
    ).is_file()


def _resolve_kimi_tokenizer_source(
    *,
    pretrained_path: str,
    model_name_or_path: str = "",
) -> str:
    from huggingface_hub import snapshot_download

    attempted: list[str] = []
    for raw_candidate in (
        pretrained_path,
        model_name_or_path,
        os.environ.get("AUDIOBENCH_MODEL_NAME_OR_PATH", ""),
        DEFAULT_KIMI_TOKENIZER_SOURCE,
    ):
        candidate = str(raw_candidate or "").strip()
        if not candidate:
            continue
        local_path = Path(candidate)
        if local_path.exists():
            if _has_kimi_tokenizer_assets(local_path):
                return str(local_path)
            attempted.append(f"{candidate} (missing tokenizer assets)")
            continue
        try:
            snapshot_path = Path(snapshot_download(candidate))
        except Exception as exc:
            attempted.append(f"{candidate} ({type(exc).__name__}: {exc})")
            continue
        if _has_kimi_tokenizer_assets(snapshot_path):
            return str(snapshot_path)
        attempted.append(f"{candidate} -> {snapshot_path} (missing tokenizer assets)")
    raise RuntimeError(
        "Could not resolve Kimi tokenizer assets for checkpoint eval. "
        f"Tried: {'; '.join(attempted) or 'no candidates'}"
    )


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with jsonlines.open(path) as reader:
        for row in reader:
            rows.append(dict(row))
    return rows


@lru_cache(maxsize=None)
def _heldout_eval_spec(
    heldout_eval_manifest: str,
    *,
    eval_tag: str | None = None,
    wandb_metric_prefix: str | None = None,
) -> dict[str, str]:
    benchmark_name = DEFAULT_BENCHMARK_NAME
    benchmark_category = BENCHMARK_CATEGORY_BY_NAME.get(
        benchmark_name, "emotion-recognition"
    )
    manifest_path = Path(str(heldout_eval_manifest or "")).expanduser()
    if manifest_path.is_file():
        with jsonlines.open(manifest_path) as reader:
            for row in reader:
                meta = dict(row.get("meta") or {})
                explicit_benchmark = str(meta.get("benchmark_name") or "").strip()
                explicit_category = str(meta.get("category") or "").strip()
                if explicit_benchmark:
                    benchmark_name = explicit_benchmark
                if explicit_category:
                    benchmark_category = explicit_category
                elif explicit_benchmark:
                    benchmark_category = BENCHMARK_CATEGORY_BY_NAME.get(
                        explicit_benchmark, benchmark_category
                    )
                break
    explicit_eval_tag = str(eval_tag or "").strip()
    eval_slug = explicit_eval_tag or benchmark_category or BENCHMARK_CATEGORY_BY_NAME.get(
        benchmark_name, "emotion-recognition"
    )
    explicit_metric_prefix = str(wandb_metric_prefix or "").strip()
    if explicit_metric_prefix:
        metric_prefix = checkpoint_eval_metric_prefix(
            wandb_metric_prefix=explicit_metric_prefix,
        )
    elif explicit_eval_tag:
        metric_prefix = checkpoint_eval_metric_prefix(eval_tag=eval_slug)
    else:
        metric_prefix = checkpoint_eval_metric_prefix()
    eval_name = f"{eval_slug}-heldout-gpt54mini"
    return {
        "eval_tag": eval_slug,
        "benchmark_name": benchmark_name,
        "benchmark_category": benchmark_category,
        "eval_name": eval_name,
        "compare_eval_name": f"{eval_name}{GENERATE_COMPARE_EVAL_SUFFIX}",
        "wandb_metric_prefix": metric_prefix,
    }


def _build_dataset_for_benchmark(benchmark_name: str):
    raise RuntimeError("In-training judge scoring is not included in this release. Use heldout_evals_json='[]' and evaluate the fixed checkpoint separately with reproduction.finetuning_lift.evaluate.")


def _whisper_feature_summary(features: list[Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"count": len(features)}
    if not features:
        return summary

    first = features[0]
    summary["first_type"] = type(first).__name__
    try:
        shape = getattr(first, "shape", None)
        if shape is not None:
            summary["first_shape"] = [int(dim) for dim in shape]
    except Exception:
        summary["first_shape"] = None

    if isinstance(first, torch.Tensor):
        summary["first_dtype"] = str(first.dtype)
        summary["first_device"] = str(first.device)
        return summary

    try:
        array = np.asarray(first)
    except Exception:
        summary["first_dtype"] = str(getattr(first, "dtype", None))
    else:
        summary["first_dtype"] = str(array.dtype)
    return summary


def _coerce_whisper_feature_to_numpy(feature: Any) -> np.ndarray:
    if isinstance(feature, torch.Tensor):
        return feature.detach().to(dtype=torch.float32).cpu().numpy()

    array = np.asarray(feature)
    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)
    return array


def _prepare_single_batch_whisper_inputs(features: list[Any]) -> list[list[np.ndarray]]:
    normalized = [_coerce_whisper_feature_to_numpy(feature) for feature in list(features or [])]
    return [normalized]


def _clip_label(index: int) -> str:
    if 0 <= index < 26:
        return chr(ord("A") + index)
    return str(index + 1)


def _resolve_content_order(msg: dict[str, Any], meta: dict[str, Any]) -> str:
    order = str(
        msg.get("content_order")
        or meta.get("content_order")
        or _infer_content_order_from_meta(meta)
    ).strip().lower()
    if order in {"prompt-first", "text-first", "question-first"}:
        return "prompt-first"
    return "audio-first"


def _infer_content_order_from_meta(meta: dict[str, Any]) -> str:
    synthetic_markers = {
        str(meta.get("category") or "").strip().lower(),
        str(meta.get("source_category") or "").strip().lower(),
        str(meta.get("benchmark_name") or "").strip(),
        str(meta.get("source_benchmark_name") or "").strip(),
    }
    if any(marker.startswith("synthetic-") for marker in synthetic_markers):
        return "prompt-first"
    if any(marker.startswith("Synthetic") for marker in synthetic_markers):
        return "prompt-first"
    return "audio-first"


def _resolve_pre_audio_texts(msg: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    raw_value = msg.get("pre_audio_texts")
    if raw_value is None:
        raw_value = meta.get("pre_audio_texts")
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        value = raw_value.strip()
        return [value] if value else []
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    return []


def _checkpoint_messages(
    msg: dict[str, Any],
    *,
    meta: dict[str, Any],
    prompt: str,
) -> list[dict[str, Any]]:
    """Build native Kimi messages, preferring an explicit held-out conversation.

    The flat ``audio`` + ``text`` manifest representation cannot preserve a
    benchmark prompt that places text on both sides of the audio.  When a
    held-out row supplies ``conversation``, retain that exact sequence while
    replacing its audio references with the resolved runtime paths.
    """

    audio_paths = list(msg.get("audio") or [])
    raw_conversation = msg.get("conversation")
    if isinstance(raw_conversation, list) and raw_conversation:
        messages: list[dict[str, Any]] = []
        audio_index = 0
        for raw_message in raw_conversation:
            if not isinstance(raw_message, dict):
                continue
            role = str(raw_message.get("role") or "user").strip() or "user"
            if role == "assistant":
                continue
            message_type = str(raw_message.get("message_type") or "").strip()
            if message_type == "text":
                content = str(raw_message.get("content") or "").strip()
                if content:
                    messages.append(
                        {"role": role, "message_type": "text", "content": content}
                    )
            elif message_type == "audio":
                if audio_index >= len(audio_paths):
                    raise ValueError(
                        "Explicit checkpoint conversation contains more audio "
                        "messages than resolved audio paths"
                    )
                messages.append(
                    {
                        "role": role,
                        "message_type": "audio",
                        "content": audio_paths[audio_index],
                    }
                )
                audio_index += 1
        if audio_index != len(audio_paths):
            raise ValueError(
                "Explicit checkpoint conversation consumed "
                f"{audio_index} of {len(audio_paths)} resolved audio paths"
            )
        if not messages:
            raise ValueError("Explicit checkpoint conversation produced no messages")
        return messages

    content_order = _resolve_content_order(msg, meta)
    pre_audio_texts = _resolve_pre_audio_texts(msg, meta)
    messages = [
        {"role": "user", "message_type": "text", "content": text}
        for text in pre_audio_texts
    ]
    if prompt and content_order == "prompt-first":
        messages.append(
            {"role": "user", "message_type": "text", "content": prompt}
        )
    multiple = len(audio_paths) > 1
    for index, audio_path in enumerate(audio_paths):
        if multiple:
            messages.append(
                {
                    "role": "user",
                    "message_type": "text",
                    "content": f"Clip {_clip_label(index)}:",
                }
            )
        messages.append(
            {"role": "user", "message_type": "audio", "content": audio_path}
        )
    if prompt and content_order != "prompt-first":
        messages.append(
            {"role": "user", "message_type": "text", "content": prompt}
        )
    return messages


def _benchmark_sampling_enabled(meta: dict[str, Any]) -> bool:
    return str(meta.get("_sampling_mode", "adapter-default")) == "benchmark-default"


def _meta_optional_float(meta: dict[str, Any], key: str) -> float | None:
    value = meta.get(key)
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def _apply_generate_path_benchmark_sampling(
    meta: dict[str, Any], sampling_params: dict[str, Any]
) -> dict[str, Any]:
    updated = dict(sampling_params)
    if not _benchmark_sampling_enabled(meta):
        return updated

    temperature = _meta_optional_float(meta, "_benchmark_temperature")
    if temperature is None:
        temperature = 1.0
    if temperature is not None:
        updated["text_temperature"] = temperature
    elif updated.get("text_temperature") == 0.0:
        updated.pop("text_temperature", None)
    return updated


def _coerce_text_response(response: Any) -> str:
    if isinstance(response, str):
        return response

    if isinstance(response, dict):
        for key in ("response", "text", "answer", "content", "output"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value
        raise TypeError(f"Unsupported response dict keys: {sorted(response)}")

    if isinstance(response, (list, tuple)):
        for value in response:
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, dict):
                try:
                    return _coerce_text_response(value)
                except TypeError:
                    continue
        raise TypeError("No text content found in response tuple/list")

    raise TypeError(f"Unsupported response type: {type(response).__name__}")


def _ensure_openai_key() -> None:
    if os.environ.get("OPENAI_API_KEY"):
        return
    raise RuntimeError(
        "Checkpoint eval requires OPENAI_API_KEY because gpt-5.4-mini judging is part "
        "of the training lifecycle now. In Modal, attach the `openai-api-key` secret."
    )


def active_wandb_run_id() -> str | None:
    try:
        import wandb
    except Exception:
        return None
    run = getattr(wandb, "run", None)
    run_id = getattr(run, "id", None)
    if run_id is None:
        return None
    return str(run_id)


def ensure_checkpoint_eval_wandb_metrics_defined(
    *,
    eval_tag: str | None = None,
    wandb_metric_prefix: str | None = None,
) -> bool:
    try:
        import wandb
    except Exception:
        return False

    run = getattr(wandb, "run", None)
    if run is None:
        return False

    run_id = getattr(run, "id", None)
    if run_id is None:
        return False
    run_id = str(run_id)
    metric_names = checkpoint_eval_metric_names(
        eval_tag=eval_tag,
        wandb_metric_prefix=wandb_metric_prefix,
    )
    defined_key = f"{run_id}:{metric_names['checkpoint_step']}"
    if defined_key in _DEFINED_CHECKPOINT_EVAL_WANDB_RUN_IDS:
        return True

    wandb.define_metric(CHECKPOINT_EVAL_WANDB_STEP_METRIC)
    wandb.define_metric(metric_names["checkpoint_step"])
    for metric_name in (
        metric_names["overall_accuracy_pct"],
        metric_names["accuracy_on_valid_pct"],
        metric_names["unparseable_or_unsure_pct"],
        metric_names["step0_phase"],
        metric_names["step0_wandb_ready"],
    ):
        wandb.define_metric(metric_name, step_metric=CHECKPOINT_EVAL_WANDB_STEP_METRIC)
    _DEFINED_CHECKPOINT_EVAL_WANDB_RUN_IDS.add(defined_key)
    return True


class LocalKimiCheckpointModel:
    def __init__(
        self,
        *,
        model,
        pretrained_path: str,
        model_name_or_path: str = "",
        sampling_params: dict[str, Any] | None = None,
    ) -> None:
        from transformers import AutoTokenizer

        from kimia_infer.models.tokenizer.glm4_tokenizer import Glm4Tokenizer
        from kimia_infer.utils.data import KimiAContent
        from kimia_infer.utils.sampler import KimiASampler
        from kimia_infer.utils.special_tokens import instantiate_extra_tokens

        self._kimi_content_cls = KimiAContent
        self._sampler_cls = KimiASampler
        self._audio_tokenizer = Glm4Tokenizer("THUDM/glm-4-voice-tokenizer").to(
            torch.cuda.current_device()
        )
        self.tokenizer_source = _resolve_kimi_tokenizer_source(
            pretrained_path=pretrained_path,
            model_name_or_path=model_name_or_path,
        )
        self._text_tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_source,
            trust_remote_code=True,
        )
        self._extra_tokens = instantiate_extra_tokens(self._text_tokenizer)
        self._model = _unwrap_model(model)
        self._whisper_model = _resolve_whisper_model(self._model)
        self._kimia_token_offset = int(self._model.config.kimia_token_offset)
        self._kimia_text_audiodelaytokens = int(
            self._model.config.kimia_mimo_audiodelaytokens
        )
        self._eod_ids = [
            self._extra_tokens.msg_end,
            self._extra_tokens.media_end,
        ]

        self._sampling_params = dict(
            sampling_params
            or {
                "audio_temperature": 0.8,
                "audio_top_k": 10,
                "text_temperature": 0.0,
                "text_top_k": 5,
                "audio_repetition_penalty": 1.0,
                "audio_repetition_window_size": 64,
                "text_repetition_penalty": 1.0,
                "text_repetition_window_size": 16,
                "max_new_tokens": 32,
            }
        )

    def _tokenize_text(self, text: str | None) -> list[int]:
        if text is None:
            return []
        return self._text_tokenizer.encode(text, bos=False, eos=False)

    def _tokenize_audio(self, audio_path: str) -> list[int]:
        wav_tokens = self._audio_tokenizer.tokenize(audio_path=audio_path)
        wav_tokens = wav_tokens + self._kimia_token_offset
        return wav_tokens.squeeze(0).cpu().numpy().tolist()

    def _extract_whisper_feat(self, audio_path: str) -> np.ndarray:
        import librosa

        wav = librosa.load(audio_path, sr=16000)[0]
        wav_tensor = torch.tensor(wav).unsqueeze(0)[:, :]
        wav_tensor = wav_tensor.to(torch.cuda.current_device())
        features = self._whisper_model.tokenize_waveform(wav_tensor)
        # `tokenize_waveform()` already returns Whisper hidden states in the eval path.
        # The vendored Kimi forward packs `[batch, time, 1280]` -> `[batch, time/4, 5120]`
        # for the adaptor; re-running Whisper on these tensors causes the recurring
        # `350 x 5120` vs `2 x 5120` shape mismatch during step-0 / save-time eval.
        return features.detach().to(dtype=torch.float32).cpu().numpy()

    def _tokenize_message(
        self,
        *,
        role: str,
        message_type: str | None,
        content: str | None,
        tokenize_role: bool,
        has_ct_token: bool,
        has_msg_end_token: bool,
        output_type: str,
    ):
        msg = self._kimi_content_cls()
        has_loss = role == "assistant"
        if tokenize_role:
            if role == "user":
                msg.audio_append(self._extra_tokens.kimia_user_msg_start)
                msg.text_append(self._extra_tokens.kimia_text_blank)
            elif role == "assistant":
                msg.audio_append(self._extra_tokens.kimia_assistant_msg_start)
                msg.text_append(self._extra_tokens.kimia_text_blank)
            else:
                raise NotImplementedError(f"Unsupported role: {role}")

        if message_type == "text":
            text_tokens = self._tokenize_text(content)
            msg.text_extend(text_tokens, has_loss)
            msg.audio_extend([self._extra_tokens.kimia_text_blank] * len(text_tokens))
            if role == "assistant":
                msg.text_append(self._extra_tokens.kimia_text_eos, has_loss)
                msg.audio_append(self._extra_tokens.kimia_text_blank)
        elif message_type == "audio":
            speech_tokens = self._tokenize_audio(str(content))
            msg.audio_append(self._extra_tokens.media_begin)
            msg.audio_extend(
                speech_tokens,
                is_continuous=True,
                audio_token_loss_mask=has_loss,
            )
            msg.audio_append(
                self._extra_tokens.media_end,
                audio_token_loss_mask=has_loss,
            )
            msg.text_extend(
                [self._extra_tokens.kimia_text_blank] * (len(speech_tokens) + 2)
            )
            if has_ct_token:
                if output_type == "text":
                    msg.audio_append(self._extra_tokens.kimia_speech_ct_id)
                else:
                    msg.audio_append(self._extra_tokens.kimia_speech_ctd_id)
                msg.text_append(self._extra_tokens.kimia_text_blank)
            msg.continuous_feature.append(self._extract_whisper_feat(str(content)))
        elif message_type is None:
            pass
        else:
            raise NotImplementedError(f"Unsupported message_type: {message_type}")

        if has_msg_end_token:
            msg.audio_append(self._extra_tokens.msg_end)
            msg.text_append(self._extra_tokens.kimia_text_blank)
        return msg

    def _build_prompt(self, messages: list[dict[str, Any]], *, output_type: str = "text"):
        chunks = []
        previous_role = None
        for idx, message in enumerate(messages):
            role = str(message["role"])
            tokenize_role = previous_role is None or role != previous_role
            next_role = (
                str(messages[idx + 1]["role"]) if idx + 1 < len(messages) else None
            )
            has_ct_token = idx == len(messages) - 1 or next_role != role
            has_msg_end_token = has_ct_token
            chunks.append(
                self._tokenize_message(
                    role=role,
                    message_type=message.get("message_type"),
                    content=message.get("content"),
                    tokenize_role=tokenize_role,
                    has_ct_token=has_ct_token,
                    has_msg_end_token=has_msg_end_token,
                    output_type=output_type,
                )
            )
            previous_role = role

        chunks.append(
            self._tokenize_message(
                role="assistant",
                message_type=None,
                content=None,
                tokenize_role=True,
                has_ct_token=False,
                has_msg_end_token=False,
                output_type=output_type,
            )
        )
        prompt = chunks[0]
        for chunk in chunks[1:]:
            prompt.merge(chunk)
        return prompt

    @torch.inference_mode()
    def _generate_loop(
        self,
        *,
        audio_input_ids: torch.Tensor,
        text_input_ids: torch.Tensor,
        is_continuous_mask: torch.Tensor,
        continuous_feature: list[list[np.ndarray]],
        output_type: str,
    ) -> tuple[list[int], list[int]]:
        sampler = self._sampler_cls(
            audio_top_k=int(self._sampling_params["audio_top_k"]),
            audio_temperature=float(self._sampling_params["audio_temperature"]),
            audio_repetition_penalty=float(
                self._sampling_params["audio_repetition_penalty"]
            ),
            audio_repetition_window_size=int(
                self._sampling_params["audio_repetition_window_size"]
            ),
            text_top_k=int(self._sampling_params["text_top_k"]),
            text_temperature=float(self._sampling_params["text_temperature"]),
            text_repetition_penalty=float(
                self._sampling_params["text_repetition_penalty"]
            ),
            text_repetition_window_size=int(
                self._sampling_params["text_repetition_window_size"]
            ),
        )

        max_new_tokens = int(self._sampling_params.get("max_new_tokens", 32))
        text_stream_finished = False
        previous_audio_tokens = torch.zeros(
            (4096,), dtype=torch.int, device=torch.cuda.current_device()
        )
        text_previous_tokens = torch.zeros(
            (4096,), dtype=torch.int, device=torch.cuda.current_device()
        )
        decoder_input_audio_ids = audio_input_ids.clone()
        decoder_input_text_ids = text_input_ids.clone()
        decoder_position_ids = (
            torch.arange(
                0,
                decoder_input_audio_ids.shape[1],
                device=torch.cuda.current_device(),
            )
            .unsqueeze(0)
            .long()
        )
        decoder_input_whisper_feature = continuous_feature
        decoder_is_continuous_mask = is_continuous_mask
        past_key_values = None
        last_position_id = decoder_input_audio_ids.shape[1] - 1
        valid_text_length = 0
        valid_audio_length = 0

        for i in range(max_new_tokens):
            # Save-hook eval reuses the live training model. The LoRA training path sets
            # `config.use_cache = False`, but autoregressive decoding still needs
            # `past_key_values`, so force cached inference here instead of inheriting the
            # training-time setting.
            model_outputs = self._model(
                input_ids=decoder_input_audio_ids,
                text_input_ids=decoder_input_text_ids,
                whisper_input_feature=decoder_input_whisper_feature,
                is_continuous_mask=decoder_is_continuous_mask,
                position_ids=decoder_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=False,
            )
            if hasattr(model_outputs, "to_tuple"):
                model_outputs = model_outputs.to_tuple()
            if not isinstance(model_outputs, (tuple, list)) or len(model_outputs) < 3:
                raise RuntimeError(
                    "Checkpoint eval generation expected cached tuple outputs, "
                    f"got {type(model_outputs).__name__} "
                    f"len={len(model_outputs) if isinstance(model_outputs, (tuple, list)) else 'n/a'}"
                )
            audio_logits, text_logits, past_key_values = model_outputs[:3]
            next_token_text = sampler.sample_text_logits(
                text_logits,
                recent_tokens=text_previous_tokens[:i] if i > 0 else None,
            )
            next_audio_token = sampler.sample_audio_logits(
                audio_logits,
                recent_tokens=previous_audio_tokens[:i] if i > 0 else None,
            )
            if text_stream_finished:
                next_token_text.fill_(self._extra_tokens.kimia_text_blank)
            elif next_token_text.item() == self._extra_tokens.kimia_text_eos:
                text_stream_finished = True
            else:
                valid_text_length += 1

            text_previous_tokens[i : i + 1] = next_token_text
            if i < self._kimia_text_audiodelaytokens:
                next_audio_token.fill_(self._extra_tokens.kimia_text_blank)
            else:
                if output_type == "text":
                    next_audio_token.fill_(self._extra_tokens.kimia_text_blank)
                else:
                    valid_audio_length += 1
            previous_audio_tokens[i : i + 1] = next_audio_token

            audio_stream_finished = next_audio_token.item() in self._eod_ids
            if (
                output_type == "text"
                and text_stream_finished
                or output_type == "both"
                and audio_stream_finished
            ):
                break

            decoder_input_audio_ids = next_audio_token.unsqueeze(1)
            decoder_input_text_ids = next_token_text.unsqueeze(1)
            decoder_position_ids = (
                torch.zeros(1, 1, device=torch.cuda.current_device())
                .fill_(last_position_id + 1)
                .long()
                .view(1, 1)
            )
            last_position_id += 1
            decoder_input_whisper_feature = None
            decoder_is_continuous_mask = None

        return_audio_tokens = (
            previous_audio_tokens[
                self._kimia_text_audiodelaytokens : valid_audio_length
                + self._kimia_text_audiodelaytokens
            ]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        return_text_tokens = (
            text_previous_tokens[:valid_text_length].detach().cpu().numpy().tolist()
        )
        return return_audio_tokens, return_text_tokens

    def _decode_text(self, text_tokens: list[int]) -> str:
        valid_text_ids: list[int] = []
        for token in text_tokens:
            if token == self._extra_tokens.kimia_text_eos:
                break
            if token < self._kimia_token_offset:
                valid_text_ids.append(token)
        return self._text_tokenizer.decode(valid_text_ids)

    def __call__(self, msg: dict[str, Any]) -> dict[str, Any]:
        prompt = _require_checkpoint_eval_prompt(
            msg.get("text", ""),
            msg.get("meta", {}) or {},
        )
        meta = dict(msg.get("meta", {}) or {})
        messages = _checkpoint_messages(msg, meta=meta, prompt=prompt)
        prompt_tensor = self._build_prompt(messages, output_type="text")
        raw_continuous_feature = list(prompt_tensor.continuous_feature or [])
        feature_summary = _whisper_feature_summary(raw_continuous_feature)
        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = (
            prompt_tensor.to_tensor()
        )
        audio_input_ids = audio_input_ids.to(torch.cuda.current_device())
        text_input_ids = text_input_ids.to(torch.cuda.current_device())
        is_continuous_mask = is_continuous_mask.to(torch.cuda.current_device())
        try:
            continuous_feature = _prepare_single_batch_whisper_inputs(
                raw_continuous_feature
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to normalize whisper_input_feature for live checkpoint eval "
                f"summary={feature_summary}: {type(exc).__name__}: {exc}"
            ) from exc
        was_training = bool(getattr(self._model, "training", False))
        self._model.eval()
        try:
            try:
                _, text_tokens = self._generate_loop(
                    audio_input_ids=audio_input_ids,
                    text_input_ids=text_input_ids,
                    is_continuous_mask=is_continuous_mask,
                    continuous_feature=continuous_feature,
                    output_type="text",
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed during live checkpoint generation "
                    f"with whisper_input_feature_summary={feature_summary}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        finally:
            if was_training:
                self._model.train()
        response = self._decode_text(text_tokens)
        return {
            "prompt": prompt,
            "response_text": response,
            "request_payload": None,
            "response_payload": None,
            "response_status": "ok",
            "response_error": None,
        }


class KimiGenerateCheckpointModel:
    def __init__(self, *, checkpoint_dir: str, model_source: str = "") -> None:
        from kimia_infer.api.kimia import KimiAudio

        checkpoint_path = Path(checkpoint_dir)
        resolved_model_source = str(checkpoint_path)
        if not (checkpoint_path / "modeling_moonshot_kimia.py").is_file():
            resolved_model_source = str(model_source).strip() or str(checkpoint_path)
        self.tokenizer_source = resolved_model_source
        self._model = KimiAudio(
            model_path=resolved_model_source,
            load_detokenizer=False,
        )
        self._sampling_params = {
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

    def __call__(self, msg: dict[str, Any]) -> dict[str, Any]:
        meta = dict(msg.get("meta") or {})
        prompt = _require_checkpoint_eval_prompt(
            msg.get("text", ""),
            meta,
        )
        messages = _checkpoint_messages(msg, meta=meta, prompt=prompt)

        sampling_params = _apply_generate_path_benchmark_sampling(
            meta,
            dict(self._sampling_params),
        )
        if str(meta.get("task", "")).upper() == "MQA":
            sampling_params["max_new_tokens"] = 32

        _, response = self._model.generate(
            messages,
            **sampling_params,
            output_type="text",
        )
        return {
            "prompt": prompt,
            "response_text": _coerce_text_response(response),
            "request_payload": None,
            "response_payload": None,
            "response_status": "ok",
            "response_error": None,
        }


def _result_row(item: dict[str, Any], trajectory: dict[str, Any]) -> dict[str, Any]:
    row = dict(item)
    row["prediction"] = trajectory.get("response_text")
    row["real_prompt"] = trajectory.get("prompt", item.get("question", ""))
    row["request_payload"] = trajectory.get("request_payload")
    row["response_payload"] = trajectory.get("response_payload")
    row["response_status"] = trajectory.get("response_status", "ok")
    row["response_error"] = trajectory.get("response_error")
    return row


def _build_checkpoint_eval_model(
    *,
    inference_mode: str,
    model,
    pretrained_path: str,
    model_name_or_path: str,
    checkpoint_dir: str,
):
    if inference_mode == CHECKPOINT_EVAL_INFERENCE_MODE_LOCAL_LOOP:
        return LocalKimiCheckpointModel(
            model=model,
            pretrained_path=pretrained_path,
            model_name_or_path=model_name_or_path,
        )
    if inference_mode == CHECKPOINT_EVAL_INFERENCE_MODE_KIMI_GENERATE:
        return KimiGenerateCheckpointModel(
            checkpoint_dir=checkpoint_dir,
            model_source=model_name_or_path,
        )
    raise ValueError(
        f"Unknown checkpoint inference mode {inference_mode!r}. Expected one of: "
        f"{CHECKPOINT_EVAL_INFERENCE_MODE_LOCAL_LOOP}, "
        f"{CHECKPOINT_EVAL_INFERENCE_MODE_KIMI_GENERATE}"
    )


def _overall_accuracy_pct(metrics_row: dict[str, Any]) -> float | None:
    value = metrics_row.get("overall_accuracy_pct")
    if value is None:
        value = metrics_row.get("accuracy_pct_total")
    if value is None:
        return None
    return float(value)


def _combine_checkpoint_eval_specs(
    eval_specs: list[dict[str, Any]],
    *,
    output_path: str | Path,
) -> dict[str, Any]:
    if not eval_specs:
        raise ValueError("Cannot combine checkpoint eval specs: no eval specs provided.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    next_index = 0
    for raw_spec in eval_specs:
        spec = dict(raw_spec)
        source_tag = str(spec.get("tag") or "").strip()
        manifest = str(spec.get("manifest") or "").strip()
        if not source_tag:
            raise ValueError(f"Combined checkpoint eval spec is missing tag: {spec!r}")
        if not manifest:
            raise ValueError(
                f"Combined checkpoint eval spec {source_tag!r} is missing manifest."
            )
        rows = _read_jsonl(manifest)
        source_counts[source_tag] = len(rows)
        for row in rows:
            combined_row = dict(row)
            meta = dict(combined_row.get("meta") or {})
            original_subset = str(combined_row.get("subset") or "default").strip() or "default"
            meta["source_tag"] = source_tag
            meta["source_manifest"] = manifest
            meta["source_index"] = combined_row.get("index")
            meta["source_subset"] = original_subset
            meta["source_benchmark_name"] = str(meta.get("benchmark_name") or "").strip() or None
            meta["source_category"] = str(meta.get("category") or "").strip() or None
            # Force a generic benchmark/category so the merged manifest evaluates through the
            # generic MQA path instead of pretending to be only the first task.
            meta["benchmark_name"] = COMBINED_CHECKPOINT_EVAL_BENCHMARK_NAME
            meta["category"] = COMBINED_CHECKPOINT_EVAL_CATEGORY
            combined_row["meta"] = meta
            combined_row["subset"] = f"{source_tag}::{original_subset}"
            combined_row["index"] = next_index
            combined_rows.append(combined_row)
            next_index += 1

    with jsonlines.open(output_path, mode="w") as writer:
        for row in combined_rows:
            writer.write(row)

    return {
        "manifest": str(output_path),
        "source_counts": source_counts,
        "total_examples": len(combined_rows),
    }


def _extract_judge_label(judge_result: Any) -> str:
    value = str(judge_result or "").strip().split("\n")[-1].strip().lower()
    if "yes" in value:
        return "yes"
    if "no" in value:
        return "no"
    return "invalid"


def _summarize_judged_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_examples = len(rows)
    correct = 0
    invalid = 0
    choice_extraction_total = 0
    choice_extraction_unsure_total = 0
    choice_extraction_refusal_total = 0
    for row in rows:
        judge_label = _extract_judge_label(row.get("judge_result"))
        if judge_label == "yes":
            correct += 1
        elif judge_label == "invalid":
            invalid += 1

        if row.get("normalized_choice_label") is not None:
            choice_extraction_total += 1
        classification = str(row.get("choice_extraction_classification") or "").strip().lower()
        if classification == "unsure":
            choice_extraction_unsure_total += 1
        elif classification == "refusal":
            choice_extraction_refusal_total += 1

    valid_predictions = max(0, total_examples - invalid)
    performance = {
        "combined": {
            "acc": round((correct / valid_predictions) * 100.0, 2)
            if valid_predictions
            else 0.0,
            "valid": valid_predictions,
            "total": total_examples,
            "correct": correct,
            "choice_extraction_total": choice_extraction_total,
            "choice_extraction_deterministic": 0,
            "choice_extraction_llm": choice_extraction_total,
            "choice_extraction_unsure": choice_extraction_unsure_total,
            "choice_extraction_refusal": choice_extraction_refusal_total,
        }
    }
    summary = summarize_performance_payload({"performance": performance})
    return {"performance": performance, **summary}


def _write_split_checkpoint_eval_outputs(
    *,
    eval_specs: list[dict[str, Any]],
    judged_rows: list[dict[str, Any]],
    shared_result_file: str,
    shared_judge_file: str,
    run_name: str,
    checkpoint_dir: str,
    checkpoint_name: str,
    checkpoint_step: int,
) -> list[dict[str, Any]]:
    rows_by_source: dict[str, list[dict[str, Any]]] = {}
    for row in judged_rows:
        meta = dict(row.get("meta") or {})
        source_tag = str(meta.get("source_tag") or "").strip()
        if not source_tag:
            continue
        rows_by_source.setdefault(source_tag, []).append(dict(row))

    run_paths = resolve_run_paths(run_name)
    per_eval_results: list[dict[str, Any]] = []
    for raw_spec in eval_specs:
        spec = dict(raw_spec)
        source_tag = str(spec.get("tag") or "").strip()
        manifest = str(spec.get("manifest") or "").strip()
        if not source_tag:
            continue
        source_rows = list(rows_by_source.get(source_tag) or [])
        if not source_rows:
            raise RuntimeError(
                f"No judged rows found for combined checkpoint eval source_tag={source_tag!r}."
            )

        eval_spec = _heldout_eval_spec(
            manifest,
            eval_tag=source_tag,
            wandb_metric_prefix=str(spec.get("wandb_metric_prefix") or "").strip() or None,
        )
        benchmark_name = str(
            ((source_rows[0].get("meta") or {}) if source_rows else {}).get("source_benchmark_name")
            or eval_spec["benchmark_name"]
        )
        question_set_version = str(
            ((source_rows[0].get("meta") or {}) if source_rows else {}).get("heldout_set_name")
            or ""
        )
        eval_repeat_dir = run_paths.eval_repeat_dir(
            eval_spec["eval_name"], int(checkpoint_step)
        )
        benchmark_root = Path(eval_repeat_dir) / MODEL_NAME / benchmark_name
        benchmark_root.mkdir(parents=True, exist_ok=True)

        perf_summary = _summarize_judged_rows(source_rows)
        performance_payload = {
            "task": "MQA",
            "dataset": benchmark_name,
            "model": MODEL_NAME,
            "date": now_utc_iso(),
            "performance": perf_summary["performance"],
            "eval_method": EVAL_METHOD,
        }
        summary_metrics = {key: value for key, value in perf_summary.items() if key != "performance"}
        perf_file = (
            benchmark_root / f"{MODEL_NAME}_{benchmark_name}_{EVAL_METHOD}_performance.json"
        )
        write_json(perf_file, performance_payload)

        summary_payload = {
            "model": MODEL_NAME,
            "model_path_override": checkpoint_dir,
            "question_set_version": question_set_version or None,
            "result_file": shared_result_file,
            "performance_file": str(perf_file),
            "judge_file": shared_judge_file,
            "performance": performance_payload,
            **summary_metrics,
        }
        combined_summary_payload = {
            "created_at": now_utc_iso(),
            "question_set_version": question_set_version or None,
            "eval_method": EVAL_METHOD,
            "models": [summary_payload],
        }
        summary_path = benchmark_root / "summary.json"
        combined_summary_path = Path(eval_repeat_dir) / "combined_summary.json"
        write_json(summary_path, summary_payload)
        write_json(combined_summary_path, combined_summary_payload)

        metrics_row = build_checkpoint_metrics_row(
            run_name=run_name,
            eval_name=eval_spec["eval_name"],
            repeat_index=checkpoint_step,
            benchmark_name=benchmark_name,
            eval_method=EVAL_METHOD,
            model_name=MODEL_NAME,
            checkpoint_dir=checkpoint_dir,
            checkpoint_name=checkpoint_name,
            checkpoint_step=checkpoint_step,
            model_path_override=checkpoint_dir,
            eval_tag=eval_spec["eval_tag"],
            wandb_metric_prefix=eval_spec["wandb_metric_prefix"],
            question_set_version=question_set_version or None,
            summary_path=str(summary_path),
            combined_summary_path=str(combined_summary_path),
            summary_payload=summary_metrics,
        )
        per_eval_results.append(
            {
                "status": "ok",
                "run_name": run_name,
                "checkpoint_name": checkpoint_name,
                "checkpoint_step": checkpoint_step,
                "result_file": shared_result_file,
                "performance_file": str(perf_file),
                "judge_file": shared_judge_file,
                "summary_path": str(summary_path),
                "combined_summary_path": str(combined_summary_path),
                "metrics_row": metrics_row,
            }
        )

    return per_eval_results


def _record_final_checkpoint_eval_result(
    *,
    run_paths,
    checkpoint_name: str,
    checkpoint_dir: str,
    checkpoint_step: int,
    final_result: dict[str, Any],
    log_to_wandb: bool = True,
) -> None:
    append_jsonl(run_paths.checkpoint_metrics, final_result["metrics_row"])
    if log_to_wandb and active_wandb_run_id() is not None:
        log_checkpoint_eval_to_active_wandb(
            checkpoint_step=int(checkpoint_step),
            metrics_row=final_result["metrics_row"],
        )
    current_state = read_run_state(run_paths) or {}
    checkpoint_streams = dict(current_state.get("checkpoint_eval_streams") or {})
    last_scored_checkpoints = dict(current_state.get("last_scored_checkpoints") or {})
    selected_best_checkpoints = dict(current_state.get("selected_best_checkpoints") or {})
    eval_tag_key = str(final_result["metrics_row"].get("eval_tag") or "default")
    last_payload = {
        "eval_tag": final_result["metrics_row"].get("eval_tag"),
        "eval_name": final_result["metrics_row"].get("eval_name"),
        "wandb_metric_prefix": final_result["metrics_row"].get("wandb_metric_prefix"),
        "checkpoint_name": checkpoint_name,
        "checkpoint_dir": checkpoint_dir,
        "persisted_checkpoint_dir": final_result["metrics_row"].get(
            "persisted_checkpoint_dir"
        ),
        "checkpoint_step": int(checkpoint_step),
        "accuracy_pct_total": final_result["metrics_row"].get("accuracy_pct_total"),
        "summary_path": final_result["summary_path"],
        "combined_summary_path": final_result["combined_summary_path"],
    }
    checkpoint_streams[eval_tag_key] = {**last_payload}
    last_scored_checkpoints[eval_tag_key] = dict(last_payload)
    selected_best_checkpoints[eval_tag_key] = _pick_best_checkpoint_payload(
        run_paths=run_paths,
        candidate_row=final_result["metrics_row"],
    )
    update_run_state(
        run_paths,
        phase="training",
        last_error=None,
        last_scored_checkpoint=dict(last_payload),
        last_scored_checkpoints=last_scored_checkpoints,
        selected_best_checkpoint=selected_best_checkpoints[eval_tag_key],
        selected_best_checkpoints=selected_best_checkpoints,
        checkpoint_eval_streams=checkpoint_streams,
        wandb_run_id=active_wandb_run_id(),
    )


def evaluate_checkpoint(
    *,
    heldout_eval_manifest: str,
    model,
    pretrained_path: str,
    model_name_or_path: str = "",
    checkpoint_dir: str,
    output_root: str,
    checkpoint_name: str,
    checkpoint_step: int,
    run_name: str,
    world_size: int = 1,
    rank: int = 0,
    inference_mode: str = CHECKPOINT_EVAL_INFERENCE_MODE_LOCAL_LOOP,
) -> dict[str, Any]:
    all_rows = _read_jsonl(heldout_eval_manifest)
    if not all_rows:
        raise RuntimeError(f"No held-out eval rows found in {heldout_eval_manifest}")
    eval_spec = _heldout_eval_spec(heldout_eval_manifest)
    benchmark_name = eval_spec["benchmark_name"]

    work_rows = all_rows[int(rank) :: max(1, int(world_size))]
    eval_model = _build_checkpoint_eval_model(
        inference_mode=inference_mode,
        model=model,
        pretrained_path=pretrained_path,
        model_name_or_path=model_name_or_path,
        checkpoint_dir=checkpoint_dir,
    )
    _log_checkpoint_eval(
        (
            f"step={int(checkpoint_step)} checkpoint={checkpoint_name} "
            f"assigned_rows={len(work_rows)}/{len(all_rows)} "
            f"inference_mode={inference_mode} "
            f"tokenizer_source={eval_model.tokenizer_source}"
        ),
        rank=rank,
    )
    partial_rows: list[dict[str, Any]] = []
    total_work = len(work_rows)
    for index, row in enumerate(work_rows, start=1):
        msg = {
            "index": row["index"],
            "audio": (
                row["audio_path"]
                if isinstance(row.get("audio_path"), list)
                else [row["audio_path"]]
            ),
            "text": row.get("question", ""),
            "conversation": row.get("conversation"),
            "meta": {
                "task": "MQA",
                "dataset_name": benchmark_name,
                **dict(row.get("meta") or {}),
            },
        }
        trajectory = eval_model(msg)
        partial_rows.append(_result_row(row, trajectory))
        torch.cuda.empty_cache()
        if total_work and (
            index % CHECKPOINT_EVAL_PROGRESS_EVERY == 0 or index == total_work
        ):
            _log_checkpoint_eval(
                f"step={int(checkpoint_step)} inference_progress={index}/{total_work}",
                rank=rank,
            )

    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    partial_path = output_root_path / f"rank-{int(rank):03d}-of-{int(world_size):03d}.jsonl"
    with jsonlines.open(partial_path, mode="w") as writer:
        for row in partial_rows:
            writer.write(row)
    _log_checkpoint_eval(
        f"step={int(checkpoint_step)} wrote_partial={partial_path}",
        rank=rank,
    )

    return {
        "status": "partial",
        "run_name": run_name,
        "checkpoint_name": checkpoint_name,
        "checkpoint_step": checkpoint_step,
        "rank": int(rank),
        "world_size": int(world_size),
        "partial_path": str(partial_path),
        "count": len(partial_rows),
    }


def finalize_checkpoint_evaluation(
    *,
    heldout_eval_manifest: str,
    output_root: str,
    checkpoint_dir: str,
    checkpoint_name: str,
    checkpoint_step: int,
    run_name: str,
    checkpoint_metrics_path: str,
    eval_repeat_dir: str,
    eval_name: str | None = None,
    eval_tag: str | None = None,
    wandb_metric_prefix: str | None = None,
    write_metrics_row: bool = True,
) -> dict[str, Any]:
    _ensure_openai_key()
    eval_spec = _heldout_eval_spec(
        heldout_eval_manifest,
        eval_tag=eval_tag,
        wandb_metric_prefix=wandb_metric_prefix,
    )
    benchmark_name = eval_spec["benchmark_name"]
    resolved_eval_name = str(eval_name or eval_spec["eval_name"]).strip()
    resolved_eval_tag = str(eval_spec["eval_tag"]).strip()
    resolved_metric_prefix = str(eval_spec["wandb_metric_prefix"]).strip()
    output_root_path = Path(output_root)
    partial_paths = sorted(output_root_path.glob("rank-*.jsonl"))
    if not partial_paths:
        raise RuntimeError(f"No partial checkpoint eval outputs found in {output_root}")
    all_rows: list[dict[str, Any]] = []
    for path in partial_paths:
        all_rows.extend(_read_jsonl(path))
    all_rows.sort(key=lambda row: int(row.get("index", 0)))
    question_set_version = str(
        ((all_rows[0].get("meta") or {}) if all_rows else {}).get("heldout_set_name") or ""
    )

    benchmark_root = Path(eval_repeat_dir) / MODEL_NAME / benchmark_name
    benchmark_root.mkdir(parents=True, exist_ok=True)
    result_file = benchmark_root / f"{MODEL_NAME}_{benchmark_name}.jsonl"
    with jsonlines.open(result_file, mode="w") as writer:
        for row in all_rows:
            writer.write(row)

    dataset = _build_dataset_for_benchmark(benchmark_name)
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(
            captured_stderr
        ):
            perf = dataset.evaluate(str(result_file), dump_judge=True, method=EVAL_METHOD)
    except Exception as exc:
        captured_output = "\n".join(
            part.strip()
            for part in (captured_stdout.getvalue(), captured_stderr.getvalue())
            if part.strip()
        )
        if captured_output:
            raise RuntimeError(
                "Checkpoint eval scoring failed. Captured evaluator output tail:\n"
                f"{captured_output[-4000:]}"
            ) from exc
        raise
    perf_file = benchmark_root / f"{MODEL_NAME}_{benchmark_name}_{EVAL_METHOD}_performance.json"
    write_json(perf_file, perf)

    judge_candidates = sorted(
        benchmark_root.glob(f"{MODEL_NAME}_{benchmark_name}_{EVAL_METHOD}*_judge.jsonl")
    )
    if not judge_candidates:
        raise FileNotFoundError(
            benchmark_root / f"{MODEL_NAME}_{benchmark_name}_{EVAL_METHOD}*_judge.jsonl"
        )
    judge_file = judge_candidates[-1]

    perf_summary = summarize_performance_payload(perf)
    summary_payload = {
        "model": MODEL_NAME,
        "model_path_override": checkpoint_dir,
        "question_set_version": question_set_version or None,
        "result_file": str(result_file),
        "performance_file": str(perf_file),
        "judge_file": str(judge_file),
        "performance": perf,
        **perf_summary,
    }
    combined_summary_payload = {
        "created_at": now_utc_iso(),
        "question_set_version": question_set_version or None,
        "eval_method": EVAL_METHOD,
        "models": [summary_payload],
    }
    write_json(benchmark_root / "summary.json", summary_payload)
    write_json(Path(eval_repeat_dir) / "combined_summary.json", combined_summary_payload)

    metrics_row = build_checkpoint_metrics_row(
        run_name=run_name,
        eval_name=resolved_eval_name,
        repeat_index=checkpoint_step,
        benchmark_name=benchmark_name,
        eval_method=EVAL_METHOD,
        model_name=MODEL_NAME,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        checkpoint_step=checkpoint_step,
        model_path_override=checkpoint_dir,
        eval_tag=resolved_eval_tag,
        wandb_metric_prefix=resolved_metric_prefix,
        question_set_version=question_set_version or None,
        summary_path=str(benchmark_root / "summary.json"),
        combined_summary_path=str(Path(eval_repeat_dir) / "combined_summary.json"),
        summary_payload=perf_summary,
    )
    if write_metrics_row:
        append_jsonl(checkpoint_metrics_path, metrics_row)
    return {
        "status": "ok",
        "run_name": run_name,
        "checkpoint_name": checkpoint_name,
        "checkpoint_step": checkpoint_step,
        "result_file": str(result_file),
        "performance_file": str(perf_file),
        "judge_file": str(judge_file),
        "summary_path": str(benchmark_root / "summary.json"),
        "combined_summary_path": str(Path(eval_repeat_dir) / "combined_summary.json"),
        "metrics_row": metrics_row,
    }


def _persist_evaluated_checkpoint(
    *,
    run_name: str,
    checkpoint_dir: str,
    checkpoint_name: str,
) -> str:
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(checkpoint_path)
    run_paths = resolve_run_paths(run_name)
    archive_dir = run_paths.promotion_archive_dir(checkpoint_name.replace("_", "-"))
    archive_dir.parent.mkdir(parents=True, exist_ok=True)
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    shutil.copytree(checkpoint_path, archive_dir)
    return str(archive_dir)


def run_step0_generate_path_comparison(
    *,
    heldout_eval_manifest: str,
    pretrained_path: str,
    model_name_or_path: str,
    run_name: str,
    checkpoint_dir: str,
    baseline_metrics_row: dict[str, Any],
    eval_tag: str | None = None,
    wandb_metric_prefix: str | None = None,
) -> dict[str, Any]:
    run_paths = resolve_run_paths(run_name)
    eval_spec = _heldout_eval_spec(
        heldout_eval_manifest,
        eval_tag=eval_tag,
        wandb_metric_prefix=wandb_metric_prefix,
    )
    compare_eval_name = eval_spec["compare_eval_name"]
    compare_eval_dir = run_paths.eval_repeat_dir(compare_eval_name, 0)
    partial_root = compare_eval_dir / "_partials"
    checkpoint_name = "base-model-generate"

    _log_checkpoint_eval(
        "step=0 compare starting generate-path evaluation via KimiAudio.generate"
    )
    evaluate_checkpoint(
        heldout_eval_manifest=heldout_eval_manifest,
        model=None,
        pretrained_path=pretrained_path,
        model_name_or_path=model_name_or_path,
        checkpoint_dir=checkpoint_dir,
        output_root=str(partial_root),
        checkpoint_name=checkpoint_name,
        checkpoint_step=0,
        run_name=run_name,
        world_size=1,
        rank=0,
        inference_mode=CHECKPOINT_EVAL_INFERENCE_MODE_KIMI_GENERATE,
    )
    result = finalize_checkpoint_evaluation(
        heldout_eval_manifest=heldout_eval_manifest,
        output_root=str(partial_root),
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        checkpoint_step=0,
        run_name=run_name,
        checkpoint_metrics_path=str(compare_eval_dir / "checkpoint_metrics.jsonl"),
        eval_repeat_dir=str(compare_eval_dir),
        eval_name=compare_eval_name,
        eval_tag=eval_spec["eval_tag"],
        wandb_metric_prefix=eval_spec["wandb_metric_prefix"],
    )
    batch_mode_accuracy = _overall_accuracy_pct(baseline_metrics_row)
    generate_accuracy = _overall_accuracy_pct(result.get("metrics_row") or {})
    accuracy_delta = None
    if batch_mode_accuracy is not None and generate_accuracy is not None:
        accuracy_delta = round(generate_accuracy - batch_mode_accuracy, 2)
    _log_checkpoint_eval(
        (
            "step=0 compare "
            f"batch_mode_overall_accuracy={batch_mode_accuracy} "
            f"generate_overall_accuracy={generate_accuracy} "
            f"accuracy_delta={accuracy_delta}"
        )
    )
    return result


def persist_active_wandb_run(*, run_name: str) -> str | None:
    run_id = active_wandb_run_id()
    if not run_id:
        return None
    run_paths = resolve_run_paths(run_name)
    update_run_state(run_paths, wandb_run_id=run_id)
    return run_id


def _checkpoint_eval_wandb_metrics_payload(
    *, checkpoint_step: int, metrics_row: dict[str, Any]
) -> dict[str, Any]:
    metric_names = checkpoint_eval_metric_names(
        eval_tag=metrics_row.get("eval_tag"),
        wandb_metric_prefix=metrics_row.get("wandb_metric_prefix"),
    )
    simplified_metrics = simplified_checkpoint_eval_summary(metrics_row)
    metrics = {
        CHECKPOINT_EVAL_WANDB_STEP_METRIC: int(checkpoint_step),
        metric_names["checkpoint_step"]: int(checkpoint_step),
        metric_names["overall_accuracy_pct"]: simplified_metrics.get(
            "overall_accuracy_pct"
        ),
        metric_names["accuracy_on_valid_pct"]: simplified_metrics.get(
            "accuracy_on_valid_pct"
        ),
        metric_names["unparseable_or_unsure_pct"]: simplified_metrics.get(
            "unparseable_or_unsure_pct"
        ),
    }
    return {key: value for key, value in metrics.items() if value is not None}


def log_checkpoint_eval_to_active_wandb(*, checkpoint_step: int, metrics_row: dict[str, Any]) -> None:
    import wandb

    run = getattr(wandb, "run", None)
    if run is None:
        raise RuntimeError(
            "Expected an active W&B run during checkpoint evaluation, but none was found."
        )
    ensure_checkpoint_eval_wandb_metrics_defined(
        eval_tag=metrics_row.get("eval_tag"),
        wandb_metric_prefix=metrics_row.get("wandb_metric_prefix"),
    )
    # Bind eval charts to the custom checkpoint_step metric without rewinding
    # W&B's internal run step, which may already be ahead when save hooks finish.
    wandb.log(
        _checkpoint_eval_wandb_metrics_payload(
            checkpoint_step=int(checkpoint_step),
            metrics_row=metrics_row,
        )
    )


def log_checkpoint_eval_batch_to_active_wandb(
    *, checkpoint_step: int, metrics_rows: list[dict[str, Any]]
) -> None:
    import wandb

    run = getattr(wandb, "run", None)
    if run is None:
        raise RuntimeError(
            "Expected an active W&B run during checkpoint evaluation, but none was found."
        )

    payload: dict[str, Any] = {
        CHECKPOINT_EVAL_WANDB_STEP_METRIC: int(checkpoint_step),
    }
    for metrics_row in list(metrics_rows or []):
        ensure_checkpoint_eval_wandb_metrics_defined(
            eval_tag=metrics_row.get("eval_tag"),
            wandb_metric_prefix=metrics_row.get("wandb_metric_prefix"),
        )
        payload.update(
            _checkpoint_eval_wandb_metrics_payload(
                checkpoint_step=int(checkpoint_step),
                metrics_row=metrics_row,
            )
        )
    wandb.log(payload)


def run_save_time_checkpoint_eval(
    *,
    model,
    pretrained_path: str,
    run_name: str,
    checkpoint_step: int,
    output_dir: str,
    heldout_eval_manifest: str,
    model_name_or_path: str = "",
    checkpoint_dir_override: str = "",
    checkpoint_name_override: str = "",
    eval_tag: str | None = None,
    wandb_metric_prefix: str | None = None,
    persisted_checkpoint_dir: str | None = None,
    persist_checkpoint: bool = True,
) -> dict[str, Any] | None:
    if not heldout_eval_manifest:
        return None
    heldout_manifest_path = Path(heldout_eval_manifest)
    if not heldout_manifest_path.is_file():
        raise FileNotFoundError(heldout_manifest_path)

    checkpoint_dir_path = (
        Path(str(checkpoint_dir_override).strip())
        if str(checkpoint_dir_override).strip()
        else Path(output_dir) / f"checkpoint-{int(checkpoint_step)}"
    )
    if not checkpoint_dir_path.is_dir():
        raise FileNotFoundError(checkpoint_dir_path)
    checkpoint_dir = str(checkpoint_dir_path)
    checkpoint_name = str(checkpoint_name_override).strip() or checkpoint_dir_path.name

    rank = _distributed_rank()
    world_size = _distributed_world_size()
    run_paths = resolve_run_paths(run_name)
    eval_spec = _heldout_eval_spec(
        heldout_eval_manifest,
        eval_tag=eval_tag,
        wandb_metric_prefix=wandb_metric_prefix,
    )
    eval_repeat_dir = run_paths.eval_repeat_dir(eval_spec["eval_name"], int(checkpoint_step))
    partial_root = eval_repeat_dir / "_partials"
    if rank == 0:
        _log_checkpoint_eval(
            (
                f"step={int(checkpoint_step)} start checkpoint={checkpoint_name} "
                f"world_size={int(world_size)} eval_tag={eval_spec['eval_tag']} "
                f"manifest={heldout_eval_manifest}"
            )
        )

    partial_error = None
    try:
        evaluate_checkpoint(
            heldout_eval_manifest=heldout_eval_manifest,
            model=model,
            pretrained_path=pretrained_path,
            model_name_or_path=model_name_or_path,
            checkpoint_dir=checkpoint_dir,
            output_root=str(partial_root),
            checkpoint_name=checkpoint_name,
            checkpoint_step=int(checkpoint_step),
            run_name=run_name,
            world_size=world_size,
            rank=rank,
        )
    except Exception as exc:
        partial_error = {
            "stage": "partial",
            "rank": rank,
            "checkpoint_step": int(checkpoint_step),
            "error": f"{type(exc).__name__}: {exc}",
        }

    gathered_partial_errors = _all_gather_objects(partial_error)
    first_partial_error = next(
        (item for item in gathered_partial_errors if item is not None),
        None,
    )
    if first_partial_error is not None:
        if rank == 0:
            _log_checkpoint_eval(
                (
                    f"step={int(checkpoint_step)} partial_failure "
                    f"rank={first_partial_error['rank']} error={first_partial_error['error']}"
                )
            )
            update_run_state(
                run_paths,
                phase="error",
                last_error=first_partial_error,
            )
        raise RuntimeError(
            "Checkpoint eval failed during partial inference "
            f"at step {int(checkpoint_step)} on rank {first_partial_error['rank']}: "
            f"{first_partial_error['error']}"
        )

    final_error = None
    final_result = None
    if rank == 0:
        try:
            _log_checkpoint_eval(
                f"step={int(checkpoint_step)} partial_inference_done starting_judge={EVAL_METHOD}"
            )
            persist_active_wandb_run(run_name=run_name)
            final_result = finalize_checkpoint_evaluation(
                heldout_eval_manifest=heldout_eval_manifest,
                output_root=str(partial_root),
                checkpoint_dir=checkpoint_dir,
                checkpoint_name=checkpoint_name,
                checkpoint_step=int(checkpoint_step),
                run_name=run_name,
                checkpoint_metrics_path=str(run_paths.checkpoint_metrics),
                eval_repeat_dir=str(eval_repeat_dir),
                eval_tag=eval_spec["eval_tag"],
                wandb_metric_prefix=eval_spec["wandb_metric_prefix"],
                write_metrics_row=False,
            )
            resolved_persisted_checkpoint_dir = str(persisted_checkpoint_dir or "").strip()
            if not resolved_persisted_checkpoint_dir and persist_checkpoint:
                resolved_persisted_checkpoint_dir = _persist_evaluated_checkpoint(
                    run_name=run_name,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_name=checkpoint_name,
                )
            final_result["persisted_checkpoint_dir"] = (
                resolved_persisted_checkpoint_dir or None
            )
            final_result["metrics_row"]["persisted_checkpoint_dir"] = (
                resolved_persisted_checkpoint_dir or None
            )
            _record_final_checkpoint_eval_result(
                run_paths=run_paths,
                checkpoint_name=checkpoint_name,
                checkpoint_dir=checkpoint_dir,
                checkpoint_step=int(checkpoint_step),
                final_result=final_result,
            )
            metrics_row = final_result.get("metrics_row") or {}
            _log_checkpoint_eval(
                f"step={int(checkpoint_step)} eval_tag={eval_spec['eval_tag']} complete "
                f"overall_accuracy={_overall_accuracy_pct(metrics_row)}"
            )
        except Exception as exc:
            final_error = {
                "stage": "finalize",
                "rank": rank,
                "checkpoint_step": int(checkpoint_step),
                "error": f"{type(exc).__name__}: {exc}",
            }
            _log_checkpoint_eval(
                (
                    f"step={int(checkpoint_step)} finalize_failure "
                    f"error={final_error['error']}"
                )
            )
            update_run_state(
                run_paths,
                phase=(
                    "training"
                    if _nonfatal_checkpoint_finalize_errors()
                    else "error"
                ),
                last_error=final_error,
                wandb_run_id=active_wandb_run_id(),
            )

    final_error = _broadcast_object(final_error, src=0)
    if final_error is not None:
        if _nonfatal_checkpoint_finalize_errors():
            if rank == 0:
                _log_checkpoint_eval(
                    (
                        f"step={int(checkpoint_step)} finalize_failure_nonfatal "
                        "continuing_training"
                    )
                )
            return {
                "status": "finalize_error",
                "run_name": run_name,
                "checkpoint_name": checkpoint_name,
                "checkpoint_step": int(checkpoint_step),
                "error": final_error["error"],
            }
        raise RuntimeError(
            "Checkpoint eval failed during finalize "
            f"at step {int(checkpoint_step)}: {final_error['error']}"
        )
    return final_result


def run_combined_save_time_checkpoint_eval(
    *,
    model,
    pretrained_path: str,
    run_name: str,
    checkpoint_step: int,
    output_dir: str,
    eval_specs: list[dict[str, Any]],
    model_name_or_path: str = "",
    checkpoint_dir_override: str = "",
    checkpoint_name_override: str = "",
    combined_eval_name: str = "combined-heldout-gpt54mini",
    combined_eval_tag: str = "combined-heldout",
    combined_wandb_metric_prefix: str = "checkpoint_eval/combined-heldout",
    persisted_checkpoint_dir: str | None = None,
    persist_checkpoint: bool = True,
) -> dict[str, Any] | None:
    normalized_specs = [
        dict(spec)
        for spec in list(eval_specs or [])
        if str(spec.get("manifest") or "").strip()
    ]
    if not normalized_specs:
        return None
    if len(normalized_specs) == 1:
        spec = normalized_specs[0]
        return run_save_time_checkpoint_eval(
            model=model,
            pretrained_path=pretrained_path,
            run_name=run_name,
            checkpoint_step=checkpoint_step,
            output_dir=output_dir,
            heldout_eval_manifest=str(spec.get("manifest") or "").strip(),
            model_name_or_path=model_name_or_path,
            checkpoint_dir_override=checkpoint_dir_override,
            checkpoint_name_override=checkpoint_name_override,
            eval_tag=str(spec.get("tag") or "").strip() or None,
            wandb_metric_prefix=str(spec.get("wandb_metric_prefix") or "").strip() or None,
            persisted_checkpoint_dir=persisted_checkpoint_dir,
            persist_checkpoint=persist_checkpoint,
        )

    checkpoint_dir_path = (
        Path(str(checkpoint_dir_override).strip())
        if str(checkpoint_dir_override).strip()
        else Path(output_dir) / f"checkpoint-{int(checkpoint_step)}"
    )
    if not checkpoint_dir_path.is_dir():
        raise FileNotFoundError(checkpoint_dir_path)
    checkpoint_dir = str(checkpoint_dir_path)
    checkpoint_name = str(checkpoint_name_override).strip() or checkpoint_dir_path.name

    rank = _distributed_rank()
    world_size = _distributed_world_size()
    run_paths = resolve_run_paths(run_name)
    combined_eval_repeat_dir = run_paths.eval_repeat_dir(
        combined_eval_name, int(checkpoint_step)
    )
    combined_manifest_info = _combine_checkpoint_eval_specs(
        normalized_specs,
        output_path=combined_eval_repeat_dir / "combined_heldout_eval_manifest.jsonl",
    )
    combined_manifest = str(combined_manifest_info["manifest"])
    partial_root = combined_eval_repeat_dir / "_partials"

    if rank == 0:
        source_tags = [str(spec.get("tag") or "").strip() for spec in normalized_specs]
        _log_checkpoint_eval(
            (
                f"step={int(checkpoint_step)} start checkpoint={checkpoint_name} "
                f"world_size={int(world_size)} eval_tag={combined_eval_tag} "
                f"manifest={combined_manifest} sources={source_tags}"
            )
        )

    partial_error = None
    try:
        evaluate_checkpoint(
            heldout_eval_manifest=combined_manifest,
            model=model,
            pretrained_path=pretrained_path,
            model_name_or_path=model_name_or_path,
            checkpoint_dir=checkpoint_dir,
            output_root=str(partial_root),
            checkpoint_name=checkpoint_name,
            checkpoint_step=int(checkpoint_step),
            run_name=run_name,
            world_size=world_size,
            rank=rank,
        )
    except Exception as exc:
        partial_error = {
            "stage": "partial",
            "rank": rank,
            "checkpoint_step": int(checkpoint_step),
            "error": f"{type(exc).__name__}: {exc}",
        }

    gathered_partial_errors = _all_gather_objects(partial_error)
    first_partial_error = next(
        (item for item in gathered_partial_errors if item is not None),
        None,
    )
    if first_partial_error is not None:
        if rank == 0:
            _log_checkpoint_eval(
                (
                    f"step={int(checkpoint_step)} partial_failure "
                    f"rank={first_partial_error['rank']} error={first_partial_error['error']}"
                )
            )
            update_run_state(
                run_paths,
                phase="error",
                last_error=first_partial_error,
            )
        raise RuntimeError(
            "Combined checkpoint eval failed during partial inference "
            f"at step {int(checkpoint_step)} on rank {first_partial_error['rank']}: "
            f"{first_partial_error['error']}"
        )

    final_error = None
    batched_result = None
    if rank == 0:
        try:
            _log_checkpoint_eval(
                f"step={int(checkpoint_step)} partial_inference_done starting_judge={EVAL_METHOD}"
            )
            persist_active_wandb_run(run_name=run_name)
            shared_result = finalize_checkpoint_evaluation(
                heldout_eval_manifest=combined_manifest,
                output_root=str(partial_root),
                checkpoint_dir=checkpoint_dir,
                checkpoint_name=checkpoint_name,
                checkpoint_step=int(checkpoint_step),
                run_name=run_name,
                checkpoint_metrics_path=str(run_paths.checkpoint_metrics),
                eval_repeat_dir=str(combined_eval_repeat_dir),
                eval_name=combined_eval_name,
                eval_tag=combined_eval_tag,
                wandb_metric_prefix=combined_wandb_metric_prefix,
                write_metrics_row=False,
            )
            per_eval_results = _write_split_checkpoint_eval_outputs(
                eval_specs=normalized_specs,
                judged_rows=_read_jsonl(shared_result["judge_file"]),
                shared_result_file=shared_result["result_file"],
                shared_judge_file=shared_result["judge_file"],
                run_name=run_name,
                checkpoint_dir=checkpoint_dir,
                checkpoint_name=checkpoint_name,
                checkpoint_step=int(checkpoint_step),
            )

            resolved_persisted_checkpoint_dir = str(persisted_checkpoint_dir or "").strip()
            if not resolved_persisted_checkpoint_dir and persist_checkpoint:
                resolved_persisted_checkpoint_dir = _persist_evaluated_checkpoint(
                    run_name=run_name,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_name=checkpoint_name,
                )
            for per_eval_result in per_eval_results:
                per_eval_result["persisted_checkpoint_dir"] = (
                    resolved_persisted_checkpoint_dir or None
                )
                per_eval_result["metrics_row"]["persisted_checkpoint_dir"] = (
                    resolved_persisted_checkpoint_dir or None
                )
                _record_final_checkpoint_eval_result(
                    run_paths=run_paths,
                    checkpoint_name=checkpoint_name,
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_step=int(checkpoint_step),
                    final_result=per_eval_result,
                    log_to_wandb=False,
                )
                metrics_row = per_eval_result.get("metrics_row") or {}
                _log_checkpoint_eval(
                    f"step={int(checkpoint_step)} eval_tag={metrics_row.get('eval_tag')} complete "
                    f"overall_accuracy={_overall_accuracy_pct(metrics_row)}"
                )
            if active_wandb_run_id() is not None:
                log_checkpoint_eval_batch_to_active_wandb(
                    checkpoint_step=int(checkpoint_step),
                    metrics_rows=[
                        dict(per_eval_result.get("metrics_row") or {})
                        for per_eval_result in per_eval_results
                    ],
                )

            batched_result = {
                "status": "ok",
                "run_name": run_name,
                "checkpoint_name": checkpoint_name,
                "checkpoint_step": int(checkpoint_step),
                "persisted_checkpoint_dir": resolved_persisted_checkpoint_dir or None,
                "combined_manifest": combined_manifest,
                "shared_result": shared_result,
                "per_eval_results": per_eval_results,
            }
        except Exception as exc:
            final_error = {
                "stage": "finalize",
                "rank": rank,
                "checkpoint_step": int(checkpoint_step),
                "error": f"{type(exc).__name__}: {exc}",
            }
            _log_checkpoint_eval(
                (
                    f"step={int(checkpoint_step)} finalize_failure "
                    f"error={final_error['error']}"
                )
            )
            update_run_state(
                run_paths,
                phase=("training" if _nonfatal_checkpoint_finalize_errors() else "error"),
                last_error=final_error,
                wandb_run_id=active_wandb_run_id(),
            )

    final_error = _broadcast_object(final_error, src=0)
    if final_error is not None:
        if _nonfatal_checkpoint_finalize_errors():
            if rank == 0:
                _log_checkpoint_eval(
                    (
                        f"step={int(checkpoint_step)} finalize_failure_nonfatal "
                        "continuing_training"
                    )
                )
            return {
                "status": "finalize_error",
                "run_name": run_name,
                "checkpoint_name": checkpoint_name,
                "checkpoint_step": int(checkpoint_step),
                "error": final_error["error"],
            }
        raise RuntimeError(
            "Combined checkpoint eval failed during finalize "
            f"at step {int(checkpoint_step)}: {final_error['error']}"
        )
    return batched_result


def _unwrap_model(model):
    current = model
    while hasattr(current, "module"):
        current = current.module
    return current


def _resolve_whisper_model(model):
    candidates = [
        model,
        getattr(model, "model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ]
    for candidate in candidates:
        whisper_model = getattr(candidate, "whisper_model", None)
        if whisper_model is not None:
            return whisper_model
    raise AttributeError("Expected whisper_model on the live Kimi training model")


def _require_checkpoint_eval_prompt(prompt: str | None, meta: dict[str, Any]) -> str:
    text = "" if prompt is None else str(prompt)
    if str(meta.get("task", "")).upper() != "MQA":
        return text
    if not text.strip():
        raise RuntimeError("Expected a non-empty MQA prompt for checkpoint evaluation.")
    return text


def _distributed_available() -> bool:
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()


def _distributed_rank() -> int:
    import torch.distributed as dist

    if _distributed_available():
        return int(dist.get_rank())
    return int(os.environ.get("LOCAL_RANK") or os.environ.get("RANK") or 0)


def _distributed_world_size() -> int:
    import torch.distributed as dist

    if _distributed_available():
        return int(dist.get_world_size())
    return int(os.environ.get("WORLD_SIZE") or 1)


def _all_gather_objects(payload: Any) -> list[Any]:
    import torch.distributed as dist

    if not _distributed_available():
        return [payload]
    gathered = [None] * int(dist.get_world_size())
    dist.all_gather_object(gathered, payload)
    return gathered


def _broadcast_object(payload: Any, *, src: int = 0) -> Any:
    import torch.distributed as dist

    if not _distributed_available():
        return payload
    object_list = [payload] if int(dist.get_rank()) == src else [None]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def _pick_best_checkpoint_payload(
    *,
    run_paths,
    candidate_row: dict[str, Any],
) -> dict[str, Any]:
    current_state = read_run_state(run_paths) or {}
    current_best = current_state.get("selected_best_checkpoint") or {}
    current_score = current_best.get("accuracy_pct_total")
    candidate_score = candidate_row.get("accuracy_pct_total")
    current_step = int(current_best.get("checkpoint_step") or -1)
    candidate_step = int(candidate_row.get("checkpoint_step") or -1)
    if current_score is not None and candidate_score is not None:
        if float(candidate_score) < float(current_score):
            return current_best
        if float(candidate_score) == float(current_score) and candidate_step <= current_step:
            return current_best
    elif current_score is not None and candidate_score is None:
        return current_best
    return {
        "checkpoint_name": candidate_row.get("checkpoint_name"),
        "checkpoint_dir": candidate_row.get("checkpoint_dir"),
        "persisted_checkpoint_dir": candidate_row.get("persisted_checkpoint_dir"),
        "checkpoint_step": candidate_step,
        "accuracy_pct_total": candidate_score,
        "eval_tag": candidate_row.get("eval_tag"),
        "eval_name": candidate_row.get("eval_name"),
        "wandb_metric_prefix": candidate_row.get("wandb_metric_prefix"),
        "summary_path": candidate_row.get("summary_path"),
        "combined_summary_path": candidate_row.get("combined_summary_path"),
    }
