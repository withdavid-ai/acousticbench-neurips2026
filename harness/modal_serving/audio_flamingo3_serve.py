"""Modal serving app for Audio Flamingo 3.

Deploy:  modal deploy [--env <modal-env>] modal_serving/audio_flamingo3_serve.py

This server keeps the real base/chat checkpoints on Hugging Face as the source of
truth, but reconstructs them into the native `AudioFlamingo3ForConditionalGeneration`
module layout at runtime. That removes the NVIDIA repo clone and `llava.load()`
dependency while preserving the existing base/chat toggle.

Variants:
  - AudioFlamingo3Server        — standard chat model
  - AudioFlamingo3BaseServer    — base (non-chat) model
"""

import json
from pathlib import Path

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
    run_smoke_test,
    safe_enter,
)

app = modal.App("audiobench-audio-flamingo3")

AF3_AUDIO_TOKEN_ID = 151669
AF3_BOS_TOKEN_ID = 151670
AF3_EOS_TOKEN_ID = 151645
AF3_PAD_TOKEN_ID = 151671
AF3_MAX_AUDIO_SECONDS = 600
AF3_HF_ALLOW_PATTERNS = [
    "config.json",
    "llm/*",
    "sound_tower/*",
    "sound_mm_projector/*",
]
AF3_CHAT_TEMPLATE = """{% if messages[0]['role'] != 'system' %}<|im_start|>system
You are a helpful assistant.<|im_end|>
{% endif %}{% for m in messages if m['content'] is not none %}<|im_start|>{{ m['role'] }}
{% if m['content'] is string %}{{ m['content'] }}{% else %}{% set audio = namespace(found=False) %}{% set text_buf = namespace(v='') %}{% for c in m['content'] %}{% if c.get('type') == 'audio' or 'audio' in c %}{% set audio.found = True %}{% elif c.get('type') == 'text' or 'text' in c %}{% set text_buf.v = text_buf.v + c['text'] %}{% endif %}{% endfor %}{% if audio.found %}{{ ' ' }}{% endif %}{{ text_buf.v }}{% endif %}<|im_end|>
{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant
{% endif %}"""

# Keep the newer Transformers AF3 implementation while staying on the older
# CUDA-compatible torch runtime that Modal's current driver supports.
HF_CACHE_PATH = "/hf-cache"

image = (
    base_image()
    .pip_install("transformers==5.3.0", "safetensors==0.7.0")
    .env({
        "HF_HOME": f"{HF_CACHE_PATH}/huggingface",
        "HF_HUB_CACHE": f"{HF_CACHE_PATH}/huggingface/hub",
        "TRANSFORMERS_CACHE": f"{HF_CACHE_PATH}/huggingface/hub",
    })
    .add_local_python_source("serving_utils", copy=True)
)

DEFAULT_VARIANT = "chat"
MODEL_VARIANTS = {
    "base": {
        "model_repo": "nvidia/audio-flamingo-3",
    },
    "chat": {
        "model_repo": "nvidia/audio-flamingo-3-chat",
    },
}
MODEL_REPOS = [
    {
        "repo_id": variant["model_repo"],
        "revision": hf_revision(variant["model_repo"]),
        "allow_patterns": AF3_HF_ALLOW_PATTERNS,
    }
    for variant in MODEL_VARIANTS.values()
]


def _af3_prompts_for_messages(messages, placeholder_counts: list[int], audio_token: str) -> tuple[str, str]:
    if not placeholder_counts:
        raise ValueError("AF3 requires at least one audio clip")
    return (
        flatten_messages_to_text(
            messages,
            audio_placeholder=lambda idx: f"<sound-{idx + 1}>",
            include_roles=True,
        ),
        flatten_messages_to_text(
            messages,
            audio_placeholder=lambda idx: audio_token * placeholder_counts[idx],
            include_roles=True,
        ),
    )


def _generate_text_only(
    model,
    tokenizer,
    messages,
    meta: dict,
    *,
    model_repo: str,
    max_new_tokens: int,
    extra_payload: dict | None = None,
    builder: str,
) -> dict:
    import torch

    text_model = getattr(model, "language_model", None)
    if text_model is None:
        raise ValueError("AF3 text-only fallback requires model.language_model")

    rendered_prompt = flatten_messages_to_text(
        messages,
        audio_placeholder="",
        include_roles=True,
    )
    model_prompt = rendered_prompt
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": model_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    text_inputs = tokenizer(chat_text, return_tensors="pt")

    generation_kwargs = apply_benchmark_sampling(
        meta,
        {"max_new_tokens": max_new_tokens, "do_sample": False},
    )
    model_device = getattr(model, "device", None)
    if model_device is None:
        model_device = next(text_model.parameters()).device
    with torch.inference_mode():
        outputs = text_model.generate(
            input_ids=text_inputs.input_ids.to(model_device),
            attention_mask=text_inputs.attention_mask.to(model_device),
            **generation_kwargs,
        )
    response = tokenizer.batch_decode(
        outputs[:, text_inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )[0]

    submission_payload = {
        "provider": "audio-flamingo3-native",
        "transport": "transformers-local-generate",
        "model_repo": model_repo,
        "text_only_fallback": True,
        "rendered_prompt": rendered_prompt,
        "model_prompt": model_prompt,
        "chat_text": chat_text,
        "generation_kwargs": generation_kwargs,
        "audio_placeholder_counts": [],
        "processor_inputs": {
            "input_ids_shape": list(text_inputs.input_ids.shape),
            "attention_mask_shape": list(text_inputs.attention_mask.shape),
        },
    }
    if extra_payload:
        submission_payload.update(extra_payload)

    return {
        "prompt": rendered_prompt,
        "response": response,
        "submission_payload": submission_payload,
        "submission_trace": {
            "exact_submission_available": True,
            "builder": builder,
        },
    }


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_audio_features(feature_extractor, audio_datas):
    import torch

    all_features = []
    all_masks = []
    placeholder_counts = []
    chunk_size = int(feature_extractor.sampling_rate * feature_extractor.chunk_length)
    max_windows = int(AF3_MAX_AUDIO_SECONDS // feature_extractor.chunk_length)

    for audio_data in audio_datas:
        n_samples = int(audio_data.shape[0])
        n_windows = max(1, (n_samples + chunk_size - 1) // chunk_size)
        n_windows = min(n_windows, max_windows)
        chunks = []
        time_cap = min(n_samples, n_windows * chunk_size)
        for idx in range(n_windows):
            start = idx * chunk_size
            end = min((idx + 1) * chunk_size, time_cap)
            chunks.append(audio_data[start:end])

        features = feature_extractor(
            chunks,
            sampling_rate=feature_extractor.sampling_rate,
            return_tensors="pt",
            return_attention_mask=True,
            padding="max_length",
        )
        input_features = features.input_features
        input_features_mask = features.attention_mask
        input_lengths = input_features_mask.sum(-1).to(torch.long)
        conv2_lengths = (input_lengths - 1) // 2 + 1
        post_pool_lengths = (conv2_lengths - 2) // 2 + 1

        all_features.append(input_features)
        all_masks.append(input_features_mask)
        placeholder_counts.append(int(post_pool_lengths.sum().item()))

    return (
        torch.cat(all_features, dim=0),
        torch.cat(all_masks, dim=0),
        placeholder_counts,
    )


def _build_native_audio_config(sound_tower_config: dict) -> dict:
    return {
        "model_type": "audioflamingo3_encoder",
        "hidden_size": int(sound_tower_config["d_model"]),
        "num_hidden_layers": int(sound_tower_config["encoder_layers"]),
        "num_attention_heads": int(sound_tower_config["encoder_attention_heads"]),
        "intermediate_size": int(sound_tower_config["encoder_ffn_dim"]),
        "max_source_positions": int(sound_tower_config["max_source_positions"]),
        "num_mel_bins": int(sound_tower_config["num_mel_bins"]),
        "activation_function": str(sound_tower_config.get("activation_function", "gelu")),
        "dropout": float(sound_tower_config.get("dropout", 0.0)),
        "attention_dropout": float(sound_tower_config.get("attention_dropout", 0.0)),
        "activation_dropout": float(sound_tower_config.get("activation_dropout", 0.0)),
        "initializer_range": float(sound_tower_config.get("init_std", 0.02)),
        "scale_embedding": bool(sound_tower_config.get("scale_embedding", False)),
    }


def _iter_llm_weight_paths(llm_dir: Path) -> list[Path]:
    sharded = sorted(llm_dir.glob("model-*.safetensors"))
    if sharded:
        return sharded
    single = llm_dir / "model.safetensors"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"No LLM safetensor weights found under {llm_dir}")


def _copy_safetensor_weights(model, model_path: Path):
    import torch
    from safetensors import safe_open

    state = model.state_dict()
    loaded_keys = set()
    unexpected_keys = []

    def copy_file(path: Path, rename):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for source_key in handle.keys():
                target_key = rename(source_key)
                target_tensor = state.get(target_key)
                if target_tensor is None:
                    unexpected_keys.append(target_key)
                    continue
                source_tensor = handle.get_tensor(source_key)
                if source_tensor.shape != target_tensor.shape:
                    raise ValueError(
                        f"AF3 weight shape mismatch for {target_key}: "
                        f"expected {tuple(target_tensor.shape)}, got {tuple(source_tensor.shape)}"
                    )
                with torch.no_grad():
                    target_tensor.copy_(source_tensor.to(dtype=target_tensor.dtype))
                loaded_keys.add(target_key)

    copy_file(model_path / "sound_tower" / "model.safetensors", lambda key: f"audio_tower.{key}")
    projector_key_map = {
        "layers.0.weight": "multi_modal_projector.linear_1.weight",
        "layers.0.bias": "multi_modal_projector.linear_1.bias",
        "layers.2.weight": "multi_modal_projector.linear_2.weight",
        "layers.2.bias": "multi_modal_projector.linear_2.bias",
    }
    copy_file(
        model_path / "sound_mm_projector" / "model.safetensors",
        lambda key: projector_key_map[key],
    )
    for llm_path in _iter_llm_weight_paths(model_path / "llm"):
        copy_file(llm_path, lambda key: f"language_model.{key}")

    missing_keys = sorted(set(state) - loaded_keys)
    if missing_keys or unexpected_keys:
        raise ValueError(
            "AF3 native load produced key mismatch: "
            f"missing={missing_keys[:10]} unexpected={unexpected_keys[:10]}"
        )


def _load_native_components(model_path: Path):
    import torch
    from transformers import (
        AudioFlamingo3Config,
        AudioFlamingo3ForConditionalGeneration,
        AutoConfig,
        AutoTokenizer,
        WhisperFeatureExtractor,
    )

    tokenizer = AutoTokenizer.from_pretrained(str(model_path / "llm"))
    tokenizer.chat_template = AF3_CHAT_TEMPLATE
    tokenizer.padding_side = "left"
    audio_token = tokenizer.convert_ids_to_tokens(AF3_AUDIO_TOKEN_ID)
    if not isinstance(audio_token, str):
        raise ValueError(f"Could not resolve AF3 audio token {AF3_AUDIO_TOKEN_ID}")

    text_config = AutoConfig.from_pretrained(str(model_path / "llm"))
    text_config.bos_token_id = tokenizer.bos_token_id or AF3_BOS_TOKEN_ID
    text_config.eos_token_id = tokenizer.eos_token_id or AF3_EOS_TOKEN_ID
    text_config.pad_token_id = tokenizer.pad_token_id or AF3_PAD_TOKEN_ID

    sound_tower_config = _read_json(model_path / "sound_tower" / "config.json")
    config = AudioFlamingo3Config(
        audio_config=_build_native_audio_config(sound_tower_config),
        text_config=text_config,
        audio_token_id=AF3_AUDIO_TOKEN_ID,
    )

    model = AudioFlamingo3ForConditionalGeneration(config)
    model = model.to(dtype=torch.bfloat16)
    _copy_safetensor_weights(model, model_path)
    model = model.to("cuda:0").eval()

    feature_extractor = WhisperFeatureExtractor(
        feature_size=128,
        sampling_rate=16000,
        chunk_length=30,
        hop_length=160,
        n_fft=400,
        padding_value=0.0,
        return_attention_mask=True,
    )
    return model, tokenizer, feature_extractor, audio_token


@app.function(image=image, volumes={HF_CACHE_PATH: hf_cache}, secrets=hf_secrets, timeout=7200, retries=0)
def prefetch():
    prefetch_repos(MODEL_REPOS)


class _AudioFlamingo3ServerBase:
    MODEL_REPO = ""

    @modal.enter()
    @safe_enter
    def load_model(self):
        from huggingface_hub import snapshot_download

        model_path = Path(snapshot_download(
            self.MODEL_REPO,
            revision=hf_revision(self.MODEL_REPO),
            allow_patterns=AF3_HF_ALLOW_PATTERNS,
        ))
        self.model, self.tokenizer, self.feature_extractor, self.audio_token = _load_native_components(
            model_path
        )
        hf_cache.commit()

    @modal.method()
    def generate(self, item: dict):
        import torch

        messages, audio_items, meta = decode_messages_payload(item)
        prompt = flatten_messages_to_text(messages, include_roles=True)
        err = init_error_response(self, prompt, include_traceback=bool(meta.get("_debug")))
        if err:
            return err

        if not audio_items:
            return _generate_text_only(
                self.model,
                self.tokenizer,
                messages,
                meta,
                model_repo=self.MODEL_REPO,
                max_new_tokens=512,
                builder=f"{self.__class__.__name__}.generate_text_only",
            )

        audio_datas = audio_items_to_ndarrays(audio_items, sr=self.feature_extractor.sampling_rate)
        input_features, input_features_mask, placeholder_counts = _build_audio_features(
            self.feature_extractor,
            audio_datas,
        )
        rendered_prompt, model_prompt = _af3_prompts_for_messages(
            messages,
            placeholder_counts,
            self.audio_token,
        )
        chat_text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": model_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        text_inputs = self.tokenizer(chat_text, return_tensors="pt")

        generation_kwargs = apply_benchmark_sampling(
            meta,
            {"max_new_tokens": 512, "do_sample": False},
        )
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids=text_inputs.input_ids.to(self.model.device),
                attention_mask=text_inputs.attention_mask.to(self.model.device),
                input_features=input_features.to(self.model.device, dtype=self.model.dtype),
                input_features_mask=input_features_mask.to(self.model.device),
                **generation_kwargs,
            )
        response = self.tokenizer.batch_decode(
            outputs[:, text_inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )[0]
        return {
            "prompt": rendered_prompt,
            "response": response,
            "submission_payload": {
                "provider": "audio-flamingo3-native",
                "transport": "transformers-local-generate",
                "model_repo": self.MODEL_REPO,
                "rendered_prompt": rendered_prompt,
                "model_prompt": model_prompt,
                "chat_text": chat_text,
                "generation_kwargs": generation_kwargs,
                "audio_placeholder_counts": placeholder_counts,
                "processor_inputs": {
                    "input_ids_shape": list(text_inputs.input_ids.shape),
                    "attention_mask_shape": list(text_inputs.attention_mask.shape),
                    "input_features_shape": list(input_features.shape),
                    "input_features_mask_shape": list(input_features_mask.shape),
                },
            },
            "submission_trace": {
                "exact_submission_available": True,
                "builder": f"{self.__class__.__name__}.generate",
            },
        }


@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={HF_CACHE_PATH: hf_cache},
    secrets=hf_secrets,
    timeout=600,
    scaledown_window=300,
    retries=0,
)
class AudioFlamingo3Server(_AudioFlamingo3ServerBase):
    MODEL_REPO = MODEL_VARIANTS["chat"]["model_repo"]


@app.cls(
    gpu="A100-80GB",
    image=image,
    volumes={HF_CACHE_PATH: hf_cache},
    secrets=hf_secrets,
    timeout=600,
    scaledown_window=300,
    retries=0,
)
class AudioFlamingo3BaseServer(_AudioFlamingo3ServerBase):
    MODEL_REPO = MODEL_VARIANTS["base"]["model_repo"]


SMOKE_TEST_VARIANTS = {
    "chat": (AudioFlamingo3Server, "generate"),
    "base": (AudioFlamingo3BaseServer, "generate"),
}


@app.local_entrypoint()
def smoke_test(
    audio_path: str,
    variant: str = DEFAULT_VARIANT,
    text: str = "",
    task: str = "",
    interactive: str = "",
    audio_type: str = "",
    dataset_name: str = "",
    lang: str = "",
    meta_json: str = "",
    prefetch_first: bool = False,
    audio_path_2: str = "",
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
        audio_path_2=audio_path_2,
    )


def _simple_choice_accuracy(rows: list[dict]) -> dict:
    import re

    from almeval.datasets.mqa_extraction import parse_question_options, resolve_correct_label

    total = len(rows)
    correct = 0
    for row in rows:
        options = parse_question_options(str(row.get("question", "")))
        correct_label, _ = resolve_correct_label(
            options,
            row.get("answer"),
            row.get("meta") if isinstance(row.get("meta"), dict) else None,
        )
        prediction = str(row.get("prediction") or "").upper()
        selected_label = None
        for candidate in re.findall(r"\b([A-Z])\b", prediction):
            if candidate in options:
                selected_label = candidate
                break
        if correct_label is not None and selected_label == correct_label:
            correct += 1
    return {
        "correct": correct,
        "total": total,
        "accuracy": (correct / total) if total else None,
    }


@app.local_entrypoint()
def eval_relative_accuracy(
    limit: int = 50,
    variant: str = DEFAULT_VARIANT,
    question_set_version: str = "relative acoustic perception v3",
    eval_method: str = "default",
    prefetch_first: bool = False,
    max_consecutive_errors: int = 3,
):
    import json
    import os
    import statistics
    import sys
    import tempfile
    import time
    from datetime import datetime
    from pathlib import Path

    evalkit_root = Path(__file__).resolve().parents[1]
    if str(evalkit_root) not in sys.path:
        sys.path.insert(0, str(evalkit_root))

    from almeval.datasets import build_dataset
    from common import build_smoke_test_item

    if variant not in SMOKE_TEST_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {sorted(SMOKE_TEST_VARIANTS)}")

    if question_set_version:
        os.environ["AUDIOBENCH_QUESTION_SET_VERSION"] = question_set_version
    else:
        os.environ.pop("AUDIOBENCH_QUESTION_SET_VERSION", None)

    dataset = build_dataset("AcousticPerceptionBench")
    if isinstance(dataset, list):
        raise TypeError("Expected a single AcousticPerceptionBench dataset instance")

    limit = min(int(limit), len(dataset.data))
    dataset.data = dataset.data[:limit]

    server_cls, method_name = SMOKE_TEST_VARIANTS[variant]
    if prefetch_first:
        prefetch.remote()
    server = server_cls()
    method = getattr(server, method_name)

    rows: list[dict] = []
    remote_times: list[float] = []
    errors: list[dict] = []
    consecutive_errors = 0
    started = time.perf_counter()

    for i in range(limit):
        msg = dataset[i]
        payload = build_smoke_test_item(
            audio_path="",
            audio_paths=list(msg["audio"]),
            text=msg["text"],
            task=str(msg.get("meta", {}).get("task", "")),
            interactive=str(msg.get("meta", {}).get("interactive", "")),
            audio_type=str(msg.get("meta", {}).get("audio_type", "")),
            dataset_name=str(msg.get("meta", {}).get("dataset_name", "")),
            lang=str(msg.get("meta", {}).get("lang", "")),
            meta_json=json.dumps(msg["meta"]),
        )
        row = dict(dataset.data[i])
        t0 = time.perf_counter()
        try:
            result = method.remote(item=payload)
            dt = time.perf_counter() - t0
            remote_times.append(dt)
            consecutive_errors = 0
            row["prediction"] = result.get("response")
            row["real_prompt"] = str(result.get("prompt", msg.get("text", "")))
            row["response_status"] = "ok"
            row["response_error"] = None
            row["wall_time_s"] = dt
            print(f"[{i + 1}/{limit}] ok {dt:.2f}s -> {row['prediction']!r}")
        except Exception as exc:
            dt = time.perf_counter() - t0
            consecutive_errors += 1
            row["prediction"] = None
            row["real_prompt"] = str(msg.get("text", ""))
            row["response_status"] = "error"
            row["response_error"] = str(exc)
            row["wall_time_s"] = dt
            errors.append({"index": int(msg["index"]), "error": str(exc)})
            print(f"[{i + 1}/{limit}] error {dt:.2f}s -> {exc}")
            if consecutive_errors >= max_consecutive_errors:
                rows.append(row)
                print(f"aborting after {consecutive_errors} consecutive errors")
                break
        rows.append(row)

    out_dir = Path(tempfile.gettempdir()) / f"af3-ephemeral-rad-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "Audio-Flamingo-3_AcousticPerceptionBench.jsonl"
    with result_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    exact = _simple_choice_accuracy(rows)
    judged = None
    judge_error = None
    try:
        judged = dataset.evaluate(str(result_path), dump_judge=True, method=eval_method)
    except Exception as exc:
        judge_error = str(exc)

    summary = {
        "result_path": str(result_path),
        "question_set_version": question_set_version,
        "variant": variant,
        "samples_attempted": len(rows),
        "errors": len(errors),
        "exact_choice_accuracy": exact,
        "judged_accuracy": judged,
        "judge_error": judge_error,
        "remote_timing_s": {
            "count": len(remote_times),
            "mean": statistics.mean(remote_times) if remote_times else None,
            "median": statistics.median(remote_times) if remote_times else None,
            "min": min(remote_times) if remote_times else None,
            "max": max(remote_times) if remote_times else None,
            "sum": sum(remote_times) if remote_times else None,
        },
        "wall_clock_s": time.perf_counter() - started,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
