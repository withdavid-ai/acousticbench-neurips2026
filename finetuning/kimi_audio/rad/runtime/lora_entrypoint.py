from dataclasses import dataclass, field
import json
import logging
import os
import warnings
from typing import Any, Dict, Optional

import torch
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from huggingface_hub import snapshot_download
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    get_peft_model_state_dict,
)
import transformers
from transformers import AutoTokenizer, Trainer, TrainerCallback
from transformers.integrations import deepspeed
from transformers.trainer_pt_utils import LabelSmoother

from finetune_codes.datasets import LazySupervisedDataset
from finetuning.kimi_audio.vendor_kimi.finetune_codes.model import KimiAudioModel
from finetuning.kimi_audio.rad.direct_checkpoint_eval import (
    persist_active_wandb_run,
    run_combined_save_time_checkpoint_eval,
    run_save_time_checkpoint_eval,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings(
    "ignore",
    message=r".*copying from a non-meta parameter in the checkpoint to a meta parameter.*",
    category=UserWarning,
)

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
# Whisper encoder projections, for the optional encoder-LoRA path below. The
# audio encoder does not share the language stack's naming: attention output is
# `out_proj` (not `o_proj`) and the MLP is `fc1`/`fc2` (not gate/up/down). Under
# `KimiAudioModel` these live at
#   whisper_model.speech_encoder.layers.{0..31}.self_attn.{q,k,v,out}_proj
#   whisper_model.speech_encoder.layers.{0..31}.{fc1,fc2}
# i.e. 32 layers x 6 = 192 modules. The wrapper keeps only `speech_encoder`, so
# these names cannot match a Whisper *decoder* module.
WHISPER_ENCODER_LORA_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "fc1",
    "fc2",
]

DEFAULT_KIMI_MODEL_NAME = "moonshotai/Kimi-Audio-7B-Instruct"


def _resolve_supported_model_name_or_path(value: Optional[str]) -> str:
    normalized = str(value or "").strip().rstrip("/") or DEFAULT_KIMI_MODEL_NAME
    if normalized == "moonshotai/Kimi-Audio-7B":
        raise ValueError(
            "Kimi RAD finetuning no longer supports `moonshotai/Kimi-Audio-7B`; "
            "use `moonshotai/Kimi-Audio-7B-Instruct`."
        )
    return normalized


def _heldout_eval_specs_from_json(
    heldout_eval_specs_json: str,
    legacy_manifest: str,
) -> list[dict[str, Any]]:
    raw_json = str(heldout_eval_specs_json or "").strip()
    if raw_json:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, list):
            raise ValueError("heldout_eval_specs_json must decode to a list")
        return [dict(item) for item in parsed if isinstance(item, dict)]
    manifest = str(legacy_manifest or "").strip()
    if not manifest:
        return []
    return [{"tag": "", "manifest": manifest, "wandb_metric_prefix": ""}]


def _normalize_heldout_eval_stages(raw_value: Any) -> list[str]:
    if raw_value in (None, ""):
        return ["step0", "checkpoint"]
    if isinstance(raw_value, str):
        raw_items = [raw_value]
    elif isinstance(raw_value, (list, tuple, set)):
        raw_items = list(raw_value)
    else:
        raw_items = [raw_value]
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        stage = str(raw_item or "").strip().lower().replace("_", "-")
        if not stage:
            continue
        aliases = {
            "step-0": "step0",
            "baseline": "step0",
            "save": "checkpoint",
            "save-hook": "checkpoint",
            "savehook": "checkpoint",
            "train-end": "final",
            "end": "final",
        }
        stage = aliases.get(stage, stage)
        if stage in {"all", "*"}:
            return ["step0", "checkpoint", "final"]
        if stage not in {"step0", "checkpoint", "final"}:
            raise ValueError(
                f"Unsupported held-out eval stage {raw_item!r}. Expected step0/checkpoint/final."
            )
        if stage not in seen:
            normalized.append(stage)
            seen.add(stage)
    return normalized or ["step0", "checkpoint"]


def _heldout_evals_for_stage(
    heldout_eval_specs: list[dict[str, Any]],
    *,
    stage: str,
) -> list[dict[str, Any]]:
    return [
        dict(eval_spec)
        for eval_spec in heldout_eval_specs
        if stage in _normalize_heldout_eval_stages(eval_spec.get("when"))
    ]


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=DEFAULT_KIMI_MODEL_NAME)
    model_path: str = field(
        default=None, metadata={"help": "Path to the pretrained model."}
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha."})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout."})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated LoRA target modules on the language stack."},
    )
    lora_include_audio_encoder: bool = field(
        default=False,
        metadata={
            "help": (
                "Also attach LoRA adapters to the Whisper audio encoder. "
                "False (the default) reproduces the paper: encoder fully frozen."
            )
        },
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_ratio: float = field(
        default=0.05, metadata={"help": "Ratio of evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    dataloader_pin_memory: bool = field(default=False)
    model_max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    memory_profile_steps: int = field(
        default=0,
        metadata={"help": "If >0, run this many train batches for GPU memory profiling."},
    )
    profile_only: bool = field(
        default=False,
        metadata={"help": "Run the memory profile and exit without starting Trainer.train()."},
    )
    run_name: str = field(default="", metadata={"help": "Persistent run name."})
    heldout_eval_manifest: str = field(
        default="",
        metadata={"help": "Path to the held-out eval manifest for save-time checkpoint scoring."},
    )
    heldout_eval_specs_json: str = field(
        default="",
        metadata={"help": "JSON list of held-out eval specs for save-time checkpoint scoring."},
    )
    pretrained_path: str = field(
        default="",
        metadata={"help": "Path to the pretrained checkpoint used for tokenizer/runtime assets."},
    )
    model_name_or_path_for_eval: str = field(
        default="",
        metadata={"help": "Base model repo/path to resolve Kimi tokenizer assets for save-time eval."},
    )


def maybe_zero_3(param):
    if hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


local_rank = None


def rank0_print(*args):
    if local_rank in (0, None):
        print(*args, flush=True)


def _flatten_row(row):
    return row.squeeze(0) if getattr(row, "ndim", 0) > 1 else row


def _pad_rows(rows, *, pad_value):
    flat_rows = [_flatten_row(row) for row in rows]
    max_len = max(int(row.shape[-1]) for row in flat_rows)
    batch = flat_rows[0].new_full((len(flat_rows), max_len), pad_value)
    for index, row in enumerate(flat_rows):
        batch[index, : row.shape[0]] = row
    return batch


def _collate_rad_batch(batch, *, pad_token: int) -> dict[str, Any]:
    return {
        "input_ids": _pad_rows([item["input_ids"] for item in batch], pad_value=pad_token),
        "text_input_ids": _pad_rows(
            [item["text_input_ids"] for item in batch], pad_value=pad_token
        ),
        "whisper_input_feature": [item["whisper_input_feature"] for item in batch],
        "is_continuous_mask": _pad_rows(
            [item["is_continuous_mask"] for item in batch], pad_value=False
        ),
        "labels": (
            _pad_rows([item["labels"][0] for item in batch], pad_value=pad_token),
            _pad_rows([item["labels"][1] for item in batch], pad_value=pad_token),
            _pad_rows([item["labels"][2] for item in batch], pad_value=False),
            _pad_rows([item["labels"][3] for item in batch], pad_value=False),
        ),
    }


def make_supervised_data_module(
    whisper_model, text_tokenizer, data_args, max_len, kimia_token_offset
) -> Dict:
    dataset_cls = LazySupervisedDataset
    rank0_print("Loading RAD data...")

    with open(data_args.data_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    all_data = [json.loads(line) for line in lines]

    if data_args.eval_ratio > 0:
        eval_data = all_data[: int(len(all_data) * data_args.eval_ratio)]
        train_data = all_data[int(len(all_data) * data_args.eval_ratio) :]
        assert len(eval_data) > 0, "No evaluation data found"
        assert len(train_data) > 0, "No training data found"
    else:
        eval_data = None
        train_data = all_data

    train_dataset = dataset_cls(
        train_data,
        whisper_model=whisper_model,
        text_tokenizer=text_tokenizer,
        max_len=max_len,
        kimia_token_offset=kimia_token_offset,
    )
    eval_dataset = (
        dataset_cls(
            eval_data,
            whisper_model=whisper_model,
            text_tokenizer=text_tokenizer,
            max_len=max_len,
            kimia_token_offset=kimia_token_offset,
        )
        if eval_data
        else None
    )

    def data_collator(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return _collate_rad_batch(batch, pad_token=train_dataset.pad_token)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def compute_loss(outputs, labels, num_items_in_batch=None):
    audio_logits, text_logits = outputs.logits
    audio_labels, text_labels, audio_loss_mask, text_loss_mask = labels
    audio_labels = audio_labels.to(audio_logits.device)
    text_labels = text_labels.to(text_logits.device)
    audio_loss_mask = audio_loss_mask.to(
        audio_logits.device, dtype=audio_logits.dtype
    )
    text_loss_mask = text_loss_mask.to(
        text_logits.device, dtype=text_logits.dtype
    )

    audio_loss = torch.nn.functional.cross_entropy(
        audio_logits.view(-1, audio_logits.shape[-1]),
        audio_labels.view(-1),
        reduction="none",
    )
    text_loss = torch.nn.functional.cross_entropy(
        text_logits.view(-1, text_logits.shape[-1]),
        text_labels.view(-1),
        reduction="none",
    )

    audio_loss = (audio_loss * audio_loss_mask.view(-1)).sum() / (
        audio_loss_mask.view(-1).sum() + 1e-4
    )
    text_loss = (text_loss * text_loss_mask.view(-1)).sum() / (
        text_loss_mask.view(-1).sum() + 1e-4
    )
    return audio_loss + text_loss


def _find_whisper_model(module):
    candidates = [
        module,
        getattr(module, "model", None),
        getattr(getattr(module, "base_model", None), "model", None),
    ]
    for candidate in candidates:
        whisper_model = getattr(candidate, "whisper_model", None)
        if whisper_model is not None:
            return whisper_model
    raise AttributeError("Expected whisper_model for frozen-audio LoRA training")


def _freeze_audio_encoder(module):
    whisper_model = _find_whisper_model(module)
    whisper_model.requires_grad_(False)
    for param in whisper_model.parameters():
        param.requires_grad = False
    whisper_model.to(dtype=torch.bfloat16)


def _cast_audio_encoder_bf16(module):
    """Cast the audio encoder to bf16 without freezing it.

    `_freeze_audio_encoder` does two things: it freezes the encoder and it casts
    it to bf16. The encoder-LoRA arm must not freeze it, but it must still be
    cast -- otherwise the encoder runs at its loaded dtype and precision becomes
    a second difference between the arms. The comparison is only meaningful if
    exactly one thing changes.
    """
    _find_whisper_model(module).to(dtype=torch.bfloat16)


def _freeze_base_model_for_lora(model):
    model.requires_grad_(False)
    _freeze_audio_encoder(model)


def _parse_lora_targets(raw_targets: str) -> list[str]:
    parts = [part.strip() for part in raw_targets.split(",") if part.strip()]
    return parts or list(DEFAULT_LORA_TARGET_MODULES)


def _resolve_language_stack_lora_targets(
    model,
    raw_targets: str,
    *,
    include_audio_encoder: bool = False,
) -> list[str]:
    """Resolve LoRA target module names.

    With `include_audio_encoder` set, Whisper encoder modules are resolved
    alongside the language stack using their own suffix list. Unset (the
    default) the behaviour -- including the leak guard -- is unchanged.
    """
    suffixes = _parse_lora_targets(raw_targets)
    resolved = []
    for module_name, _module in model.named_modules():
        if module_name.startswith(("model.layers.", "model.mimo_layers.")):
            if any(
                module_name == suffix or module_name.endswith(f".{suffix}")
                for suffix in suffixes
            ):
                resolved.append(module_name)
        elif include_audio_encoder and module_name.startswith("whisper_model."):
            if any(
                module_name == suffix or module_name.endswith(f".{suffix}")
                for suffix in WHISPER_ENCODER_LORA_MODULES
            ):
                resolved.append(module_name)
    resolved = sorted(set(resolved))
    if not resolved:
        raise RuntimeError(
            f"No language-stack LoRA targets resolved from suffix list: {suffixes}"
        )
    if not include_audio_encoder and any("whisper_model" in name for name in resolved):
        raise RuntimeError(
            f"Audio encoder leaked into LoRA target list: {resolved[:8]}"
        )
    if include_audio_encoder and not any("whisper_model" in n for n in resolved):
        # Guard against a silent no-op: if the encoder prefix matches nothing the
        # run would look fine while training exactly the frozen-encoder baseline.
        raise RuntimeError(
            "lora_include_audio_encoder was set but no whisper_model modules "
            "resolved; the encoder prefix or suffix list is wrong."
        )
    return resolved


def _assert_lora_constraints(model, *, include_audio_encoder: bool = False):
    """Check that only LoRA adapters are trainable.

    The "no trainable whisper_model params" assertion is skipped when encoder
    adapters were requested. The stricter invariant that *only* `.lora_` params
    are trainable still holds in both modes, so the encoder's base weights stay
    frozen either way: it is adapted, never unfrozen.
    """
    if not include_audio_encoder:
        bad_audio = [
            name
            for name, param in model.named_parameters()
            if param.requires_grad and "whisper_model" in name
        ]
        if bad_audio:
            raise RuntimeError(
                "Audio encoder / whisper_model unexpectedly trainable under LoRA: "
                + ", ".join(bad_audio[:16])
            )
    bad_non_lora = [
        name for name, param in model.named_parameters() if param.requires_grad and ".lora_" not in name
    ]
    if bad_non_lora:
        raise RuntimeError(
            "Non-LoRA weights unexpectedly trainable: " + ", ".join(bad_non_lora[:16])
        )


def _trainable_parameter_summary(model, *, include_audio_encoder: bool = False) -> dict:
    total = 0
    trainable = 0
    audio_trainable = 0
    trainable_names = []
    for name, param in model.named_parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
            if "whisper_model" in name:
                audio_trainable += count
            if len(trainable_names) < 64:
                trainable_names.append(name)
    return {
        "training_mode": (
            "lora-peft-decoder-and-audio-encoder"
            if include_audio_encoder
            else "lora-peft-frozen-audio"
        ),
        "audio_encoder_frozen": not include_audio_encoder,
        # Gate for the encoder-LoRA path: zero here with the flag set means the
        # run silently trained the baseline and its numbers must be discarded.
        "audio_trainable_params": audio_trainable,
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": round((trainable / total) * 100.0, 6) if total else None,
        "trainable_names_head": trainable_names,
    }


def _cuda_memory_stats() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {}
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "device_index": int(device),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "total_bytes": int(props.total_memory),
    }


def _run_memory_profile(
    *,
    model,
    train_dataset,
    data_collator,
    training_args,
) -> dict[str, Any] | None:
    profile_steps = max(0, int(training_args.memory_profile_steps))
    if profile_steps <= 0:
        return None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    batch_summaries = []
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(dataloader, start=1):
        if step > profile_steps:
            break
        labels = batch["labels"]
        autocast_kwargs = {
            "device_type": "cuda",
            "dtype": torch.bfloat16,
            "enabled": torch.cuda.is_available(),
        }
        with torch.autocast(**autocast_kwargs):
            outputs = model(
                input_ids=batch["input_ids"],
                text_input_ids=batch["text_input_ids"],
                whisper_input_feature=batch["whisper_input_feature"],
                is_continuous_mask=batch["is_continuous_mask"],
                labels=labels,
                use_cache=False,
                return_dict=True,
            )
            loss = compute_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        batch_summaries.append(
            {
                "step": step,
                "loss": round(float(loss.detach().cpu().item()), 6),
                **_cuda_memory_stats(),
            }
        )

    summary = {
        "profile_steps": profile_steps,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "model_max_length": training_args.model_max_length,
        "batch_summaries": batch_summaries,
        "final_memory": _cuda_memory_stats(),
    }
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(
        os.path.join(training_args.output_dir, "rad_memory_profile.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)
    rank0_print(json.dumps(summary, indent=2))
    return summary


def _save_lora_model(trainer: Trainer, tokenizer, output_dir: str) -> None:
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    os.makedirs(output_dir, exist_ok=True)
    state_dict = None
    if deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    adapter_state = get_peft_model_state_dict(model, state_dict=state_dict)
    if trainer.is_world_process_zero():
        model.save_pretrained(output_dir, state_dict=adapter_state)
        try:
            tokenizer.save_pretrained(output_dir)
        except Exception as exc:
            rank0_print(f"Skipping tokenizer.save_pretrained for LoRA adapter: {exc}")
        with open(
            os.path.join(output_dir, "rad_lora_summary.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(_trainable_parameter_summary(model), handle, indent=2)


class _CheckpointEvalCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if args.run_name:
            persist_active_wandb_run(run_name=args.run_name)
        return control

    def on_save(self, args, state, control, **kwargs):
        heldout_eval_specs = _heldout_evals_for_stage(
            _heldout_eval_specs_from_json(
                args.heldout_eval_specs_json,
                args.heldout_eval_manifest,
            ),
            stage="checkpoint",
        )
        if not heldout_eval_specs:
            return control
        model = kwargs.get("model")
        if model is None:
            raise RuntimeError("Checkpoint eval callback expected `model` in on_save kwargs")
        checkpoint_step = int(state.global_step or 0)
        if checkpoint_step <= 0:
            return control
        rank0_print(
            f"[checkpoint-eval] step={checkpoint_step} save hook fired; starting held-out eval"
        )
        if len(heldout_eval_specs) > 1:
            eval_result = run_combined_save_time_checkpoint_eval(
                model=model,
                pretrained_path=args.pretrained_path,
                model_name_or_path=args.model_name_or_path_for_eval,
                run_name=args.run_name,
                checkpoint_step=checkpoint_step,
                output_dir=args.output_dir,
                eval_specs=heldout_eval_specs,
            )
            eval_status = "ok" if eval_result is None else str(eval_result.get("status") or "ok")
            for per_eval_result in list((eval_result or {}).get("per_eval_results") or []):
                metrics_row = dict(per_eval_result.get("metrics_row") or {})
                resolved_tag = str(metrics_row.get("eval_tag") or "default")
                if eval_status == "ok":
                    rank0_print(
                        f"[checkpoint-eval] step={checkpoint_step} eval_tag={resolved_tag} save hook finished successfully"
                    )
                else:
                    rank0_print(
                        f"[checkpoint-eval] step={checkpoint_step} eval_tag={resolved_tag} save hook completed with status={eval_status}; training continuing"
                    )
            return control

        persisted_checkpoint_dir = None
        for index, eval_spec in enumerate(heldout_eval_specs):
            eval_result = run_save_time_checkpoint_eval(
                model=model,
                pretrained_path=args.pretrained_path,
                model_name_or_path=args.model_name_or_path_for_eval,
                run_name=args.run_name,
                checkpoint_step=checkpoint_step,
                output_dir=args.output_dir,
                heldout_eval_manifest=str(eval_spec.get("manifest") or "").strip(),
                eval_tag=str(eval_spec.get("tag") or "").strip() or None,
                wandb_metric_prefix=str(eval_spec.get("wandb_metric_prefix") or "").strip()
                or None,
                persisted_checkpoint_dir=persisted_checkpoint_dir,
                persist_checkpoint=index == 0,
            )
            if persisted_checkpoint_dir is None and eval_result is not None:
                persisted_checkpoint_dir = eval_result.get("persisted_checkpoint_dir")
            eval_status = "ok" if eval_result is None else str(eval_result.get("status") or "ok")
            resolved_tag = str(eval_spec.get("tag") or "").strip() or "default"
            if eval_status == "ok":
                rank0_print(
                    f"[checkpoint-eval] step={checkpoint_step} eval_tag={resolved_tag} save hook finished successfully"
                )
            else:
                rank0_print(
                    f"[checkpoint-eval] step={checkpoint_step} eval_tag={resolved_tag} save hook completed with status={eval_status}; training continuing"
                )
        return control


def _run_final_heldout_eval(
    *,
    trainer: Trainer,
    training_args: TrainingArguments,
    heldout_eval_specs: list[dict[str, Any]],
) -> None:
    final_eval_specs = _heldout_evals_for_stage(heldout_eval_specs, stage="final")
    if not final_eval_specs:
        return
    checkpoint_step = int(getattr(trainer.state, "global_step", 0) or 0)
    if checkpoint_step <= 0:
        return
    model = trainer.model
    persisted_checkpoint_dir = None
    checkpoint_dir = os.path.join(training_args.output_dir, f"checkpoint-{checkpoint_step}")
    if not os.path.isdir(checkpoint_dir):
        checkpoint_dir = training_args.output_dir
    rank0_print(
        f"[checkpoint-eval] step={checkpoint_step} train complete; starting final held-out eval"
    )
    if len(final_eval_specs) > 1:
        eval_result = run_combined_save_time_checkpoint_eval(
            model=model,
            pretrained_path=training_args.pretrained_path,
            model_name_or_path=training_args.model_name_or_path_for_eval,
            run_name=training_args.run_name,
            checkpoint_step=checkpoint_step,
            output_dir=training_args.output_dir,
            eval_specs=final_eval_specs,
            checkpoint_dir_override=checkpoint_dir,
            checkpoint_name_override=f"final-step-{checkpoint_step}",
            combined_eval_name="combined-full-heldout-gpt54mini",
            combined_eval_tag="combined-full-heldout",
            combined_wandb_metric_prefix="checkpoint_eval/combined-heldout/full",
            persist_checkpoint=False,
        )
        eval_status = "ok" if eval_result is None else str(eval_result.get("status") or "ok")
        for per_eval_result in list((eval_result or {}).get("per_eval_results") or []):
            metrics_row = dict(per_eval_result.get("metrics_row") or {})
            resolved_tag = str(metrics_row.get("eval_tag") or "default")
            if eval_status == "ok":
                rank0_print(
                    f"[checkpoint-eval] step={checkpoint_step} eval_tag={resolved_tag} final eval finished successfully"
                )
            else:
                rank0_print(
                    f"[checkpoint-eval] step={checkpoint_step} eval_tag={resolved_tag} final eval completed with status={eval_status}"
                )
        return

    for index, eval_spec in enumerate(final_eval_specs):
        eval_result = run_save_time_checkpoint_eval(
            model=model,
            pretrained_path=training_args.pretrained_path,
            model_name_or_path=training_args.model_name_or_path_for_eval,
            run_name=training_args.run_name,
            checkpoint_step=checkpoint_step,
            output_dir=training_args.output_dir,
            heldout_eval_manifest=str(eval_spec.get("manifest") or "").strip(),
            checkpoint_dir_override=checkpoint_dir,
            checkpoint_name_override=f"final-step-{checkpoint_step}",
            eval_tag=str(eval_spec.get("tag") or "").strip() or None,
            wandb_metric_prefix=str(eval_spec.get("wandb_metric_prefix") or "").strip() or None,
            persisted_checkpoint_dir=persisted_checkpoint_dir,
            persist_checkpoint=False,
        )
        if persisted_checkpoint_dir is None and eval_result is not None:
            persisted_checkpoint_dir = eval_result.get("persisted_checkpoint_dir")
        eval_status = "ok" if eval_result is None else str(eval_result.get("status") or "ok")
        resolved_tag = str(eval_spec.get("tag") or "").strip() or "default"
        if eval_status == "ok":
            rank0_print(
                f"[checkpoint-eval] step={checkpoint_step} eval_tag={resolved_tag} final eval finished successfully"
            )
        else:
            rank0_print(
                f"[checkpoint-eval] step={checkpoint_step} eval_tag={resolved_tag} final eval completed with status={eval_status}"
            )


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.model_name_or_path = _resolve_supported_model_name_or_path(
        model_args.model_name_or_path
    )
    if training_args.model_name_or_path_for_eval:
        training_args.model_name_or_path_for_eval = _resolve_supported_model_name_or_path(
            training_args.model_name_or_path_for_eval
        )

    if getattr(training_args, "deepspeed", None) and int(os.environ.get("WORLD_SIZE", 1)) == 1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    model_load_kwargs = {
        "low_cpu_mem_usage": not deepspeed.is_deepspeed_zero3_enabled(),
    }
    if os.path.exists(model_args.model_name_or_path):
        cache_path = model_args.model_name_or_path
    else:
        cache_path = snapshot_download(model_args.model_name_or_path)

    if not os.path.exists(model_args.model_path):
        raise ValueError(f"Model path {model_args.model_path} does not exist")

    rank0_print("Loading Kimi-Audio base model for RAD LoRA training")
    base_model = KimiAudioModel.from_pretrained(
        model_args.model_path,
        device_map=None,
        **model_load_kwargs,
    )
    base_model.config.use_cache = False

    include_audio = bool(model_args.lora_include_audio_encoder)
    if include_audio:
        # Freeze everything and let PEFT add adapters, including on the encoder.
        # The encoder is still cast to bf16, exactly as the baseline casts it, so
        # dtype does not become a second difference between the arms.
        base_model.requires_grad_(False)
        _cast_audio_encoder_bf16(base_model)
    else:
        _freeze_base_model_for_lora(base_model)

    if training_args.gradient_checkpointing and hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    target_modules = _resolve_language_stack_lora_targets(
        base_model,
        model_args.lora_target_modules,
        include_audio_encoder=include_audio,
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, peft_config)
    if not include_audio:
        _freeze_audio_encoder(model)
    _assert_lora_constraints(model, include_audio_encoder=include_audio)
    if torch.cuda.is_available():
        model.to(training_args.device)

    summary = _trainable_parameter_summary(model, include_audio_encoder=include_audio)
    rank0_print(
        json.dumps(
            {
                **summary,
                "lora_r": model_args.lora_r,
                "lora_alpha": model_args.lora_alpha,
                "lora_dropout": model_args.lora_dropout,
                "lora_include_audio_encoder": include_audio,
                "lora_encoder_target_count": sum(
                    1 for n in target_modules if "whisper_model" in n
                ),
                "lora_target_modules": target_modules,
            },
            indent=2,
        )
    )
    if include_audio and summary["audio_trainable_params"] == 0:
        raise RuntimeError(
            "lora_include_audio_encoder was set but no audio-encoder parameters "
            "are trainable; this run would silently reproduce the baseline."
        )
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    text_tokenizer = AutoTokenizer.from_pretrained(cache_path, trust_remote_code=True)
    data_module = make_supervised_data_module(
        whisper_model=_find_whisper_model(model),
        text_tokenizer=text_tokenizer,
        data_args=data_args,
        max_len=training_args.model_max_length,
        kimia_token_offset=model.config.kimia_token_offset,
    )
    data_collator = data_module["data_collator"]
    _run_memory_profile(
        model=model,
        train_dataset=data_module["train_dataset"],
        data_collator=data_collator,
        training_args=training_args,
    )
    if training_args.profile_only:
        return

    trainer = Trainer(
        model=model,
        args=training_args,
        compute_loss_func=compute_loss,
        data_collator=data_collator,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        callbacks=[_CheckpointEvalCallback()],
    )

    heldout_eval_specs = _heldout_eval_specs_from_json(
        training_args.heldout_eval_specs_json,
        training_args.heldout_eval_manifest,
    )
    trainer.train()
    trainer.save_state()
    _save_lora_model(trainer, text_tokenizer, training_args.output_dir)
    _run_final_heldout_eval(
        trainer=trainer,
        training_args=training_args,
        heldout_eval_specs=heldout_eval_specs,
    )


if __name__ == "__main__":
    train()
