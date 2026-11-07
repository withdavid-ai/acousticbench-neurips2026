import argparse
import json
import os
import warnings

import modal
import torch

import finetune
from transformers import TrainerCallback

from finetuning.kimi_audio.vendor_kimi.finetune_codes.model import KimiAudioModel
from finetuning.kimi_audio.rad.direct_checkpoint_eval import (
    persist_active_wandb_run,
    run_combined_save_time_checkpoint_eval,
    run_save_time_checkpoint_eval,
)

_ORIG_FROM_PRETRAINED = KimiAudioModel.from_pretrained
_VOLUME = modal.Volume.from_name("acousticbench-kimi-rad-data", create_if_missing=True)
_ORIG_FINETUNE_MODEL = finetune.KimiAudioModel
warnings.filterwarnings(
    "ignore",
    message=r".*copying from a non-meta parameter in the checkpoint to a meta parameter.*",
    category=UserWarning,
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _training_mode() -> str:
    return str(os.environ.get("AUDIOBENCH_TRAINING_MODE") or "stock-full-ft").strip()


def _audio_encoder_frozen() -> bool:
    return _env_flag("AUDIOBENCH_AUDIO_ENCODER_FROZEN", default=False)


def _heldout_eval_specs_from_env() -> list[dict[str, object]]:
    raw_json = str(os.environ.get("AUDIOBENCH_HELDOUT_EVAL_SPECS_JSON") or "").strip()
    if raw_json:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, list):
            raise ValueError("AUDIOBENCH_HELDOUT_EVAL_SPECS_JSON must decode to a list")
        return [dict(item) for item in parsed if isinstance(item, dict)]
    manifest = str(os.environ.get("AUDIOBENCH_HELDOUT_EVAL_MANIFEST", "")).strip()
    if not manifest:
        return []
    return [{"tag": "", "manifest": manifest, "wandb_metric_prefix": ""}]


def _normalize_heldout_eval_stages(raw_value: object) -> list[str]:
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
    heldout_eval_specs: list[dict[str, object]],
    *,
    stage: str,
) -> list[dict[str, object]]:
    return [
        dict(eval_spec)
        for eval_spec in heldout_eval_specs
        if stage in _normalize_heldout_eval_stages(eval_spec.get("when"))
    ]


def _trainable_parameter_summary(model):
    total = 0
    trainable = 0
    audio_trainable = 0
    trainable_names = []
    for name, param in model.named_parameters():
        count = int(param.numel())
        total += count
        if param.requires_grad:
            trainable += count
            if "whisper_model" in name:
                audio_trainable += count
            if len(trainable_names) < 64:
                trainable_names.append(name)
    return {
        "training_mode": _training_mode(),
        "audio_encoder_frozen": _audio_encoder_frozen(),
        "total_params": total,
        "trainable_params": trainable,
        "audio_trainable_params": audio_trainable,
        "trainable_pct": round((trainable / total) * 100.0, 6) if total else None,
        "trainable_names_head": trainable_names,
    }


def _maybe_patch_frozen_audio_from_pretrained() -> None:
    if not _audio_encoder_frozen():
        return

    @classmethod
    def _frozen_audio_from_pretrained(cls, *args, **kwargs):
        model = _ORIG_FROM_PRETRAINED(*args, **kwargs)
        whisper_model = getattr(model, "whisper_model", None)
        if whisper_model is None:
            raise AttributeError("Expected whisper_model on KimiAudioModel")
        whisper_model.requires_grad_(False)
        for param in whisper_model.parameters():
            param.requires_grad = False
        whisper_model.to(dtype=torch.bfloat16)
        finetune.rank0_print(json.dumps(_trainable_parameter_summary(model), indent=2))
        return model

    KimiAudioModel.from_pretrained = _frozen_audio_from_pretrained


class _CheckpointVolumeCommitCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        run_name = os.environ.get("AUDIOBENCH_RUN_NAME", "").strip()
        if run_name:
            persist_active_wandb_run(run_name=run_name)
        return control

    def on_save(self, args, state, control, **kwargs):
        heldout_eval_specs = _heldout_evals_for_stage(
            _heldout_eval_specs_from_env(),
            stage="checkpoint",
        )
        run_name = os.environ.get("AUDIOBENCH_RUN_NAME", "").strip()
        if heldout_eval_specs and run_name:
            model = kwargs.get("model")
            if model is None:
                raise RuntimeError("Checkpoint eval callback expected `model` in on_save kwargs")
            checkpoint_step = int(state.global_step or 0)
            if checkpoint_step > 0:
                print(
                    f"[checkpoint-eval] step={checkpoint_step} save hook fired; starting held-out eval",
                    flush=True,
                )
                if len(heldout_eval_specs) > 1:
                    eval_result = run_combined_save_time_checkpoint_eval(
                        model=model,
                        pretrained_path=os.environ.get(
                            "AUDIOBENCH_PRETRAINED_PATH", ""
                        ).strip(),
                        model_name_or_path=os.environ.get(
                            "AUDIOBENCH_MODEL_NAME_OR_PATH", ""
                        ).strip(),
                        run_name=run_name,
                        checkpoint_step=checkpoint_step,
                        output_dir=args.output_dir,
                        eval_specs=heldout_eval_specs,
                    )
                    eval_status = (
                        "ok" if eval_result is None else str(eval_result.get("status") or "ok")
                    )
                    for per_eval_result in list((eval_result or {}).get("per_eval_results") or []):
                        metrics_row = dict(per_eval_result.get("metrics_row") or {})
                        resolved_tag = str(metrics_row.get("eval_tag") or "default")
                        if eval_status == "ok":
                            print(
                                f"[checkpoint-eval] step={checkpoint_step} eval_tag={resolved_tag} save hook finished successfully",
                                flush=True,
                            )
                        else:
                            print(
                                f"[checkpoint-eval] step={checkpoint_step} eval_tag={resolved_tag} save hook completed with status={eval_status}; training continuing",
                                flush=True,
                            )
                    _VOLUME.commit()
                    return control

                persisted_checkpoint_dir = None
                for index, eval_spec in enumerate(heldout_eval_specs):
                    eval_result = run_save_time_checkpoint_eval(
                        model=model,
                        pretrained_path=os.environ.get("AUDIOBENCH_PRETRAINED_PATH", "").strip(),
                        model_name_or_path=os.environ.get(
                            "AUDIOBENCH_MODEL_NAME_OR_PATH", ""
                        ).strip(),
                        run_name=run_name,
                        checkpoint_step=checkpoint_step,
                        output_dir=args.output_dir,
                        heldout_eval_manifest=str(eval_spec.get("manifest") or "").strip(),
                        eval_tag=str(eval_spec.get("tag") or "").strip() or None,
                        wandb_metric_prefix=str(
                            eval_spec.get("wandb_metric_prefix") or ""
                        ).strip()
                        or None,
                        persisted_checkpoint_dir=persisted_checkpoint_dir,
                        persist_checkpoint=index == 0,
                    )
                    if persisted_checkpoint_dir is None and eval_result is not None:
                        persisted_checkpoint_dir = eval_result.get("persisted_checkpoint_dir")
                    eval_status = (
                        "ok" if eval_result is None else str(eval_result.get("status") or "ok")
                    )
                    resolved_tag = str(eval_spec.get("tag") or "").strip() or "default"
                    if eval_status == "ok":
                        print(
                            f"[checkpoint-eval] step={checkpoint_step} eval_tag={resolved_tag} save hook finished successfully",
                            flush=True,
                        )
                    else:
                        print(
                            f"[checkpoint-eval] step={checkpoint_step} eval_tag={resolved_tag} save hook completed with status={eval_status}; training continuing",
                            flush=True,
                        )
        _VOLUME.commit()
        return control


_ORIG_TRAIN = finetune.train


def _train_with_volume_checkpoint_commits(*args, **kwargs):
    orig_trainer_cls = finetune.Trainer
    finetune.KimiAudioModel = KimiAudioModel

    class _TrainerWithCommit(orig_trainer_cls):
        def __init__(self, *trainer_args, **trainer_kwargs):
            callbacks = list(trainer_kwargs.pop("callbacks", []) or [])
            callbacks.append(_CheckpointVolumeCommitCallback())
            trainer_kwargs["callbacks"] = callbacks
            super().__init__(*trainer_args, **trainer_kwargs)

    finetune.Trainer = _TrainerWithCommit
    try:
        return _ORIG_TRAIN(*args, **kwargs)
    finally:
        finetune.Trainer = orig_trainer_cls
        finetune.KimiAudioModel = _ORIG_FINETUNE_MODEL


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--freeze-audio-encoder", action="store_true")
    args, remaining = parser.parse_known_args()
    if args.freeze_audio_encoder:
        os.environ["AUDIOBENCH_AUDIO_ENCODER_FROZEN"] = "1"

    _maybe_patch_frozen_audio_from_pretrained()
    finetune.train = _train_with_volume_checkpoint_commits

    import sys

    sys.argv = [sys.argv[0], *remaining]
    finetune.train()


if __name__ == "__main__":
    main()
