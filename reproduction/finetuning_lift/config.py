"""Declarative specification for the Table 2 reproduction and encoder-LoRA ablation.

Everything that defines an experiment lives here as data. No other module in this
package hardcodes a hyperparameter, a split, or a per-task branch.

Data uses the AcousticBench release schema: a
self-contained directory of question JSONL plus mirrored audio, carrying its own
`split` field. There is no database and no S3 access anywhere in this package.
The release's eval-split sizes match Table 2's reported `n` exactly:

    RAD 600    SER-Classify 300    SER-Intensity 200    RSQ 500

Research-tool configuration; see docs/reproduction.md for provenance and scope:
  - Stored paper scores are comparison targets, not proof of run provenance.

The two arms differ by EXACTLY ONE field: `lora_include_audio_encoder`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------
# Release location
#
# The release is ~5.2 GB, almost all audio, so it is referenced in place rather
# than copied into the repo. Override with ACOUSTICBENCH_DATA.
# --------------------------------------------------------------------------

RELEASE_ROOT = Path(
    os.environ.get("ACOUSTICBENCH_DATA")
    or REPO_ROOT / "archive"
).expanduser()


def questions_path(slice_name: str) -> Path:
    return RELEASE_ROOT / "questions" / f"acousticbench_{slice_name}.jsonl"


# --------------------------------------------------------------------------
# Local research recipe. Shared by all four specialists and both arms.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    """Local fixed-step research settings, not a verified record of submitted runs."""

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"

    # "standard attention & MLP projections of the decoder"
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    weight_decay: float = 0.1
    learning_rate: float = 5e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03

    precision: str = "bf16"
    gradient_checkpointing: bool = True

    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    model_max_length: int = 512

    logging_steps: int = 1

    # NOTE (documentation vs implementation) -- training length:
    #
    # Appendix K states runs stop "once the logged training loss stays at or
    # below 1e-4 for 50 consecutive logging events". No such rule exists in the
    # codebase: `finetuning/` contains no early-stopping callback, and the only
    # callback (`_CheckpointEvalCallback`) never inspects loss nor sets
    # `control.should_training_stop`. Length is set entirely by `--max_steps`.
    #
    # Rather than guess the paper's step count, every run saves every
    # `save_steps` and `evaluate.py` scores each checkpoint. That yields the
    # paper's step for free and shows the surrounding curve, so a reproduction
    # gap can be attributed to step selection rather than confounded with it.
    save_steps: int = 50

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


RECIPE = Recipe()

# The paper reports means over 3 runs with +/-2 sample SD error bars (Appendix L).
SEEDS: tuple[int, ...] = (1, 2, 3)


# --------------------------------------------------------------------------
# Models
#
# The experiment is model x arm x task. Kimi-Audio is the paper's model; a
# second model tests whether the encoder finding is architecture-specific or
# general.
#
# Both models share the shape "audio encoder feeding a language decoder, with a
# speech-generation head we never use" -- Kimi's MIMO audio decoder and Qwen's
# Talker are both dead weight for text-output MCQA and are disabled.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Model:
    key: str
    label: str
    hf_checkpoint: str

    # Modal app/class serving the un-finetuned model.
    serving_app: str
    serving_class: str

    # Modal app/class serving a trained checkpoint. `ft_serving_class` takes a
    # `model_path` parameter pointing at a merged checkpoint on the volume.
    ft_serving_app: str | None
    ft_serving_class: str | None

    # Modal parameter name the FT serving class expects. Kimi is served from a
    # merged checkpoint (`model_path`); Qwen is served as base + LoRA adapter
    # (`adapter_path`), because merging a 30B model for each of 8 checkpoints
    # would write ~480GB to encode ~50MB of difference per arm.
    ft_path_param: str

    # Which training backend this model uses. They are not interchangeable:
    # `kimi-rad` is the paper's launcher and is Kimi-specific end to end
    # (semantic-code extraction, Whisper features, its own entrypoint).
    trainer: str

    description: str


KIMI = Model(
    key="kimi",
    label="Kimi-Audio-7B",
    hf_checkpoint="moonshotai/Kimi-Audio-7B-Instruct",
    serving_app="audiobench-kimi-audio",
    serving_class="KimiAudioServer",
    ft_serving_app="audiobench-kimi-audio-ft",
    ft_serving_class="KimiAudioFtServer",
    ft_path_param="model_path",
    trainer="kimi-rad",
    description=(
        "The paper's model. Whisper Large v3 audio encoder + Qwen2-7B decoder. "
        "Dense 7B. The MIMO audio decoder is unused."
    ),
)

QWEN3_OMNI = Model(
    key="qwen3omni",
    label="Qwen3-Omni-30B-A3B",
    hf_checkpoint="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    serving_app="audiobench-qwen3-omni",
    serving_class="Qwen3OmniServer",
    ft_serving_app="audiobench-qwen3-omni-ft",
    ft_serving_class="Qwen3OmniFtServer",
    ft_path_param="adapter_path",
    trainer="ms-swift",
    description=(
        "Second architecture. AuT audio encoder + MoE Thinker (30B total, ~3B "
        "active). The Talker is disabled: text-output MCQA never uses it."
    ),
)

MODELS: dict[str, Model] = {model.key: model for model in (KIMI, QWEN3_OMNI)}

# Kimi is the paper's model, so it stays the default for every entry point.
DEFAULT_MODEL = KIMI.key


# --------------------------------------------------------------------------
# Arms - the single experimental variable
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    key: str
    label: str
    lora_include_audio_encoder: bool
    description: str


# Nicknames. The three models under comparison are STOCK (untrained), READOUT
# (decoder LoRA) and REWIRE (decoder + encoder LoRA). They name the hypothesis:
# READOUT learns to read out a fixed audio representation, REWIRE changes the
# representation itself. Used in every table and log line, because `base` and
# `baseline` differ by two letters and mean entirely different models.
#
# The CLI keys stay `base` / `baseline` / `encoder_lora`: checkpoints on the
# Modal volume are named after them (`rad-baseline-seed1`), so renaming the keys
# would orphan existing runs.

BASELINE = Arm(
    key="baseline",
    label="READOUT (decoder LoRA, frozen encoder)",
    lora_include_audio_encoder=False,
    description=(
        "Reproduction of the submitted paper. Audio encoder fully frozen: no "
        "adapter, no gradient. LoRA on the language decoder only."
    ),
)

ENCODER_LORA = Arm(
    key="encoder_lora",
    label="REWIRE (decoder + audio-encoder LoRA)",
    lora_include_audio_encoder=True,
    description=(
        "Ablation. Identical data and recipe; LoRA adapters additionally "
        "attached to the Whisper audio encoder (192 modules)."
    ),
)

ARMS: dict[str, Arm] = {arm.key: arm for arm in (BASELINE, ENCODER_LORA)}


# --------------------------------------------------------------------------
# Tasks
#
# `slice_name` selects the release JSONL and matches `question_set.slice` on
# every record. Train/eval membership comes from each record's `split` field,
# so no split logic lives in this package.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    key: str
    label: str
    slice_name: str
    train_count: int
    eval_count: int
    category: str
    benchmark_name: str
    clips_per_question: int

    # Fixed training length (see the note on Recipe about Appendix K).
    max_steps: int

    # Table 2 targets, as (mean, two_sigma). Table 2 reports no error bar for
    # values that were identical across all three runs.
    paper_base: tuple[float, float]
    paper_lora: tuple[float, float]
    paper_lift: tuple[float, float]
    human: float

    @property
    def bundle_name(self) -> str:
        return f"{self.key}-{self.train_count}"


TASKS: dict[str, Task] = {
    "rad": Task(
        key="rad",
        label="RAD",
        slice_name="rad",
        train_count=600,
        eval_count=600,
        category="acoustic-perception",
        benchmark_name="AcousticPerceptionBench",
        clips_per_question=2,
        max_steps=1000,
        paper_base=(69.2, 0.0),
        paper_lora=(98.7, 0.0),
        paper_lift=(29.5, 0.0),
        human=99.0,
    ),
    "ser_classify": Task(
        key="ser_classify",
        label="SER-Classify",
        slice_name="ser_classify",
        train_count=300,
        eval_count=300,
        category="emotion-recognition",
        benchmark_name="EmotionRecognitionBench",
        clips_per_question=1,
        max_steps=1000,
        paper_base=(70.3, 0.7),
        paper_lora=(75.2, 0.4),
        paper_lift=(4.9, 0.4),
        human=87.1,
    ),
    "ser_intensity": Task(
        key="ser_intensity",
        label="SER-Intensity",
        slice_name="ser_intensity",
        train_count=200,
        eval_count=200,
        category="emotion-recognition",
        benchmark_name="EmotionRecognitionBench",
        clips_per_question=2,
        max_steps=1000,
        paper_base=(72.3, 0.6),
        paper_lora=(92.5, 0.0),
        paper_lift=(20.2, 0.6),
        human=91.4,
    ),
    "rsq": Task(
        key="rsq",
        label="RSQ",
        slice_name="rsq",
        train_count=500,
        eval_count=500,
        category="speech-quality",
        benchmark_name="SpeechQualityBench",
        clips_per_question=2,
        max_steps=1000,
        paper_base=(54.8, 0.8),
        paper_lora=(98.4, 0.0),
        paper_lift=(43.6, 0.8),
        human=93.6,
    ),
}

# The release also ships `ser_classify_text`, the transcript-only SER control
# (300/300). It has no Table 2 row and no audio, so it is not a task here.
RELEASE_ONLY_SLICES: tuple[str, ...] = ("ser_classify_text",)


# --------------------------------------------------------------------------
# Modal / storage layout
# --------------------------------------------------------------------------

MODAL_ENV = os.environ.get("MODAL_ENVIRONMENT", "acousticbench")
VOLUME_NAME = "acousticbench-kimi-rad-data"
HF_SECRET_NAME = "huggingface-secret"
BASE_MODEL = "moonshotai/Kimi-Audio-7B-Instruct"

# Materialized once by the existing `prepare-model` step; shared by all runs.
PRETRAINED_PATH = "/data/output/pretrained_hf"

# Where the served model reads audio from inside the training volume.
VOLUME_AUDIO_ROOT = "/data/audio"

# ms-swift bundles live on their own volume, mounted at /data in the training
# container. Clip paths are baked into the JSONL, so this template must match
# where `modal volume put` actually places them: /bundles/<name> on the volume,
# which appears as /data/bundles/<name> in the container.
#
# Getting this wrong is not obvious from the failure. swift reports a missing
# audio file as "Failed to retrieve the dataset. You can avoid this issue by
# increasing `max_length`", with the real FileNotFoundError a hundred lines up.
SWIFT_VOLUME_NAME = "acousticbench-swift-data"
SWIFT_VOLUME_AUDIO_ROOT_TEMPLATE = "/data/bundles/{bundle_name}/audio"

# ms-swift writes LoRA adapters here; served directly rather than merged.
SWIFT_ADAPTER_PATH_TEMPLATE = "/data/runs/{run_name}/checkpoint-{checkpoint}"

LOCAL_BUNDLE_ROOT = REPO_ROOT / "reproduction/finetuning_lift/data"
RESULTS_ROOT = REPO_ROOT / "reproduction/finetuning_lift/results"

# The base (un-finetuned) model, evaluated to reproduce Table 2's Base column.
BASE_ARM_KEY = "base"

# The launcher wires each bundle's held-out manifest into an in-training eval
# that fires at step 0 and at every checkpoint. We turn it off.
#
# It is not merely redundant with `evaluate.py`, it is broken: removing W&B left
# the finalize path asserting an active W&B run, so each pass runs inference,
# pays for a GPT judge, raises "Expected an active W&B run", and discards the
# result. Non-fatal, but ~15 wasted evals per run.
#
# `[]` is honoured by the launcher as an explicit disable (an unset value falls
# back to the bundle's manifest). Scoring is `evaluate.py`, on the full split,
# after training.
HELDOUT_EVALS_JSON = "[]"

# A run is uniquely identified by (task, arm, seed).
RUN_NAME_TEMPLATE = "{task}-{arm}-seed{seed}"


def run_name(task_key: str, arm_key: str, seed: int) -> str:
    return RUN_NAME_TEMPLATE.format(task=task_key, arm=arm_key, seed=seed)


def all_runs() -> list[tuple[Task, Arm, int]]:
    """Every (task, arm, seed) combination, in a stable order."""
    return [
        (task, arm, seed)
        for task in TASKS.values()
        for arm in ARMS.values()
        for seed in SEEDS
    ]


def recipe_for(arm: Arm, seed: int) -> Recipe:
    """The recipe is identical across arms; only the arm flag and seed vary.

    Kept as a function so that any future arm-specific override is forced to be
    explicit and visible here rather than scattered through the runner.
    """
    del arm, seed  # no arm- or seed-dependent hyperparameters by design
    return replace(RECIPE)
