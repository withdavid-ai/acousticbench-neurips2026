"""Qwen3-Omni LoRA training on Modal via ms-swift.

The paper's launcher cannot train Qwen: it is Kimi-specific end to end (semantic
code extraction, Whisper feature handling, its own entrypoint). ms-swift supports
Qwen3-Omni with LoRA and exposes the switch this experiment needs.

The arm mapping:

    READOUT  192 targets   48 decoder layers x {q,k,v,o_proj}
    REWIRE   320 targets   the same 192, plus 128 =
                           32 audio-encoder layers x {q,k,v,out_proj}

REWIRE's target set is a strict superset of READOUT's, so the arms differ by
exactly the audio encoder.

**`freeze_vit` does NOT separate the arms**, despite being ms-swift's name for the
visual/audio tower and reading like the obvious switch. It governs base-weight
freezing for full finetuning and has no say over which modules PEFT wraps. With
suffix-style targets (`--target_modules q_proj k_proj v_proj o_proj`) those names
match everywhere -- decoder, audio tower, talker, code2wav vocoder -- and reading
the saved adapters showed both arms with 192 audio_tower tensors, an identical
21.9546M trainable count, and the same train_loss. The two arms were one model.
`--target_regex` cannot be used instead: in 4.4.2 the regex lands in
`target_modules` and is iterated character by character.

Hence fully-qualified module names, and hence the checks at the end of `train`,
which read the saved adapter rather than the log: the run fails if REWIRE produced
no `thinker.audio_tower` adapters, or if READOUT produced any. Grepping stdout is
not enough -- it cannot tell "the encoder was not adapted" from "the log never
named the modules". `freeze_vit` is still passed, so its value is recorded
alongside each run, but nothing depends on it.

`freeze_aligner` stays True in both arms. The aligner projects audio into the
decoder's token space; leaving it frozen keeps the audio encoder as the single
variable, which is the whole point of the comparison.

Steps:
    probe          introspect module names and trainable params for both arms
    upload-bundle  sync a local ms-swift bundle to the training volume
    train          run `swift sft` for one (task, arm, seed)

Usage:
    modal run reproduction/finetuning_lift/swift_train.py::main --step probe
    modal run reproduction/finetuning_lift/swift_train.py::main \
        --step upload-bundle --bundle-dir reproduction/finetuning_lift/data/rad-600-swift
    modal run reproduction/finetuning_lift/swift_train.py::main \
        --step train --task rad --arm baseline --seed 1
"""

from __future__ import annotations

import modal

APP_NAME = "acousticbench-qwen3-omni-swift"
MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

DATA_MOUNT = "/data"
# Mounted away from /root/.cache: Modal's pip_install runs uv, which writes
# /root/.cache/uv, and a volume cannot mount over a non-empty path. See the
# comment in harness/modal_serving/qwen3_omni_serve.py.
HF_CACHE_MOUNT = "/hf-cache"
HF_HOME = f"{HF_CACHE_MOUNT}/huggingface"

app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name(
    "acousticbench-swift-data", create_if_missing=True
)
hf_cache = modal.Volume.from_name("audiobench-hf-cache", create_if_missing=True)

HF_ENV = {
    # ms-swift resolves model IDs through ModelScope by default. Left alone it
    # ignores the HF cache entirely and re-downloads all 15 shards (~75GB) to
    # /root/.cache/modelscope, which is not a mounted volume -- so every run
    # would pay it again (~600GB across both arms). USE_HF=1 sends it to the
    # HuggingFace hub, where the weights are already cached on the volume.
    "USE_HF": "1",
    "HF_HOME": HF_HOME,
    "HF_HUB_CACHE": f"{HF_HOME}/hub",
    "HUGGINGFACE_HUB_CACHE": f"{HF_HOME}/hub",
    "TRANSFORMERS_CACHE": f"{HF_HOME}/hub",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "libsndfile1")
    .pip_install(
        "torch==2.11.0",
        "torchaudio==2.11.0",
        "torchvision==0.26.0",
        "transformers==5.3.0",
        "accelerate",
        "peft",
        # deepspeed deliberately NOT installed. Its presence makes swift reach
        # for fused CUDA ops, which need a full CUDA toolchain in the image:
        #   MissingCUDAException: CUDA_HOME does not exist, unable to compile
        #   CUDA op(s)
        # It buys nothing here. Only the LoRA parameters train (~22M of 35B), so
        # a single H200 holds the model plus optimizer state without ZeRO.
        "librosa",
        "soundfile",
        "av",
        "qwen-omni-utils[decord]==0.0.9",
        "ms-swift",
    )
    .env(HF_ENV)
)

hf_secrets = [modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])]

# 30B MoE: ~60GB of BF16 weights before optimizer state and activations. LoRA
# keeps gradients and optimizer state small, and gradient checkpointing keeps
# activations down, so one H200 (141GB) is enough.
TRAIN_GPU = "H200"


# --------------------------------------------------------------------------
# probe -- learn the real module layout before training anything
# --------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={HF_CACHE_MOUNT: hf_cache},
    secrets=hf_secrets,
    timeout=3600,
)
def probe() -> dict:
    """Report Qwen3-Omni's module layout: which prefix owns the audio encoder.

    Built on the meta device from config alone, so no weights are read and no GPU
    is needed -- module *names* are all that is in question here.

    The point is to locate the audio tower by inspection rather than by guessing
    from another model's naming. ms-swift's `freeze_vit` claims to control it,
    and this is what lets that claim be checked against the parameter counts a
    real run reports.
    """
    import collections

    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen3OmniMoeForConditionalGeneration

    config = AutoConfig.from_pretrained(MODEL_ID)
    with init_empty_weights():
        model = Qwen3OmniMoeForConditionalGeneration._from_config(config)

    linear_names = [
        name
        for name, module in model.named_modules()
        if module.__class__.__name__.startswith("Linear")
    ]

    # Group by the two-level prefix, which is where thinker/talker splits into
    # audio tower vs vision tower vs language model.
    by_prefix: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    examples: dict[str, str] = {}
    for name in linear_names:
        parts = name.split(".")
        prefix = ".".join(parts[:2])
        by_prefix[prefix][name.rsplit(".", 1)[-1]] += 1
        examples.setdefault(prefix, name)

    params_by_prefix = collections.Counter()
    for name, param in model.named_parameters():
        params_by_prefix[".".join(name.split(".")[:2])] += param.numel()

    return {
        "model": MODEL_ID,
        "top_level_children": [name for name, _ in model.named_children()],
        "linear_by_prefix": {
            prefix: {
                "total": sum(counts.values()),
                "suffixes": dict(counts),
                "example": examples[prefix],
            }
            for prefix, counts in sorted(by_prefix.items())
        },
        "params_by_prefix": dict(params_by_prefix),
    }


@app.function(image=image, timeout=1800)
def swift_help(pattern: str = "") -> dict:
    """Report the installed ms-swift version and its real `sft` flags.

    The published docs and the installed package disagree -- `--train_type lora`
    came back as `remaining_argv`, meaning the parser had never heard of it. Ask
    the CLI that will actually run instead of trusting a docs page.
    """
    import re
    import subprocess

    version = subprocess.run(
        ["python", "-c", "import swift; print(swift.__version__)"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()

    helptext = subprocess.run(
        ["swift", "sft", "--help"], capture_output=True, text=True, check=False
    )
    blob = helptext.stdout + helptext.stderr
    flags = sorted(set(re.findall(r"--[a-zA-Z][\w-]*", blob)))

    # `swift sft --help` only advertises --tuner_backend: the parser is built
    # from dataclasses at runtime, so the authoritative flag list is the
    # dataclass fields themselves.
    import dataclasses

    # Do not guess the import path: read what the installed CLI actually does.
    sft_source = ""
    try:
        import swift.cli.sft as sft_cli

        sft_source = open(sft_cli.__file__).read()
    except Exception as exc:  # noqa: BLE001
        sft_source = f"<unreadable: {exc}>"

    # swift 4.4.2 moved arguments out of swift.llm (`swift.cli.sft` now does
    # `from swift.pipelines import sft_main`). Rather than chase the layout,
    # walk the package for the dataclass that actually defines `freeze_vit` --
    # the field this experiment turns on.
    import importlib
    import pkgutil

    # swift 4.4.2 composes its CLI from several argument dataclasses rather than
    # one, so search for every dataclass declaring any field this experiment
    # needs, and report where each lives.
    import importlib
    import pkgutil

    WANTED = {
        "train_type", "sft_type", "freeze_vit", "freeze_aligner", "freeze_llm",
        "target_modules", "lora_rank", "lora_alpha", "dataset", "max_steps",
    }

    found: dict[str, dict] = {}
    scanned: list[str] = []
    arg_class = None
    try:
        import swift

        for info in pkgutil.walk_packages(swift.__path__, prefix="swift."):
            try:
                module = importlib.import_module(info.name)
            except Exception:  # noqa: BLE001 - optional deps, keep walking
                continue
            scanned.append(info.name)
            for attr_name, attr in vars(module).items():
                if not (dataclasses.is_dataclass(attr) and isinstance(attr, type)):
                    continue
                names = {f.name for f in dataclasses.fields(attr)}
                hit = sorted(WANTED & names)
                if not hit:
                    continue
                key = f"{attr.__module__}.{attr.__name__}"
                if key not in found:
                    found[key] = {"has": hit, "n_fields": len(names)}
                if attr.__name__ == "SftArguments" and "freeze_vit" in hit:
                    arg_class = attr
    except Exception:  # noqa: BLE001
        pass

    field_names: list[str] = []
    defaults: dict[str, str] = {}
    if arg_class is not None:
        for field in dataclasses.fields(arg_class):
            field_names.append(field.name)
            if field.default is not dataclasses.MISSING:
                defaults[field.name] = repr(field.default)

    keys = (
        "train_type", "sft_type", "tuner", "lora", "target", "freeze",
        "vit", "aligner", "adapter", "dataset", "max_steps",
        "save_steps", "output_dir", "torch_dtype", "gradient_checkpointing",
        "learning_rate", "warmup", "batch_size", "max_length", "seed",
    )
    interesting = sorted(
        name for name in field_names if any(key in name for key in keys)
    )
    matched = [name for name in field_names if pattern and pattern in name]

    return {
        "swift_version": version,
        "arg_class": f"{arg_class.__module__}.{arg_class.__name__}" if arg_class else None,
        "n_fields": len(field_names),
        "interesting_fields": {
            name: defaults.get(name, "<required>") for name in interesting
        },
        "matched": matched,
        "sft_cli_source": sft_source[:400],
        "modules_scanned": len(scanned),
        "dataclasses_with_wanted_fields": found,
        "help_head": blob[:300],
    }


# --------------------------------------------------------------------------
# upload-bundle
# --------------------------------------------------------------------------


@app.function(
    image=image,
    volumes={DATA_MOUNT: data_volume},
    timeout=3600,
)
def verify_bundle(bundle_name: str) -> dict:
    """Check an uploaded bundle: row counts, and that every clip resolves."""
    import json
    from pathlib import Path

    data_volume.reload()
    root = Path(DATA_MOUNT) / "bundles" / bundle_name
    if not root.is_dir():
        raise FileNotFoundError(root)

    summary = {"bundle": bundle_name, "root": str(root)}
    for name in ("swift_train.jsonl", "swift_heldout.jsonl"):
        path = root / name
        rows = [json.loads(line) for line in path.open() if line.strip()]
        missing = [
            audio
            for row in rows
            for audio in row["audios"]
            if not Path(audio).is_file()
        ]
        tag_mismatch = [
            index
            for index, row in enumerate(rows)
            if row["messages"][0]["content"].count("<audio>") != len(row["audios"])
        ]
        summary[name] = {
            "rows": len(rows),
            "missing_audio": len(missing),
            "tag_count_mismatch": len(tag_mismatch),
            "example_missing": missing[:3],
        }
    return summary


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------

# Table 9, mirrored from the Kimi runs so the two models share a recipe. See
# reproduction/finetuning_lift/config.py for provenance.
RECIPE = {
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "learning_rate": 5e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,
    "weight_decay": 0.1,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    # NOT Table 9's 512. That value is impossible here, and this is the one
    # recipe value that cannot transfer between the two models.
    #
    # Qwen3-Omni's AuT encoder emits audio tokens at 12.5Hz, so a question costs
    # roughly 12.5 tokens per second of audio plus prompt text. Measured clip
    # durations in the release:
    #
    #   task           clips   mean audio   max     -> tokens (mean/max)
    #   rad              2        5.8s       9s          73 /  113
    #   rsq              2        7.8s       9s          98 /  113
    #   ser_classify     1       12.8s      45.7s       160 /  571
    #   ser_intensity    2       48.4s      65s         605 /  813
    #
    # At 512 every SER-Intensity example is truncated away and swift fails with
    # "Failed to retrieve the dataset. You can avoid this issue by increasing
    # `max_length`". 2048 clears the worst case (813 + prompt) with headroom.
    #
    # Kimi keeps 512 because it consumes audio as discrete semantic codes rather
    # than as tokens counted against this budget, and its runs converged. The
    # asymmetry is therefore forced by the architectures, not chosen -- see
    # RESULTS.md, "Comparability across models".
    "max_length": 2048,
    "logging_steps": 1,
    "save_steps": 50,
}


# Layer counts from `--step probe`. Asserted against the saved adapter after
# every run, so a wrong count fails loudly rather than silently under-targeting.
DECODER_LAYERS = 48
AUDIO_ENCODER_LAYERS = 32
DECODER_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
AUDIO_ENCODER_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "out_proj")


def _lora_target_modules(*, include_audio_encoder: bool) -> list[str]:
    """Fully-qualified LoRA targets for one arm.

    READOUT gets decoder attention. REWIRE gets the same plus audio-encoder
    attention -- a strict superset, so the arms differ by exactly the encoder.
    """
    targets = [
        f"thinker.model.layers.{layer}.self_attn.{projection}"
        for layer in range(DECODER_LAYERS)
        for projection in DECODER_PROJECTIONS
    ]
    if include_audio_encoder:
        targets += [
            f"thinker.audio_tower.layers.{layer}.self_attn.{projection}"
            for layer in range(AUDIO_ENCODER_LAYERS)
            for projection in AUDIO_ENCODER_PROJECTIONS
        ]
    return targets


@app.function(
    image=image,
    gpu=TRAIN_GPU,
    volumes={DATA_MOUNT: data_volume, HF_CACHE_MOUNT: hf_cache},
    secrets=hf_secrets,
    timeout=86400,
)
def train(
    *,
    bundle_name: str,
    run_name: str,
    freeze_vit: bool,
    max_steps: int,
    seed: int,
) -> dict:
    """Run `swift sft` for one (task, arm, seed).

    `freeze_vit` selects the arm here -- True for READOUT, False for REWIRE -- but
    only by choosing the target-module list. The flag itself does not control
    which modules get adapters (see the module docstring); it is forwarded and
    recorded so each run's config is visible, and nothing relies on its effect.
    """
    import json
    import re
    import shutil
    import subprocess
    from pathlib import Path

    data_volume.reload()
    bundle = Path(DATA_MOUNT) / "bundles" / bundle_name
    train_file = bundle / "swift_train.jsonl"
    if not train_file.is_file():
        raise FileNotFoundError(train_file)

    output_dir = Path(DATA_MOUNT) / "runs" / run_name
    # Start clean. Verification reads the saved adapter to decide whether the
    # encoder was adapted, so a checkpoint left behind by an earlier run would be
    # attributed to this one -- which already happened once and reported a failed
    # run's config as verified.
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "swift", "sft",
        "--model", MODEL_ID,
        # `--tuner_type`, not `--train_type`. ms-swift 4.4.2 renamed it, and the
        # published docs still say `train_type`; passing that yields
        # `ValueError: remaining_argv: ['--train_type', 'lora']`. The real field
        # list comes from `swift.arguments.sft_args.SftArguments`, which
        # `--step swift-help` dumps, because `swift sft --help` advertises only
        # `--tuner_backend`.
        "--tuner_type", "lora",
        "--dataset", str(train_file),
        # `--target_regex`, NOT `--target_modules`.
        #
        # `--target_modules q_proj k_proj v_proj o_proj` matches those suffixes
        # EVERYWHERE: thinker.model (decoder), thinker.audio_tower (encoder),
        # talker, and code2wav (vocoder). Verified by reading the saved adapter:
        # both arms produced 192 audio_tower LoRA tensors and an identical
        # 21.9546M trainable count, and both reached train_loss 1.025 -- i.e.
        # the two arms were the same model.
        #
        # `freeze_vit` does NOT fix that. It governs base-weight freezing for
        # full finetuning; it has no say over which modules PEFT wraps. So the
        # scoping has to be explicit, and it is the regex that separates the
        # arms:
        #
        #   READOUT  decoder attention only
        #   REWIRE   decoder attention + audio-encoder attention
        #
        # The encoder's attention output is `out_proj`, not `o_proj` (see
        # `--step probe`), so it is named separately. talker and code2wav are
        # excluded from both arms: the talker is disabled at inference and the
        # vocoder is never used, so adapting them is waste and a confound.
        # Fully-qualified module names, one per target, rather than suffixes or a
        # regex.
        #
        # Suffixes do not work: `--target_modules q_proj k_proj v_proj o_proj`
        # matches those names EVERYWHERE -- decoder, audio tower, talker, and the
        # code2wav vocoder. Verified by reading the saved adapters: both arms
        # produced 192 audio_tower LoRA tensors, an identical 21.9546M trainable
        # count, and the same train_loss 1.025. The arms were one model.
        #
        # `freeze_vit` does not separate them either: it governs base-weight
        # freezing for full finetuning and has no say over which modules PEFT
        # wraps.
        #
        # `--target_regex` is not usable in 4.4.2: the regex ends up in
        # target_modules and is iterated character by character
        # ("Target modules {'r','.','v',...} not found").
        #
        # So enumerate the modules, which is also what the Kimi entrypoint does.
        # PEFT suffix-matches list entries, and a fully-qualified name matches
        # only itself. Layer counts come from `--step probe`: 48 decoder layers,
        # 32 audio-encoder layers. The encoder's attention output is `out_proj`,
        # not `o_proj`. talker and code2wav are excluded from both arms -- the
        # talker is disabled at inference and the vocoder is never used.
        "--target_modules", *_lora_target_modules(include_audio_encoder=not freeze_vit),
        "--torch_dtype", "bfloat16",
        "--gradient_checkpointing", "true",
        "--max_steps", str(max_steps),
        "--seed", str(seed),
        "--output_dir", str(output_dir),
        "--save_strategy", "steps",
        "--report_to", "none",
        # Plain torch AdamW: no deepspeed in the image, so nothing tries to
        # build fused kernels at runtime.
        "--optim", "adamw_torch",
    ]
    for key, value in RECIPE.items():
        cmd += [f"--{key}", str(value)]

    print("[swift] " + " ".join(cmd), flush=True)

    # Stream rather than capture. A 1000-step run on a 30B model is long enough
    # that buffering the whole log until exit makes a live run indistinguishable
    # from a hung one, and the log is still needed afterwards for the
    # trainable-parameter check.
    lines: list[str] = []
    process = subprocess.Popen(
        cmd,
        cwd=str(bundle),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line)
        print(line, end="", flush=True)
    process.wait()

    log = "".join(lines)
    data_volume.commit()

    # Verify what was actually adapted, from the saved adapter rather than from
    # the log. Grepping stdout cannot distinguish "the encoder was not adapted"
    # from "the log never named the modules"; the checkpoint can.
    #
    # swift prints e.g.
    #   model_parameter_info: PeftModelForCausalLM: 35281.7731M Params
    #   (21.9546M Trainable [0.0622%]), 1.9205M Buffers.
    trainable = None
    match = re.search(r"\(([\d.]+)M Trainable", log)
    if match:
        trainable = float(match.group(1)) * 1e6

    checkpoints = sorted(
        (p for p in output_dir.glob("**/checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    encoder_targets: list[str] = []
    adapter_target_modules: list[str] = []
    if checkpoints:
        adapter_config = checkpoints[-1] / "adapter_config.json"
        if adapter_config.is_file():
            cfg = json.loads(adapter_config.read_text())
            raw = cfg.get("target_modules") or []
            adapter_target_modules = sorted(raw) if isinstance(raw, list) else [str(raw)]
            encoder_targets = [
                name for name in adapter_target_modules if "audio_tower" in name
            ]
        # The weight names are the ground truth: whatever PEFT actually saved.
        weights = checkpoints[-1] / "adapter_model.safetensors"
        if weights.is_file():
            try:
                from safetensors import safe_open

                with safe_open(str(weights), framework="pt") as handle:
                    keys = list(handle.keys())
                encoder_targets = sorted(
                    {k for k in keys if "audio_tower" in k}
                )
                adapter_target_modules = keys[:8]
            except Exception as exc:  # noqa: BLE001
                print(f"[verify] could not read adapter weights: {exc}", flush=True)

    result = {
        "run_name": run_name,
        "bundle": bundle_name,
        "freeze_vit": freeze_vit,
        "arm": "READOUT" if freeze_vit else "REWIRE",
        "max_steps": max_steps,
        "seed": seed,
        "returncode": process.returncode,
        "trainable_parameters": trainable,
        "audio_tower_adapter_count": len(encoder_targets),
        "audio_tower_adapters_sample": encoder_targets[:4],
        "adapter_keys_sample": adapter_target_modules[:6],
        "checkpoints_found": [p.name for p in checkpoints][-5:],
        "output_dir": str(output_dir),
        "checkpoints": [p.name for p in checkpoints][:40],
    }
    if process.returncode != 0:
        raise RuntimeError(f"swift sft failed for {run_name}: {result}")

    # The guard that matters. `freeze_vit=False` is an assertion about someone
    # else's flag; if it silently fails to reach the audio tower, REWIRE trains
    # the exact READOUT configuration and the ablation reports a null result that
    # means nothing. Fail loudly instead.
    if not freeze_vit and not encoder_targets:
        raise RuntimeError(
            f"{run_name}: freeze_vit=False but no thinker.audio_tower modules "
            f"were adapted; this run would silently reproduce READOUT. {result}"
        )
    if freeze_vit and encoder_targets:
        raise RuntimeError(
            f"{run_name}: freeze_vit=True but audio_tower modules were adapted; "
            f"the frozen-encoder arm is not frozen. {result}"
        )
    return result


@app.local_entrypoint()
def main(
    step: str = "probe",
    task: str = "",
    arm: str = "baseline",
    seed: int = 1,
    bundle_name: str = "",
    max_steps: int = 1000,
):
    import json

    if step == "probe":
        print(json.dumps(probe.remote(), indent=2))
        return

    if step == "swift-help":
        print(json.dumps(swift_help.remote(), indent=2))
        return

    if step == "verify-bundle":
        if not bundle_name:
            raise SystemExit("--bundle-name is required for verify-bundle")
        print(json.dumps(verify_bundle.remote(bundle_name), indent=2))
        return

    if step == "train":
        if not task or not bundle_name:
            raise SystemExit("--task and --bundle-name are required for train")
        if arm not in {"baseline", "encoder_lora"}:
            raise SystemExit(f"unknown arm {arm!r}")
        print(
            json.dumps(
                train.remote(
                    bundle_name=bundle_name,
                    run_name=f"{task}-{arm}-seed{seed}-qwen3omni",
                    # baseline == READOUT == encoder frozen.
                    freeze_vit=(arm == "baseline"),
                    max_steps=max_steps,
                    seed=seed,
                ),
                indent=2,
            )
        )
        return

    raise SystemExit(f"unknown step {step!r}")
