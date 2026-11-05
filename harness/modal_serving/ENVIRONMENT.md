# Modal serving

The `*_serve.py` files in this directory deploy each open-weight model as a [Modal](https://modal.com) app on an NVIDIA A100-80GB GPU. These are the serving implementations used by the evaluation harness; the additional checkpoint servers support the research tools. The harness calls them with `python run.py --backend modal`. The Docker servers in `docker/` serve the same models locally (see the main README).

You deploy the apps into your own Modal workspace; nothing in this repository points at the authors' workspace.

## Prerequisites

```bash
pip install -e ".[modal]"        # or use the locked environment from the main README
modal token new                  # authenticate the Modal CLI
modal environment create acousticbench   # optional; skip it to use your default environment
export MODAL_ENVIRONMENT=acousticbench   # only if you created a separate environment
```

`run.py --backend modal` looks the apps up in the environment given by `--modal-environment`, else `MODAL_ENVIRONMENT` (shell or `.env`), else your Modal profile's default environment. It must be the environment the apps were deployed into. The `modal deploy` and `modal run` commands below pass `--env "$MODAL_ENVIRONMENT"`; drop that option if you use the default environment.

## Hugging Face token

The apps do not reference any named Modal secret. When you run `modal deploy`, `common.py` reads `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) from your local shell and attaches it to the app with `modal.Secret.from_dict`. Export it before deploying:

```bash
export HF_TOKEN=hf_...
```

If neither variable is set, the apps deploy without a token and download weights anonymously, which can fail or be rate-limited. To change the token later, redeploy.

## Deploy

Run from the repository root:

```bash
modal deploy --env "$MODAL_ENVIRONMENT" harness/modal_serving/qwen3_omni_serve.py
```

| `--model` | Serve file | Modal app | Hugging Face ID |
|-----------|------------|-----------|-----------------|
| `Qwen3-Omni` | `qwen3_omni_serve.py` | `audiobench-qwen3-omni` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |
| `Kimi-Audio` | `kimi_audio_serve.py` | `audiobench-kimi-audio` | `moonshotai/Kimi-Audio-7B-Instruct` |
| `MiniCPM-o-4.5` | `minicpm_o_serve.py` | `audiobench-minicpm-o` | `openbmb/MiniCPM-o-4_5` |
| `Step-Audio-2-mini` | `step_audio2_serve.py` | `audiobench-step-audio2` | `stepfun-ai/Step-Audio-2-mini` |
| `Voxtral-Small-24B`, `Voxtral-Mini-3B` | `voxtral_serve.py` | `audiobench-voxtral` | `mistralai/Voxtral-Small-24B-2507`, `mistralai/Voxtral-Mini-3B-2507` |
| `MiMo-Audio-7B` | `mimo_audio_serve.py` | `audiobench-mimo-audio` | `XiaomiMiMo/MiMo-Audio-7B-Instruct` (+ `XiaomiMiMo/MiMo-Audio-Tokenizer`) |
| `Phi-4-Multimodal` | `phi4mm_serve.py` | `audiobench-phi4mm` | `microsoft/Phi-4-multimodal-instruct` |
| `Qwen2-Audio-7B-Instruct` | `qwen2_audio_serve.py` | `audiobench-qwen2-audio` | `Qwen/Qwen2-Audio-7B-Instruct` |
| `Audio-Flamingo-3` | `audio_flamingo3_serve.py` | `audiobench-audio-flamingo3` | `nvidia/audio-flamingo-3-chat` |
| `Kimi-Audio-LoRA-*` (all five) | `kimi_lora_serve.py` | `audiobench-kimi-lora` | `moonshotai/Kimi-Audio-7B-Instruct` + adapters |

To deploy every model in the table:

```bash
for f in qwen3_omni kimi_audio minicpm_o step_audio2 voxtral mimo_audio phi4mm qwen2_audio audio_flamingo3 kimi_lora; do
  modal deploy --env "$MODAL_ENVIRONMENT" "harness/modal_serving/${f}_serve.py"
done
```

The app names are fixed in the serve files and in the matching `harness/models/*.py` clients. If you rename an app, change both.

### Kimi-Audio LoRA adapters

`kimi_lora_serve.py` copies the adapters into its image at deploy time, so supply independently obtained adapters first so that `archive/models/<adapter>/adapter_model.safetensors` exists for `rad`, `rsq`, `ser_classify`, `ser_intensity`, and `simple_audioqa`. Because `archive/` always exists in a clone, deploying this file without the adapters fails with a `FileNotFoundError` that lists the paths it checked. On first use of each adapter the app merges and exports the adapted model to the `audiobench-kimi-lora-adapters` volume; you can do this ahead of time:

```bash
modal run --env "$MODAL_ENVIRONMENT" harness/modal_serving/kimi_lora_serve.py::prefetch --adapter-name rad
```

### Pre-downloading weights (optional)

Every serve file has a `prefetch` function that downloads the weights into the shared `audiobench-hf-cache` volume, so the first benchmark request does not wait for the download:

```bash
modal run --env "$MODAL_ENVIRONMENT" harness/modal_serving/qwen3_omni_serve.py::prefetch
```

Each serve file also has a `smoke_test` local entrypoint (`modal run ... ::smoke_test --help` lists its arguments).

## Run

```bash
python run.py --backend modal --model Qwen3-Omni --task rsq --max-samples 3
python run.py --backend modal --modal-environment acousticbench --model Qwen3-Omni --task rsq --max-samples 3
```

## Volumes

Created automatically on first deploy:

| Volume | Contents |
|--------|----------|
| `audiobench-hf-cache` | Hugging Face model weights shared by all apps |
| `audiobench-kimi-lora-adapters` | Kimi-Audio LoRA adapters and merged exports |

## Images

All apps run on one NVIDIA A100-80GB GPU per container, without request batching. The images are built from the stacks below. The serve files and `common.py` are authoritative for package versions and upstream repository commits. Every Hugging Face checkpoint is loaded at the commit listed in `HF_REVISIONS` in `serving_utils.py`, which the Modal and Docker servers share.

| Stack | Defined in | Used by |
|-------|------------|---------|
| Base: Python 3.11, PyTorch 2.4.1, Transformers 4.52.4, ffmpeg, libsndfile | `common.base_image()` | Phi-4-Multimodal, Audio-Flamingo-3 (Transformers 5.3.0), MiniCPM-o-4.5 (PyTorch 2.8.0, Transformers 4.51.0), Step-Audio-2-mini (PyTorch 2.5.1, Transformers 4.49.0), Qwen2-Audio-7B-Instruct (PyTorch 2.5.1) |
| Modern: Python 3.11, PyTorch 2.11.0, Transformers 5.3.0 | `common.modern_image()` | Qwen3-Omni (`qwen-omni-utils`), Voxtral (`mistral_common[audio]`) |
| `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` + MiMo-Audio source + flash-attn | `mimo_audio_serve.py` | MiMo-Audio-7B |
| `moonshotai/kimi-audio:v0.1` + Kimi-Audio source | `kimi_audio_serve.py`, `kimi_lora_serve.py` | Kimi-Audio, Kimi-Audio LoRAs (+ `peft` and the vendored finetuning code in `kimi_vendor/`) |

Decoding is greedy on every server (`do_sample=False`, or text temperature 0 for Kimi-Audio).
