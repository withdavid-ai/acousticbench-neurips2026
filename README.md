# AcousticBench

**Measuring Acoustic Perception in Large Audio Language Models — NeurIPS 2026**

AcousticBench contains 3,200 two-choice audio questions about pitch, loudness,
tempo, dynamic range, vocal emotion, and recording quality. A separate
600-question transcript-only diagnostic tests lexical shortcuts.

- [Dataset on Hugging Face](https://huggingface.co/datasets/davidai-labs/AcousticBench) — gated access; accept its data-use agreement.
- [Croissant 1.1 + Responsible AI metadata](dataset/croissant.json)
- [Reproduction scope and known differences](docs/reproduction.md)
- [Modal model serving](harness/modal_serving/ENVIRONMENT.md)

| Task | Questions | Train / eval | Audio clips per question |
|---|---:|---:|---:|
| RAD | 1,200 | 600 / 600 | 2 |
| SER-Classify | 600 | 300 / 300 | 1 |
| SER-Intensity | 400 | 200 / 200 | 2 |
| RSQ | 1,000 | 500 / 500 | 2 |
| SER-Classify-Text (diagnostic) | 600 | 300 / 300 | 0 |

Leaderboard evaluation uses **both halves** (`--split all`). Each complete task
has equal A/B answers; individual emotion categories and halves need not.
The diagnostic is excluded from aggregates. Question-ID halves can share
speakers and RAD source clips; they are not the speaker-disjoint experiment.

## Install

Python 3.11 or newer is required. Open-weight inference needs Linux and a
suitable NVIDIA GPU (the original serving configuration used A100-80GB).
Offline analysis and API clients also run on macOS.

```bash
git clone https://github.com/withdavid-ai/acousticbench-neurips2026.git
cd acousticbench-neurips2026
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[api,modal,dev]'
```

For the existing pinned dependency resolution, use `uv sync --frozen --extra api
--extra modal --extra dev --python 3.11`. `uv.lock` and `requirements.lock` record
an environment specification, not proof of every historical provider environment.
Optional local audio validation requires `pip install soundfile librosa`.
GPU packages are installed inside the serving containers.

## Recompute scores offline

No dataset download, API key, GPU, or paid inference is needed:

```bash
python scripts/make_paper_tables.py --check
python -m pytest -q
python scripts/check_public_release.py
```

The generator checks **550 published cells** from 270 reference run files.
These cover the main leaderboard, parseability, run variability, original Kimi
specialists, and Kimi cross-task transfer. Its table IDs retain the original
numbering; the final paper added other tables. Human values are reported
constants, not recomputed from individual ratings.

The reference records retain question IDs, answer keys, extracted choices,
correctness and task/split fields. **Free-form responses, prompts, transcripts,
error payloads and internal metadata are not published here.** A non-empty
`response_text` placeholder preserves response-presence accounting; it is not a
model answer and must not be rejudged. See [reference provenance](reference_trajectories/README.md).

## Download the dataset

Request access on Hugging Face and accept the dataset terms, then authenticate:

```bash
hf auth login
hf download davidai-labs/AcousticBench --repo-type dataset --local-dir archive
python scripts/validate_archive.py --archive archive --json-out results/archive_validation.json
```

Audio and question files stay outside Git. The MIT code license does **not**
grant rights to the dataset. Its data-use agreement prohibits training and
fine-tuning; split names document historical experiments, not training permission.
Training utilities are supplied for methodological transparency and use with
independently licensed data where training is permitted.

## Run a model

```bash
cp example.env .env
# Fill only the keys you need. Never commit .env.
python run.py --list-models
python run.py --list-tasks
python run.py --model Gemini-3.1-Pro --task rad --split all \
  --max-samples 10 --runs 1 --verify-checksums
```

`OPENAI_API_KEY` is used for the answer-extraction judge, including when the
answering model is open-weight. Gemini also requires `GEMINI_API_KEY`. The cascade
and GPT-4o-Audio use `OPENAI_API_KEY`. Provider calls incur charges. Increase sample
counts only after a smoke test; a full sweep is not run automatically.

For an open-weight model, deploy a server into **your own** Modal environment
using [these instructions](harness/modal_serving/ENVIRONMENT.md), then run:

```bash
python run.py --model Qwen3-Omni --backend modal \
  --modal-environment acousticbench --task rad --split all --max-samples 10
```

Alternatively, Docker on Linux uses the model-specific Dockerfiles:

```bash
bash scripts/docker_serve.sh qwen3_omni --build-only
bash scripts/docker_serve.sh qwen3_omni --run-only
python run.py --model Qwen3-Omni --backend docker --task rad --max-samples 10
```

Docker helper scripts require Bash 4+ and NVIDIA Container Toolkit. Endpoint
variables and ports are listed in `example.env`. Export `HF_TOKEN` in your shell
before deploying containers that need gated model weights. Serving images run
upstream model code; inspect it and use isolated environments.

Use `--runs 3 --split all` for the main benchmark and `--compare-reference` for
item-level comparisons. New provider outputs can differ from saved runs. The
retired GPT-4o-Audio snapshot cannot be freshly reproduced.

## Research tools and coverage

- `harness/`: released prompt loader, API adapters, scoring, Docker/Modal serving.
- `scripts/`: archive validation, table regeneration and run comparisons.
- `finetuning/kimi_audio/`: Kimi training runtime and source utilities.
- `reproduction/finetuning_lift/`: fixed-step Kimi/Qwen research tools, dataset
  conversion, checkpoint evaluation and describe-then-answer diagnostics.

**The fixed-step research runs are not established as the training runs behind
all submitted rebuttal values.** In particular, local encoder-ablation outputs
previously differed from submitted results. Do not treat a training script's
configuration as verified provenance for those tables. MOSS training, external
transfer, acoustic-feature probes, and speaker-disjoint experiment pipelines are
not complete in the available release sources. Consult the paper for their aggregate results. See
[the coverage matrix](docs/reproduction.md).

Adapter weights are not bundled, and no verified adapter download location is
claimed. Kimi LoRA serving expects user-supplied adapters; see its serving code.

## Citation

```bibtex
@inproceedings{acousticbench2026,
  title = {AcousticBench: Measuring Acoustic Perception in Large Audio Language Models},
  author = {Rahul, Anirudh and Vidyarthi, Arth and Horgan, Michael and Jaju, Suyash and Zhou, Yuhao},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026},
  url = {https://github.com/withdavid-ai/acousticbench-neurips2026}
}
```

## License and contact

Code: [MIT](LICENSE), with [third-party notices](THIRD_PARTY_NOTICES.md) for
vendored components. Dataset and upstream model weights retain their separate
terms. Contact: **research@withdavid.ai**. Please report security/privacy issues
privately rather than attaching sensitive data to a public issue.
