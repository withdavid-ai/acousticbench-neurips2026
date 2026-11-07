# Reproduction scope

## Evidence levels

Recomputing an aggregate from stored labels is different from rerunning model
inference, reproducing training, or independently validating human annotations.
This release preserves that distinction.

| Result | Available here | Limitation |
|---|---|---|
| Main leaderboard, variability, parseability | Sanitized per-item reference scores; offline table generator | Raw responses withheld; cannot re-run extraction from these placeholders |
| Original Kimi specialists and cross-task transfer | Sanitized references and score generator | Original training logs/checkpoints are not bundled |
| Describe-then-answer diagnostic | Prompt construction, sampling, scoring and runner source | Requires provider access or deployed model servers; raw responses not bundled |
| Kimi/Qwen training research tools | Dataset conversion, LoRA training source, checkpoint evaluation/serving | Local fixed-step experiments are not verified provenance for the submitted encoder table |
| MOSS and Qwen cross-architecture reported results | See paper for aggregate values | MOSS training implementation and raw outputs are not available in this package |
| Submitted encoder ablation | See paper for aggregate values | Do not substitute the different local fixed-step results |
| Speaker-disjoint retraining, external transfer, acoustic-feature probe | See paper for aggregate values | Complete manifests/predictions/pipelines not present in available release sources |
| Human accuracy, agreement and difficulty correlations | Human accuracy constants in table generator; see paper for other aggregates | Individual ratings unavailable; no independent recomputation claimed |

Results without underlying records are not independently recomputed by this repository.

## Historical reference differences

- The cascade's 42.9% SER-Classify result uses the precomputed transcript
  diagnostic. Its reference IDs are the text diagnostic IDs. Running the live
  audio-transcription path is a different experiment; the table checker warns.
- GPT-4o-Audio retains an earlier option order for 20 SER-Classify questions and
  is scored with its corresponding key. That snapshot was unavailable for reruns.
- Other affected SER-Classify results use the corrected label order. MiMo's
  SER-Classify and SER-Intensity records were rerun with greedy decoding.
- Historical references include deterministic answer extraction and some
  refusal/error reruns. New runs default to the LLM extractor. The provided
  helper scripts for rejudging/rerunning are optional and change the procedure.
- Matching an aggregate does not identify a training run. All four local Qwen
  decoder-only aggregates match the submitted results, but the local adapted
  RSQ result differed. This release does not silently equate those run lineages.

## Training utilities

The dataset agreement does not permit recipients to train on AcousticBench.
The following commands illustrate the tooling for **separately authorized data**
in the documented question/audio schema. They do not start jobs unless invoked.

```bash
pip install -e '.[research]'
export ACOUSTICBENCH_DATA=/path/to/authorized-data
export MODAL_ENVIRONMENT=acousticbench
python -m reproduction.finetuning_lift.bundle --help
python -m reproduction.finetuning_lift.swift_bundle --help
python -m reproduction.finetuning_lift.train --help
python -m reproduction.finetuning_lift.evaluate --help
```

Use a fixed, preselected checkpoint for reported evaluation. Do not select a
checkpoint based on held-out accuracy. The tools can inspect checkpoint curves
for exploratory research, which is not the claimed paper selection protocol.

Kimi's public launcher keeps in-training held-out judge evaluation disabled
(`heldout_evals_json="[]"`). Its former dataset-backed scoring and operator
serving validation are not shipped; those optional operations fail explicitly
rather than connecting to private infrastructure. Evaluate separately using the
public research evaluator and the checkpoint servers:

- `harness/modal_serving/kimi_audio_ft_serve.py`
- `harness/modal_serving/qwen3_omni_ft_serve.py`

Deploy servers and train into the same user-owned Modal environment. Training
uses user-created `huggingface-secret` and (where requested)
`acousticbench-wandb`/`openai-api-key` secrets. No credential values or author
workspace identifiers are supplied. W&B can be disabled by the research driver.
Model downloads, training and evaluation consume compute and may incur costs.

The fixed-step Qwen recipe is in `swift_train.py`; the Kimi research recipe is in
`config.py`. These record research-tool settings, not an authenticated
configuration for each submitted experiment. They must not be relabeled as such.

## Describe-then-answer diagnostic

```bash
export ACOUSTICBENCH_DATA="$PWD/archive"
python -m reproduction.finetuning_lift.cot --help
python -m reproduction.finetuning_lift.cot --model gemini --task rad \
  --condition direct --limit 10
```

The `research` extra includes `google-genai` for the Gemini diagnostic. This is a separate diagnostic with its own deterministic parsers and generation
settings, not the headline judge-based evaluation. It uses fixed held-out
question samples. Inspect `cot.py` for the prompt replacement and sampling seeds.

## Validation performed for this code release

Clean local installation, offline tests, syntax/static checks, sanitized reference
score regeneration, dataset contract checks and a public-content scan are run
before publication. This does not claim a new paid inference sweep, GPU training
run, or independent recomputation of results whose raw records are unavailable.
