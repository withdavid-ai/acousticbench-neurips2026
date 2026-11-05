# Third-party notices

The project code is MIT-licensed. Existing third-party copyright and license
headers are retained in both copies of the Kimi model implementation:

- `harness/modal_serving/kimi_vendor/finetune_codes/`
- `finetuning/kimi_audio/vendor_kimi/finetune_codes/`

These contain code derived from MoonshotAI's Kimi-Audio and Qwen/Hugging Face
implementations, with MIT and Apache-2.0 notices in `modeling_kimia.py`. Local
modifications support the training/serving wrapper and Whisper feature contract.
See https://github.com/MoonshotAI/Kimi-Audio and the included Apache-2.0 text in
`licenses/Apache-2.0.txt`. The repository's MIT license does not replace those
notices.

Serving images download additional upstream code and model checkpoints at run
time. Their licenses remain applicable; model names, repositories and pinned
Hub revisions are in `harness/modal_serving/serving_utils.py` and the Dockerfiles.
Dataset access is governed by the separate Hugging Face data-use agreement.
