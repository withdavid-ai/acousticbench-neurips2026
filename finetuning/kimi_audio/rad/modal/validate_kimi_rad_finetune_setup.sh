#!/usr/bin/env bash
# Local structural check; does not deploy or start a training job.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
python3 -m py_compile finetuning/kimi_audio/rad/modal/kimi_rad_finetune.py reproduction/finetuning_lift/bundle.py
python3 -m reproduction.finetuning_lift.bundle --help >/dev/null
python3 -c 'from finetuning.kimi_audio.rad.modal.kimi_rad_finetune import app; assert app is not None; print("Kimi launcher imports successfully (no remote job run).")'
