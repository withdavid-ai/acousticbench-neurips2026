# syntax=docker/dockerfile:1.7
#
# Docker build for the Kimi-Audio LoRA server from
# modal_serving/kimi_lora_serve.py.
#
# Each running container serves exactly ONE LoRA adapter (rad / rsq /
# ser_classify / ser_intensity) selected at run time via the
# ADAPTER_NAME env var. This mirrors how the Modal class is parameterized
# with `adapter_name=...`.
#
# Build context: harness/modal_serving/   (so COPY serving_utils.py works)
#
#   cd harness   # from the repository root
#   docker build \
#       -f modal_serving/docker/kimi_lora/Dockerfile \
#       -t acousticbench-kimi-lora:latest \
#       modal_serving
#
# Run (one container per adapter):
#   docker run --gpus '"device=0"' \
#       -e ADAPTER_NAME=rad \
#       -v ~/.cache/huggingface:/hf-cache/huggingface \
#       -v "$PWD/../archive":/adapters:ro \
#       -p 8015:8000 \
#       acousticbench-kimi-lora:latest

FROM moonshotai/kimi-audio:v0.1

ENV HF_HOME=/hf-cache/huggingface \
    HF_HUB_CACHE=/hf-cache/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/hf-cache/huggingface/hub \
    TRANSFORMERS_CACHE=/hf-cache/huggingface/hub \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    ADAPTER_NAME=rad \
    ADAPTER_ROOT=/adapters \
    PRETRAINED_ROOT=/var/lib/kimi-lora-exports/pretrained_hf \
    EXPORT_ROOT=/var/lib/kimi-lora-exports/exported

# Match the main AcousticBench finetuning image: clone Kimi-Audio, initialize
# submodules, install its requirements, then install the package editable.
# The clone is pinned to the last upstream commit on or before 2026-05-07.
ARG KIMI_AUDIO_GIT_SHA=349251e1d8f4f98d58fda59246381faecd7392e0
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://github.com/MoonshotAI/Kimi-Audio.git /opt/Kimi-Audio \
    && cd /opt/Kimi-Audio \
    && git checkout "${KIMI_AUDIO_GIT_SHA}" \
    && git submodule update --init --recursive \
    && python -m pip install -r requirements.txt \
    && python -m pip install -e .

WORKDIR /app

COPY docker/kimi_lora/requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

COPY serving_utils.py /app/serving_utils.py
COPY kimi_vendor /app/kimi_vendor
COPY docker/kimi_lora/app.py /app/app.py

VOLUME ["/hf-cache", "/adapters", "/var/lib/kimi-lora-exports"]
EXPOSE 8000

# Generous start-period: first boot exports the LoRA-merged checkpoint, which
# can take several minutes on cold cache. Subsequent boots reuse the cached
# export from the /var/lib/kimi-lora-exports volume if mounted.
HEALTHCHECK --interval=30s --timeout=10s --start-period=1500s --retries=3 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5); sys.exit(0 if r.status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
