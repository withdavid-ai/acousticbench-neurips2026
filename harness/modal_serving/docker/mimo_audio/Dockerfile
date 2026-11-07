# syntax=docker/dockerfile:1.7
#
# Reproducible Docker build of the MiMo-Audio server from
# modal_serving/mimo_audio_serve.py.
#
# Build context: harness/modal_serving/   (so COPY serving_utils.py works)
#
#   cd harness   # from the repository root
#   docker build \
#       -f modal_serving/docker/mimo_audio/Dockerfile \
#       -t acousticbench-mimo-audio:latest \
#       modal_serving
#
# Run:
#   docker run --gpus '"device=0"' \
#       -v ~/.cache/huggingface:/hf-cache \
#       -p 8001:8000 \
#       -e HF_TOKEN=$HF_TOKEN \
#       acousticbench-mimo-audio:latest

ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

FROM ${BASE_IMAGE}

ENV HF_HOME=/hf-cache/huggingface \
    HF_HUB_CACHE=/hf-cache/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/hf-cache/huggingface/hub \
    TRANSFORMERS_CACHE=/hf-cache/huggingface/hub \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/MiMo-Audio

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/mimo_audio/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
RUN pip install flash-attn==2.7.4.post1 --no-build-isolation

# Clone MiMo-Audio repo for model source code, pinned to the last upstream
# commit on or before 2026-05-07 (2025-09-20).
ARG MIMO_AUDIO_GIT_SHA=62d956b4a1a45419bee5e41f477078c3684dbbcc
RUN git clone https://github.com/XiaomiMiMo/MiMo-Audio.git /app/MiMo-Audio \
    && git -C /app/MiMo-Audio checkout "${MIMO_AUDIO_GIT_SHA}"

COPY serving_utils.py /app/serving_utils.py
COPY docker/mimo_audio/app.py /app/app.py

VOLUME ["/hf-cache"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5); sys.exit(0 if r.status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
