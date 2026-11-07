# syntax=docker/dockerfile:1.7
#
# Reproducible Docker build of the Voxtral-Small-24B server from
# modal_serving/voxtral_serve.py.
#
# Build context: harness/modal_serving/   (so COPY serving_utils.py works)
#
#   cd harness   # from the repository root
#   docker build \
#       -f modal_serving/docker/voxtral_small_24b/Dockerfile \
#       -t acousticbench-voxtral-small-24b:latest \
#       modal_serving
#
# Run:
#   docker run --gpus '"device=0"' \
#       -v ~/.cache/huggingface:/hf-cache \
#       -p 8001:8000 \
#       -e HF_TOKEN=$HF_TOKEN \
#       acousticbench-voxtral-small-24b:latest

ARG PYTHON_BASE=python:3.11.10-slim

FROM ${PYTHON_BASE}

ARG TORCH_INDEX_URL=

ENV HF_HOME=/hf-cache/huggingface \
    HF_HUB_CACHE=/hf-cache/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/hf-cache/huggingface/hub \
    TRANSFORMERS_CACHE=/hf-cache/huggingface/hub \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/voxtral_small_24b/requirements.txt /app/requirements.txt
RUN if [ -n "$TORCH_INDEX_URL" ]; then \
        python -m pip install --extra-index-url "$TORCH_INDEX_URL" -r /app/requirements.txt; \
    else \
        python -m pip install -r /app/requirements.txt; \
    fi

COPY serving_utils.py /app/serving_utils.py
COPY docker/voxtral_small_24b/app.py /app/app.py

VOLUME ["/hf-cache"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5); sys.exit(0 if r.status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
