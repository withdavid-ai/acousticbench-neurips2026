# syntax=docker/dockerfile:1.7
#
# Reproducible Docker build of the MiniCPM-o-4.5 server from
# modal_serving/minicpm_o_serve.py.
#
# Build context: harness/modal_serving/   (so COPY serving_utils.py works)
#
#   cd harness   # from the repository root
#   docker build \
#       -f modal_serving/docker/minicpm_o/Dockerfile \
#       -t acousticbench-minicpm-o:latest \
#       modal_serving
#
# Run:
#   docker run --gpus '"device=0"' \
#       -v ~/.cache/huggingface:/hf-cache \
#       -p 8001:8000 \
#       -e HF_TOKEN=$HF_TOKEN \
#       acousticbench-minicpm-o:latest

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

COPY docker/minicpm_o/requirements.txt /app/requirements.txt
RUN python -m pip install setuptools==82.0.1 wheel==0.47.0 numpy==2.4.4 cython==3.2.4 && \
    python -m pip install torch==2.8.0 torchaudio==2.8.0 && \
    python -m pip install --no-build-isolation "minicpmo-utils[all]==1.0.6" && \
    python -m pip install -r /app/requirements.txt

COPY serving_utils.py /app/serving_utils.py
COPY docker/minicpm_o/app.py /app/app.py

VOLUME ["/hf-cache"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5); sys.exit(0 if r.status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
