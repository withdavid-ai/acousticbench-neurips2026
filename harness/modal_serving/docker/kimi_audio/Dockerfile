# syntax=docker/dockerfile:1.7
#
# Docker build for the Kimi-Audio server from modal_serving/kimi_audio_serve.py.
#
# Uses Moonshot's official base image which includes torch, transformers,
# and all Kimi-Audio core dependencies.
#
# Build context: harness/modal_serving/   (so COPY serving_utils.py works)
#
#   cd harness   # from the repository root
#   docker build \
#       -f modal_serving/docker/kimi_audio/Dockerfile \
#       -t acousticbench-kimi-audio:latest \
#       modal_serving
#
# Run:
#   docker run --gpus '"device=0"' \
#       -v ~/.cache/huggingface:/hf-cache/huggingface \
#       -p 8002:8000 \
#       -e HF_TOKEN=$HF_TOKEN \
#       acousticbench-kimi-audio:latest

FROM moonshotai/kimi-audio:v0.1

ENV HF_HOME=/hf-cache/huggingface \
    HF_HUB_CACHE=/hf-cache/huggingface/hub \
    HUGGINGFACE_HUB_CACHE=/hf-cache/huggingface/hub \
    TRANSFORMERS_CACHE=/hf-cache/huggingface/hub \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Install Kimi-Audio Python package (no-deps: base image has all deps), pinned
# to the last upstream commit on or before 2026-05-07 (2025-06-03).
ARG KIMI_AUDIO_GIT_SHA=349251e1d8f4f98d58fda59246381faecd7392e0
RUN python -m pip install --no-deps \
    "git+https://github.com/MoonshotAI/Kimi-Audio.git@${KIMI_AUDIO_GIT_SHA}"

WORKDIR /app

COPY docker/kimi_audio/requirements.txt /app/requirements.txt
RUN python -m pip install -r /app/requirements.txt

COPY serving_utils.py /app/serving_utils.py
COPY docker/kimi_audio/app.py /app/app.py

VOLUME ["/hf-cache"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
    CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=5); sys.exit(0 if r.status==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
