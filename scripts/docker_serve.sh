#!/usr/bin/env bash
#
# One-shot Docker build + run + health-check for AcousticBench models.
#
# Usage:
#   ./scripts/docker_serve.sh qwen3_omni              # build + run + wait
#   ./scripts/docker_serve.sh qwen3_omni --build-only  # just build the image
#   ./scripts/docker_serve.sh qwen3_omni --stop         # stop the container
#   ./scripts/docker_serve.sh --list                    # list available models
#
# Prerequisites:
#   - Docker with NVIDIA Container Toolkit (nvidia-docker)
#   - GPU with enough VRAM for the model
#   - HuggingFace model weights downloaded to ~/.cache/huggingface
#     (the container mounts this directory; first run downloads automatically)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_CONTEXT="${REPO_ROOT}/harness/modal_serving"
DOCKER_DIR="${BUILD_CONTEXT}/docker"
HF_CACHE="${HOME}/.cache/huggingface"
HEALTH_TIMEOUT=600  # seconds to wait for model to load

# ── Model registry ──────────────────────────────────────────────────────────
# Add new models here. Entry format:
#   image_name|port|gpu|env_var[|dockerfile_dir[|adapter_name]]
#
# Optional fields:
#   dockerfile_dir   Subdirectory under harness/modal_serving/docker/ that
#                    contains the Dockerfile. Defaults to the registry key.
#                    Multiple registry keys can share one dir (e.g. Kimi-Audio
#                    LoRA adapters use kimi_lora/Dockerfile).
#   adapter_name     If set, the container is started with
#                    -e ADAPTER_NAME=<name> and -v <repo>/archive:/adapters:ro
#                    so a Kimi-Audio LoRA server can pick its adapter.
declare -A MODELS=(
    [qwen3_omni]="acousticbench-qwen3-omni|8001|0|ACOUSTICBENCH_QWEN3_OMNI_ENDPOINT"
    [kimi_audio]="acousticbench-kimi-audio|8002|0|ACOUSTICBENCH_KIMI_AUDIO_ENDPOINT"
    [qwen2_audio]="acousticbench-qwen2-audio|8004|0|ACOUSTICBENCH_QWEN2_AUDIO_7B_INSTRUCT_ENDPOINT"
    [minicpm_o]="acousticbench-minicpm-o|8005|0|ACOUSTICBENCH_MINICPM_O_4_5_ENDPOINT"
    [phi4mm]="acousticbench-phi4mm|8006|0|ACOUSTICBENCH_PHI_4_MULTIMODAL_ENDPOINT"
    [voxtral_small_24b]="acousticbench-voxtral-small-24b|8008|0|ACOUSTICBENCH_VOXTRAL_SMALL_24B_ENDPOINT"
    [voxtral_mini_3b]="acousticbench-voxtral-mini-3b|8009|0|ACOUSTICBENCH_VOXTRAL_MINI_3B_ENDPOINT"
    [audio_flamingo3]="acousticbench-audio-flamingo3|8010|0|ACOUSTICBENCH_AUDIO_FLAMINGO_3_ENDPOINT"
    [kimi_lora_rad]="acousticbench-kimi-lora|8014|0|ACOUSTICBENCH_KIMI_AUDIO_LORA_RAD_ENDPOINT|kimi_lora|rad"
    [kimi_lora_rsq]="acousticbench-kimi-lora|8015|0|ACOUSTICBENCH_KIMI_AUDIO_LORA_RSQ_ENDPOINT|kimi_lora|rsq"
    [kimi_lora_ser_classify]="acousticbench-kimi-lora|8016|0|ACOUSTICBENCH_KIMI_AUDIO_LORA_SER_CLASSIFY_ENDPOINT|kimi_lora|ser_classify"
    [kimi_lora_ser_intensity]="acousticbench-kimi-lora|8017|0|ACOUSTICBENCH_KIMI_AUDIO_LORA_SER_INTENSITY_ENDPOINT|kimi_lora|ser_intensity"
    [kimi_lora_simple_audioqa]="acousticbench-kimi-lora|8018|0|ACOUSTICBENCH_KIMI_AUDIO_LORA_SIMPLE_AUDIOQA_ENDPOINT|kimi_lora|simple_audioqa"
    [mimo_audio]="acousticbench-mimo-audio|8019|0|ACOUSTICBENCH_MIMO_AUDIO_7B_ENDPOINT"
    [step_audio2]="acousticbench-step-audio2|8020|0|ACOUSTICBENCH_STEP_AUDIO_2_MINI_ENDPOINT"
)

# run.py model name for each registry key (used in the smoke-test hint).
declare -A MODEL_NAMES=(
    [qwen3_omni]="Qwen3-Omni"
    [kimi_audio]="Kimi-Audio"
    [qwen2_audio]="Qwen2-Audio-7B-Instruct"
    [minicpm_o]="MiniCPM-o-4.5"
    [phi4mm]="Phi-4-Multimodal"
    [voxtral_small_24b]="Voxtral-Small-24B"
    [voxtral_mini_3b]="Voxtral-Mini-3B"
    [audio_flamingo3]="Audio-Flamingo-3"
    [mimo_audio]="MiMo-Audio-7B"
    [step_audio2]="Step-Audio-2-mini"
)

_list_models() {
    echo "Available Docker models:"
    for model in "${!MODELS[@]}"; do
        IFS='|' read -r image port gpu env_var dockerfile_dir adapter_name <<< "${MODELS[$model]}"
        local extra=""
        if [[ -n "${adapter_name:-}" ]]; then
            extra=", adapter=${adapter_name}"
        fi
        echo "  ${model}  (image=${image}, port=${port}, gpu=${gpu}${extra})"
    done | sort
}

_resolve_model() {
    local model="$1"
    if [[ -z "${MODELS[$model]+x}" ]]; then
        echo "Error: unknown model '${model}'"
        _list_models
        exit 1
    fi
    IFS='|' read -r IMAGE PORT GPU ENV_VAR DOCKERFILE_DIR ADAPTER_NAME <<< "${MODELS[$model]}"
    DOCKERFILE_DIR="${DOCKERFILE_DIR:-$model}"
    ADAPTER_NAME="${ADAPTER_NAME:-}"
    DOCKERFILE="${DOCKER_DIR}/${DOCKERFILE_DIR}/Dockerfile"
    CONTAINER_NAME="acousticbench-${model//_/-}"
}

_build() {
    echo "Building ${IMAGE}:latest ..."
    docker build \
        -f "${DOCKERFILE}" \
        -t "${IMAGE}:latest" \
        "${BUILD_CONTEXT}"
    echo "Built ${IMAGE}:latest"
}

_stop() {
    echo "Stopping ${CONTAINER_NAME} ..."
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    echo "Stopped."
}

_run() {
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

    mkdir -p "${HF_CACHE}"

    local adapter_args=()
    if [[ -n "${ADAPTER_NAME}" ]]; then
        local archive_dir="${REPO_ROOT}/archive"
        local export_cache="${HOME}/.cache/acousticbench/kimi-lora-exports"
        mkdir -p "${export_cache}"
        if [[ ! -d "${archive_dir}" ]]; then
            echo "Error: ${archive_dir} not found. The Kimi-Audio LoRA container"
            echo "needs the unzipped adapter archive mounted at /adapters."
            echo "See README \"Download the data\" step."
            exit 1
        fi
        adapter_args=(
            -e "ADAPTER_NAME=${ADAPTER_NAME}"
            -v "${archive_dir}:/adapters:ro"
            -v "${export_cache}:/var/lib/kimi-lora-exports"
        )
    fi

    echo "Starting ${CONTAINER_NAME} on GPU ${GPU}, port ${PORT} ..."
    if [[ -n "${ADAPTER_NAME}" ]]; then
        echo "  adapter=${ADAPTER_NAME}"
    fi

    docker run -d \
        --name "${CONTAINER_NAME}" \
        --init \
        --gpus "\"device=${GPU}\"" \
        -v "${HF_CACHE}:/hf-cache/huggingface" \
        -p "${PORT}:8000" \
        ${HF_TOKEN:+-e HF_TOKEN="${HF_TOKEN}"} \
        "${adapter_args[@]}" \
        "${IMAGE}:latest"

    echo "Container started. Waiting for model to load ..."
    _wait_ready
}

_wait_ready() {
    local url="http://localhost:${PORT}/health"
    local elapsed=0
    local interval=10

    while (( elapsed < HEALTH_TIMEOUT )); do
        local health
        health=$(curl -s "${url}" 2>/dev/null || echo "")

        if echo "${health}" | grep -q '"ready"'; then
            echo ""
            echo "======================================"
            echo "  Model ready!"
            echo "  Endpoint: http://localhost:${PORT}"
            echo ""
            echo "  Export for the harness:"
            echo "    export ${ENV_VAR}=http://localhost:${PORT}"
            echo ""
            echo "  Quick smoke test:"
            if [[ -n "${ADAPTER_NAME}" ]]; then
                local lora_model
                case "${ADAPTER_NAME}" in
                    rad) lora_model="Kimi-Audio-LoRA-RAD" ;;
                    rsq) lora_model="Kimi-Audio-LoRA-RSQ" ;;
                    ser_classify) lora_model="Kimi-Audio-LoRA-SER-Classify" ;;
                    ser_intensity) lora_model="Kimi-Audio-LoRA-SER-Intensity" ;;
                    simple_audioqa) lora_model="Kimi-Audio-LoRA-Simple-AudioQA" ;;
                    *) lora_model="Kimi-Audio-LoRA-${ADAPTER_NAME}" ;;
                esac
                echo "    ${ENV_VAR}=http://localhost:${PORT} python run.py --backend docker --model ${lora_model} --task ${ADAPTER_NAME} --split eval --max-samples 3"
            else
                echo "    ${ENV_VAR}=http://localhost:${PORT} python run.py --backend docker --model ${MODEL_NAMES[$MODEL]:-<model>} --task rsq --max-samples 3"
            fi
            echo "======================================"
            return 0
        fi

        if echo "${health}" | grep -q '"init_error"'; then
            echo ""
            echo "ERROR: Model failed to initialize."
            echo "${health}" | python3 -m json.tool 2>/dev/null || echo "${health}"
            echo ""
            echo "Check logs: docker logs ${CONTAINER_NAME}"
            return 1
        fi

        printf "\r  [%3ds / %ds] Loading ..." "${elapsed}" "${HEALTH_TIMEOUT}"
        sleep "${interval}"
        elapsed=$(( elapsed + interval ))
    done

    echo ""
    echo "ERROR: Timed out after ${HEALTH_TIMEOUT}s waiting for model."
    echo "Check logs: docker logs ${CONTAINER_NAME}"
    return 1
}

# ── Main ────────────────────────────────────────────────────────────────────

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model> [--build-only|--stop|--run-only]"
    echo "       $0 --list"
    exit 1
fi

if [[ "$1" == "--list" ]]; then
    _list_models
    exit 0
fi

MODEL="$1"
shift
ACTION="${1:---default}"

_resolve_model "${MODEL}"

case "${ACTION}" in
    --build-only)
        _build
        ;;
    --run-only)
        _run
        ;;
    --stop)
        _stop
        ;;
    --default)
        _build
        _run
        ;;
    *)
        echo "Unknown action: ${ACTION}"
        echo "Usage: $0 <model> [--build-only|--stop|--run-only]"
        exit 1
        ;;
esac
