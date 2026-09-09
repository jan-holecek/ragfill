#!/bin/bash
set -e

set -a
source .env
set +a

DEVICE=${DEVICE:-cpu}

if [ "$DEVICE" = "rocm" ] && [ "$ROCM_IGPU" = "1" ]; then
    echo "Configuring ROCm iGPU support"

    HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
    HCC_AMDGPU_TARGET="${HCC_AMDGPU_TARGET:-gfx1100}"

    sed -i "s|HSA_OVERRIDE_GFX_VERSION=.*|HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}|" .env
    sed -i "s|HCC_AMDGPU_TARGET=.*|HCC_AMDGPU_TARGET=${HCC_AMDGPU_TARGET}|" .env
    sed -i "s|ROCM_IGPU=.*|ROCM_IGPU=1|" .env
else
    sed -i "s|HSA_OVERRIDE_GFX_VERSION=.*|HSA_OVERRIDE_GFX_VERSION=|" .env
    sed -i "s|HCC_AMDGPU_TARGET=.*|HCC_AMDGPU_TARGET=|" .env
    sed -i "s|ROCM_IGPU=.*|ROCM_IGPU=0|" .env
fi

echo "Starting services (device: ${DEVICE})"
sudo docker compose --profile ${DEVICE} up -d

if [ "$DEVICE" = "cuda" ]; then
    echo "Waiting for vLLM generation"
    until curl -fsS http://localhost:8000/health > /dev/null 2>&1; do
        echo "Waiting..."
        sleep 10
    done

    echo "Waiting for vLLM embedding"
    until curl -fsS http://localhost:8001/health > /dev/null 2>&1; do
        echo "Waiting..."
        sleep 10
    done

    sed -i "s|LLM__LITELLM_MODEL=.*|LLM__LITELLM_MODEL=openai/${LLM__HF_MODEL}|" .env
    sed -i "s|EMBEDDING__LITELLM_MODEL=.*|EMBEDDING__LITELLM_MODEL=openai/${EMBEDDING__HF_MODEL}|" .env
    sed -i "s|REWRITE__LITELLM_MODEL=.*|REWRITE__LITELLM_MODEL=openai/${LLM__HF_MODEL}|" .env
    sed -i "s|LLM__API_BASE=.*|LLM__API_BASE=http://localhost:8000/v1|" .env
    sed -i "s|EMBEDDING__API_BASE=.*|EMBEDDING__API_BASE=http://localhost:8001/v1|" .env
    sed -i "s|REWRITE__API_BASE=.*|REWRITE__API_BASE=http://localhost:8000/v1|" .env

else
    echo "Waiting for Ollama"
    until curl -fsS http://localhost:11434/api/tags > /dev/null 2>&1; do
        echo "Waiting..."
        sleep 5
    done

    echo "Pulling models"
    sudo docker exec ragfill-ollama ollama pull ${EMBEDDING__MODEL}
    sudo docker exec ragfill-ollama ollama pull ${LLM__MODEL}

    if [ -n "${LLM__CONTEXT}" ] && [ -n "${LLM__CUSTOM_MODEL}" ]; then
        echo "Creating custom LLM model with extended context"
        sudo docker exec ragfill-ollama sh -c "printf 'FROM ${LLM__MODEL}\nPARAMETER num_ctx ${LLM__CONTEXT}' > /tmp/Modelfile && ollama create ${LLM__CUSTOM_MODEL} -f /tmp/Modelfile"
    fi

    if [ -n "${REWRITE__CONTEXT}" ] && [ -n "${REWRITE__CUSTOM_MODEL}" ]; then
        echo "Creating custom rewrite model with limited context"
        sudo docker exec ragfill-ollama sh -c "printf 'FROM ${LLM__MODEL}\nPARAMETER num_ctx ${REWRITE__CONTEXT}' > /tmp/Modelfile && ollama create ${REWRITE__CUSTOM_MODEL} -f /tmp/Modelfile"
    fi

    sed -i "s|LLM__LITELLM_MODEL=.*|LLM__LITELLM_MODEL=ollama/${LLM__CUSTOM_MODEL:-${LLM__MODEL}}|" .env
    sed -i "s|EMBEDDING__LITELLM_MODEL=.*|EMBEDDING__LITELLM_MODEL=ollama/${EMBEDDING__MODEL}|" .env
    sed -i "s|REWRITE__LITELLM_MODEL=.*|REWRITE__LITELLM_MODEL=ollama/${REWRITE__CUSTOM_MODEL:-${LLM__CUSTOM_MODEL:-${LLM__MODEL}}}|" .env
    sed -i "s|LLM__API_BASE=.*|LLM__API_BASE=http://localhost:11434|" .env
    sed -i "s|EMBEDDING__API_BASE=.*|EMBEDDING__API_BASE=http://localhost:11434|" .env
    sed -i "s|REWRITE__API_BASE=.*|REWRITE__API_BASE=http://localhost:11434|" .env
fi

echo ""
echo "Device:          ${DEVICE}$([ "$DEVICE" = "rocm" ] && [ "$ROCM_IGPU" = "1" ] && echo " (iGPU: ${HSA_OVERRIDE_GFX_VERSION})")"
echo "LLM model:       ${LLM__CUSTOM_MODEL:-${LLM__HF_MODEL:-${LLM__MODEL}}}"
echo "Rewrite model:   ${REWRITE__CUSTOM_MODEL:-${LLM__CUSTOM_MODEL:-${LLM__MODEL}}} (ctx: ${REWRITE__CONTEXT:-default})"
echo "Embedding model: ${EMBEDDING__HF_MODEL:-${EMBEDDING__MODEL}}"
echo "LiteLLM:         http://localhost:4000"
echo "MongoDB:         mongodb://localhost:27017"
echo "Elasticsearch:   http://localhost:9200"