#!/bin/bash
# entrypoint.sh — Download model from HuggingFace if not already present, then start server.
set -e

MODELS_DIR="${MODELS_DIR:-/models}"
MODEL_REPO="${IMBE_ASR_MODEL:-trunk-reporter/imbe-asr-base-512d-p25}"

# Map model repo to recommended LM params if not explicitly set
case "$MODEL_REPO" in
    *large-1024d*)
        : "${IMBE_ASR_LM_ALPHA:=0.7}"
        : "${IMBE_ASR_LM_BETA:=2.0}"
        LM_FILE="lm/5gram.bin"
        ;;
    *base-512d-p25*)
        : "${IMBE_ASR_LM_ALPHA:=0.5}"
        : "${IMBE_ASR_LM_BETA:=1.0}"
        LM_FILE="lm/3gram.bin"
        ;;
    *base-512d*)
        : "${IMBE_ASR_LM_ALPHA:=0.7}"
        : "${IMBE_ASR_LM_BETA:=2.0}"
        LM_FILE="lm/3gram.bin"
        ;;
    *)
        : "${IMBE_ASR_LM_ALPHA:=0.5}"
        : "${IMBE_ASR_LM_BETA:=1.0}"
        LM_FILE="lm/3gram.bin"
        ;;
esac

export IMBE_ASR_LM_ALPHA IMBE_ASR_LM_BETA

# Download model if checkpoint not present
CHECKPOINT="${IMBE_ASR_CHECKPOINT:-${MODELS_DIR}/model_int8.onnx}"
if [ ! -f "$CHECKPOINT" ]; then
    echo "Model not found at $CHECKPOINT — downloading $MODEL_REPO from HuggingFace..."
    python3 -c "
import os, sys
from huggingface_hub import snapshot_download
repo = os.environ.get('IMBE_ASR_MODEL', 'trunk-reporter/imbe-asr-base-512d-p25')
local_dir = os.environ.get('MODELS_DIR', '/models')
print(f'Downloading {repo} → {local_dir}')
snapshot_download(repo, local_dir=local_dir)
print('Download complete.')
"
else
    echo "Model found at $CHECKPOINT"
fi

# Set LM paths from downloaded model dir if not explicitly set
if [ -z "$IMBE_ASR_LM_PATH" ] && [ -f "${MODELS_DIR}/${LM_FILE}" ]; then
    export IMBE_ASR_LM_PATH="${MODELS_DIR}/${LM_FILE}"
    export IMBE_ASR_UNIGRAMS_PATH="${MODELS_DIR}/lm/unigrams.txt"
    echo "Using LM: $IMBE_ASR_LM_PATH (alpha=$IMBE_ASR_LM_ALPHA, beta=$IMBE_ASR_LM_BETA)"
fi

# Set default checkpoint/stats from models dir if not explicitly set
export IMBE_ASR_CHECKPOINT="${IMBE_ASR_CHECKPOINT:-${MODELS_DIR}/model_int8.onnx}"
export IMBE_ASR_STATS="${IMBE_ASR_STATS:-${MODELS_DIR}/stats.npz}"

echo "Starting imbe-asr server (model=$MODEL_REPO, device=${IMBE_ASR_DEVICE:-cuda})..."
exec uvicorn server.app:app --host 0.0.0.0 --port 8000
