#!/bin/bash
# vast_launch.sh -- One-shot vast.ai instance setup + sweep launch
#
# Usage (from eddie):
#   # 1. Push code + data to instance:
#   bash scripts/vast_push.sh HOST PORT
#
#   # 2. Run this script:
#   ssh -p PORT root@HOST 'bash /workspace/imbe_asr/scripts/vast_launch.sh'
#
# Override defaults via env:
#   SWEEP_ID=luxprimatech/imbe-asr/XXXXX NUM_GPUS=1 ssh -p PORT root@HOST \
#       'bash /workspace/imbe_asr/scripts/vast_launch.sh'

set -euo pipefail

cd /workspace/imbe_asr

# ---- Config ----
SWEEP_ID="${SWEEP_ID:-luxprimatech/imbe-asr/qo59skgh}"

export SWEEP_EPOCHS="${SWEEP_EPOCHS:-5}"
export SWEEP_BATCH_SIZE="${SWEEP_BATCH_SIZE:-32}"
export SWEEP_ACCUM_STEPS="${SWEEP_ACCUM_STEPS:-8}"
export SWEEP_WORKERS="${SWEEP_WORKERS:-4}"
export MMAP_DIR="data/packed"
export WANDB_API_KEY="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"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

echo "=== IMBE-ASR Quick Launch ==="
echo "GPUs: $NUM_GPUS | Sweep: $SWEEP_ID"
echo "Batch: ${SWEEP_BATCH_SIZE} x ${SWEEP_ACCUM_STEPS} accum"

# ---- Step 1: Deps + decompress data in parallel ----
echo "[1/2] Installing deps + preparing data..."

(apt-get update -qq && apt-get install -y -qq pigz 2>/dev/null) &
APT_PID=$!
pip install -q wandb 2>&1 | tail -1 &
PIP_PID=$!

mkdir -p "$MMAP_DIR"
if [ ! -f "$MMAP_DIR/all.features.bin" ]; then
    if ls "$MMAP_DIR"/chunks/features.gz.* 1>/dev/null 2>&1; then
        echo "  Reassembling + decompressing chunks..."
        cat "$MMAP_DIR"/chunks/features.gz.* | pigz -d -p $(nproc) \
            > "$MMAP_DIR/all.features.bin"
        rm -rf "$MMAP_DIR/chunks"
        echo "  Data: $(du -sh $MMAP_DIR)"
    else
        echo "ERROR: No data found. Run vast_push.sh first."
        exit 1
    fi
else
    echo "  Data present: $(du -sh $MMAP_DIR)"
fi

wait $PIP_PID 2>/dev/null || true
wait $APT_PID 2>/dev/null || true

# ---- Step 2: Launch agents ----
echo "[2/2] Launching sweep agents..."
wandb login --relogin "$WANDB_API_KEY" 2>/dev/null

pkill -9 -f "wandb agent" 2>/dev/null || true
pkill -9 -f "sweep_agent" 2>/dev/null || true
sleep 1

for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  GPU $i -> /tmp/sweep_gpu${i}.log"
    CUDA_VISIBLE_DEVICES=$i nohup wandb agent --count 20 "$SWEEP_ID" \
        > /tmp/sweep_gpu${i}.log 2>&1 &
done

echo ""
echo "=== Ready. Monitor: tail -f /tmp/sweep_gpu*.log ==="
