#!/bin/bash
# vast_sweep.sh -- Run W&B hyperparameter sweep on a vast.ai instance
#
# Required env vars:
#   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY  -- OVH S3 credentials
#   WANDB_API_KEY                              -- Weights & Biases API key
#
# Optional env vars:
#   SWEEP_ID       -- existing sweep ID to join (skips creation)
#   SWEEP_COUNT    -- number of trials to run (default: 20)
#   SWEEP_EPOCHS   -- epochs per trial (default: 5)
#   SWEEP_BATCH_SIZE -- per-GPU batch size (default: 16)
#   SWEEP_ACCUM_STEPS -- gradient accumulation (default: 4)
#
# Usage:
#   export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... WANDB_API_KEY=...
#   bash scripts/vast_sweep.sh

set -euo pipefail

S3_ENDPOINT="https://s3.us-east-va.io.cloud.ovh.us"
S3_BUCKET="s3://breezy-peter/imbe-asr"
DATA_DIR="data/packed"

SWEEP_COUNT="${SWEEP_COUNT:-20}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-5}"
SWEEP_BATCH_SIZE="${SWEEP_BATCH_SIZE:-16}"
SWEEP_ACCUM_STEPS="${SWEEP_ACCUM_STEPS:-4}"

export SWEEP_EPOCHS SWEEP_BATCH_SIZE SWEEP_ACCUM_STEPS
export MMAP_DIR="$DATA_DIR"

echo "=== IMBE-ASR W&B Sweep Launcher ==="
echo "Trials: $SWEEP_COUNT, Epochs/trial: $SWEEP_EPOCHS"
echo "Batch: $SWEEP_BATCH_SIZE x $SWEEP_ACCUM_STEPS accum"

# ---- Install dependencies ----
echo "Installing dependencies..."
pip install -q torch numpy pyyaml wandb awscli

# ---- Pull data from S3 ----
mkdir -p "$DATA_DIR"

if [ ! -f "$DATA_DIR/all.features.bin" ]; then
    echo "Downloading training data from S3..."
    aws s3 cp "$S3_BUCKET/packed/all.features.bin" "$DATA_DIR/all.features.bin" \
        --endpoint-url "$S3_ENDPOINT"
    aws s3 cp "$S3_BUCKET/packed/all.meta.pkl" "$DATA_DIR/all.meta.pkl" \
        --endpoint-url "$S3_ENDPOINT"
    echo "Data downloaded: $(du -sh $DATA_DIR)"
else
    echo "Data already present: $(du -sh $DATA_DIR)"
fi

# ---- Create or join sweep ----
if [ -n "${SWEEP_ID:-}" ]; then
    echo "Joining existing sweep: $SWEEP_ID"
else
    echo "Creating new sweep..."
    SWEEP_ID=$(wandb sweep configs/sweep.yaml --project imbe-asr 2>&1 | \
        grep -oP 'wandb agent \K\S+' || true)
    if [ -z "$SWEEP_ID" ]; then
        echo "ERROR: Failed to create sweep. Run manually:"
        echo "  wandb sweep configs/sweep.yaml --project imbe-asr"
        exit 1
    fi
    echo "Created sweep: $SWEEP_ID"
fi

# ---- Run sweep agent ----
echo ""
echo "Launching sweep agent ($SWEEP_COUNT trials)..."
wandb agent --count "$SWEEP_COUNT" "$SWEEP_ID"

echo "Sweep complete."
