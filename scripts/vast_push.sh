#!/bin/bash
# vast_push.sh -- Push code + compressed data chunks to a vast.ai instance
#
# Sends code from eddie, data chunks from sarah, all in parallel.
# Then run vast_launch.sh on the instance to decompress + start.
#
# Usage:
#   bash scripts/vast_push.sh HOST PORT
#
# Example:
#   bash scripts/vast_push.sh ssh1.vast.ai 24030

set -euo pipefail

HOST="${1:?Usage: vast_push.sh HOST PORT}"
PORT="${2:?Usage: vast_push.sh HOST PORT}"
SSH="ssh -o StrictHostKeyChecking=no -p $PORT root@$HOST"
SCP="scp -o StrictHostKeyChecking=no -P $PORT"

SARAH="${SARAH_HOST:-user@training-server}"
SARAH_CHUNKS="/mnt/disk/imbe_asr/data/packed/chunks"
SARAH_META="/mnt/disk/imbe_asr/data/packed/all.meta.pkl"

echo "=== Pushing to $HOST:$PORT ==="

# Create dirs on instance
$SSH 'mkdir -p /workspace/imbe_asr/data/packed/chunks'

# ---- Push code from eddie (fast, <5s) ----
echo "[1/3] Pushing code..."
rsync -avz --quiet -e "ssh -o StrictHostKeyChecking=no -p $PORT" \
    --exclude data/ --exclude checkpoints/ --exclude .git/ --exclude __pycache__/ \
    ~/imbe_asr/ root@$HOST:/workspace/imbe_asr/ &
CODE_PID=$!

# ---- Push data chunks from sarah in parallel ----
echo "[2/3] Pushing data chunks from sarah (5 x 2GB parallel)..."
CHUNKS=$(ssh $SARAH "ls $SARAH_CHUNKS/features.gz.*")
for chunk in $CHUNKS; do
    fname=$(basename "$chunk")
    echo "  $fname..."
    ssh $SARAH "scp -o StrictHostKeyChecking=no -P $PORT $chunk root@$HOST:/workspace/imbe_asr/data/packed/chunks/$fname" &
done

# Push meta file (small, 166MB)
echo "[3/3] Pushing meta..."
ssh $SARAH "scp -o StrictHostKeyChecking=no -P $PORT $SARAH_META root@$HOST:/workspace/imbe_asr/data/packed/all.meta.pkl" &

# Wait for everything
wait
echo ""
echo "=== Push complete. Now run: ==="
echo "  ssh -p $PORT root@$HOST 'bash /workspace/imbe_asr/scripts/vast_launch.sh'"
