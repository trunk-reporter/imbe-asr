#!/bin/bash
# Download and prepare LibriSpeech train-clean-100 for IMBE-ASR training.
#
# Prerequisites:
#   - generate_pairs.py from ~/p25_audio_quality/ (IMBE-encodes audio to NPZ)
#   - libimbe.so built in vocoder/
#
# Usage:
#   bash scripts/prepare_librispeech.sh

set -euo pipefail

DATA_DIR="${1:-data}"
LIBRISPEECH_URL="https://www.openslr.org/resources/12/train-clean-100.tar.gz"

mkdir -p "$DATA_DIR"

# Download LibriSpeech if not present
if [ ! -d "$DATA_DIR/LibriSpeech/train-clean-100" ]; then
    echo "Downloading LibriSpeech train-clean-100..."
    wget -c "$LIBRISPEECH_URL" -O "$DATA_DIR/train-clean-100.tar.gz"
    tar xzf "$DATA_DIR/train-clean-100.tar.gz" -C "$DATA_DIR"
    rm "$DATA_DIR/train-clean-100.tar.gz"
    echo "Downloaded to $DATA_DIR/LibriSpeech/train-clean-100"
else
    echo "LibriSpeech already present at $DATA_DIR/LibriSpeech/train-clean-100"
fi

# Generate IMBE pairs (NPZ files with frame_vectors + audio)
if [ ! -d "$DATA_DIR/pairs" ]; then
    echo "Generating IMBE pairs..."
    python ~/p25_audio_quality/generate_pairs.py \
        --librispeech "$DATA_DIR/LibriSpeech/train-clean-100" \
        --output "$DATA_DIR/pairs" \
        --workers 12
    echo "Generated pairs to $DATA_DIR/pairs"
else
    echo "Pairs already present at $DATA_DIR/pairs"
fi

# Precompute 170-dim raw IMBE params from frame_vectors
echo "Precomputing 170-dim raw IMBE parameters..."
python -m src.precompute --pairs-dir "$DATA_DIR/pairs" --workers 12

echo ""
echo "Done. Ready for training:"
echo "  python -m src.train --pairs-dir $DATA_DIR/pairs --librispeech-dir $DATA_DIR/LibriSpeech/train-clean-100"
