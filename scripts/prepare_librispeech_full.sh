#!/bin/bash
# Download and prepare LibriSpeech train-clean-360 and train-other-500.
#
# Generates TSV manifests from .trans.txt files, IMBE-encodes via
# imbe_encode.py, then precomputes 170-dim raw IMBE parameters.
#
# Prerequisites:
#   - vocoder/libimbe.so (symlink or copy)
#   - Python deps: soundfile, scipy, numpy
#
# Usage:
#   bash scripts/prepare_librispeech_full.sh [data_dir]

set -euo pipefail

DATA_DIR="${1:-data}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

LS_BASE_URL="https://www.openslr.org/resources/12"

# --------------------------------------------------------------------------- #
#  Download                                                                     #
# --------------------------------------------------------------------------- #

download_split() {
    local split="$1"
    local url="${LS_BASE_URL}/${split}.tar.gz"

    if [ -d "$DATA_DIR/LibriSpeech/$split" ]; then
        echo "LibriSpeech $split already present"
        return
    fi

    echo "Downloading LibriSpeech $split..."
    wget -c "$url" -O "$DATA_DIR/${split}.tar.gz"
    tar xzf "$DATA_DIR/${split}.tar.gz" -C "$DATA_DIR"
    rm "$DATA_DIR/${split}.tar.gz"
    echo "Downloaded to $DATA_DIR/LibriSpeech/$split"
}

mkdir -p "$DATA_DIR"
download_split "train-clean-360"
download_split "train-other-500"

# --------------------------------------------------------------------------- #
#  Generate manifests                                                           #
# --------------------------------------------------------------------------- #

generate_manifest() {
    local split="$1"
    local ls_dir="$DATA_DIR/LibriSpeech/$split"
    local manifest="$DATA_DIR/${split}_manifest.tsv"

    if [ -f "$manifest" ]; then
        echo "Manifest already exists: $manifest"
        return
    fi

    echo "Generating manifest for $split..."
    python3 -c "
import sys
from pathlib import Path

ls_dir = Path('$ls_dir')
out = open('$manifest', 'w')
out.write('audio_path\tutterance_id\ttranscript\tgroup_key\tstart\tend\n')

n = 0
for trans_file in sorted(ls_dir.glob('*/*/*.trans.txt')):
    speaker = trans_file.parent.parent.name
    with open(trans_file) as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) != 2:
                continue
            utt_id, text = parts
            flac_path = trans_file.parent / (utt_id + '.flac')
            if not flac_path.exists():
                continue
            out.write('%s\t%s\t%s\t%s\t\t\n' % (flac_path, utt_id, text, speaker))
            n += 1

out.close()
print('  %d utterances in manifest' % n)
"
}

generate_manifest "train-clean-360"
generate_manifest "train-other-500"

# --------------------------------------------------------------------------- #
#  IMBE encode                                                                  #
# --------------------------------------------------------------------------- #

encode_split() {
    local split="$1"
    local pairs_dir="$2"
    local manifest="$DATA_DIR/${split}_manifest.tsv"

    if [ -d "$pairs_dir" ] && [ "$(find "$pairs_dir" -name '*.npz' | head -1)" ]; then
        echo "IMBE pairs already exist: $pairs_dir"
    else
        echo "IMBE-encoding $split..."
        python3 "$SCRIPT_DIR/imbe_encode.py" \
            --manifest "$manifest" \
            --output "$pairs_dir" \
            --workers 12
    fi
}

encode_split "train-clean-360" "$DATA_DIR/pairs_ls360"
encode_split "train-other-500" "$DATA_DIR/pairs_ls500"

# --------------------------------------------------------------------------- #
#  Precompute 170-dim raw IMBE params                                           #
# --------------------------------------------------------------------------- #

echo "Precomputing raw IMBE parameters for train-clean-360..."
python3 -m src.precompute --pairs-dir "$DATA_DIR/pairs_ls360" --workers 12

echo "Precomputing raw IMBE parameters for train-other-500..."
python3 -m src.precompute --pairs-dir "$DATA_DIR/pairs_ls500" --workers 12

# --------------------------------------------------------------------------- #
#  Done                                                                         #
# --------------------------------------------------------------------------- #

echo ""
echo "Done. New pairs directories:"
echo "  $DATA_DIR/pairs_ls360/"
echo "  $DATA_DIR/pairs_ls500/"
echo ""
echo "Add to data config YAML:"
echo "  - pairs_dir: $DATA_DIR/pairs_ls360"
echo "    transcript_source: librispeech"
echo "    librispeech_dir: $DATA_DIR/LibriSpeech/train-clean-360"
echo "  - pairs_dir: $DATA_DIR/pairs_ls500"
echo "    transcript_source: librispeech"
echo "    librispeech_dir: $DATA_DIR/LibriSpeech/train-other-500"
