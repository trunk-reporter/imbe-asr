#!/usr/bin/env python3
"""Download and prepare GigaSpeech S subset (250h) for IMBE-ASR training.

Uses HuggingFace datasets library to download GigaSpeech, strips punctuation
tags, generates a TSV manifest, then IMBE-encodes and precomputes raw params.

Prerequisites:
    pip install datasets soundfile
    # Accept license at https://huggingface.co/datasets/speechcolab/gigaspeech
    # huggingface-cli login

Usage:
    python scripts/prepare_gigaspeech.py --output data --subset s

    # Smaller test run:
    python scripts/prepare_gigaspeech.py --output data --subset xs
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


# GigaSpeech uses these tags for punctuation
GS_PUNCT_TAGS = re.compile(
    r"<(COMMA|PERIOD|QUESTIONMARK|EXCLAMATIONPOINT|SIL|MUSIC|NOISE|OTHER)>",
    re.IGNORECASE)


def normalize_transcript(text):
    """Normalize GigaSpeech transcript to match CTC tokenizer vocabulary."""
    text = str(text).upper()
    # Remove GigaSpeech punctuation/noise tags
    text = GS_PUNCT_TAGS.sub("", text)
    text = text.replace("-", " ")
    text = re.sub(r"[^A-Z0-9 ']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_audio_segments(dataset, audio_dir, manifest_path):
    """Extract audio segments from HF dataset and write manifest.

    Args:
        dataset: HuggingFace dataset split
        audio_dir: Directory to write WAV segments
        manifest_path: Path for output TSV manifest
    """
    os.makedirs(audio_dir, exist_ok=True)

    with open(manifest_path, "w") as out:
        out.write("audio_path\tutterance_id\ttranscript\tgroup_key\tstart\tend\n")

        n_written = 0
        n_skipped = 0

        for i, example in enumerate(dataset):
            # Extract audio
            audio_data = example["audio"]
            audio_array = np.array(audio_data["array"], dtype=np.float64)
            sr = audio_data["sampling_rate"]

            # Get segment/show identifiers
            segment_id = example.get("segment_id",
                                     example.get("id", "seg_%08d" % i))
            audio_id = example.get("audio_id", "unknown")

            # Use first part of audio_id as group key (show/source)
            # GigaSpeech audio_ids look like "POD0000000001_S0000000"
            group_key = audio_id.split("_")[0] if "_" in audio_id else audio_id

            # Normalize transcript
            text = normalize_transcript(example.get("text", ""))
            if len(text) < 2:
                n_skipped += 1
                continue

            # Skip very short or very long segments
            duration = len(audio_array) / sr
            if duration < 0.2 or duration > 40.0:
                n_skipped += 1
                continue

            # Write audio to WAV
            wav_path = Path(audio_dir) / group_key / ("%s.wav" % segment_id)
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(wav_path), audio_array, sr)

            out.write("%s\t%s\t%s\t%s\t\t\n" % (
                wav_path, segment_id, text, group_key))
            n_written += 1

            if (i + 1) % 5000 == 0:
                print("  %d processed, %d written, %d skipped" %
                      (i + 1, n_written, n_skipped))

    print("Manifest: %d segments -> %s (skipped %d)" %
          (n_written, manifest_path, n_skipped))
    return n_written


def main():
    parser = argparse.ArgumentParser(
        description="Prepare GigaSpeech for IMBE-ASR training")
    parser.add_argument("--output", default="data",
                        help="Base output directory")
    parser.add_argument("--subset", default="s",
                        choices=["xs", "s", "m", "l", "xl"],
                        help="GigaSpeech subset size (default: s = 250h)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use existing manifest")
    parser.add_argument("--skip-encode", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    audio_dir = os.path.join(args.output, "gigaspeech_segments")
    manifest_path = os.path.join(args.output, "gigaspeech_manifest.tsv")

    if not args.skip_download:
        if os.path.exists(manifest_path):
            print("Manifest already exists: %s" % manifest_path)
        else:
            print("Loading GigaSpeech '%s' from HuggingFace..." % args.subset)
            try:
                from datasets import load_dataset
            except ImportError:
                print("ERROR: 'datasets' package not installed.")
                print("  pip install datasets")
                sys.exit(1)

            try:
                ds = load_dataset(
                    "speechcolab/gigaspeech",
                    args.subset,
                    split="train",
                )
            except Exception as e:
                if "gated" in str(e).lower() or "access" in str(e).lower():
                    print("\nERROR: GigaSpeech is a gated dataset.")
                    print("1. Visit https://huggingface.co/datasets/speechcolab/gigaspeech")
                    print("2. Accept the license agreement")
                    print("3. Run: huggingface-cli login")
                    print("4. Re-run this script")
                    sys.exit(1)
                raise

            print("GigaSpeech %s: %d examples" % (args.subset, len(ds)))
            extract_audio_segments(ds, audio_dir, manifest_path)

    if args.skip_encode:
        print("Skipping IMBE encoding (--skip-encode)")
        return

    # IMBE encode
    pairs_dir = os.path.join(args.output, "pairs_gigaspeech")
    print("\nIMBE-encoding GigaSpeech segments...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run([
        sys.executable, os.path.join(script_dir, "imbe_encode.py"),
        "--manifest", manifest_path,
        "--output", pairs_dir,
        "--workers", str(args.workers),
    ], check=True)

    # Precompute raw params
    print("\nPrecomputing 170-dim raw IMBE parameters...")
    subprocess.run([
        sys.executable, "-m", "src.precompute",
        "--pairs-dir", pairs_dir,
        "--workers", str(args.workers),
    ], check=True)

    print("\nDone. GigaSpeech pairs at: %s" % pairs_dir)


if __name__ == "__main__":
    main()
