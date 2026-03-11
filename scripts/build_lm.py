#!/usr/bin/env python3
"""Build a KenLM n-gram language model from P25 transcripts.

Extracts transcripts from P25 NPZ pairs, HF datasets, and/or LibriSpeech.
Normalizes text to match CTC tokenizer vocabulary, trains a KenLM model.

Requires KenLM binaries (lmplz, build_binary).

Usage:
    python scripts/build_lm.py \
        --p25-pairs data/p25_raw \
        --librispeech data/LibriSpeech/train-clean-100 \
        --output data/lm/
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path


def normalize_transcript(text):
    """Normalize text to match CTC tokenizer vocabulary."""
    text = str(text).upper()
    text = text.replace("-", " ")
    text = re.sub(r"[^A-Z0-9 ']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_p25_npz(pairs_dir):
    """Extract transcripts from P25 NPZ pair files."""
    import numpy as np

    pairs_dir = Path(pairs_dir)
    transcripts = []
    n_files = 0

    for npz_path in sorted(pairs_dir.glob("*.npz")):
        try:
            d = np.load(str(npz_path), allow_pickle=True)
            text = str(d["transcript"])
            text = normalize_transcript(text)
            if len(text) >= 3:
                transcripts.append(text)
            n_files += 1
        except Exception:
            continue

    print("  P25 NPZ: %d transcripts from %d files" % (len(transcripts), n_files))
    return transcripts


def extract_librispeech(ls_dir):
    """Extract transcripts from LibriSpeech .trans.txt files."""
    ls_dir = Path(ls_dir)
    transcripts = []

    for trans_file in ls_dir.glob("*/*/*.trans.txt"):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    text = normalize_transcript(parts[1])
                    if len(text) >= 3:
                        transcripts.append(text)

    print("  LibriSpeech: %d transcripts" % len(transcripts))
    return transcripts


def build_arpa(corpus_path, output_dir, order=5):
    """Train KenLM ARPA model and convert to binary format."""
    lmplz = shutil.which("lmplz")
    build_binary = shutil.which("build_binary")

    if not lmplz or not build_binary:
        for prefix in ["/usr/local/bin", os.path.expanduser("~/.local/bin")]:
            lp = os.path.join(prefix, "lmplz")
            bb = os.path.join(prefix, "build_binary")
            if os.path.isfile(lp) and os.path.isfile(bb):
                lmplz, build_binary = lp, bb
                break

    if not lmplz or not build_binary:
        print("ERROR: KenLM binaries (lmplz, build_binary) not found.")
        print("Install: git clone https://github.com/kpu/kenlm.git && "
              "cd kenlm && mkdir build && cd build && cmake .. && make -j4")
        sys.exit(1)

    arpa_path = os.path.join(output_dir, "%dgram.arpa" % order)
    binary_path = os.path.join(output_dir, "%dgram.bin" % order)

    print("Training %d-gram model..." % order)
    with open(corpus_path, "r") as corpus_f, open(arpa_path, "w") as arpa_f:
        proc = subprocess.run(
            [lmplz, "-o", str(order), "--discount_fallback"],
            stdin=corpus_f, stdout=arpa_f, stderr=subprocess.PIPE, text=True,
        )
        if proc.returncode != 0:
            print("  lmplz stderr: %s" % proc.stderr[:1000])
            raise RuntimeError("lmplz failed with code %d" % proc.returncode)

    print("  ARPA: %s (%.1f MB)" %
          (arpa_path, os.path.getsize(arpa_path) / 1e6))

    print("Converting to binary...")
    proc = subprocess.run(
        [build_binary, arpa_path, binary_path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print("  Warning: binary conversion failed, using ARPA")
        return arpa_path

    print("  Binary: %s (%.1f MB)" %
          (binary_path, os.path.getsize(binary_path) / 1e6))
    return binary_path


def extract_unigrams(transcripts, min_count=2):
    """Extract word unigrams from transcripts for pyctcdecode lexicon."""
    word_counts = Counter()
    for text in transcripts:
        for word in text.split():
            word_counts[word] += 1

    unigrams = sorted(w for w, c in word_counts.items() if c >= min_count)
    print("  Unigrams: %d words (min_count=%d)" % (len(unigrams), min_count))
    return unigrams


def main():
    parser = argparse.ArgumentParser(
        description="Build KenLM language model from transcripts")
    parser.add_argument("--p25-pairs", default=None)
    parser.add_argument("--librispeech", default=None)
    parser.add_argument("--corpus", default=None,
                        help="Pre-existing text corpus (skip extraction)")
    parser.add_argument("--output", default="data/lm/")
    parser.add_argument("--order", type=int, default=5)
    parser.add_argument("--min-word-count", type=int, default=2)
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    corpus_path = os.path.join(args.output, "corpus.txt")
    unigrams_path = os.path.join(args.output, "unigrams.txt")

    if args.corpus:
        corpus_path = args.corpus
        print("Using existing corpus: %s" % corpus_path)
        with open(corpus_path) as f:
            transcripts = [line.strip() for line in f if line.strip()]
        print("  %d lines" % len(transcripts))
    else:
        transcripts = []

        if args.p25_pairs:
            print("Extracting P25 NPZ transcripts...")
            transcripts.extend(extract_p25_npz(args.p25_pairs))

        if args.librispeech:
            print("Extracting LibriSpeech transcripts...")
            transcripts.extend(extract_librispeech(args.librispeech))

        if not transcripts:
            print("ERROR: No transcripts found. "
                  "Provide --p25-pairs, --librispeech, or --corpus.")
            sys.exit(1)

        unique = list(set(transcripts))
        print("\nTotal: %d transcripts, %d unique" %
              (len(transcripts), len(unique)))
        transcripts = unique

        with open(corpus_path, "w") as f:
            for text in transcripts:
                f.write(text + "\n")
        print("Corpus saved: %s" % corpus_path)

    # Unigrams
    unigrams = extract_unigrams(transcripts, min_count=args.min_word_count)
    with open(unigrams_path, "w") as f:
        for word in unigrams:
            f.write(word + "\n")
    print("Unigrams saved: %s" % unigrams_path)

    if args.skip_build:
        print("Skipping LM training (--skip-build).")
        return

    model_path = build_arpa(corpus_path, args.output, order=args.order)

    print("\nDone. LM model: %s" % model_path)
    print("Unigrams: %s" % unigrams_path)


if __name__ == "__main__":
    main()
