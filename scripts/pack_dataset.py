#!/usr/bin/env python3
"""Pack NPZ dataset files into a single memory-mapped binary for fast loading.

Reads scan cache pickle files (or scans NPZ files directly) and produces:
  - {name}.features.bin  -- flat float32 binary of all raw_params concatenated
  - {name}.meta.pkl      -- pickle with per-utterance metadata for indexing

Usage:
    python scripts/pack_dataset.py \
        --data-config configs/data_expanded.yaml \
        --output-dir data/packed
"""

import argparse
import os
import pickle
import struct
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# Add project root to path so we can import src modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tokenizer import encode

RAW_PARAM_DIM = 170


def _load_librispeech_transcripts(librispeech_dir):
    """Load transcripts from LibriSpeech .trans.txt files."""
    ls_dir = Path(librispeech_dir)
    transcripts = {}
    for trans_file in ls_dir.glob("*/*/*.trans.txt"):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1].upper()
    return transcripts


def _scan_one_npz(args):
    """Worker: scan a single NPZ file. Returns entry tuple or None."""
    npz_path, group_key, transcript_source, transcripts = args
    try:
        d = np.load(npz_path, allow_pickle=True)
        if "raw_params" not in d:
            return None
        n_frames = d["raw_params"].shape[0]

        utt_id = Path(npz_path).stem
        if transcript_source == "librispeech":
            if utt_id not in transcripts:
                return None
            text = transcripts[utt_id]
        elif transcript_source == "embedded":
            if "transcript" not in d:
                return None
            text = str(d["transcript"]).strip().upper()
        else:
            return None
    except Exception:
        return None

    tokens = encode(text)
    if len(tokens) == 0:
        return None

    return (npz_path, n_frames, tokens, group_key)


def _scan_source_for_pack(source_cfg, min_frames=10, max_frames=2000,
                          workers=12):
    """Scan one data source and return list of (npz_path, n_frames, tokens, group_key).

    Loads from scan cache if available, otherwise scans NPZ files in parallel.
    """
    pairs_dir = Path(source_cfg["pairs_dir"])
    transcript_source = source_cfg.get("transcript_source", "embedded")

    if not pairs_dir.exists():
        print("Warning: pairs_dir not found: %s" % pairs_dir)
        return []

    # Try loading from scan cache
    cache_path = Path(str(pairs_dir) + ".scan_cache.pkl")
    all_entries = None

    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                all_entries = pickle.load(f)
            print("    (cached: %d entries from %s)" %
                  (len(all_entries), cache_path.name))
        except Exception:
            all_entries = None

    # Full scan if no cache -- parallel
    if all_entries is None:
        print("    No cache found, scanning NPZ files in %s ..." % pairs_dir)
        transcripts = {}
        if transcript_source == "librispeech":
            ls_dir = source_cfg.get("librispeech_dir")
            if ls_dir:
                transcripts = _load_librispeech_transcripts(ls_dir)

        # Build work items
        work = []
        for group_dir in sorted(pairs_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            group_key = group_dir.name
            for npz_path in sorted(group_dir.glob("**/*.npz")):
                work.append((str(npz_path), group_key,
                             transcript_source, transcripts))

        print("    %d files to scan with %d workers..." %
              (len(work), workers))

        with Pool(workers) as pool:
            results = pool.map(_scan_one_npz, work, chunksize=256)

        all_entries = [r for r in results if r is not None]

        # Save cache for next time
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(all_entries, f, protocol=4)
            print("    (saved cache: %d entries to %s)" %
                  (len(all_entries), cache_path.name))
        except Exception:
            pass

    # Apply frame filtering
    entries = [(path, n_frames, tokens, group_key)
               for path, n_frames, tokens, group_key in all_entries
               if min_frames <= n_frames <= max_frames]

    return entries


def pack_sources(data_config, output_dir):
    """Pack all data sources into memory-mapped binary files.

    Produces one pair of files per source (named after the pairs_dir basename)
    plus a combined "all" pair that merges everything.
    """
    import yaml

    if isinstance(data_config, (str, Path)):
        with open(data_config) as f:
            data_config = yaml.safe_load(f)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    min_frames = data_config.get("min_frames", 10)
    max_frames = data_config.get("max_frames", 2000)

    all_meta = []
    cumulative_offset = 0  # in frames

    # Single combined binary file
    combined_name = "all"
    bin_path = output_dir / ("%s.features.bin" % combined_name)
    meta_path = output_dir / ("%s.meta.pkl" % combined_name)

    print("Packing to: %s" % bin_path)

    with open(bin_path, "wb") as bin_f:
        for src in data_config["sources"]:
            pairs_dir_name = Path(src["pairs_dir"]).name
            print("\nSource: %s" % src["pairs_dir"])

            entries = _scan_source_for_pack(src, min_frames, max_frames)
            print("  %d entries after filtering" % len(entries))

            for i, (npz_path, n_frames, tokens, group_key) in enumerate(entries):
                try:
                    d = np.load(npz_path)
                    raw_params = d["raw_params"].astype(np.float16)
                except Exception as e:
                    print("  Warning: failed to load %s: %s" % (npz_path, e))
                    continue

                actual_frames = raw_params.shape[0]
                if raw_params.shape[1] != RAW_PARAM_DIM:
                    print("  Warning: unexpected dim %d in %s, skipping" %
                          (raw_params.shape[1], npz_path))
                    continue

                # Write raw bytes
                bin_f.write(raw_params.tobytes())

                all_meta.append({
                    "offset": cumulative_offset,
                    "n_frames": actual_frames,
                    "tokens": tokens,
                    "group_key": group_key,
                    "npz_path": npz_path,
                })

                cumulative_offset += actual_frames

                if (i + 1) % 5000 == 0:
                    print("    %d / %d packed ..." % (i + 1, len(entries)))

    # Save metadata
    with open(meta_path, "wb") as f:
        pickle.dump(all_meta, f, protocol=4)

    total_bytes = cumulative_offset * RAW_PARAM_DIM * 2  # float16 = 2 bytes
    print("\nDone! %d utterances, %d total frames" %
          (len(all_meta), cumulative_offset))
    print("Binary size: %.2f GB" % (total_bytes / (1024**3)))
    print("Files:")
    print("  %s" % bin_path)
    print("  %s" % meta_path)

    return all_meta


def main():
    parser = argparse.ArgumentParser(
        description="Pack NPZ dataset into memory-mapped binary format"
    )
    parser.add_argument(
        "--data-config", required=True,
        help="Path to data config YAML (same format as training config)"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for .features.bin and .meta.pkl files"
    )
    args = parser.parse_args()

    pack_sources(args.data_config, args.output_dir)


if __name__ == "__main__":
    main()
