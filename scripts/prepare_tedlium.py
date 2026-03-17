#!/usr/bin/env python3
"""Download and prepare TEDLIUM Release 3 for IMBE-ASR training.

Uses HuggingFace datasets library to download TEDLIUM 3 (the openslr.org
archive is no longer reliably available). Extracts audio segments,
generates a TSV manifest, IMBE-encodes, and precomputes raw parameters.

TEDLIUM 3: 2,351 TED talks, ~452 hours, 16kHz audio.

Usage:
    python scripts/prepare_tedlium.py --output data

    # If you have the archive locally:
    python scripts/prepare_tedlium.py \
        --tedlium-dir data/TEDLIUM_release-3 \
        --output data
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def normalize_transcript(text):
    """Normalize TEDLIUM text to match CTC tokenizer vocabulary.

    Strips disfluency/noise markers specific to TEDLIUM:
      {BREATH}, {COUGH}, {NOISE}, {SMACK}, {UH}, etc.
      <sil>, <noise>, etc.
      (2), (3) -- hesitation repeat markers
    """
    text = str(text)
    # Remove braced markers: {BREATH}, {COUGH}, {NOISE}, {SMACK}, {UH}
    text = re.sub(r"\{[^}]*\}", "", text)
    # Remove angle-bracket markers: <sil>, <noise>, etc.
    text = re.sub(r"<[^>]*>", "", text)
    # Remove parenthesized repeat markers: (2), (3)
    text = re.sub(r"\(\d+\)", "", text)
    # Standard normalization
    text = text.upper()
    text = text.replace("-", " ")
    text = re.sub(r"[^A-Z0-9 ']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------- #
#  HuggingFace path: download via datasets library                              #
# --------------------------------------------------------------------------- #

def build_manifest_hf(output_dir, manifest_path):
    """Download TEDLIUM 3 via HuggingFace and write TSV manifest.

    Uses jpalgo/tedlium-original which is TEDLIUM 3 in Parquet format.
    Fields: audio, text, speaker_id, gender, file, id
    The text field contains disfluency markers like {BREATH}, <sil>, (2)
    which normalize_transcript strips out.
    """
    from datasets import load_dataset

    print("Loading TEDLIUM 3 from HuggingFace (jpalgo/tedlium-original)...")
    ds = load_dataset("jpalgo/tedlium-original", split="train")
    print("TEDLIUM 3: %d segments" % len(ds))

    audio_dir = os.path.join(output_dir, "tedlium_segments")
    os.makedirs(audio_dir, exist_ok=True)

    with open(manifest_path, "w") as out:
        out.write("audio_path\tutterance_id\ttranscript\tgroup_key\tstart\tend\n")

        n_written = 0
        n_skipped = 0

        for i, example in enumerate(ds):
            audio_data = example["audio"]
            audio_array = np.array(audio_data["array"], dtype=np.float64)
            sr = audio_data["sampling_rate"]

            text = normalize_transcript(example.get("text", ""))
            if len(text) < 2:
                n_skipped += 1
                continue

            duration = len(audio_array) / sr
            if duration < 0.2 or duration > 40.0:
                n_skipped += 1
                continue

            if audio_array.ndim > 1:
                audio_array = audio_array.mean(axis=1)

            # speaker_id is the talk name (e.g., "BeauLotto_2009G")
            speaker_id = str(example.get("speaker_id", "spk_%05d" % i))
            # id contains talk-start-end-label, clean for filesystem
            raw_id = str(example.get("id", "tedlium_%08d" % i))
            utt_id = re.sub(r"[^\w\-]", "_", raw_id)

            group_key = speaker_id

            wav_path = Path(audio_dir) / group_key / ("%s.wav" % utt_id)
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(wav_path), audio_array, sr)

            out.write("%s\t%s\t%s\t%s\t\t\n" % (
                wav_path, utt_id, text, group_key))
            n_written += 1

            if (i + 1) % 5000 == 0:
                print("  %d processed, %d written, %d skipped" %
                      (i + 1, n_written, n_skipped))

    print("Manifest: %d segments -> %s (skipped %d)" %
          (n_written, manifest_path, n_skipped))
    return n_written


# --------------------------------------------------------------------------- #
#  Local archive path: parse STM + segment SPH                                  #
# --------------------------------------------------------------------------- #

def parse_stm(stm_path):
    """Parse a TEDLIUM STM file into segment dicts."""
    segments = []
    with open(stm_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";;"):
                continue

            parts = line.split(None, 6)
            if len(parts) < 7:
                continue

            talk_id = parts[0]
            speaker = parts[2]
            start = float(parts[3])
            end = float(parts[4])
            label = parts[5]
            transcript = parts[6]

            if "ignore_time_segment_in_scoring" in label:
                continue

            transcript = normalize_transcript(transcript)
            if len(transcript) < 2:
                continue

            segments.append({
                "talk_id": talk_id,
                "speaker": speaker,
                "start": start,
                "end": end,
                "transcript": transcript,
            })

    return segments


def segment_audio(sph_path, segments, audio_dir):
    """Read SPH file and write per-segment WAV files."""
    try:
        audio, sr = sf.read(str(sph_path), dtype="float64")
    except Exception:
        try:
            wav_tmp = str(sph_path) + ".tmp.wav"
            subprocess.run(
                ["sox", str(sph_path), "-r", "16000", "-c", "1", wav_tmp],
                check=True, capture_output=True)
            audio, sr = sf.read(wav_tmp, dtype="float64")
            os.unlink(wav_tmp)
        except Exception as e:
            print("  Error reading %s: %s" % (sph_path.name, e))
            return []

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    results = []
    for seg in segments:
        start_sample = int(seg["start"] * sr)
        end_sample = int(seg["end"] * sr)

        if end_sample <= start_sample or end_sample > len(audio):
            continue

        chunk = audio[start_sample:end_sample]
        if len(chunk) < sr * 0.1:
            continue

        utt_id = "%s_%06d_%06d" % (
            seg["talk_id"],
            int(seg["start"] * 100),
            int(seg["end"] * 100))

        wav_path = Path(audio_dir) / seg["talk_id"] / ("%s.wav" % utt_id)
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(wav_path), chunk, sr)

        results.append((str(wav_path), utt_id, seg))

    return results


def build_manifest_local(tedlium_dir, audio_dir, manifest_path):
    """Parse all STM files, segment audio, write TSV manifest."""
    tedlium_dir = Path(tedlium_dir)
    stm_dir = tedlium_dir / "data" / "stm"
    sph_dir = tedlium_dir / "data" / "sph"

    if not stm_dir.exists():
        stm_dir = tedlium_dir / "legacy" / "stm"
        sph_dir = tedlium_dir / "legacy" / "sph"

    stm_files = sorted(stm_dir.glob("*.stm"))
    print("Found %d STM files" % len(stm_files))

    with open(manifest_path, "w") as out:
        out.write("audio_path\tutterance_id\ttranscript\tgroup_key\tstart\tend\n")

        n_total = 0
        for idx, stm_file in enumerate(stm_files):
            talk_id = stm_file.stem
            sph_path = sph_dir / ("%s.sph" % talk_id)

            if not sph_path.exists():
                print("  Warning: SPH not found for %s" % talk_id)
                continue

            segments = parse_stm(str(stm_file))
            if not segments:
                continue

            results = segment_audio(sph_path, segments, audio_dir)

            for wav_path, utt_id, seg in results:
                out.write("%s\t%s\t%s\t%s\t\t\n" % (
                    wav_path, utt_id, seg["transcript"], seg["talk_id"]))
                n_total += 1

            if (idx + 1) % 100 == 0:
                print("  %d/%d talks, %d segments" %
                      (idx + 1, len(stm_files), n_total))

    print("Manifest: %d segments -> %s" % (n_total, manifest_path))
    return n_total


def main():
    parser = argparse.ArgumentParser(
        description="Prepare TEDLIUM 3 for IMBE-ASR training")
    parser.add_argument("--output", default="data",
                        help="Base output directory")
    parser.add_argument("--tedlium-dir", default=None,
                        help="Path to existing TEDLIUM_release-3 directory "
                             "(skips HF download, uses local STM/SPH files)")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--skip-encode", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    manifest_path = os.path.join(args.output, "tedlium_manifest.tsv")

    if os.path.exists(manifest_path):
        print("Manifest already exists: %s" % manifest_path)
    elif args.tedlium_dir:
        # Local archive path
        audio_dir = os.path.join(args.output, "tedlium_segments")
        build_manifest_local(args.tedlium_dir, audio_dir, manifest_path)
    else:
        # HuggingFace download path
        build_manifest_hf(args.output, manifest_path)

    if args.skip_encode:
        print("Skipping IMBE encoding (--skip-encode)")
        return

    # IMBE encode
    pairs_dir = os.path.join(args.output, "pairs_tedlium")
    print("\nIMBE-encoding TEDLIUM segments...")
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

    print("\nDone. TEDLIUM pairs at: %s" % pairs_dir)


if __name__ == "__main__":
    main()
