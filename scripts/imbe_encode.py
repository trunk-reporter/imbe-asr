#!/usr/bin/env python3
"""Generic audio-to-IMBE encoder for any speech dataset.

Accepts a TSV manifest and IMBE-encodes each utterance, producing NPZ files
with frame_vectors (N, 8) and transcript. Works with any audio format that
soundfile can read (WAV, FLAC, SPH, etc.) at any sample rate.

Manifest TSV format (header line required):
    audio_path  utterance_id  transcript  group_key  start  end

  - start/end are in seconds (float). Use 0/0 or empty for whole file.
  - group_key is used for output directory structure (speaker ID, talk ID, etc.)

Output: output_dir/{group_key}/{utterance_id}.npz with keys:
  - frame_vectors: (N, 8) int16
  - transcript: string (uppercased, normalized)

Usage:
    python scripts/imbe_encode.py \
        --manifest data/tedlium_manifest.tsv \
        --output data/pairs_tedlium \
        --workers 12
"""

import argparse
import ctypes
import csv
import os
import re
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# --------------------------------------------------------------------------- #
#  IMBE vocoder interface (per-process, loaded lazily)
# --------------------------------------------------------------------------- #

_lib = None

IMBE_FRAME_SAMPLES = 160  # 20ms at 8kHz
IMBE_SR = 8000


def _get_lib():
    """Load libimbe.so, trying project vocoder/ first."""
    global _lib
    if _lib is not None:
        return _lib
    candidates = [
        Path(__file__).resolve().parent.parent / "vocoder" / "libimbe.so",
        Path("/mnt/disk/p25_train/vocoder/libimbe.so"),
    ]
    for p in candidates:
        if p.exists():
            _lib = ctypes.CDLL(str(p))
            break
    if _lib is None:
        raise FileNotFoundError(
            "libimbe.so not found. Checked: %s" %
            ", ".join(str(p) for p in candidates))

    _lib.imbe_create.restype = ctypes.c_void_p
    _lib.imbe_destroy.argtypes = [ctypes.c_void_p]
    _lib.imbe_encode.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
    ]
    return _lib


def normalize_transcript(text):
    """Normalize text to match CTC tokenizer vocabulary."""
    text = str(text).upper()
    text = text.replace("-", " ")
    text = re.sub(r"[^A-Z0-9 ']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_audio(audio, sr):
    """IMBE-encode audio array, returning frame_vectors (N, 8).

    Args:
        audio: 1-D float64 array (any sample rate)
        sr: sample rate of audio

    Returns:
        frame_vectors: (N, 8) int16 array of IMBE codewords
    """
    # Resample to 8kHz if needed
    if sr != IMBE_SR:
        if sr == 16000:
            audio_8k = resample_poly(audio, 1, 2)
        elif sr == 48000:
            audio_8k = resample_poly(audio, 1, 6)
        elif sr == 44100:
            audio_8k = resample_poly(audio, 80, 441)
        else:
            audio_8k = resample_poly(audio, IMBE_SR, sr)
    else:
        audio_8k = audio

    audio_8k = audio_8k.astype(np.float64)
    n_frames = len(audio_8k) // IMBE_FRAME_SAMPLES
    if n_frames < 2:
        return None

    audio_8k = audio_8k[:n_frames * IMBE_FRAME_SAMPLES]
    audio_int16 = np.clip(audio_8k * 32768, -32768, 32767).astype(np.int16)

    lib = _get_lib()
    enc = lib.imbe_create()
    fv_enc = (ctypes.c_int16 * 8)()
    snd_in = (ctypes.c_int16 * IMBE_FRAME_SAMPLES)()
    frame_vectors = np.zeros((n_frames, 8), dtype=np.int16)

    for i in range(n_frames):
        chunk = audio_int16[i * IMBE_FRAME_SAMPLES:(i + 1) * IMBE_FRAME_SAMPLES]
        for j in range(IMBE_FRAME_SAMPLES):
            snd_in[j] = ctypes.c_int16(int(chunk[j]))
        lib.imbe_encode(enc, fv_enc, snd_in)
        for j in range(8):
            frame_vectors[i, j] = fv_enc[j]

    lib.imbe_destroy(enc)

    # IMBE encoder has a 2-frame analysis delay: output frame N describes
    # input audio at frame N-2. Drop the first 2 frames so frame_vectors[0]
    # aligns with audio[0]. See ~/p25_audio_quality/train/measure_encoder_delay.py
    if frame_vectors.shape[0] > 2:
        frame_vectors = frame_vectors[2:]

    return frame_vectors


def _process_row(row):
    """Process one manifest row. Called by multiprocessing pool."""
    audio_path, utt_id, transcript, group_key, start, end, out_dir = row

    out_path = os.path.join(out_dir, group_key, "%s.npz" % utt_id)
    if os.path.exists(out_path):
        return out_path, 0, "skip"

    try:
        # Read audio (soundfile handles WAV, FLAC, SPH, etc.)
        info = sf.info(audio_path)
        sr = info.samplerate

        if start and end and float(end) > 0:
            start_sample = int(float(start) * sr)
            end_sample = int(float(end) * sr)
            audio, sr = sf.read(audio_path, start=start_sample,
                                stop=end_sample, dtype="float64")
        else:
            audio, sr = sf.read(audio_path, dtype="float64")

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

    except Exception as e:
        return audio_path, 0, "read error: %s" % e

    transcript = normalize_transcript(transcript)
    if len(transcript) < 2:
        return audio_path, 0, "empty transcript"

    frame_vectors = encode_audio(audio, sr)
    if frame_vectors is None:
        return audio_path, 0, "too short"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path,
                        frame_vectors=frame_vectors,
                        transcript=transcript)

    return out_path, frame_vectors.shape[0], "ok"


def encode_manifest(manifest_path, output_dir, workers=None):
    """Encode all rows in a manifest TSV file.

    Args:
        manifest_path: Path to TSV file with header row
        output_dir: Base output directory
        workers: Number of parallel workers (default: cpu_count - 1)
    """
    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append((
                r["audio_path"],
                r["utterance_id"],
                r["transcript"],
                r["group_key"],
                r.get("start", ""),
                r.get("end", ""),
                output_dir,
            ))

    print("Manifest: %d utterances" % len(rows))

    n_workers = workers or max(1, cpu_count() - 1)
    print("Workers: %d" % n_workers)

    n_ok = n_skip = n_err = 0
    with Pool(n_workers) as pool:
        for i, (path, n_frames, status) in enumerate(
                pool.imap_unordered(_process_row, rows)):
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                if n_err <= 10:
                    print("  Error: %s: %s" % (path, status))

            if (i + 1) % 1000 == 0:
                print("  %d/%d (%d ok, %d skip, %d err)" %
                      (i + 1, len(rows), n_ok, n_skip, n_err))

    print("\nDone. %d encoded, %d skipped, %d errors." %
          (n_ok, n_skip, n_err))
    return n_ok, n_skip, n_err


def main():
    parser = argparse.ArgumentParser(
        description="IMBE-encode audio from a manifest TSV file")
    parser.add_argument("--manifest", required=True,
                        help="TSV manifest (audio_path, utterance_id, "
                             "transcript, group_key, start, end)")
    parser.add_argument("--output", required=True,
                        help="Output directory for NPZ files")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    encode_manifest(args.manifest, args.output, args.workers)


if __name__ == "__main__":
    main()
