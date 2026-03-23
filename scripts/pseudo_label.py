#!/usr/bin/env python3
"""P25 pseudo-labeling pipeline: .tap files → (raw_params, transcript) NPZ pairs.

Runs Whisper + Qwen3-ASR locally on all files. When they agree, accepts
consensus. When they disagree (>30% edit distance), flags for later
ElevenLabs + full reconciliation.

Usage:
    python3 scripts/pseudo_label.py \
        --tap-dir ~/trunk-recorder/audio/ \
        --output-dir data/p25_labeled \
        --workers 4
"""

import argparse
import io
import json
import os
import struct
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.inference import _read_tap_file, TAP_MAGIC, TAP_HEADER_SIZE

# IMBE decode imports
from src.precompute import decode_frame_vectors

# ---- Config ----
WHISPER_URL = "http://localhost:8766/v1/audio/transcriptions"
QWEN3_URL = "http://localhost:8765/v1/audio/transcriptions"
IMBE_SR = 8000
DISAGREEMENT_THRESHOLD = 0.30  # edit distance ratio above this → flag
MIN_FRAMES = 15  # skip very short files


def tap_to_wav(tap_path):
    """Decode .tap file to WAV bytes via libimbe.

    Returns (wav_bytes, raw_params, tgid) or (None, None, None) on failure.
    """
    try:
        fv, tgid = _read_tap_file(str(tap_path), strip_silence=True)
    except Exception:
        return None, None, None

    if fv.shape[0] < MIN_FRAMES:
        return None, None, None

    raw_params = decode_frame_vectors(fv)

    # Synthesize audio from IMBE for ASR
    # Use libimbe decode: frame_vectors → int16 audio at 8kHz
    try:
        import ctypes
        lib_paths = [
            "./vocoder/libimbe.so",
            "/mnt/disk/p25_train/vocoder/libimbe.so",
            os.path.expanduser("~/p25_audio_quality/vocoder/libimbe.so"),
        ]
        lib = None
        for lp in lib_paths:
            if os.path.exists(lp):
                lib = ctypes.CDLL(lp)
                break
        if lib is None:
            return None, None, None

        lib.imbe_create.restype = ctypes.c_void_p
        lib.imbe_destroy.argtypes = [ctypes.c_void_p]
        lib.imbe_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_int16)]

        dec = lib.imbe_create()
        FRAME_SAMPLES = 160
        fv_buf = (ctypes.c_int16 * 8)()
        snd_buf = (ctypes.c_int16 * FRAME_SAMPLES)()

        audio = np.zeros(fv.shape[0] * FRAME_SAMPLES, dtype=np.int16)
        for i in range(fv.shape[0]):
            for j in range(8):
                fv_buf[j] = ctypes.c_int16(int(fv[i, j]))
            lib.imbe_decode(dec, fv_buf, snd_buf)
            for j in range(FRAME_SAMPLES):
                audio[i * FRAME_SAMPLES + j] = snd_buf[j]

        lib.imbe_destroy(dec)
    except Exception as e:
        print("libimbe error: %s" % e)
        return None, None, None

    # Convert to WAV bytes
    buf = io.BytesIO()
    sf.write(buf, audio.astype(np.float32) / 32768.0, IMBE_SR, format="WAV")
    wav_bytes = buf.getvalue()

    return wav_bytes, raw_params, tgid


def transcribe_whisper(wav_bytes):
    """Send WAV to local Whisper server."""
    try:
        resp = requests.post(
            WHISPER_URL,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            timeout=60)
        if resp.status_code == 200:
            return resp.json().get("text", "").strip().upper()
    except Exception:
        pass
    return None


def transcribe_qwen3(wav_bytes):
    """Send WAV to local Qwen3-ASR server."""
    try:
        resp = requests.post(
            QWEN3_URL,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
            timeout=60)
        if resp.status_code == 200:
            return resp.json().get("text", "").strip().upper()
    except Exception:
        pass
    return None


def edit_distance_ratio(a, b):
    """Normalized edit distance between two strings (0=identical, 1=completely different)."""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0

    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n] / max(m, n)


def process_one(tap_path, output_dir, flagged_dir):
    """Process a single .tap file. Returns (status, tap_path)."""
    name = tap_path.stem

    # Check if already processed
    out_path = output_dir / ("%s.npz" % name)
    if out_path.exists():
        return "skip", str(tap_path)

    # Decode to WAV + raw_params
    wav_bytes, raw_params, tgid = tap_to_wav(tap_path)
    if wav_bytes is None:
        return "fail_decode", str(tap_path)

    # Transcribe with both ASR systems in parallel
    with ThreadPoolExecutor(max_workers=2) as pool:
        whisper_fut = pool.submit(transcribe_whisper, wav_bytes)
        qwen3_fut = pool.submit(transcribe_qwen3, wav_bytes)
        whisper_text = whisper_fut.result()
        qwen3_text = qwen3_fut.result()

    if whisper_text is None and qwen3_text is None:
        return "fail_asr", str(tap_path)

    # If only one succeeded, use it
    if whisper_text is None:
        transcript = qwen3_text
        source = "qwen3_only"
    elif qwen3_text is None:
        transcript = whisper_text
        source = "whisper_only"
    else:
        # Both succeeded — check agreement
        dist = edit_distance_ratio(whisper_text, qwen3_text)

        if dist <= DISAGREEMENT_THRESHOLD:
            # Agree — use whisper (generally more accurate on clean speech)
            transcript = whisper_text
            source = "consensus"
        else:
            # Disagree — flag for later ElevenLabs reconciliation
            flag_path = flagged_dir / ("%s.json" % name)
            with open(flag_path, "w") as f:
                json.dump({
                    "tap_path": str(tap_path),
                    "whisper": whisper_text,
                    "qwen3": qwen3_text,
                    "edit_distance": dist,
                    "tgid": int(tgid),
                    "n_frames": raw_params.shape[0],
                }, f)

            # Still save with whisper text, mark as flagged
            transcript = whisper_text
            source = "flagged"

    # Save NPZ
    np.savez(str(out_path),
             raw_params=raw_params.astype(np.float32),
             transcript=transcript,
             tgid=int(tgid),
             source=source)

    return source, str(tap_path)


def main():
    parser = argparse.ArgumentParser(
        description="P25 pseudo-labeling: .tap → (raw_params, transcript) NPZ")
    parser.add_argument("--tap-dir", required=True,
                        help="Directory with .tap files (recursive)")
    parser.add_argument("--output-dir", default="data/p25_labeled",
                        help="Output directory for NPZ files")
    parser.add_argument("--flagged-dir", default=None,
                        help="Directory for flagged disagreements "
                             "(default: output_dir/flagged)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N files (0=all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    flagged_dir = Path(args.flagged_dir) if args.flagged_dir else output_dir / "flagged"
    flagged_dir.mkdir(parents=True, exist_ok=True)

    # Check servers
    for name, url in [("Whisper", WHISPER_URL.replace("/v1/audio/transcriptions", "/health")),
                      ("Qwen3", QWEN3_URL.replace("/v1/audio/transcriptions", "/health"))]:
        try:
            resp = requests.get(url, timeout=5)
            print("%s: %s" % (name, resp.json().get("status", "?")))
        except Exception as e:
            print("WARNING: %s not reachable: %s" % (name, e))

    # Find all .tap files
    tap_dir = Path(args.tap_dir)
    tap_files = sorted(tap_dir.rglob("*.tap"))
    if args.limit > 0:
        tap_files = tap_files[:args.limit]

    print("\nProcessing %d .tap files..." % len(tap_files))
    print("Output: %s" % output_dir)
    print("Flagged: %s\n" % flagged_dir)

    counts = {"consensus": 0, "flagged": 0, "whisper_only": 0,
              "qwen3_only": 0, "fail_decode": 0, "fail_asr": 0, "skip": 0}
    t0 = time.time()

    for i, tap_path in enumerate(tap_files):
        status, path = process_one(tap_path, output_dir, flagged_dir)
        counts[status] = counts.get(status, 0) + 1

        if (i + 1) % 50 == 0 or (i + 1) == len(tap_files):
            dt = time.time() - t0
            rate = (i + 1) / dt
            eta = (len(tap_files) - i - 1) / rate if rate > 0 else 0
            print("[%d/%d] %.1f files/s, ETA %.0fm | %s" % (
                i + 1, len(tap_files), rate, eta / 60,
                " ".join("%s=%d" % (k, v) for k, v in sorted(counts.items()) if v > 0)))

    dt = time.time() - t0
    print("\nDone in %.0fs (%.1f files/s)" % (dt, len(tap_files) / dt))
    print("Results: %s" % json.dumps(counts, indent=2))
    print("\nFlagged files (need ElevenLabs): %d" % counts.get("flagged", 0))


if __name__ == "__main__":
    main()
