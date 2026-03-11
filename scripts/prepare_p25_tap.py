#!/usr/bin/env python3
"""Convert P25 TAP files to training NPZ with raw_params + ASR transcripts.

Pipeline:
  1. Read TAP JSON -> extract u[0..7] per frame
  2. Decode through libimbe -> 170-dim raw_params + 8kHz synthesized audio
  3. Send WAV to ASR server (localhost:8765) for transcription
  4. Save NPZ: raw_params, transcript, tgid, src_id

Usage:
    python scripts/prepare_p25_tap.py \
        --tap-dir ~/p25_audio_quality/dataset \
        --output-dir data/p25_raw \
        --workers 6
"""

import argparse
import ctypes
import io
import json
import os
import sys
import wave
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import requests

MAX_H = 56
RAW_PARAM_DIM = 170
IMBE_SAMPLES_PER_FRAME = 160  # 8kHz * 20ms

_lib = None


def _get_lib():
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
        raise FileNotFoundError("libimbe.so not found")

    _lib.imbe_create.restype = ctypes.c_void_p
    _lib.imbe_destroy.argtypes = [ctypes.c_void_p]
    _lib.imbe_decode_params.restype = ctypes.c_int
    _lib.imbe_decode_params.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
    ]
    return _lib


def process_tap(tap_path):
    """Convert one TAP file -> raw_params + 8kHz WAV bytes."""
    try:
        with open(tap_path) as f:
            meta = json.load(f)
    except Exception as e:
        return None, str(e)

    frames = meta.get("frames", [])
    n_frames = len(frames)
    if n_frames < 5:
        return None, "too_short"

    lib = _get_lib()
    dec = lib.imbe_create()

    fv = (ctypes.c_int16 * 8)()
    snd = (ctypes.c_int16 * IMBE_SAMPLES_PER_FRAME)()
    f0c = ctypes.c_float()
    nhc = ctypes.c_int16()
    nbc = ctypes.c_int16()
    vuv = (ctypes.c_int16 * MAX_H)()
    sa = (ctypes.c_int16 * MAX_H)()

    raw_params = np.zeros((n_frames, RAW_PARAM_DIM), dtype=np.float32)
    audio_8k = np.zeros(n_frames * IMBE_SAMPLES_PER_FRAME, dtype=np.int16)

    for i, frame in enumerate(frames):
        u = frame["u"]
        for j in range(8):
            fv[j] = ctypes.c_int16(int(u[j]))

        lib.imbe_decode_params(
            dec, fv, snd,
            ctypes.byref(f0c), ctypes.byref(nhc),
            ctypes.byref(nbc), vuv, sa,
        )

        L = min(nhc.value, MAX_H)
        raw_params[i, 0] = f0c.value
        raw_params[i, 1] = L
        for j in range(L):
            raw_params[i, 2 + j] = sa[j] / 4.0
            raw_params[i, 58 + j] = vuv[j]
            raw_params[i, 114 + j] = 1.0

        offset = i * IMBE_SAMPLES_PER_FRAME
        for j in range(IMBE_SAMPLES_PER_FRAME):
            audio_8k[offset + j] = snd[j]

    lib.imbe_destroy(dec)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(8000)
        wf.writeframes(audio_8k.tobytes())
    wav_bytes = buf.getvalue()

    return {
        "raw_params": raw_params,
        "wav_bytes": wav_bytes,
        "tgid": meta.get("tgid", 0),
        "src_id": meta.get("src_id", 0),
        "n_frames": n_frames,
        "tap_path": tap_path,
    }, "ok"


def transcribe_via_server(wav_bytes, asr_url):
    """Send WAV to ASR server, return transcript string."""
    resp = requests.post(
        asr_url,
        files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        data={
            "model": "qwen3-asr-p25",
            "language": "English",
            "response_format": "json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    text = resp.json().get("text", "").strip()
    return text


def main():
    parser = argparse.ArgumentParser(
        description="Convert P25 TAP files to training NPZ")
    parser.add_argument("--tap-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--asr-url",
                        default="http://localhost:8765/v1/audio/transcriptions")
    parser.add_argument("--min-frames", type=int, default=25)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument("--min-clean-pct", type=float, default=70)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    tap_dir = Path(args.tap_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find and pre-filter TAP files
    tap_files = sorted(tap_dir.glob("*/*.tap"))
    print("Found %d TAP files" % len(tap_files))

    filtered = []
    for tap_path in tap_files:
        try:
            with open(tap_path) as f:
                meta = json.load(f)
            n = meta["n_frames"]
            errs = meta.get("error_summary", {})
            clean = errs.get("clean", 0)
            clean_pct = clean / max(n, 1) * 100
            if (n >= args.min_frames and n <= args.max_frames
                    and clean_pct >= args.min_clean_pct):
                filtered.append(str(tap_path))
        except Exception:
            continue

    print("After filtering: %d calls (%d-%d frames, >=%.0f%% clean)" %
          (len(filtered), args.min_frames, args.max_frames, args.min_clean_pct))

    if args.limit:
        filtered = filtered[:args.limit]

    # Verify ASR server
    try:
        r = requests.get(
            args.asr_url.replace("/v1/audio/transcriptions", "/health"),
            timeout=5)
        print("ASR server: %s" % r.json().get("model", "unknown"))
    except Exception as e:
        print("WARNING: ASR server not reachable: %s" % e)

    n_ok = n_skip = n_err = 0

    for batch_start in range(0, len(filtered), args.batch_size):
        batch_paths = filtered[batch_start:batch_start + args.batch_size]

        with Pool(min(args.workers, len(batch_paths))) as pool:
            batch_results = pool.map(process_tap, batch_paths)

        for (result, status), tap_path in zip(batch_results, batch_paths):
            if status != "ok" or result is None:
                n_skip += 1
                continue

            try:
                text = transcribe_via_server(result["wav_bytes"], args.asr_url)

                if len(text) < 2:
                    n_skip += 1
                    continue

                tap_name = Path(tap_path).stem
                out_path = output_dir / ("%s.npz" % tap_name)
                np.savez_compressed(
                    str(out_path),
                    raw_params=result["raw_params"],
                    transcript=text.upper(),
                    tgid=result["tgid"],
                    src_id=result["src_id"],
                )
                n_ok += 1
            except Exception as e:
                n_err += 1
                if n_err <= 5:
                    print("  Error: %s: %s" % (tap_path, e))

        total = batch_start + len(batch_paths)
        print("  %d/%d  (%d ok, %d skip, %d err)" %
              (total, len(filtered), n_ok, n_skip, n_err))

    print("\nDone. %d saved, %d skipped, %d errors." % (n_ok, n_skip, n_err))
    print("Output: %s" % output_dir)


if __name__ == "__main__":
    main()
