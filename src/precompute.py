#!/usr/bin/env python3
"""Precompute 170-dim raw IMBE parameters from frame_vectors in NPZ files.

Reads frame_vectors from existing NPZ files, decodes through libimbe to get
f0, L, spectral amplitudes, per-harmonic V/UV flags, and a binary harmonic
mask. Saves as raw_params alongside the original data.

Output per frame (170 dims):
  [0]       f0 (fundamental frequency in Hz)
  [1]       L  (number of harmonics, 0-56)
  [2:58]    sa[0..55] spectral amplitudes (padded with 0)
  [58:114]  v_uv[0..55] voiced/unvoiced flags (padded with 0)
  [114:170] mask[0..55] binary harmonic validity mask (1=real, 0=pad)

Usage:
    python -m src.precompute --pairs-dir data/pairs --workers 12
"""

import argparse
import ctypes
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

MAX_HARMONICS = 56
RAW_PARAM_DIM = 2 + MAX_HARMONICS + MAX_HARMONICS + MAX_HARMONICS  # 170

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
        raise FileNotFoundError(
            "libimbe.so not found. Checked: %s" %
            ", ".join(str(p) for p in candidates))

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


def decode_frame_vectors(frame_vectors):
    """Decode (N, 8) frame vectors into (N, 170) raw IMBE parameters.

    Output layout per frame:
      [0]       f0
      [1]       L
      [2:58]    sa[0..55]   (zero-padded spectral amplitudes)
      [58:114]  v_uv[0..55] (zero-padded voicing flags)
      [114:170] mask[0..55] (1.0 for valid harmonics, 0.0 for padding)
    """
    n_frames = frame_vectors.shape[0]
    raw_params = np.zeros((n_frames, RAW_PARAM_DIM), dtype=np.float32)

    lib = _get_lib()
    dec = lib.imbe_create()

    fv_dec = (ctypes.c_int16 * 8)()
    snd_dec = (ctypes.c_int16 * 160)()
    fund_freq = ctypes.c_float()
    num_harms = ctypes.c_int16()
    num_bands = ctypes.c_int16()
    v_uv = (ctypes.c_int16 * MAX_HARMONICS)()
    sa = (ctypes.c_int16 * MAX_HARMONICS)()

    for i in range(n_frames):
        for j in range(8):
            fv_dec[j] = ctypes.c_int16(int(frame_vectors[i, j]))

        lib.imbe_decode_params(
            dec, fv_dec, snd_dec,
            ctypes.byref(fund_freq), ctypes.byref(num_harms),
            ctypes.byref(num_bands), v_uv, sa,
        )

        f0 = fund_freq.value
        L = min(num_harms.value, MAX_HARMONICS)

        raw_params[i, 0] = f0
        raw_params[i, 1] = L
        for j in range(L):
            raw_params[i, 2 + j] = sa[j] / 4.0  # Q14.2 -> float
            raw_params[i, 58 + j] = v_uv[j]
            raw_params[i, 114 + j] = 1.0  # harmonic mask

    lib.imbe_destroy(dec)
    return raw_params


def process_one(npz_path):
    """Add raw_params to one NPZ file."""
    npz_path = str(npz_path)
    try:
        d = np.load(npz_path, allow_pickle=True)
        if "raw_params" in d and d["raw_params"].shape[1] == RAW_PARAM_DIM:
            return npz_path, "skip"
        if "frame_vectors" not in d:
            return npz_path, "no_fv"

        fv = d["frame_vectors"]
        raw_params = decode_frame_vectors(fv)

        save_dict = {k: d[k] for k in d.files}
        save_dict["raw_params"] = raw_params
        np.savez_compressed(npz_path, **save_dict)

        return npz_path, "ok"
    except Exception as e:
        return npz_path, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Precompute raw IMBE params from frame_vectors")
    parser.add_argument("--pairs-dir", required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pairs_dir = Path(args.pairs_dir)
    npz_files = sorted(pairs_dir.glob("**/*.npz"))
    if args.limit:
        npz_files = npz_files[:args.limit]

    print("Files: %d" % len(npz_files))

    n_workers = args.workers or max(1, cpu_count() - 1)
    print("Workers: %d" % n_workers)

    n_ok = n_skip = n_err = 0
    with Pool(n_workers) as pool:
        for i, (path, status) in enumerate(pool.imap_unordered(process_one, npz_files)):
            if status == "ok":
                n_ok += 1
            elif status == "skip":
                n_skip += 1
            else:
                n_err += 1
                if n_err <= 5:
                    print("  Error: %s: %s" % (path, status))

            if (i + 1) % 1000 == 0:
                print("  %d/%d (%d ok, %d skip, %d err)" %
                      (i + 1, len(npz_files), n_ok, n_skip, n_err))

    print("\nDone. %d updated, %d skipped, %d errors." % (n_ok, n_skip, n_err))


if __name__ == "__main__":
    main()
