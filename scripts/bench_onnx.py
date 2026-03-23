#!/usr/bin/env python3
"""Benchmark ONNX Runtime inference on edge devices.

Tests fp32 and int8 ONNX models. Needs only onnxruntime + numpy (no PyTorch).

Usage:
    python3 scripts/bench_onnx.py --model model.onnx --stats stats.npz
    python3 scripts/bench_onnx.py --model model_int8.onnx --stats stats.npz
"""

import argparse
import os
import time

import numpy as np


def get_memory_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 0


CTC_VOCAB = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '"


def ctc_greedy_decode(log_probs):
    """Greedy CTC decode from (T, V) log_probs."""
    ids = np.argmax(log_probs, axis=-1)
    chars = []
    prev = -1
    for idx in ids:
        if idx != 0 and idx != prev:  # skip blank and repeats
            if 1 <= idx <= len(CTC_VOCAB):
                chars.append(CTC_VOCAB[idx - 1])
        prev = idx
    return "".join(chars).strip()


def bench_duration(session, stats, duration_s, n_runs=10, warmup=2):
    n_frames = int(duration_s * 50)
    mean, std = stats

    features = np.random.randn(n_frames, 170).astype(np.float32)
    features = ((features - mean) / std).astype(np.float32)
    feat_batch = features.reshape(1, n_frames, 170)
    lengths = np.array([n_frames], dtype=np.int64)

    for _ in range(warmup):
        session.run(None, {"features": feat_batch, "lengths": lengths})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {"features": feat_batch, "lengths": lengths})
        times.append(time.perf_counter() - t0)

    times_ms = [t * 1000 for t in times]
    return {
        "duration_s": duration_s,
        "n_frames": n_frames,
        "mean_ms": np.mean(times_ms),
        "std_ms": np.std(times_ms),
        "min_ms": np.min(times_ms),
        "max_ms": np.max(times_ms),
        "rtf": np.mean(times) / duration_s,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX IMBE-ASR")
    parser.add_argument("--model", required=True, help="ONNX model path")
    parser.add_argument("--stats", required=True, help="stats.npz path")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--npz", default=None,
                        help="NPZ file with raw_params+transcript for accuracy test")
    args = parser.parse_args()

    import onnxruntime as ort

    mem_before = get_memory_mb()
    t0 = time.perf_counter()

    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 4
    sess_opts.intra_op_num_threads = 4
    session = ort.InferenceSession(args.model, sess_opts)

    load_time = time.perf_counter() - t0
    mem_after = get_memory_mb()

    model_size = os.path.getsize(args.model) / (1024 * 1024)

    print("=" * 60)
    print("IMBE-ASR ONNX Inference Benchmark")
    print("=" * 60)
    print("Model: %s (%.1f MB)" % (os.path.basename(args.model), model_size))
    print("Load time: %.2fs" % load_time)
    print("Memory: %.0f MB (loaded) / %.0f MB (baseline)" %
          (mem_after, mem_before))
    print()

    s = np.load(args.stats)
    stats = (s["mean"], s["std"])

    # Speed benchmark
    durations = [1, 2, 5, 10, 30]
    print("%-12s %8s %8s %8s %8s %8s" %
          ("Duration", "Mean", "Std", "Min", "Max", "RTF"))
    print("-" * 60)

    for dur in durations:
        r = bench_duration(session, stats, dur, n_runs=args.runs)
        print("%-12s %7.1fms %7.1fms %7.1fms %7.1fms %7.4fx" %
              ("%ds (%d fr)" % (dur, r["n_frames"]),
               r["mean_ms"], r["std_ms"], r["min_ms"], r["max_ms"], r["rtf"]))

    # Accuracy test on real data
    if args.npz:
        print("\n--- Accuracy Test ---")
        data = np.load(args.npz)
        raw_params = data["raw_params"].astype(np.float32)
        transcript = str(data["transcript"])

        # Normalize
        raw_params = (raw_params - stats[0]) / stats[1]
        feat_batch = raw_params.reshape(1, -1, 170).astype(np.float32)
        lengths = np.array([raw_params.shape[0]], dtype=np.int64)

        out = session.run(None, {"features": feat_batch, "lengths": lengths})
        log_probs = out[0][0]  # (T, V)
        out_len = int(out[1][0])

        hyp = ctc_greedy_decode(log_probs[:out_len])
        print("REF: %s" % transcript)
        print("HYP: %s" % hyp)

    print()
    print("RTF < 1.0 = faster than real-time")
    print("Memory after benchmark: %.0f MB" % get_memory_mb())


if __name__ == "__main__":
    main()
