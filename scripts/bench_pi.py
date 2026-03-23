#!/usr/bin/env python3
"""Benchmark inference speed on edge devices (Pi, etc).

Tests model loading time, single-call inference, and batch throughput
at various simulated call durations. Reports wall time and memory usage.

Usage:
    python3 scripts/bench_pi.py --checkpoint checkpoints/best.pth
    python3 scripts/bench_pi.py --checkpoint checkpoints/best.pth --fp16
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ConformerCTC
from src.tokenizer import decode_greedy


def get_memory_mb():
    """Get current RSS in MB (Linux only)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return 0


def bench_inference(model, stats, device, duration_s, dtype, n_runs=10, warmup=2):
    """Benchmark inference for a given call duration."""
    n_frames = int(duration_s * 50)  # 50 fps (20ms frames)
    mean, std = stats

    # Generate random IMBE-like features
    features = np.random.randn(n_frames, 170).astype(np.float32)
    features = (features - mean) / std

    feat_t = torch.tensor(features, dtype=dtype).unsqueeze(0).to(device)
    lengths = torch.tensor([n_frames], dtype=torch.long).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(feat_t, lengths)

    # Timed runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.no_grad():
            log_probs, out_lengths = model(feat_t, lengths)
        times.append(time.perf_counter() - t0)

    # Decode one to show it works
    with torch.no_grad():
        log_probs, out_lengths = model(feat_t, lengths)
        hyp = decode_greedy(log_probs[0, :out_lengths[0].item()])

    times_ms = [t * 1000 for t in times]
    return {
        "duration_s": duration_s,
        "n_frames": n_frames,
        "mean_ms": np.mean(times_ms),
        "std_ms": np.std(times_ms),
        "min_ms": np.min(times_ms),
        "max_ms": np.max(times_ms),
        "rtf": np.mean(times) / duration_s,
        "sample_hyp": hyp,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark IMBE-ASR on edge device")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fp16", action="store_true", help="Use float16 weights")
    parser.add_argument("--runs", type=int, default=10, help="Runs per duration")
    args = parser.parse_args()

    device = torch.device("cpu")

    # Measure load time
    mem_before = get_memory_mb()
    t0 = time.perf_counter()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    model = ConformerCTC(
        input_dim=cfg["input_dim"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
        n_layers=cfg["n_layers"], conv_kernel=cfg["conv_kernel"],
        vocab_size=cfg["vocab_size"], dropout=0.0,
        subsample=cfg.get("subsample", False),
    )
    model.load_state_dict(ckpt["model_state_dict"])

    compute_dtype = torch.float32
    if args.fp16:
        model = model.half()
        compute_dtype = torch.float16

    model.eval()
    load_time = time.perf_counter() - t0
    mem_after = get_memory_mb()

    n_params = sum(p.numel() for p in model.parameters())
    dtype_name = "float16" if args.fp16 else "float32"

    print("=" * 60)
    print("IMBE-ASR Inference Benchmark")
    print("=" * 60)
    print("Model: %s params (%.1fM)" % (f"{n_params:,}", n_params / 1e6))
    print("Config: d=%d, layers=%d, heads=%d, ff=%d" %
          (cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["d_ff"]))
    print("Dtype: %s" % dtype_name)
    print("Device: %s" % device)
    print("Load time: %.2fs" % load_time)
    print("Memory: %.0f MB (model) / %.0f MB (baseline)" %
          (mem_after, mem_before))
    print()

    # Load stats
    stats_path = os.path.join(os.path.dirname(args.checkpoint), "stats.npz")
    s = np.load(stats_path)
    stats = (s["mean"], s["std"])

    # Benchmark at various durations
    durations = [1, 2, 5, 10, 30]
    print("%-12s %8s %8s %8s %8s %8s" %
          ("Duration", "Mean", "Std", "Min", "Max", "RTF"))
    print("-" * 60)

    for dur in durations:
        r = bench_inference(model, stats, device, dur, compute_dtype,
                            n_runs=args.runs)
        print("%-12s %7.1fms %7.1fms %7.1fms %7.1fms %7.4fx" %
              ("%ds (%d fr)" % (dur, r["n_frames"]),
               r["mean_ms"], r["std_ms"], r["min_ms"], r["max_ms"], r["rtf"]))

    print()
    print("RTF < 1.0 = faster than real-time")
    print("Memory after benchmark: %.0f MB" % get_memory_mb())


if __name__ == "__main__":
    main()
