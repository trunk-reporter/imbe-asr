#!/usr/bin/env python3
"""Export IMBE-ASR model to ONNX and optionally quantize to int8.

Usage:
    python3 scripts/export_onnx.py --checkpoint checkpoints/sarah_1024d/best.pth
    python3 scripts/export_onnx.py --checkpoint checkpoints/sarah_1024d/best.pth --quantize
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import ConformerCTC


def main():
    parser = argparse.ArgumentParser(description="Export IMBE-ASR to ONNX")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None, help="Output path (default: same dir as checkpoint)")
    parser.add_argument("--quantize", action="store_true", help="Also produce int8 quantized model")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    out_dir = args.output or os.path.dirname(args.checkpoint)
    os.makedirs(out_dir, exist_ok=True)

    # Load model
    print("Loading checkpoint: %s" % args.checkpoint)
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
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print("Model: %s params (%.1fM)" % (f"{n_params:,}", n_params / 1e6))
    print("Config: d=%d, layers=%d, heads=%d, ff=%d" %
          (cfg["d_model"], cfg["n_layers"], cfg["n_heads"], cfg["d_ff"]))

    # Dummy input for tracing
    seq_len = 500  # 10 seconds
    dummy_features = torch.randn(1, seq_len, 170)
    dummy_lengths = torch.tensor([seq_len], dtype=torch.long)

    # Verify forward pass
    with torch.no_grad():
        log_probs, out_lengths = model(dummy_features, dummy_lengths)
    print("Forward pass OK: output shape %s" % str(log_probs.shape))

    # Export to ONNX
    onnx_path = os.path.join(out_dir, "model.onnx")
    print("\nExporting to %s ..." % onnx_path)
    t0 = time.time()

    torch.onnx.export(
        model,
        (dummy_features, dummy_lengths),
        onnx_path,
        input_names=["features", "lengths"],
        output_names=["log_probs", "out_lengths"],
        dynamic_axes={
            "features": {0: "batch", 1: "time"},
            "lengths": {0: "batch"},
            "log_probs": {0: "batch", 1: "time"},
            "out_lengths": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print("Exported: %.1f MB (%.1fs)" % (onnx_size, time.time() - t0))

    # Validate ONNX model
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX validation passed")

    # Test ONNX Runtime inference matches PyTorch
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path)
    ort_out = sess.run(None, {
        "features": dummy_features.numpy(),
        "lengths": dummy_lengths.numpy(),
    })
    pt_lp = log_probs.numpy()
    ort_lp = ort_out[0]
    max_diff = np.max(np.abs(pt_lp - ort_lp))
    print("PyTorch vs ONNX max diff: %.6f" % max_diff)

    if args.quantize:
        print("\nQuantizing to int8...")
        t0 = time.time()
        int8_path = os.path.join(out_dir, "model_int8.onnx")

        from onnxruntime.quantization import quantize_dynamic, QuantType
        quantize_dynamic(
            onnx_path,
            int8_path,
            weight_type=QuantType.QInt8,
        )

        int8_size = os.path.getsize(int8_path) / (1024 * 1024)
        print("Quantized: %.1f MB (%.1fs)" % (int8_size, time.time() - t0))
        print("Compression: %.1fx" % (onnx_size / int8_size))

        # Test int8 accuracy
        sess_int8 = ort.InferenceSession(int8_path)
        int8_out = sess_int8.run(None, {
            "features": dummy_features.numpy(),
            "lengths": dummy_lengths.numpy(),
        })
        int8_lp = int8_out[0]
        max_diff_int8 = np.max(np.abs(pt_lp - int8_lp))
        print("PyTorch vs int8 max diff: %.6f" % max_diff_int8)

    # Copy stats.npz alongside
    stats_src = os.path.join(os.path.dirname(args.checkpoint), "stats.npz")
    stats_dst = os.path.join(out_dir, "stats.npz")
    if os.path.exists(stats_src) and stats_src != stats_dst:
        import shutil
        shutil.copy2(stats_src, stats_dst)
        print("Copied stats.npz")

    print("\nDone. Files in %s/" % out_dir)


if __name__ == "__main__":
    main()
