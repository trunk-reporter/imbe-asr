#!/usr/bin/env python3
"""Single-file and pseudo-streaming inference.

Supports:
  - Single NPZ file transcription
  - Pseudo-streaming demo (progressive prefix decoding)
  - Batch evaluation on validation set

Usage:
    # Single file
    python -m src.inference \
        --checkpoint checkpoints/best.pth \
        --tap-file path/to/call.tap

    # Streaming demo
    python -m src.inference \
        --checkpoint checkpoints/best.pth \
        --npz path/to/file.npz \
        --stream --chunk-ms 500

    # Batch validation
    python -m src.inference \
        --checkpoint checkpoints/best.pth \
        --all-val --pairs-dir data/pairs
"""

import argparse
import sys
import time

import numpy as np
import torch

from .model import ConformerCTC
from .tokenizer import VOCAB_SIZE, decode_greedy


def load_model(checkpoint_path, device=None):
    """Load model and stats from checkpoint.

    Returns:
        model: ConformerCTC model in eval mode
        device: torch device
        stats: (mean, std) tuple or None
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = ConformerCTC(
        input_dim=cfg["input_dim"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
        n_layers=cfg["n_layers"], conv_kernel=cfg["conv_kernel"],
        vocab_size=cfg["vocab_size"], dropout=0.0,
        subsample=cfg.get("subsample", False),
    ).to(device).eval()
    model.load_state_dict(ckpt["model_state_dict"])

    return model, device, ckpt


def load_stats(stats_path):
    """Load normalization stats from NPZ file."""
    s = np.load(stats_path)
    return s["mean"], s["std"]


def transcribe(model, features, device):
    """Transcribe a single utterance.

    Args:
        model: ConformerCTC model
        features: (T, D) numpy array, already normalized
        device: torch device

    Returns:
        Transcribed text string
    """
    x = torch.from_numpy(features).unsqueeze(0).to(device)
    lengths = torch.tensor([features.shape[0]], device=device)

    with torch.no_grad():
        log_probs, out_lengths = model(x, lengths)

    return decode_greedy(log_probs[0, :out_lengths[0]].cpu())


def stream_transcribe(model, features, device, chunk_frames=25,
                      sleep_factor=1.0):
    """Simulate streaming transcription with progressive prefixes.

    Args:
        model: ConformerCTC model
        features: (T, D) numpy array, normalized
        device: torch device
        chunk_frames: frames per chunk (25 = 500ms at 50fps)
        sleep_factor: multiply real-time delay (0 = no delay)
    """
    T = features.shape[0]
    total_dur = T * 0.020
    prev_line_len = 0

    for end in range(chunk_frames, T + chunk_frames, chunk_frames):
        end = min(end, T)
        elapsed = end * 0.020

        x = torch.from_numpy(features[:end]).unsqueeze(0).to(device)
        lengths = torch.tensor([end], device=device)

        with torch.no_grad():
            log_probs, out_lengths = model(x, lengths)

        hyp = decode_greedy(log_probs[0, :out_lengths[0]].cpu())

        sys.stdout.write('\r' + ' ' * (prev_line_len + 20) + '\r')
        line = "[%4.1fs/%4.1fs] %s" % (elapsed, total_dur, hyp)
        sys.stdout.write(line)
        sys.stdout.flush()
        prev_line_len = len(line)

        if end >= T:
            break

        if sleep_factor > 0:
            time.sleep(chunk_frames * 0.020 * sleep_factor)

    sys.stdout.write('\n')
    return hyp


def main():
    parser = argparse.ArgumentParser(
        description="IMBE-ASR inference and streaming demo")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", default=None,
                        help="Stats NPZ (default: same dir as checkpoint)")
    parser.add_argument("--npz", default=None, help="NPZ file to transcribe")
    parser.add_argument("--tap-file", default=None, help="TAP file to transcribe")
    parser.add_argument("--stream", action="store_true",
                        help="Enable streaming demo mode")
    parser.add_argument("--chunk-ms", type=int, default=500)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed (0=instant, 1=realtime)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--all-val", action="store_true",
                        help="Run on validation samples")
    parser.add_argument("--pairs-dir", default="data/pairs")
    parser.add_argument("--librispeech-dir",
                        default="data/LibriSpeech/train-clean-100")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, device, ckpt = load_model(args.checkpoint, device)

    # Load stats
    if args.stats:
        stats_path = args.stats
    else:
        import os
        stats_path = os.path.join(os.path.dirname(args.checkpoint), "stats.npz")
    mean, std = load_stats(stats_path)

    chunk_frames = args.chunk_ms // 20

    print("Model: epoch %d, best WER=%.1f%%" %
          (ckpt["epoch"] + 1, ckpt["best_wer"]))
    print("Device: %s, chunk: %dms (%d frames)\n" %
          (device, args.chunk_ms, chunk_frames))

    if args.all_val:
        from pathlib import Path
        from .dataset import get_speaker_split

        _, val_speakers = get_speaker_split(args.pairs_dir, 0.1)
        transcripts = {}
        for tf in Path(args.librispeech_dir).glob("*/*/*.trans.txt"):
            with open(tf) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        transcripts[parts[0]] = parts[1].upper()

        samples = []
        for spk in sorted(val_speakers):
            for p in sorted(Path(args.pairs_dir).glob("%s/*/*.npz" % spk)):
                if p.stem in transcripts:
                    d = np.load(str(p))
                    if "raw_params" in d and d["raw_params"].shape[1] == 170:
                        samples.append((str(p), p.stem, transcripts[p.stem]))
                if len(samples) >= 10:
                    break
            if len(samples) >= 10:
                break

        for npz_path, utt_id, ref in samples:
            d = np.load(npz_path)
            feats = (d["raw_params"].astype(np.float32) - mean) / std
            dur = feats.shape[0] * 0.020

            print("=" * 70)
            print("[%s] (%.1fs)" % (utt_id, dur))
            print("REF: %s" % ref)
            print()

            if args.stream:
                stream_transcribe(model, feats, device,
                                  chunk_frames=chunk_frames,
                                  sleep_factor=args.speed)
            else:
                hyp = transcribe(model, feats, device)
                print("HYP: %s" % hyp)
            print()

    elif args.npz:
        d = np.load(args.npz)
        if "raw_params" not in d:
            print("ERROR: NPZ has no raw_params key")
            sys.exit(1)

        feats = (d["raw_params"].astype(np.float32) - mean) / std
        dur = feats.shape[0] * 0.020
        print("File: %s (%.1fs, %d frames)" %
              (args.npz, dur, feats.shape[0]))

        # Show reference if available
        from pathlib import Path
        utt_id = Path(args.npz).stem
        for tf in Path(args.librispeech_dir).glob("*/*/*.trans.txt"):
            with open(tf) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2 and parts[0] == utt_id:
                        print("REF: %s" % parts[1].upper())
                        break
        print()

        if args.stream:
            stream_transcribe(model, feats, device,
                              chunk_frames=chunk_frames,
                              sleep_factor=args.speed)
        else:
            hyp = transcribe(model, feats, device)
            print("HYP: %s" % hyp)

    elif args.tap_file:
        # Decode TAP file through libimbe first
        from .precompute import decode_frame_vectors
        import json

        with open(args.tap_file) as f:
            meta = json.load(f)

        frames = meta.get("frames", [])
        fv = np.array([[fr["u"][j] for j in range(8)] for fr in frames],
                       dtype=np.int16)
        raw_params = decode_frame_vectors(fv)
        feats = (raw_params.astype(np.float32) - mean) / std

        dur = feats.shape[0] * 0.020
        print("TAP: %s (%.1fs, %d frames, tgid=%s)" %
              (args.tap_file, dur, feats.shape[0],
               meta.get("tgid", "?")))
        print()

        if args.stream:
            stream_transcribe(model, feats, device,
                              chunk_frames=chunk_frames,
                              sleep_factor=args.speed)
        else:
            hyp = transcribe(model, feats, device)
            print("HYP: %s" % hyp)

    else:
        parser.print_help()
        print("\nProvide --npz, --tap-file, or --all-val")


if __name__ == "__main__":
    main()
