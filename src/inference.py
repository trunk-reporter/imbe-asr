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
import struct
import sys
import time

import numpy as np
import torch

from .model import ConformerCTC
from .tokenizer import VOCAB_SIZE, decode_greedy

# Binary TAP file format (trunk-recorder imbe_tap file mode):
#   Header: uint32 magic (0x494D4245), uint32 version (1)
#   Frames: uint32 seq, uint32 tgid, uint32 src_id, uint32 flags, uint16 u[8]
TAP_MAGIC = 0x494D4245
TAP_HEADER_FMT = "<II"
TAP_HEADER_SIZE = struct.calcsize(TAP_HEADER_FMT)
TAP_FRAME_FMT = "<IIII8H"
TAP_FRAME_SIZE = struct.calcsize(TAP_FRAME_FMT)


def _read_tap_file(path, strip_silence=True):
    """Read a binary TAP file and return (frame_vectors, tgid).

    Args:
        path: Path to .tap file
        strip_silence: If True, decode through libimbe and remove
            zero-energy frames (P25 signaling/preamble/silence).

    Returns:
        fv: (N, 8) int16 array of IMBE codewords
        tgid: talkgroup ID from first frame
    """
    with open(path, "rb") as f:
        data = f.read()

    # Try binary format first
    if len(data) >= TAP_HEADER_SIZE:
        magic, version = struct.unpack(TAP_HEADER_FMT, data[:TAP_HEADER_SIZE])
        if magic == TAP_MAGIC:
            n_frames = (len(data) - TAP_HEADER_SIZE) // TAP_FRAME_SIZE
            frames = []
            tgid = 0
            for i in range(n_frames):
                off = TAP_HEADER_SIZE + i * TAP_FRAME_SIZE
                fields = struct.unpack(TAP_FRAME_FMT,
                                       data[off:off + TAP_FRAME_SIZE])
                seq, ftgid, src_id, flags = fields[:4]
                u = list(fields[4:])
                if i == 0:
                    tgid = ftgid
                frames.append(u)
            fv = np.array(frames, dtype=np.int16)
            if strip_silence:
                fv = _strip_silence_frames(fv)
            return fv, tgid

    # Fallback: JSON format (old TAP files)
    import json
    with open(path) as f:
        meta = json.load(f)
    frames = meta.get("frames", [])
    fv = np.array([[fr["u"][j] for j in range(8)] for fr in frames],
                   dtype=np.int16)
    if strip_silence:
        fv = _strip_silence_frames(fv)
    return fv, meta.get("tgid", 0)


def _strip_silence_frames(fv):
    """Remove P25 signaling/silence frames from IMBE codeword array.

    P25 interleaves voice IMBE frames with signaling (LICH, HDU, LDU headers)
    that decode to zero spectral energy. These break temporal continuity and
    confuse the ASR model. We detect them by decoding through libimbe and
    checking for zero energy in the spectral amplitude bands.

    Args:
        fv: (N, 8) int16 array of IMBE codewords

    Returns:
        Filtered (M, 8) int16 array with silence frames removed.
    """
    if len(fv) == 0:
        return fv

    from .precompute import decode_frame_vectors

    raw_params = decode_frame_vectors(fv)
    # Spectral amplitudes are in raw_params[:, 2:58]
    energy = np.sum(np.abs(raw_params[:, 2:58]), axis=1)
    voice_mask = energy > 0
    return fv[voice_mask]


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
    parser.add_argument("--watch", default=None,
                        help="Watch directory for new .tap files and transcribe")
    parser.add_argument("--min-frames", type=int, default=10,
                        help="Min frames to attempt transcription (default: 10)")
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
            feats = d["raw_params"].astype(np.float32)[2:]  # encoder delay
            feats = (feats - mean) / std
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

        feats = d["raw_params"].astype(np.float32)[2:]  # encoder delay
        feats = (feats - mean) / std
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

        fv, tgid = _read_tap_file(args.tap_file)
        if fv.shape[0] < args.min_frames:
            print("TAP: %s -- too short (%d frames), skipping" %
                  (args.tap_file, fv.shape[0]))
        else:
            raw_params = decode_frame_vectors(fv)
            feats = (raw_params.astype(np.float32) - mean) / std

            dur = feats.shape[0] * 0.020
            print("TAP: %s (%.1fs, %d frames, tgid=%s)" %
                  (args.tap_file, dur, feats.shape[0], tgid))
            print()

            if args.stream:
                stream_transcribe(model, feats, device,
                                  chunk_frames=chunk_frames,
                                  sleep_factor=args.speed)
            else:
                hyp = transcribe(model, feats, device)
                print("HYP: %s" % hyp)

    elif args.watch:
        from pathlib import Path
        from .precompute import decode_frame_vectors
        import os

        watch_dir = Path(args.watch)
        if not watch_dir.exists():
            print("ERROR: watch directory not found: %s" % watch_dir)
            sys.exit(1)

        print("Watching %s for new .tap files..." % watch_dir)
        print("Press Ctrl-C to stop.\n")

        seen = set()
        # Pre-populate with existing files so we only transcribe new ones
        for p in watch_dir.rglob("*.tap"):
            seen.add(str(p))
        print("(skipping %d existing files)\n" % len(seen))

        try:
            while True:
                new_files = []
                for p in sorted(watch_dir.rglob("*.tap")):
                    sp = str(p)
                    if sp not in seen:
                        seen.add(sp)
                        new_files.append(p)

                for tap_path in new_files:
                    # Wait briefly for file to finish writing
                    time.sleep(0.2)
                    try:
                        fv, tgid = _read_tap_file(str(tap_path))
                    except Exception as e:
                        print("[ERR] %s: %s" % (tap_path.name, e))
                        continue

                    if fv.shape[0] < args.min_frames:
                        continue

                    raw_params = decode_frame_vectors(fv)
                    feats = (raw_params.astype(np.float32) - mean) / std

                    dur = feats.shape[0] * 0.020
                    t0 = time.time()
                    hyp = transcribe(model, feats, device)
                    dt = (time.time() - t0) * 1000

                    ts = tap_path.stem.split("-")[1].split("_")[0]
                    print("[TG=%s] (%.1fs, %.0fms) %s" %
                          (tgid, dur, dt, tap_path.name))
                    print("  >> %s" % hyp)
                    print()
                    sys.stdout.flush()

                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nDone.")

    else:
        parser.print_help()
        print("\nProvide --npz, --tap-file, --watch, or --all-val")


if __name__ == "__main__":
    main()
