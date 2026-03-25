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
        --dvcf-file path/to/call.dvcf

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
import json
import os
import struct
import sys
import time

import numpy as np
import torch

from .model import ConformerCTC
from .tokenizer import VOCAB_SIZE, decode_greedy


class OnnxModel:
    """Wrapper around an ONNX InferenceSession that mimics the PyTorch model
    interface enough for callers to detect format and run inference."""

    def __init__(self, session):
        self.session = session
        self.is_onnx = True

    def __call__(self, x_np, lengths_np):
        """Run ONNX inference.

        Args:
            x_np: float32 numpy array of shape (B, T, 170)
            lengths_np: int64 numpy array of shape (B,)

        Returns:
            log_probs: (B, T', vocab_size) numpy array
            output_lengths: (B,) numpy array
        """
        input_names = [inp.name for inp in self.session.get_inputs()]
        feeds = {input_names[0]: x_np}
        if len(input_names) > 1:
            feeds[input_names[1]] = lengths_np
        outputs = self.session.run(None, feeds)
        # outputs[0] = logits or log_probs, outputs[1] = output_lengths (if present)
        logits = outputs[0]
        # Apply log_softmax if the output looks like raw logits
        # (check if values sum to ~1 along last dim after exp)
        from scipy.special import log_softmax as sp_log_softmax
        log_probs = sp_log_softmax(logits, axis=-1)
        out_lengths = outputs[1] if len(outputs) > 1 else np.array([log_probs.shape[1]])
        return log_probs, out_lengths

# SymbolStream v2 binary format (.dvcf files)
# See ~/symbolstream/SPEC.md §3 for full specification.
#
# Each message: 8-byte header + payload
#   Header: magic[2] ('SY' = 0x5359), version (0x02), msg_type, payload_len (uint32 LE)
#   msg_type 0x01 = CODEC_FRAME, 0x02 = CALL_START, 0x03 = CALL_END, 0x04 = HEARTBEAT
#
# CODEC_FRAME payload: 24-byte sssp_codec_hdr_t + param_count × uint32 params
#   codec_hdr: talkgroup(4), src_id(4), call_id(4), timestamp_us(8),
#              codec_type(1), param_count(1), errs(1), flags(1)
#   For IMBE (codec_type=0): param_count=8, params = u[0..7]

SSSP_MAGIC = b'SY'
SSSP_VERSION = 0x02
SSSP_HEADER_SIZE = 8
SSSP_HEADER_FMT = "<2sBBI"  # magic(2), version(1), msg_type(1), payload_len(4)
SSSP_MSG_CODEC_FRAME = 0x01
SSSP_MSG_CALL_START = 0x02
SSSP_MSG_CALL_END = 0x03
SSSP_CODEC_HDR_SIZE = 24
SSSP_CODEC_HDR_FMT = "<IIIQBBBx"  # talkgroup, src_id, call_id, timestamp_us, codec_type, param_count, errs, flags(pad)
# Note: the struct is packed with errs(1)+flags(1) = 2 bytes at offsets 22-23.
# Using explicit unpack below for clarity.


def _read_dvcf_file(path, strip_silence=True):
    """Read a SymbolStream v2 binary .dvcf file and return (frame_vectors, tgid).

    Parses the SSSP v2 message stream, extracts CODEC_FRAME messages, and
    returns the IMBE codeword parameters as a numpy array suitable for
    decode_frame_vectors().

    Args:
        path: Path to .dvcf file (SymbolStream v2 binary format)
        strip_silence: If True, decode through libimbe and remove
            zero-energy frames (P25 signaling/preamble/silence).

    Returns:
        fv: (N, 8) int16 array of IMBE codewords u[0..7]
        tgid: talkgroup ID from first CODEC_FRAME (0 if no frames)
    """
    with open(path, "rb") as f:
        data = f.read()

    frames = []
    tgid = 0
    pos = 0
    dlen = len(data)

    while pos + SSSP_HEADER_SIZE <= dlen:
        # Parse 8-byte message header
        magic = data[pos:pos + 2]
        if magic != SSSP_MAGIC:
            # Try to resync: scan forward for 'SY' + version byte
            found = False
            for scan in range(pos + 1, dlen - 2):
                if data[scan:scan + 2] == SSSP_MAGIC and data[scan + 2] == SSSP_VERSION:
                    pos = scan
                    found = True
                    break
            if not found:
                break
            continue

        version = data[pos + 2]
        msg_type = data[pos + 3]
        payload_len = struct.unpack_from("<I", data, pos + 4)[0]
        pos += SSSP_HEADER_SIZE

        # Bounds check for payload
        if pos + payload_len > dlen:
            break  # truncated message

        if msg_type == SSSP_MSG_CODEC_FRAME:
            if payload_len >= SSSP_CODEC_HDR_SIZE:
                # Parse 24-byte codec header
                tg = struct.unpack_from("<I", data, pos)[0]
                src_id = struct.unpack_from("<I", data, pos + 4)[0]
                call_id = struct.unpack_from("<I", data, pos + 8)[0]
                timestamp_us = struct.unpack_from("<Q", data, pos + 12)[0]
                codec_type = data[pos + 20]
                param_count = data[pos + 21]
                errs = data[pos + 22]
                flags = data[pos + 23]

                if len(frames) == 0:
                    tgid = tg

                # Extract param_count × uint32 parameters
                params_offset = pos + SSSP_CODEC_HDR_SIZE
                expected_bytes = param_count * 4
                if params_offset + expected_bytes <= pos + payload_len:
                    params = list(struct.unpack_from(
                        "<%dI" % param_count, data, params_offset
                    ))
                    frames.append(params)

        # Skip to next message regardless of type
        pos += payload_len

    if not frames:
        fv = np.zeros((0, 8), dtype=np.int16)
        return fv, tgid

    # Pad frames to consistent width (should be 8 for IMBE)
    max_params = max(len(f) for f in frames)
    for i, f in enumerate(frames):
        if len(f) < max_params:
            frames[i] = f + [0] * (max_params - len(f))

    fv = np.array(frames, dtype=np.int16)

    if strip_silence:
        fv = _strip_silence_frames(fv)

    return fv, tgid


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
    """Load model from checkpoint.

    Supports three checkpoint formats, detected by file extension:
      - ``.pth``  — PyTorch checkpoint (dict with model_state_dict + config)
      - ``.safetensors`` — SafeTensors state dict; config.json must sit in the
        same directory with keys: input_dim, d_model, n_heads, d_ff, n_layers,
        conv_kernel_size (or conv_kernel), vocab_size, dropout, use_subsampling.
      - ``.onnx`` — ONNX model loaded via onnxruntime.  Returns an
        :class:`OnnxModel` wrapper instead of a PyTorch model.

    Returns:
        model: ConformerCTC (PyTorch) or OnnxModel (ONNX), in eval mode
        device: torch device (or "cpu" for ONNX)
        ckpt_info: dict with at least ``config``; for .pth also contains
                   ``epoch``, ``best_wer``, etc.
    """
    ext = os.path.splitext(checkpoint_path)[1].lower()

    # ----- ONNX -----
    if ext == ".onnx":
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(checkpoint_path, providers=providers)

        # Try to load companion config.json
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
        cfg = {}
        if os.path.isfile(config_path):
            with open(config_path) as f:
                cfg = json.load(f)

        model = OnnxModel(session)
        onnx_device = torch.device("cpu")  # device not used for ONNX inference
        ckpt_info = {"config": cfg, "epoch": cfg.get("epoch", -1),
                     "best_wer": cfg.get("best_wer", 0)}
        return model, onnx_device, ckpt_info

    # ----- SafeTensors -----
    if ext == ".safetensors":
        from safetensors.torch import load_file

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        state_dict = load_file(checkpoint_path, device=str(device))

        # Load config.json from same directory
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                "SafeTensors checkpoint requires config.json in the same "
                "directory: %s" % config_path
            )
        with open(config_path) as f:
            cfg = json.load(f)

        # Normalise config key names (config.json may use conv_kernel_size
        # or conv_kernel, and use_subsampling or subsample)
        conv_kernel = cfg.get("conv_kernel",
                              cfg.get("conv_kernel_size", 31))
        subsample = cfg.get("subsample",
                            cfg.get("use_subsampling", False))

        model = ConformerCTC(
            input_dim=cfg["input_dim"], d_model=cfg["d_model"],
            n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
            n_layers=cfg["n_layers"], conv_kernel=conv_kernel,
            vocab_size=cfg["vocab_size"], dropout=0.0,
            subsample=subsample,
        ).to(device).eval()
        model.load_state_dict(state_dict)

        ckpt_info = {"config": cfg, "epoch": cfg.get("epoch", -1),
                     "best_wer": cfg.get("best_wer", 0)}
        return model, device, ckpt_info

    # ----- PyTorch .pth (default / backward compat) -----
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
        model: ConformerCTC or OnnxModel
        features: (T, D) numpy array, already normalized
        device: torch device

    Returns:
        Transcribed text string
    """
    if getattr(model, "is_onnx", False):
        x_np = features[np.newaxis, :, :].astype(np.float32)
        lengths_np = np.array([features.shape[0]], dtype=np.int64)
        log_probs, out_lengths = model(x_np, lengths_np)
        # Convert back to torch for decode_greedy
        lp = torch.from_numpy(log_probs[0, :int(out_lengths[0])])
        return decode_greedy(lp)

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

        if getattr(model, "is_onnx", False):
            x_np = features[:end][np.newaxis, :, :].astype(np.float32)
            lengths_np = np.array([end], dtype=np.int64)
            log_probs_np, out_lengths_np = model(x_np, lengths_np)
            lp = torch.from_numpy(log_probs_np[0, :int(out_lengths_np[0])])
            hyp = decode_greedy(lp)
        else:
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
    parser.add_argument("--dvcf-file", default=None, help="TAP file to transcribe")
    parser.add_argument("--watch", default=None,
                        help="Watch directory for new .dvcf files and transcribe")
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

    epoch = ckpt.get("epoch", -1)
    best_wer = ckpt.get("best_wer", 0) or 0
    is_onnx = getattr(model, "is_onnx", False)
    fmt_label = "ONNX" if is_onnx else "PyTorch"
    print("Model (%s): epoch %d, best WER=%.1f%%" %
          (fmt_label, epoch + 1, best_wer))
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

    elif args.dvcf_file:
        # Decode TAP file through libimbe first
        from .precompute import decode_frame_vectors

        fv, tgid = _read_dvcf_file(args.dvcf_file)
        if fv.shape[0] < args.min_frames:
            print("TAP: %s -- too short (%d frames), skipping" %
                  (args.dvcf_file, fv.shape[0]))
        else:
            raw_params = decode_frame_vectors(fv)
            feats = (raw_params.astype(np.float32) - mean) / std

            dur = feats.shape[0] * 0.020
            print("TAP: %s (%.1fs, %d frames, tgid=%s)" %
                  (args.dvcf_file, dur, feats.shape[0], tgid))
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

        print("Watching %s for new .dvcf files..." % watch_dir)
        print("Press Ctrl-C to stop.\n")

        seen = set()
        # Pre-populate with existing files so we only transcribe new ones
        for p in watch_dir.rglob("*.dvcf"):
            seen.add(str(p))
        print("(skipping %d existing files)\n" % len(seen))

        try:
            while True:
                new_files = []
                for p in sorted(watch_dir.rglob("*.dvcf")):
                    sp = str(p)
                    if sp not in seen:
                        seen.add(sp)
                        new_files.append(p)

                for tap_path in new_files:
                    # Wait briefly for file to finish writing
                    time.sleep(0.2)
                    try:
                        fv, tgid = _read_dvcf_file(str(tap_path))
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
        print("\nProvide --npz, --dvcf-file, --watch, or --all-val")


if __name__ == "__main__":
    main()
