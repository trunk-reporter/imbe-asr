#!/usr/bin/env python3
"""Live P25 transcription from trunk-recorder IMBE tap socket.

Reads raw IMBE frames from the Unix DGRAM socket, groups by talkgroup,
decodes through libimbe -> 170-dim raw params, and transcribes in real-time.

Usage:
    python -m src.live \
        --checkpoint checkpoints/best.pth \
        --socket /tmp/imbe_tap.sock
"""

import argparse
import os
import socket
import struct
import sys
import time
from collections import defaultdict

import numpy as np
import torch

from .inference import load_model, load_stats, transcribe
from .precompute import decode_frame_vectors

# struct imbe_tap_frame (packed, 36 bytes):
#   uint32  magic
#   uint32  seq
#   uint16  tgid
#   uint32  src_id
#   uint16  errs
#   uint16  E0
#   uint16  ET
#   uint16  u[8]
FRAME_FMT = "<IIHI HHH8H"
FRAME_SIZE = struct.calcsize(FRAME_FMT)
MAGIC = 0x494D4245

# Gap between frames that indicates end of transmission (seconds)
CALL_GAP = 1.0


def main():
    parser = argparse.ArgumentParser(
        description="Live P25 transcription from IMBE tap")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--socket", default="/tmp/imbe_tap.sock")
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-frames", type=int, default=10,
                        help="Min frames before transcribing (default: 10)")
    parser.add_argument("--stream", action="store_true",
                        help="Show progressive transcription during call")
    parser.add_argument("--stream-interval", type=float, default=1.0,
                        help="Seconds between stream updates (default: 1.0)")
    args = parser.parse_args()

    # Load model
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, device, ckpt = load_model(args.checkpoint, device)

    if args.stats:
        stats_path = args.stats
    else:
        stats_path = os.path.join(os.path.dirname(args.checkpoint), "stats.npz")
    mean, std = load_stats(stats_path)

    print("IMBE-ASR Live Transcription")
    print("Model: epoch %d, best WER=%.1f%%, device=%s" %
          (ckpt["epoch"] + 1, ckpt["best_wer"], device))
    print("Listening on %s" % args.socket)
    print("-" * 70)

    # Connect to the tap socket (as a second listener)
    # The socket already exists (trunk-recorder created it), so we bind
    # our own socket and trunk-recorder sends to it via DGRAM
    # Actually -- trunk-recorder sends to a fixed path. We need to be
    # the one bound to that path. If it's already bound, we can't rebind.
    # Check if socket exists and if we can receive from it.

    sock_path = args.socket

    # Take over the socket (remove stale, bind fresh)
    if os.path.exists(sock_path):
        os.unlink(sock_path)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.bind(sock_path)
    sock.settimeout(CALL_GAP)

    print("Listening on %s" % sock_path)
    print("Waiting for IMBE frames from trunk-recorder...")
    print()

    # Track active calls by talkgroup
    calls = defaultdict(lambda: {
        "frames": [],
        "src_id": 0,
        "last_time": 0,
    })

    def transcribe_call(tgid, call):
        """Transcribe accumulated frames for a call."""
        frames = call["frames"]
        if len(frames) < args.min_frames:
            return

        fv = np.array(frames, dtype=np.int16)
        raw_params = decode_frame_vectors(fv)
        feats = (raw_params.astype(np.float32) - mean) / std
        dur = len(frames) * 0.020

        t0 = time.time()
        hyp = transcribe(model, feats, device)
        dt = (time.time() - t0) * 1000

        print("[TG=%s src=%s] (%.1fs, %d frames, %.0fms)" %
              (tgid, call["src_id"], dur, len(frames), dt))
        print("  >> %s" % hyp)
        print()
        sys.stdout.flush()

    try:
        while True:
            try:
                data = sock.recv(256)
            except socket.timeout:
                # Check for completed calls
                now = time.time()
                finished = []
                for tgid, call in calls.items():
                    if call["frames"] and (now - call["last_time"]) > CALL_GAP:
                        transcribe_call(tgid, call)
                        finished.append(tgid)
                for tgid in finished:
                    calls[tgid]["frames"] = []
                continue

            if len(data) < FRAME_SIZE:
                continue

            fields = struct.unpack(FRAME_FMT, data[:FRAME_SIZE])
            magic, seq, tgid, src_id, errs, E0, ET, *u = fields

            if magic != MAGIC:
                continue

            # Skip lost frames
            if errs >= 0xFFFF:
                continue

            call = calls[tgid]
            call["frames"].append(u)
            call["src_id"] = src_id
            call["last_time"] = time.time()

            # Stream mode: show progressive transcription
            if args.stream and len(call["frames"]) >= args.min_frames:
                if len(call["frames"]) % int(args.stream_interval / 0.020) == 0:
                    fv = np.array(call["frames"], dtype=np.int16)
                    raw_params = decode_frame_vectors(fv)
                    feats = (raw_params.astype(np.float32) - mean) / std
                    hyp = transcribe(model, feats, device)
                    dur = len(call["frames"]) * 0.020
                    line = "[TG=%s %.1fs] %s" % (tgid, dur, hyp)
                    sys.stdout.write('\r' + ' ' * 120 + '\r')
                    sys.stdout.write(line)
                    sys.stdout.flush()

    except KeyboardInterrupt:
        # Transcribe any remaining calls
        for tgid, call in calls.items():
            if call["frames"]:
                if args.stream:
                    sys.stdout.write('\n')
                transcribe_call(tgid, call)
        print("\nDone.")
    finally:
        sock.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)


if __name__ == "__main__":
    main()
