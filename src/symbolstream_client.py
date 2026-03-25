#!/usr/bin/env python3
"""Symbolstream client -- TCP server for live IMBE-ASR transcription.

Listens for incoming TCP connections from the trunk-recorder symbolstream
plugin (JSON mode). Accumulates IMBE frames by talkgroup, transcribes on
call_end, prints results to stdout.

The symbolstream plugin connects TO this server and pushes frames.

Usage:
    python -m src.symbolstream_client \
        --checkpoint checkpoints/best.pth \
        --port 9090

    # With progressive transcription during calls:
    python -m src.symbolstream_client \
        --checkpoint checkpoints/best.pth \
        --port 9090 --stream

Symbolstream plugin config (in trunk-recorder config.json):
    {
        "name": "symbolstream",
        "library": "libsymbolstream_plugin",
        "streams": [{
            "address": "<this machine's IP>",
            "port": 9090,
            "TGID": 0,
            "useTCP": true,
            "sendJSON": true
        }]
    }
"""

import argparse
import json
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


def recv_exact(sock, n):
    """Read exactly n bytes from socket, or return None on disconnect."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def read_message(sock):
    """Read one symbolstream JSON message + optional codewords.

    Returns:
        (metadata_dict, codewords_or_None) on success
        None on disconnect
    """
    # 4 bytes: JSON length
    header = recv_exact(sock, 4)
    if header is None:
        return None

    json_len = struct.unpack("<I", header)[0]
    if json_len > 65536:
        # Sanity check -- corrupted stream
        return None

    # N bytes: JSON metadata
    json_bytes = recv_exact(sock, json_len)
    if json_bytes is None:
        return None

    try:
        meta = json.loads(json_bytes)
    except json.JSONDecodeError:
        return None

    # codec_frame events have 32 bytes of codewords appended
    codewords = None
    if meta.get("event") == "codec_frame":
        cw_bytes = recv_exact(sock, 32)
        if cw_bytes is None:
            return None
        codewords = struct.unpack("<8I", cw_bytes)

    return meta, codewords


def main():
    parser = argparse.ArgumentParser(
        description="Symbolstream receiver for live IMBE-ASR transcription")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-frames", type=int, default=10,
                        help="Min frames before transcribing (default: 10)")
    parser.add_argument("--stream", action="store_true",
                        help="Show progressive transcription during calls")
    parser.add_argument("--stream-interval", type=float, default=1.0,
                        help="Seconds between stream updates (default: 1.0)")
    parser.add_argument("--output-dir", default=None,
                        help="Write transcription JSON files to this directory")
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

    print("IMBE-ASR Symbolstream Client")
    print("Model: epoch %d, best WER=%.1f%%, device=%s" %
          (ckpt["epoch"] + 1, ckpt["best_wer"], device))
    print("Listening on %s:%d" % (args.host, args.port))
    print("-" * 70)

    def transcribe_call(tgid, call):
        """Transcribe accumulated frames for a completed call."""
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

        if args.stream:
            # Clear the progressive line
            sys.stdout.write("\r" + " " * 120 + "\r")

        print("[TG=%s src=%s] (%.1fs, %d frames, %.0fms)" %
              (tgid, call["src_id"], dur, len(frames), dt))
        print("  >> %s" % hyp)
        print()
        sys.stdout.flush()

        # Write transcription to disk if --output-dir is set
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            ts = int(time.time())
            fname = "%s_%s_%d.json" % (tgid, call["src_id"], ts)
            fpath = os.path.join(args.output_dir, fname)
            result = {
                "talkgroup": tgid,
                "src_id": call["src_id"],
                "duration": round(dur, 2),
                "frames": len(frames),
                "inference_ms": round(dt, 1),
                "transcript": hyp,
                "timestamp": ts,
                "source": "symbolstream_client"
            }
            with open(fpath, "w") as out:
                json.dump(result, out, indent=2)
            print("  -> %s" % fpath)

    def stream_update(tgid, call):
        """Show progressive transcription for an in-progress call."""
        frames = call["frames"]
        if len(frames) < args.min_frames:
            return

        fv = np.array(frames, dtype=np.int16)
        raw_params = decode_frame_vectors(fv)
        feats = (raw_params.astype(np.float32) - mean) / std
        dur = len(frames) * 0.020

        hyp = transcribe(model, feats, device)
        line = "[TG=%s %.1fs] %s" % (tgid, dur, hyp)
        sys.stdout.write("\r" + " " * 120 + "\r")
        sys.stdout.write(line)
        sys.stdout.flush()

    # TCP server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    print("Waiting for symbolstream connection...")

    try:
        while True:
            conn, addr = server.accept()
            print("Connected: %s:%d" % addr)

            calls = defaultdict(lambda: {
                "frames": [],
                "src_id": 0,
                "last_stream": 0,
            })

            try:
                while True:
                    result = read_message(conn)
                    if result is None:
                        print("Disconnected: %s:%d" % addr)
                        break

                    meta, codewords = result
                    event = meta.get("event", "")
                    tgid = meta.get("talkgroup", 0)

                    if event == "call_start":
                        calls[tgid]["frames"] = []
                        calls[tgid]["src_id"] = meta.get("src", 0)

                    elif event == "codec_frame":
                        call = calls[tgid]
                        call["frames"].append(codewords)
                        call["src_id"] = meta.get("src", call["src_id"])

                        # Progressive streaming
                        if args.stream:
                            now = time.time()
                            if now - call["last_stream"] >= args.stream_interval:
                                stream_update(tgid, call)
                                call["last_stream"] = now

                    elif event == "call_end":
                        call = calls[tgid]
                        if meta.get("encrypted", False):
                            print("[TG=%s] (encrypted, skipped)" % tgid)
                        else:
                            transcribe_call(tgid, call)
                        calls[tgid]["frames"] = []

            except ConnectionResetError:
                print("Connection reset: %s:%d" % addr)
            finally:
                conn.close()

            # Transcribe any in-progress calls from the dropped connection
            for tgid, call in calls.items():
                if call["frames"]:
                    if args.stream:
                        sys.stdout.write("\n")
                    transcribe_call(tgid, call)

            print("Waiting for symbolstream connection...")

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.close()


if __name__ == "__main__":
    main()
