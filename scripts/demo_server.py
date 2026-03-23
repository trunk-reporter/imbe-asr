#!/usr/bin/env python3
"""Live P25 transcription + audio demo server.

Relay between the trunk-recorder imbe_asr plugin (TCP) and web browsers
(WebSocket). Receives interleaved audio packets and JSON text lines from
the plugin, and forwards them to connected browsers.

Protocol from plugin (TCP port 9090):
  - Audio: 4-byte header ('A' + tgid_hi + tgid_lo + 0) + 320 bytes PCM int16 @ 8kHz
  - Text:  JSON line terminated by \n (partial or final transcriptions)

Protocol to browser (WebSocket port 8081):
  - Binary frames: audio packets (header + PCM, same as plugin format)
  - Text frames: JSON messages (transcriptions + metadata)

Modes:
  --watch DIR        Standalone mode (watches .tap files, runs own inference)
  (no --watch)       Plugin mode (reads from plugin TCP socket)

Usage:
    # Plugin mode (recommended):
    python3 scripts/demo_server.py --port 8080

    # Standalone mode (no plugin needed):
    python3 scripts/demo_server.py --checkpoint checkpoints/p25_finetuned/best.pth \
        --watch ~/trunk-recorder/audio/
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import websockets
from websockets.server import serve
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

AUDIO_HEADER_SIZE = 4

# Connected WebSocket clients
clients = set()

# Beam search decoder (loaded lazily)
beam_decoder = None

def get_beam_decoder():
    global beam_decoder
    if beam_decoder is not None:
        return beam_decoder
    try:
        from pyctcdecode import build_ctcdecoder
        lm_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'lm', '5gram.bin')
        unigrams_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'lm', 'unigrams.txt')
        if not os.path.exists(lm_path):
            print("KenLM not found at %s, using greedy decode" % lm_path)
            return None
        VOCAB = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '"
        labels = [''] + list(VOCAB)
        with open(unigrams_path) as f:
            unigrams = [line.strip() for line in f if line.strip()]
        beam_decoder = build_ctcdecoder(labels, kenlm_model_path=lm_path,
                                         unigrams=unigrams, alpha=0.15, beta=1.0)
        print("Beam search decoder loaded (KenLM + %d unigrams)" % len(unigrams))
    except Exception as e:
        print("Beam search not available: %s" % e)
    return beam_decoder


def beam_decode_logprobs(b64_logprobs, T):
    """Decode base64-encoded log_probs with beam search + KenLM."""
    import base64
    import numpy as np
    decoder = get_beam_decoder()
    if decoder is None:
        return None
    raw = base64.b64decode(b64_logprobs)
    lp = np.frombuffer(raw, dtype=np.float32).reshape(T, 39)
    return decoder.decode(lp, beam_width=32)


async def ws_handler(websocket):
    clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        clients.discard(websocket)


async def broadcast_text(msg):
    if clients:
        data = json.dumps(msg)
        await asyncio.gather(
            *[c.send(data) for c in clients],
            return_exceptions=True)


async def broadcast_binary(data):
    if clients:
        await asyncio.gather(
            *[c.send(data) for c in clients],
            return_exceptions=True)


def serve_http(port):
    html_dir = str(Path(__file__).parent.parent / "demo")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=html_dir, **kwargs)
        def log_message(self, format, *args):
            pass

    server = HTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()


async def read_plugin_tcp(host, port):
    """Read interleaved audio + JSON from the plugin's TCP socket."""
    while True:
        try:
            reader, writer = await asyncio.open_connection(host, port, limit=1024*1024)
            print("Connected to plugin at %s:%d" % (host, port))

            while True:
                # Peek at first byte to determine packet type
                header = await reader.readexactly(1)

                if header[0] == ord('A'):
                    # Audio packet: read remaining 3 header bytes + 320 bytes PCM
                    rest = await reader.readexactly(3 + 160 * 2)
                    packet = header + rest
                    tgid = (packet[1] << 8) | packet[2]
                    await broadcast_binary(packet)

                elif header[0] == ord('{'):
                    # JSON line: read until newline
                    line = header + await reader.readline()
                    line = line.decode("utf-8", errors="replace").strip()
                    if line:
                        try:
                            msg = json.loads(line)
                            msg["timestamp"] = time.time()
                            kind = msg.get("type", "?")
                            tgid = msg.get("tgid", "?")
                            text = msg.get("text", "")

                            if kind == "final" and "logprobs" in msg:
                                # Strip logprobs before sending to browser
                                if "logprobs" in msg: del msg["logprobs"]
                                if "T" in msg: del msg["T"]
                                print("[TG=%s] FINAL: %s" % (tgid, text[:80]))
                            elif kind == "final":
                                print("[TG=%s] FINAL: %s" % (tgid, text[:80]))
                            elif kind == "partial":
                                pass  # quiet for partials

                            await broadcast_text(msg)
                        except json.JSONDecodeError:
                            pass
                else:
                    # Unknown byte, skip rest of line
                    await reader.readline()

        except asyncio.IncompleteReadError:
            print("Plugin disconnected")
        except (ConnectionRefusedError, OSError) as e:
            pass  # quiet retry

        await asyncio.sleep(2)


async def watch_and_transcribe(watch_dir, model, device, mean, std,
                                min_frames=15):
    """Standalone mode: poll .tap files and transcribe."""
    import numpy as np
    import torch
    from src.inference import _read_tap_file, transcribe
    from src.precompute import decode_frame_vectors

    watch_path = Path(watch_dir)
    seen = set()
    for p in watch_path.rglob("*.tap"):
        seen.add(str(p))
    print("Skipping %d existing files" % len(seen))

    while True:
        new_files = []
        for p in sorted(watch_path.rglob("*.tap")):
            sp = str(p)
            if sp not in seen:
                seen.add(sp)
                new_files.append(p)

        for tap_path in new_files:
            await asyncio.sleep(0.2)
            try:
                fv, tgid = _read_tap_file(str(tap_path))
            except Exception:
                continue
            if fv.shape[0] < min_frames:
                continue

            raw_params = decode_frame_vectors(fv)
            feats = (raw_params.astype(np.float32) - mean) / std
            dur = feats.shape[0] * 0.020
            t0 = time.time()
            hyp = transcribe(model, feats, device)
            dt = (time.time() - t0) * 1000

            if not hyp.strip():
                continue

            msg = {
                "type": "final",
                "tgid": int(tgid),
                "text": hyp,
                "duration": round(dur, 1),
                "latency_ms": round(dt),
                "timestamp": time.time(),
            }
            print("[TG=%s] (%.1fs) %s" % (tgid, dur, hyp[:80]))
            await broadcast_text(msg)

        await asyncio.sleep(0.5)


async def main():
    parser = argparse.ArgumentParser(description="Live P25 demo server")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--watch", default=None)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ws-port", type=int, default=8081)
    parser.add_argument("--plugin-host", default="localhost")
    parser.add_argument("--plugin-port", type=int, default=9090)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    plugin_mode = args.watch is None

    if not plugin_mode:
        import numpy as np
        import torch
        from src.inference import load_model, load_stats
        device = torch.device(args.device)
        model, device, ckpt = load_model(args.checkpoint, device)
        stats_path = args.stats or os.path.join(
            os.path.dirname(args.checkpoint), "stats.npz")
        s = np.load(stats_path)
        mean, std = s["mean"], s["std"]
        print("Model: epoch %d, WER=%.1f%%" %
              (ckpt["epoch"] + 1, ckpt["best_wer"]))
        print("Watching: %s" % args.watch)
    else:
        print("Plugin mode: %s:%d" % (args.plugin_host, args.plugin_port))
        model = device = mean = std = None

    print("HTTP:      http://localhost:%d" % args.port)
    print("WebSocket: ws://localhost:%d\n" % args.ws_port)

    http_thread = threading.Thread(target=serve_http, args=(args.port,),
                                   daemon=True)
    http_thread.start()

    async with serve(ws_handler, "0.0.0.0", args.ws_port):
        if plugin_mode:
            await read_plugin_tcp(args.plugin_host, args.plugin_port)
        else:
            await watch_and_transcribe(args.watch, model, device, mean, std)


if __name__ == "__main__":
    asyncio.run(main())
