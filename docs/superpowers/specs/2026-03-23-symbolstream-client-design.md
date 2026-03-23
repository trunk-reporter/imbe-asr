# symbolstream_client design

## Purpose

TCP client that receives JSON-mode symbolstream plugin output from trunk-recorder, accumulates IMBE frames by call, and transcribes on call completion. Proof-of-concept / dev tool -- validates the end-to-end pipeline from live radio to text via symbolstream.

## File

`src/symbolstream_client.py` -- single module, runnable as `python -m src.symbolstream_client`.

## Protocol

Connects to symbolstream plugin over TCP (JSON mode, `sendJSON=true`). The wire format is:

### Codec frame

```
[4 bytes] json_length (uint32 LE)
[N bytes] JSON: {"event":"codec_frame","talkgroup":N,"src":N,"codec_type":0,"errs":N,"short_name":"..."}
[32 bytes] u[0..7] (8 x uint32 LE) -- IMBE codewords
```

### Call start (JSON only, no codewords)

```
[4 bytes] json_length (uint32 LE)
[N bytes] JSON: {"event":"call_start","talkgroup":N,"freq":N,"short_name":"..."}
```

### Call end (JSON only, no codewords)

```
[4 bytes] json_length (uint32 LE)
[N bytes] JSON: {"event":"call_end","talkgroup":N,"src":N,"duration":N.N,"error_count":N,"encrypted":bool,...}
```

## State machine

Per-talkgroup call state:

```
idle -> accumulating (on call_start or first codec_frame)
accumulating -> transcribing (on call_end)
transcribing -> idle (after output)
```

Frame buffer is a list of `u[0..7]` codeword arrays, keyed by talkgroup. `call_start` resets the buffer. `call_end` triggers transcription and clears the buffer.

If `codec_frame` arrives without a prior `call_start` (e.g., client connected mid-call), accumulate anyway -- call_end still triggers transcription.

## Inference pipeline

Same path as existing `src/live.py`:

1. Accumulated `u[0..7]` codewords (list of 8-element int arrays)
2. `decode_frame_vectors()` from `src/precompute.py` -- libimbe decode to 170-dim raw params
3. Normalize with `stats.npz` (per-dimension z-score)
4. Model forward pass (PyTorch, single batch)
5. Greedy CTC decode via `src/tokenizer.decode_greedy()`

Reuses `load_model()`, `load_stats()`, `transcribe()` from `src/inference.py`.

## Progressive streaming (optional)

With `--stream` flag, transcribe every `--stream-interval` seconds (default 1.0) during an active call, printing progressive results with carriage return overwrite. Final transcription on call_end prints on a new line. Same approach as `live.py --stream`.

## Connection handling

- Connect to `host:port` on startup
- On connection failure or disconnect: log, wait with linear backoff (1s, 2s, 3s, ... capped at 10s), retry
- Clean shutdown on SIGINT (Ctrl+C): transcribe any in-progress calls, then exit

## CLI

```
python -m src.symbolstream_client \
    --checkpoint checkpoints/best.pth \
    --host 127.0.0.1 \
    --port 9090 \
    --stream \
    --stream-interval 1.0 \
    --min-frames 10 \
    --device cpu \
    --stats path/to/stats.npz
```

All arguments except `--checkpoint` have sensible defaults.

## Output

Stdout, one block per completed call:

```
[TG=12345 src=67890] (3.2s, 160 frames, 45ms)
  >> MEDIC 61 RESPOND TO 1234 MAIN STREET
```

Fields: talkgroup, source radio ID, call duration, frame count, inference time in ms.

Encrypted calls (from call_end metadata) are skipped with a note:

```
[TG=12345] (encrypted, skipped)
```

## Not in scope

- UDP mode
- Binary wire format (sendJSON=false)
- Webhook / MQTT / external output
- ONNX Runtime inference (use PyTorch for POC)
- Multi-server connections
- C implementation (future -- port protocol handling into inference/imbe_asr.c)

## Dependencies

No new dependencies. Uses: socket, struct, json (stdlib) + existing src/ modules.

## Testing

Manual testing against a running trunk-recorder instance with symbolstream configured. Can also test with a simple Python script that replays recorded frames over TCP in symbolstream JSON format.
