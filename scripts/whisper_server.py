#!/usr/bin/env python3
"""Whisper transcription server using openai-whisper.

OpenAI-compatible API on port 8766, runs alongside Qwen3-ASR on 8765.

Endpoints:
    POST /v1/audio/transcriptions  — transcription
    GET  /health                   — health check
"""

import io
import logging
import time

import numpy as np
import torch
import whisper
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

MODEL_SIZE = "large-v3"

log.info("Loading Whisper %s on cuda..." % MODEL_SIZE)
model = whisper.load_model(MODEL_SIZE, device="cuda")
log.info("Whisper model loaded. VRAM: %d MB" %
         (torch.cuda.memory_allocated() / 1e6))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "whisper-%s" % MODEL_SIZE})


@app.route("/v1/audio/transcriptions", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "no file provided"}), 400

    audio_file = request.files["file"]

    # Save to temp file (whisper needs a file path)
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    try:
        t0 = time.time()
        result = model.transcribe(tmp_path, language="en", fp16=True)
        text = result["text"].strip()
        dt = time.time() - t0

        log.info("Transcribed in %.1fs: %s" % (dt, text[:80]))
        return jsonify({"text": text})
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8766, threaded=False)
