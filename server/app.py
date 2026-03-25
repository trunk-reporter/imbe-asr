#!/usr/bin/env python3
"""FastAPI transcription server for IMBE ASR.

Endpoints:
    POST /v1/audio/transcriptions  — upload a .tap file, get text back
    GET  /health                   — model/server status

Environment variables:
    IMBE_ASR_CHECKPOINT  — path to model checkpoint (required)
    IMBE_ASR_STATS       — path to stats.npz (default: <checkpoint_dir>/stats.npz)
    IMBE_ASR_DEVICE      — torch device (default: auto-detect cuda/cpu)
    IMBE_ASR_MIN_FRAMES  — minimum voice frames to attempt transcription (default: 10)
"""

import os
import sys
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Add project root to path so we can import src.*
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference import load_model, load_stats, transcribe, _read_tap_file
from src.precompute import decode_frame_vectors

logger = logging.getLogger("imbe_asr")

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_model = None
_device = None
_mean = None
_std = None
_ckpt_info = {}
_min_frames = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global _model, _device, _mean, _std, _ckpt_info, _min_frames

    checkpoint_path = os.environ.get("IMBE_ASR_CHECKPOINT")
    if not checkpoint_path:
        logger.error("IMBE_ASR_CHECKPOINT not set")
        raise RuntimeError("IMBE_ASR_CHECKPOINT environment variable is required")

    stats_path = os.environ.get(
        "IMBE_ASR_STATS",
        os.path.join(os.path.dirname(checkpoint_path), "stats.npz"),
    )

    device_str = os.environ.get("IMBE_ASR_DEVICE")
    device = torch.device(device_str) if device_str else None

    _min_frames = int(os.environ.get("IMBE_ASR_MIN_FRAMES", "10"))

    logger.info("Loading model from %s", checkpoint_path)
    _model, _device, ckpt = load_model(checkpoint_path, device)
    _mean, _std = load_stats(stats_path)
    _ckpt_info = {
        "checkpoint": checkpoint_path,
        "epoch": ckpt.get("epoch", -1) + 1,
        "best_wer": ckpt.get("best_wer", None),
    }
    logger.info(
        "Model loaded: epoch %d, best_wer=%.1f%%, device=%s",
        _ckpt_info["epoch"],
        _ckpt_info.get("best_wer") or 0,
        _device,
    )
    yield


app = FastAPI(
    title="IMBE ASR Server",
    description="P25 IMBE vocoder → text transcription",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class TranscriptionResponse(BaseModel):
    text: str
    duration: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str | None = None
    checkpoint: str | None = None
    epoch: int | None = None
    best_wer: float | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if _model is not None else "not_ready",
        model_loaded=_model is not None,
        device=str(_device) if _device else None,
        **_ckpt_info,
    )


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read uploaded file to a temp location
    import tempfile

    suffix = Path(file.filename).suffix if file.filename else ".tap"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        t0 = time.time()

        # Parse TAP file → IMBE frame vectors
        fv, tgid = _read_tap_file(tmp_path)

        if fv.shape[0] < _min_frames:
            raise HTTPException(
                status_code=422,
                detail="Too few voice frames (%d < %d minimum)" % (fv.shape[0], _min_frames),
            )

        # Decode frame vectors → 170-dim raw parameters via libimbe
        raw_params = decode_frame_vectors(fv)
        feats = (raw_params.astype(np.float32) - _mean) / _std

        duration = feats.shape[0] * 0.020  # 20ms per frame

        # Run CTC inference
        text = transcribe(_model, feats, _device)

        elapsed = time.time() - t0
        logger.info(
            "Transcribed %s: %.1fs audio, %d frames, %.0fms inference, tgid=%s",
            file.filename, duration, feats.shape[0], elapsed * 1000, tgid,
        )

        return TranscriptionResponse(text=text, duration=round(duration, 3))

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Transcription failed for %s", file.filename)
        raise HTTPException(status_code=500, detail="Transcription failed: %s" % str(e))
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Run with: uvicorn server.app:app --host 0.0.0.0 --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
