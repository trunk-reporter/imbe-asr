FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip python3.11-dev \
    build-essential cmake \
    libboost-program-options-dev libboost-system-dev libboost-thread-dev \
    libsndfile1 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir onnxruntime-gpu && \
    pip install --no-cache-dir pyctcdecode https://github.com/kpu/kenlm/archive/master.zip

COPY src/ /app/src/
COPY server/ /app/server/

# libimbe.so is NOT included in the public image (reverse-engineered proprietary codec).
# Required only for raw codeword input (.tap / live symbolstream).
# Not needed for .dvcf transcription via tr-plugin-dvcf (the primary use case).
# Mount at runtime if needed: -v /path/to/libimbe.so:/app/libimbe.so

# Models and LM are mounted at runtime via volumes or env vars.
# See docs.luxprimatech.com/#/imbe-asr-setup for deployment guide.
ENV IMBE_ASR_CHECKPOINT=/models/model_int8.onnx
ENV IMBE_ASR_STATS=/models/stats.npz
ENV IMBE_ASR_DEVICE=cuda

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
