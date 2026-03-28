FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install Python 3.11 + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip python3.11-dev \
    build-essential cmake git \
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
COPY server/entrypoint.sh /app/entrypoint.sh
COPY vocoder/ /app/vocoder/

# Build libimbe.so from trunk-recorder's open-source op25 vocoder implementation.
# This is the same reverse-engineered IMBE codec used by trunk-recorder itself.
RUN git clone --depth 1 --filter=blob:none --sparse \
        https://github.com/TrunkRecorder/trunk-recorder.git /tmp/tr && \
    cd /tmp/tr && git sparse-checkout set lib/op25_repeater/lib/imbe_vocoder && \
    cd /app/vocoder && make VOCODER_DIR=/tmp/tr/lib/op25_repeater/lib/imbe_vocoder && \
    cp /app/vocoder/libimbe.so /app/libimbe.so && \
    rm -rf /tmp/tr

# Models and LM are mounted at runtime via volumes or env vars.
# See docs.luxprimatech.com/#/imbe-asr-setup for deployment guide.
ENV IMBE_ASR_DEVICE=cuda
ENV IMBE_ASR_MODEL=trunk-reporter/imbe-asr-base-512d-p25
ENV MODELS_DIR=/models

RUN pip install --no-cache-dir huggingface_hub && \
    chmod +x /app/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
