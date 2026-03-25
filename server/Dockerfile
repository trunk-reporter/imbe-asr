FROM python:3.11-slim

WORKDIR /app

# System deps for libimbe
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy project code
COPY src/ /app/src/
COPY server/ /app/server/
COPY vocoder/libimbe.so /app/vocoder/libimbe.so

# Checkpoint and stats are mounted at runtime
ENV IMBE_ASR_CHECKPOINT=/models/best.pth
ENV IMBE_ASR_STATS=/models/stats.npz

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
