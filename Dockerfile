# Slim image: do NOT bake model weights. First-run caches to /runpod-volume.
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Python deps (pin for reproducibility)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      runpod==1.7.13 \
      diffusers==0.35.1 \
      transformers==4.56.0 \
      accelerate==1.10.1 \
      huggingface_hub==0.34.4 \
      hf_transfer==0.1.9 \
      fastapi==0.116.1 uvicorn==0.35.0 \
      boto3==1.40.21 botocore==1.40.21 \
      aiohttp==3.12.15 aiohttp-retry==2.9.1

# Where the handler lives
WORKDIR /app
COPY handler.py /app/handler.py

# --- Runtime env ---
# Use volume-backed caches so the image stays small; first start fills them.
ENV HF_HOME=/runpod-volume/hf_cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf_cache \
    TRANSFORMERS_CACHE=/runpod-volume/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_OFFLINE=0

# (Optional) set your model IDs/filenames via env here or in the Runpod template:
# ENV CHROMA_BASE_ID=lodestones/Chroma
# ENV AIO_REPO=Phr00t/Chroma-Rapid-AIO
# ENV AIO_FILENAME_PRI=Chroma-Rapid-AIO-v2.safetensors
# ENV AIO_FILENAME_ALT=Chroma-Rapid-AIO.safetensors

# Runpod worker wiring
ENV RP_HANDLER="handler.run" \
    RP_MAX_CONCURRENCY=1 \
    RUNPOD_WEB_CONCURRENCY=1 \
    PORT=3000

EXPOSE 3000

# Healthcheck for the Testing phase
HEALTHCHECK --start-period=30s --interval=30s --timeout=5s --retries=10 \
  CMD curl -fsS http://localhost:3000/health || exit 1

# Start the serverless worker
CMD ["python", "-u", "-m", "runpod"]
