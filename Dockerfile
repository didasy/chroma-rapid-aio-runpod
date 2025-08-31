# Small & fast: no baked weights; first run caches to /runpod-volume
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps (pin the versions that work with Chroma in diffusers)
RUN python -m pip install -U pip setuptools wheel && \
    pip install --no-cache-dir \
      runpod==1.7.13 \
      aiohttp==3.12.15 aiohttp-retry==2.9.1 \
      diffusers==0.35.1 \
      transformers==4.56.0 \
      accelerate==1.10.1 \
      huggingface_hub==0.34.4 \
      hf_transfer==0.1.9 \
      safetensors==0.4.5 \
      Pillow==10.4.0 \
      sentencepiece==0.2.0 \
      protobuf==5.27.4 \
      boto3==1.35.14 botocore==1.35.14

# App code
WORKDIR /app
COPY handler.py /app/handler.py

# Cache & runtime env — first run downloads to the volume, then reuses
ENV HF_HOME=/runpod-volume/hf_cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/hf_cache \
    TRANSFORMERS_CACHE=/runpod-volume/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_OFFLINE=0 \
    CHROMA_LOCAL_DIR=/runpod-volume/chroma \
    AIO_LOCAL_PATH=/runpod-volume/chroma_aio.safetensors

# Runpod worker entrypoint (don’t change)
ENV RP_HANDLER="handler:handler" \
    RP_MAX_CONCURRENCY=1 \
    RUNPOD_WEB_CONCURRENCY=1

CMD ["python", "-u", "-m", "runpod"]
