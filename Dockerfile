FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ca-certificates curl && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
      runpod==1.7.13 \
      aiohttp==3.12.15 aiohttp-retry==2.9.1 \
      fastapi==0.116.1 uvicorn==0.35.0 \
      diffusers==0.35.1 transformers==4.56.0 accelerate==1.10.1 \
      huggingface_hub==0.34.4 hf_transfer==0.1.9 \
      safetensors==0.4.5 Pillow==10.4.0 \
      boto3==1.35.14 botocore==1.35.14 \
      sentencepiece==0.2.0 protobuf==5.27.4
WORKDIR /app
COPY handler.py /app/handler.py
ENV CHROMA_BASE_ID=lodestones/Chroma \
    CHROMA_LOCAL_DIR=/tmp/chroma \
    AIO_REPO=Phr00t/Chroma-Rapid-AIO \
    AIO_LOCAL_PATH=/tmp/chroma_aio.safetensors \
    HF_HOME=/tmp/hf_cache \
    HUGGINGFACE_HUB_CACHE=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_OFFLINE=0 \
    PORT=3000 \
    ENABLE_HTTP=1
EXPOSE 3000
CMD ["python","-u","handler.py"]
