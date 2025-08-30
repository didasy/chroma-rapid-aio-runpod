# CUDA 12.6 + cuDNN 9 + PyTorch 2.6 (Runpod-friendly)
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/weights/huggingface \
    HUGGINGFACE_HUB_CACHE=/weights/huggingface \
    TRANSFORMERS_CACHE=/weights/huggingface \
    DIFFUSERS_CACHE=/weights/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Diffusers (Chroma) + S3 client
RUN pip install --upgrade pip && pip install \
    "diffusers==0.35.1" \
    "transformers>=4.44.2,<4.46" \
    "accelerate>=1.0.0" \
    "huggingface_hub>=0.24.6" \
    "safetensors>=0.4.5" \
    "sentencepiece>=0.2.0" \
    "pillow>=10.4.0" \
    "boto3>=1.34.0" \
    "runpod>=1.6.0"

RUN mkdir -p /weights/huggingface

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python", "-u", "handler.py"]
