# CUDA 12.6 + cuDNN 9 + PyTorch 2.6
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/weights/huggingface \
    HUGGINGFACE_HUB_CACHE=/weights/huggingface \
    TRANSFORMERS_CACHE=/weights/huggingface \
    DIFFUSERS_CACHE=/weights/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True \
    CHROMA_LOCAL_DIR=/opt/chroma \
    AIO_LOCAL_PATH=/opt/aio/chroma_aio.safetensors \
    HF_HUB_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Core libs + hf_transfer (fixes previous build error) + S3 client + runpod
RUN python -m pip install --upgrade pip && pip install \
    "diffusers==0.35.1" \
    "transformers>=4.44.2,<4.46" \
    "accelerate>=1.0.0" \
    "huggingface_hub>=0.34.4" \
    "safetensors>=0.4.5" \
    "sentencepiece>=0.2.0" \
    "pillow>=10.4.0" \
    "boto3>=1.34.0" \
    "runpod>=1.6.0" \
    "hf_transfer>=0.1.5"

# Deterministic, baked-in locations
RUN mkdir -p /opt/chroma /opt/aio /app /weights/hf /weights/huggingface
WORKDIR /app

# --- Snapshot Chroma base (PUBLIC) ---
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="lodestones/Chroma",
    local_dir="/opt/chroma",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "model_index.json",
        "*.json",
        "ae.safetensors",
        "vae/*",
        "text_encoder/*",
        "tokenizer/*",
        "*.safetensors",
    ],
)
print("Chroma snapshot -> /opt/chroma")
PY

# --- Snapshot AIO transformer weights (PUBLIC) ---
RUN python - <<'PY'
import os, glob, shutil
from huggingface_hub import snapshot_download
tmp="/opt/aio_src"; dst="/opt/aio"; os.makedirs(dst, exist_ok=True)
snapshot_download(
    repo_id="Phr00t/Chroma-Rapid-AIO",
    local_dir=tmp,
    local_dir_use_symlinks=False,
    allow_patterns=["Chroma-Rapid-AIO-v2.safetensors","Chroma-Rapid-AIO.safetensors"],
)
cands=glob.glob(os.path.join(tmp, "*.safetensors"))
if not cands:
    raise SystemExit("No AIO .safetensors downloaded")
best=max(cands, key=lambda p: os.path.getsize(p))
shutil.copy2(best, os.path.join(dst, "chroma_aio.safetensors"))
shutil.rmtree(tmp, ignore_errors=True)
print("AIO checkpoint -> /opt/aio/chroma_aio.safetensors")
PY

# Optional: trim hub caches
RUN rm -rf /root/.cache/huggingface

# App
COPY handler.py /app/handler.py

# Runtime is fully offline; handler loads from /opt/chroma and /opt/aio/chroma_aio.safetensors
CMD ["python", "-u", "handler.py"]
