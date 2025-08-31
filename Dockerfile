# CUDA 12.6 + cuDNN 9 + PyTorch 2.6 (Runpod-friendly)
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# ---------- Base env & caches ----------
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/weights/huggingface \
    HUGGINGFACE_HUB_CACHE=/weights/huggingface \
    TRANSFORMERS_CACHE=/weights/huggingface \
    DIFFUSERS_CACHE=/weights/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# ---------- System deps ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# ---------- Python deps ----------
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        "huggingface_hub>=0.23" \
        "diffusers>=0.31.0" \
        "transformers>=4.42.0" \
        "accelerate>=0.31.0" \
        "safetensors>=0.4.3" \
        "Pillow>=10.3.0" \
        runpod \
        boto3 \
        botocore

# ---------- Folders ----------
RUN mkdir -p /weights/huggingface /opt/chroma /opt/aio /app
WORKDIR /app

# ---------- Download/bake model assets (network ON only for these RUN lines) ----------
# Chroma base snapshot (lodestones/Chroma) → /opt/chroma
RUN HF_HUB_OFFLINE=0 python - <<'PY'
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
        "*.safetensors"
    ],
)
print("Chroma snapshot -> /opt/chroma")
PY

# AIO checkpoint (Phr00t/Chroma-Rapid-AIO) → /opt/aio/chroma_aio.safetensors
# Try a few known filenames; keep the largest successfully fetched.
RUN HF_HUB_OFFLINE=0 python - <<'PY'
import os, shutil, tempfile
from huggingface_hub import hf_hub_download

repo = "Phr00t/Chroma-Rapid-AIO"
candidates = [
    "Chroma-Rapid-AIO-v2.safetensors",
    "Chroma-Rapid-AIO.safetensors",
    "chroma-rapid-aio.safetensors",
]
tmp = tempfile.mkdtemp()
downloaded = []
for fn in candidates:
    try:
        p = hf_hub_download(repo_id=repo, filename=fn, local_dir=tmp, local_dir_use_symlinks=False)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            downloaded.append(p)
    except Exception:
        pass

if not downloaded:
    raise SystemExit("Failed to download any AIO checkpoint from Phr00t/Chroma-Rapid-AIO")

dst_dir = "/opt/aio"
os.makedirs(dst_dir, exist_ok=True)
best = max(downloaded, key=lambda p: os.path.getsize(p))
shutil.copy2(best, os.path.join(dst_dir, "chroma_aio.safetensors"))
shutil.rmtree(tmp, ignore_errors=True)
print("AIO checkpoint -> /opt/aio/chroma_aio.safetensors")
PY

# ---------- Sanity checks (fail build early if assets missing) ----------
RUN test -f /opt/chroma/model_index.json && \
    test -f /opt/aio/chroma_aio.safetensors

# ---------- Runtime env (offline + baked paths) ----------
ENV CHROMA_LOCAL_DIR=/opt/chroma \
    AIO_LOCAL_PATH=/opt/aio/chroma_aio.safetensors \
    HF_HUB_OFFLINE=1

# (Optional) Backblaze S3 defaults — keep secrets in the template, not in the image
ENV S3_ENDPOINT_URL=https://s3.us-west-000.backblazeb2.com \
    S3_REGION=us-west-000

# ---------- App ----------
COPY handler.py /app/handler.py

# ---------- Entrypoint ----------
CMD ["python", "-u", "handler.py"]
