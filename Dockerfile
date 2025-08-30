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

# Core libs + HF hub + S3 client + runpod
RUN python -m pip install --upgrade pip && pip install \
    "diffusers==0.35.1" \
    "transformers>=4.44.2,<4.46" \
    "accelerate>=1.0.0" \
    "huggingface_hub>=0.24.6" \
    "safetensors>=0.4.5" \
    "sentencepiece>=0.2.0" \
    "pillow>=10.4.0" \
    "boto3>=1.34.0" \
    "runpod>=1.6.0"

# Create deterministic, baked-in locations
RUN mkdir -p /opt/chroma /opt/aio /app /weights/hf /weights/huggingface
WORKDIR /app

# ---------- Bake Chroma base snapshot (requires HF token at build time) ----------
# Pass the token with:  docker build --build-arg HF_TOKEN=hf_xxx -t yourimage .
ARG HF_TOKEN
# Snapshot only the needed files into /opt/chroma; no symlinks; no runtime network needed
RUN HF_TOKEN=$HF_TOKEN python - <<'PY'
import os, sys, shutil
from huggingface_hub import snapshot_download

repo_id = "lodestones/Chroma"
local_dir = "/opt/chroma"
allow = [
    "model_index.json",
    "*.json",
    "ae.safetensors",      # VAE (naming may evolve)
    "vae/*",
    "text_encoder/*",
    "tokenizer/*",
    "*.safetensors",
]
snapshot_download(
    repo_id=repo_id,
    token=os.getenv("HF_TOKEN"),
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    allow_patterns=allow,
)
print("Chroma base snapshot ready at", local_dir)
PY

# ---------- Bake AIO transformer weights ----------
# Pull Phr00t/Chroma-Rapid-AIO and place one .safetensors file at /opt/aio/chroma_aio.safetensors
RUN python - <<'PY'
import os, shutil, glob
from huggingface_hub import snapshot_download

repo_id = "Phr00t/Chroma-Rapid-AIO"
tmp_dir = "/opt/aio_src"
dst_dir = "/opt/aio"
dst_file = os.path.join(dst_dir, "chroma_aio.safetensors")

os.makedirs(dst_dir, exist_ok=True)
snapshot_download(
    repo_id=repo_id,
    local_dir=tmp_dir,
    local_dir_use_symlinks=False,
    allow_patterns=["Chroma-Rapid-AIO-v2.safetensors","Chroma-Rapid-AIO.safetensors"],
)
candidates = glob.glob(os.path.join(tmp_dir, "*.safetensors"))
if not candidates:
    raise SystemExit("No AIO .safetensors found in snapshot.")
# pick the largest (v2 usually)
best = max(candidates, key=lambda p: os.path.getsize(p))
shutil.copy2(best, dst_file)
shutil.rmtree(tmp_dir, ignore_errors=True)
print("AIO checkpoint ready at", dst_file)
PY

# Optional: remove HF caches so only baked paths remain
RUN rm -rf /root/.cache/huggingface

# App code
COPY handler.py /app/handler.py

# Runtime is fully offline; handler reads /opt/chroma and /opt/aio/chroma_aio.safetensors
CMD ["python", "-u", "handler.py"]
