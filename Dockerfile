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

# Core libs + HF hub + S3 client + runpod + hf_transfer (✨ NEW)
RUN python -m pip install --upgrade pip && pip install \
    "diffusers==0.35.1" \
    "transformers>=4.44.2,<4.46" \
    "accelerate>=1.0.0" \
    "huggingface_hub>=0.24.6" \
    "safetensors>=0.4.5" \
    "sentencepiece>=0.2.0" \
    "pillow>=10.4.0" \
    "boto3>=1.34.0" \
    "runpod>=1.6.0" \
    "hf_transfer>=0.1.5"

# Create deterministic, baked-in locations
RUN mkdir -p /opt/chroma /opt/flux /opt/aio /app /weights/hf /weights/huggingface
WORKDIR /app

# --- Snapshot Chroma base (public) ---
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="lodestones/Chroma",
    local_dir="/opt/chroma",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "model_index.json",
        "*.json",
        "ae.safetensors",   # VAE file name, if present
        "vae/*",            # VAE folder, if present
        "text_encoder/*",
        "tokenizer/*",
        "*.safetensors",
    ],
)
print("Chroma snapshot -> /opt/chroma")
PY

# --- Snapshot FLUX.1-schnell (public) ---
RUN python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="black-forest-labs/FLUX.1-schnell",
    local_dir="/opt/flux",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "model_index.json","*.json","*.safetensors",
        "text_encoder/*","tokenizer/*","vae/*","ae.safetensors"
    ],
)
print("Flux schnell snapshot -> /opt/flux")
PY

# --- Merge any missing pieces into /opt/chroma ---
RUN python - <<'PY'
import os, shutil
pairs = [
    ("/opt/flux/text_encoder", "/opt/chroma/text_encoder"),
    ("/opt/flux/tokenizer",    "/opt/chroma/tokenizer"),
    ("/opt/flux/vae",          "/opt/chroma/vae"),
]
for src, dst in pairs:
    if os.path.isdir(src) and not os.path.exists(dst):
        shutil.copytree(src, dst)
ae_src = "/opt/flux/ae.safetensors"
ae_dst = "/opt/chroma/ae.safetensors"
if os.path.exists(ae_src) and not os.path.exists(ae_dst):
    shutil.copy2(ae_src, ae_dst)
print("Ensured /opt/chroma has VAE + text encoder/tokenizer if needed")
PY

# --- Snapshot AIO transformer weights (public) ---
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

# Optional: trim hub caches so only baked paths remain
RUN rm -rf /root/.cache/huggingface

# App code
COPY handler.py /app/handler.py

# Fully offline at runtime; handler loads from /opt/chroma and /opt/aio/chroma_aio.safetensors
CMD ["python", "-u", "handler.py"]
