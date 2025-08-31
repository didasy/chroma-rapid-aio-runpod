import os, uuid, time, threading, logging, io, urllib.request, shutil
from typing import Any, Dict, List, Optional

import runpod
import torch
from PIL import Image
from huggingface_hub import snapshot_download, hf_hub_download
from botocore.config import Config as BotoConfig
import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("chroma-aio")

# ---- Model/env ----
HF_BASE_ID = os.getenv("CHROMA_BASE_ID", "lodestones/Chroma")
CHROMA_LOCAL_DIR = os.getenv("CHROMA_LOCAL_DIR", "/runpod-volume/chroma")

AIO_REPO = os.getenv("AIO_REPO", "Phr00t/Chroma-Rapid-AIO")
AIO_FILENAMES = [
    os.getenv("AIO_FILENAME_PRI", "Chroma-Rapid-AIO-v2.safetensors"),
    os.getenv("AIO_FILENAME_ALT", "Chroma-Rapid-AIO.safetensors"),
    "chroma-rapid-aio.safetensors",
]
AIO_LOCAL_PATH = os.getenv("AIO_LOCAL_PATH", "/runpod-volume/chroma_aio.safetensors")
AIO_URL = os.getenv("AIO_URL")  # optional direct URL (B2, S3, etc.)

HF_TOKEN = os.getenv("HF_TOKEN")  # optional auth/rate-limit bypass
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- S3 ----
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_REGION = os.getenv("S3_REGION", "us-west-000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "chroma/outputs/")
S3_PRESIGN_EXPIRES = int(os.getenv("S3_PRESIGN_EXPIRES", "3600"))

# ---- Lazy pipeline ----
_PIPE = None
_PIPE_LOCK = threading.Lock()

def _import_diffusers():
    from diffusers import DiffusionPipeline
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
    except Exception:
        FlowMatchEulerDiscreteScheduler = None
    try:
        from diffusers import ChromaPipeline as _ChromaPipeline
    except Exception:
        _ChromaPipeline = None
    try:
        from diffusers import ChromaTransformer2DModel as _ChromaTransformer2DModel
    except Exception:
        _ChromaTransformer2DModel = None
    try:
        from diffusers import Transformer2DModel as _Transformer2DModel
    except Exception:
        _Transformer2DModel = None
    return (DiffusionPipeline, FlowMatchEulerDiscreteScheduler,
            _ChromaPipeline, _ChromaTransformer2DModel, _Transformer2DModel)

def _ensure_chroma_base_local() -> str:
    if os.path.exists(os.path.join(CHROMA_LOCAL_DIR, "model_index.json")):
        return CHROMA_LOCAL_DIR
    os.makedirs(CHROMA_LOCAL_DIR, exist_ok=True)
    log.info(f"Downloading base model: {HF_BASE_ID} -> {CHROMA_LOCAL_DIR}")
    snapshot_download(
        repo_id=HF_BASE_ID,
        local_dir=CHROMA_LOCAL_DIR,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "model_index.json", "*.json", "ae.safetensors", "vae/*",
            "text_encoder/*", "tokenizer/*", "*.safetensors",
        ],
        token=HF_TOKEN or None,
    )
    return CHROMA_LOCAL_DIR

def _download_url(url: str, dst_path: str, chunk: int = 8 * 1024 * 1024):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp = f"{dst_path}.part"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)
    os.replace(tmp, dst_path)

def _ensure_aio_checkpoint() -> str:
    """Prefer AIO_URL if provided; else fetch from HF by known filenames."""
    if os.path.exists(AIO_LOCAL_PATH) and os.path.getsize(AIO_LOCAL_PATH) > 0:
        return AIO_LOCAL_PATH

    os.makedirs(os.path.dirname(AIO_LOCAL_PATH), exist_ok=True)

    if AIO_URL:
        log.info(f"Downloading AIO from URL -> {AIO_LOCAL_PATH}")
        _download_url(AIO_URL, AIO_LOCAL_PATH)
        if os.path.getsize(AIO_LOCAL_PATH) == 0:
            raise RuntimeError("Downloaded AIO file is empty.")
        return AIO_LOCAL_PATH

    last_err = None
    for fn in AIO_FILENAMES:
        try:
            log.info(f"Fetching AIO from HF: {AIO_REPO}:{fn}")
            p = hf_hub_download(
                repo_id=AIO_REPO,
                filename=fn,
                local_dir=os.path.dirname(AIO_LOCAL_PATH),
                local_dir_use_symlinks=False,
                token=HF_TOKEN or None,
            )
            if os.path.exists(p) and os.path.getsize(p) > 0:
                if p != AIO_LOCAL_PATH:
                    os.replace(p, AIO_LOCAL_PATH)
                return AIO_LOCAL_PATH
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to download AIO from HF ({AIO_FILENAMES}): {last_err}")

def _build_pipeline():
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    DiffusionPipeline, FMEDS, CPipe, CTrans, T2D = _import_diffusers()

    ckpt = _ensure_aio_checkpoint()
    local_model_root = _ensure_chroma_base_local()

    if CTrans is not None:
        transformer = CTrans.from_single_file(ckpt, torch_dtype=DTYPE)
    elif T2D is not None:
        transformer = T2D.from_single_file(ckpt, torch_dtype=DTYPE)
    else:
        raise RuntimeError("No compatible transformer class; update diffusers or use a wheel that includes Chroma classes.")

    if CPipe is not None:
        pipe = CPipe.from_pretrained(
            local_model_root,
            transformer=transformer,
            torch_dtype=DTYPE,
            local_files_only=True,
            trust_remote_code=True,
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(
            local_model_root,
            torch_dtype=DTYPE,
            local_files_only=True,
            trust_remote_code=True,
        )
        if hasattr(pipe, "transformer"):
            pipe.transformer = transformer

    if FMEDS is not None and hasattr(pipe, "scheduler"):
        pipe.scheduler = FMEDS.from_config(pipe.scheduler.config)

    pipe.to(DEVICE, dtype=DTYPE)
    _PIPE = pipe
    log.info("Chroma pipeline ready.")
    return _PIPE

# ---- S3 helpers ----
def _s3_client():
    if not all([S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET]):
        raise RuntimeError("Missing S3 envs: S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET")
    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        config=BotoConfig(s3={"addressing_style": "virtual"}, signature_version="s3v4"),
    )

def _upload_and_presign(img: Image.Image, fmt: str = "PNG") -> str:
    key = f"{S3_PREFIX.rstrip('/')}/{int(time.time())}-{uuid.uuid4().hex}.{fmt.lower()}".lstrip("/")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    client = _s3_client()
    client.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue(),
                      ContentType=f"image/{fmt.lower()}")
    return client.generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=S3_PRESIGN_EXPIRES
    )

# ---- Inference ----
def generate_images(prompt: str, negative_prompt: Optional[str] = None,
                    steps: int = 6, num_images: int = 1,
                    width: Optional[int] = None, height: Optional[int] = None,
                    guidance_scale: Optional[float] = None) -> List[Image.Image]:
    with _PIPE_LOCK:
        pipe = _build_pipeline()

    gen = {"prompt": prompt, "num_inference_steps": int(steps)}
    if negative_prompt: gen["negative_prompt"] = negative_prompt
    if guidance_scale is not None: gen["guidance_scale"] = float(guidance_scale)
    if width: gen["width"] = int(width)
    if height: gen["height"] = int(height)

    out = pipe(**gen)
    images = out.images if hasattr(out, "images") else out
    if not isinstance(images, list): images = [images]
    if num_images and len(images) > num_images: images = images[:num_images]
    return images

# ---- Runpod handler ----
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    data = event.get("input") or {}
    try:
        if data.get("warmup"):
            # Pre-download assets; returns when ready
            _ensure_aio_checkpoint()
            _ensure_chroma_base_local()
            return {"status": "ok", "warmed": True}

        prompt = data.get("prompt")
        if not prompt or not isinstance(prompt, str):
            return {"error": "Missing 'prompt' (string) in input."}

        imgs = generate_images(
            prompt=prompt,
            negative_prompt=data.get("negative_prompt"),
            steps=int(data.get("steps", 6)),
            num_images=int(data.get("num_images", 1)),
            width=data.get("width"),
            height=data.get("height"),
            guidance_scale=data.get("guidance_scale"),
        )

        fmt = str(data.get("output_format", "PNG")).upper()
        urls = [_upload_and_presign(im, fmt=fmt) for im in imgs]
        return {"status": "ok", "count": len(urls), "results": [{"url": u} for u in urls]}
    except Exception as e:
        log.exception("Handler error")
        return {"status": "error", "error": str(e)}

runpod.serverless.start({"handler": handler})
