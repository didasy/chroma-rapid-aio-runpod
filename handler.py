import os, io, time, uuid, stat, tarfile, shutil, logging, threading, urllib.request
from typing import Any, Dict, List, Optional

import runpod
import torch
from PIL import Image
from huggingface_hub import snapshot_download, hf_hub_download
import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s")
log = logging.getLogger("chroma-aio")

CHROMA_BASE_ID = os.getenv("CHROMA_BASE_ID", "lodestones/Chroma")
CHROMA_LOCAL_DIR = os.getenv("CHROMA_LOCAL_DIR", "/tmp/chroma")
AIO_REPO = os.getenv("AIO_REPO", "Phr00t/Chroma-Rapid-AIO")
AIO_FILENAMES = [
    os.getenv("AIO_FILENAME_PRI", "Chroma-Rapid-AIO-v2.safetensors"),
    os.getenv("AIO_FILENAME_ALT", "Chroma-Rapid-AIO.safetensors"),
    "chroma-rapid-aio.safetensors",
]
AIO_LOCAL_PATH = os.getenv("AIO_LOCAL_PATH", "/tmp/chroma_aio.safetensors")
AIO_URL = os.getenv("AIO_URL")
SNAPSHOT_URL = os.getenv("SNAPSHOT_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

HF_HOME = os.getenv("HF_HOME", "/tmp/hf_cache")

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
S3_REGION = os.getenv("S3_REGION", "us-west-000")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_PREFIX = os.getenv("S3_PREFIX", "chroma/outputs")
S3_PRESIGN_EXPIRES = int(os.getenv("S3_PRESIGN_EXPIRES", "3600"))
S3_USE_SSL = os.getenv("S3_USE_SSL", "1")
S3_VERIFY_SSL = os.getenv("S3_VERIFY_SSL", "1")
S3_CONTENT_DISPOSITION = os.getenv("S3_CONTENT_DISPOSITION", "inline")

MIN_FREE_GB = float(os.getenv("MIN_FREE_GB", "50"))

DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_PIPE = None
_PIPE_LOCK = threading.Lock()

def _free_gb(path: str) -> float:
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)

def _ensure_free_space(path: str, need_gb: float):
    if _free_gb(path) < need_gb:
        raise RuntimeError("not enough free space")

def _safe_symlink(target: str, link: str):
    if os.path.islink(link) or os.path.exists(link):
        try:
            os.remove(link)
        except Exception:
            os.chmod(link, stat.S_IWUSR | stat.S_IRUSR)
            os.remove(link)
    os.symlink(target, link)

def _download_url(url: str, dst_path: str, chunk: int = 8 * 1024 * 1024) -> int:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp = f"{dst_path}.part"
    total = 0
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)
            total += len(b)
    os.replace(tmp, dst_path)
    return total

def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            fp = os.path.join(root, fn)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total

def _have_s3() -> bool:
    return all([S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET])

def _parse_verify(v: str):
    v = str(v).strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    if v in ("1", "true", "yes", "on"):
        return True
    return v

def _s3_client():
    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        use_ssl=(str(S3_USE_SSL).strip().lower() not in ("0", "false", "no", "off")),
        verify=_parse_verify(S3_VERIFY_SSL),
        config=BotoConfig(s3={"addressing_style": "virtual"}, signature_version="s3v4"),
    )

def _upload_and_presign(img: Image.Image, fmt: str = "PNG") -> str:
    key = f"{S3_PREFIX.rstrip('/')}/{int(time.time())}-{uuid.uuid4().hex}.{fmt.lower()}".lstrip("/")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    c = _s3_client()
    extra = {"ContentType": f"image/{fmt.lower()}"}
    cd = S3_CONTENT_DISPOSITION.strip()
    if cd:
        extra["ContentDisposition"] = cd
    c.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue(), **extra)
    return c.generate_presigned_url("get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=S3_PRESIGN_EXPIRES)

def _ensure_aio(metrics: Dict[str, Any]) -> str:
    if os.path.exists(AIO_LOCAL_PATH) and os.path.getsize(AIO_LOCAL_PATH) > 0:
        metrics["aio"] = {"source": "local", "size_bytes": os.path.getsize(AIO_LOCAL_PATH), "seconds": 0}
        return AIO_LOCAL_PATH
    _ensure_free_space("/tmp", max(20.0, MIN_FREE_GB))
    if AIO_URL:
        t0 = time.time()
        size = _download_url(AIO_URL, AIO_LOCAL_PATH)
        metrics["aio"] = {"source": "url", "size_bytes": size, "seconds": time.time() - t0}
        return AIO_LOCAL_PATH
    last_err = None
    for fn in AIO_FILENAMES:
        try:
            t0 = time.time()
            p = hf_hub_download(repo_id=AIO_REPO, filename=fn, local_dir=None, local_dir_use_symlinks=True, token=HF_TOKEN or None)
            size = os.path.getsize(p)
            _safe_symlink(p, AIO_LOCAL_PATH)
            metrics["aio"] = {"source": "huggingface", "size_bytes": size, "seconds": time.time() - t0}
            return AIO_LOCAL_PATH
        except Exception as e:
            last_err = e
    raise RuntimeError(f"failed to download AIO: {last_err}")

def _ensure_snapshot(metrics: Dict[str, Any]) -> str:
    mi = os.path.join(CHROMA_LOCAL_DIR, "model_index.json")
    if os.path.exists(mi):
        metrics["snapshot"] = {"source": "local", "size_bytes": _dir_size_bytes(CHROMA_LOCAL_DIR), "seconds": 0}
        return CHROMA_LOCAL_DIR
    _ensure_free_space("/tmp", max(10.0, MIN_FREE_GB))
    os.makedirs(CHROMA_LOCAL_DIR, exist_ok=True)
    if SNAPSHOT_URL:
        tmp = "/tmp/chroma-snapshot.tar.gz"
        t0 = time.time()
        size = _download_url(SNAPSHOT_URL, tmp)
        with tarfile.open(tmp, "r:gz") as tf:
            tf.extractall(CHROMA_LOCAL_DIR)
        os.remove(tmp)
        metrics["snapshot"] = {"source": "url", "size_bytes": size, "seconds": time.time() - t0}
        return CHROMA_LOCAL_DIR
    t0 = time.time()
    snapshot_download(repo_id=CHROMA_BASE_ID, local_dir=CHROMA_LOCAL_DIR, local_dir_use_symlinks=True, allow_patterns=["model_index.json","*.json","ae.safetensors","vae/*","text_encoder/*","tokenizer/*","*.safetensors"], token=HF_TOKEN or None)
    metrics["snapshot"] = {"source": "huggingface", "size_bytes": _dir_size_bytes(CHROMA_LOCAL_DIR), "seconds": time.time() - t0}
    return CHROMA_LOCAL_DIR

def _import_diffusers():
    from diffusers import DiffusionPipeline
    try:
        from diffusers import FlowMatchEulerDiscreteScheduler
    except Exception:
        FlowMatchEulerDiscreteScheduler = None
    try:
        from diffusers import ChromaPipeline as CPipe
    except Exception:
        CPipe = None
    try:
        from diffusers import ChromaTransformer2DModel as CTrans
    except Exception:
        CTrans = None
    try:
        from diffusers import Transformer2DModel as T2D
    except Exception:
        T2D = None
    return DiffusionPipeline, FlowMatchEulerDiscreteScheduler, CPipe, CTrans, T2D

def _build_pipeline(metrics: Dict[str, Any]):
    global _PIPE
    if _PIPE is not None:
        return _PIPE
    ck_m = {}
    sp_m = {}
    ckpt = _ensure_aio(ck_m)
    _ensure_snapshot(sp_m)
    metrics.update(ck_m)
    metrics.update(sp_m)
    DiffusionPipeline, FMEDS, CPipe, CTrans, T2D = _import_diffusers()
    if CTrans is not None:
        transformer = CTrans.from_single_file(ckpt, torch_dtype=DTYPE)
    elif T2D is not None:
        transformer = T2D.from_single_file(ckpt, torch_dtype=DTYPE)
    else:
        raise RuntimeError("no compatible transformer class")
    if CPipe is not None:
        pipe = CPipe.from_pretrained(CHROMA_LOCAL_DIR, transformer=transformer, torch_dtype=DTYPE, local_files_only=True, trust_remote_code=True)
    else:
        pipe = DiffusionPipeline.from_pretrained(CHROMA_LOCAL_DIR, torch_dtype=DTYPE, local_files_only=True, trust_remote_code=True)
        if hasattr(pipe, "transformer"):
            pipe.transformer = transformer
    if FMEDS is not None and hasattr(pipe, "scheduler"):
        pipe.scheduler = FMEDS.from_config(pipe.scheduler.config)
    pipe.to(DEVICE, dtype=DTYPE)
    _PIPE = pipe
    return _PIPE

def generate_images(prompt: str, negative_prompt: Optional[str] = None, steps: int = 6, num_images: int = 1, width: Optional[int] = None, height: Optional[int] = None, guidance_scale: Optional[float] = None) -> List[Image.Image]:
    metrics = {}
    with _PIPE_LOCK:
        pipe = _build_pipeline(metrics)
    gen = {"prompt": prompt, "num_inference_steps": int(steps)}
    if negative_prompt:
        gen["negative_prompt"] = negative_prompt
    if guidance_scale is not None:
        gen["guidance_scale"] = float(guidance_scale)
    if width:
        gen["width"] = int(width)
    if height:
        gen["height"] = int(height)
    out = pipe(**gen)
    images = out.images if hasattr(out, "images") else out
    if not isinstance(images, list):
        images = [images]
    if num_images and len(images) > num_images:
        images = images[:num_images]
    return images

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    data = event.get("input") or {}
    try:
        if data.get("warmup"):
            m = {}
            with _PIPE_LOCK:
                _build_pipeline(m)
            return {"status": "ok", "warmed": True, "metrics": m}
        prompt = data.get("prompt")
        if not prompt or not isinstance(prompt, str):
            with _PIPE_LOCK:
                _build_pipeline({})
            return {"status": "ok", "warmed": True}
        imgs = generate_images(prompt=prompt, negative_prompt=data.get("negative_prompt"), steps=int(data.get("steps", 6)), num_images=int(data.get("num_images", 1)), width=data.get("width"), height=data.get("height"), guidance_scale=data.get("guidance_scale"))
        if not all([S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET]):
            return {"status": "error", "error": "missing S3 envs"}
        fmt = str(data.get("output_format", "PNG")).upper()
        urls = [_upload_and_presign(im, fmt=fmt) for im in imgs]
        return {"status": "ok", "count": len(urls), "results": [{"url": u} for u in urls]}
    except Exception as e:
        log.exception("handler error")
        return {"status": "error", "error": str(e)}

runpod.serverless.start({"handler": handler})
