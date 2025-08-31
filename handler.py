import os, io, time, uuid, stat, tarfile, shutil, logging, threading, urllib.request
from typing import Any, Dict, List, Optional

import runpod
import torch
from PIL import Image
from huggingface_hub import snapshot_download, hf_hub_download
try:
    from huggingface_hub.constants import HF_HUB_CACHE as HUB_CACHE_PATH_CONST  # best-effort
except Exception:
    HUB_CACHE_PATH_CONST = None

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import JSONResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s")
log = logging.getLogger("chroma-hybrid")

# ------------------ CACHE ROOT (use the enlarged container disk) ------------------
CACHE_ROOT = os.getenv("CACHE_ROOT", "/cache")
os.makedirs(CACHE_ROOT, exist_ok=True)

# ------------------ Env / Config (derive defaults from CACHE_ROOT) ----------------
CHROMA_BASE_ID   = os.getenv("CHROMA_BASE_ID", "lodestones/Chroma")
CHROMA_LOCAL_DIR = os.getenv("CHROMA_LOCAL_DIR", os.path.join(CACHE_ROOT, "chroma"))

AIO_REPO         = os.getenv("AIO_REPO", "Phr00t/Chroma-Rapid-AIO")
AIO_FILENAMES    = [
    os.getenv("AIO_FILENAME_PRI", "Chroma-Rapid-AIO-v2.safetensors"),
    os.getenv("AIO_FILENAME_ALT", "Chroma-Rapid-AIO.safetensors"),
    "chroma-rapid-aio.safetensors",
]
AIO_LOCAL_PATH   = os.getenv("AIO_LOCAL_PATH", os.path.join(CACHE_ROOT, "chroma_aio.safetensors"))
AIO_URL          = os.getenv("AIO_URL")
SNAPSHOT_URL     = os.getenv("SNAPSHOT_URL")
HF_TOKEN         = os.getenv("HF_TOKEN")

HF_HOME = os.getenv("HF_HOME", os.path.join(CACHE_ROOT, "hf"))
os.environ.setdefault("HF_HOME", HF_HOME)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.getenv("HUGGINGFACE_HUB_CACHE", os.path.join(CACHE_ROOT, "hf")))
os.environ.setdefault("HF_HUB_CACHE",          os.getenv("HF_HUB_CACHE",          os.path.join(CACHE_ROOT, "hf")))
os.environ.setdefault("TRANSFORMERS_CACHE",    os.getenv("TRANSFORMERS_CACHE",    os.path.join(CACHE_ROOT, "hf", "transformers")))

S3_ENDPOINT_URL        = os.getenv("S3_ENDPOINT_URL")
S3_REGION              = os.getenv("S3_REGION", "us-west-000")
S3_ACCESS_KEY          = os.getenv("S3_ACCESS_KEY")
S3_SECRET_KEY          = os.getenv("S3_SECRET_KEY")
S3_BUCKET              = os.getenv("S3_BUCKET")
S3_PREFIX              = os.getenv("S3_PREFIX", "chroma/outputs")
S3_PRESIGN_EXPIRES     = int(os.getenv("S3_PRESIGN_EXPIRES", "3600"))
S3_USE_SSL             = os.getenv("S3_USE_SSL", "1")
S3_VERIFY_SSL          = os.getenv("S3_VERIFY_SSL", "1")
S3_CONTENT_DISPOSITION = os.getenv("S3_CONTENT_DISPOSITION", "inline")

MIN_FREE_GB      = float(os.getenv("MIN_FREE_GB", "50"))
DTYPE            = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"

ENABLE_HTTP      = os.getenv("ENABLE_HTTP", "1")
PORT             = int(os.getenv("PORT", "3000"))

# Ensure important dirs exist
os.makedirs(CHROMA_LOCAL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(AIO_LOCAL_PATH), exist_ok=True)

# ------------------ Globals ------------------
_PIPE = None
_PIPE_LOCK = threading.Lock()
HISTORY: Dict[str, Dict[str, Any]] = {}

# ------------------ Utils ------------------
def _free_gb(path: str) -> float:
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)

def _ensure_free_space(path: str, need_gb: float):
    free = _free_gb(path)
    if free < need_gb:
        raise RuntimeError(f"not enough free space at {path}: need {need_gb:.1f} GB, have {free:.1f} GB")

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

def _path_report() -> Dict[str, Any]:
    hub_env = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HUB_CACHE")
    hub_resolved = str(HUB_CACHE_PATH_CONST) if HUB_CACHE_PATH_CONST is not None else hub_env
    return {
        "cache_root": CACHE_ROOT,
        "hf_home": os.environ.get("HF_HOME"),
        "hub_cache_env": hub_env,
        "hub_cache_resolved": hub_resolved,
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
        "chroma_local_dir": CHROMA_LOCAL_DIR,
        "aio_local_path": AIO_LOCAL_PATH,
        "free_gb_cache_root": round(_free_gb(CACHE_ROOT), 2),
        "free_gb_rootfs": round(_free_gb("/"), 2),
    }

# ------------------ S3 ------------------
def _have_s3() -> bool:
    return all([S3_ENDPOINT_URL, S3_ACCESS_KEY, S3_SECRET_KEY, S3_BUCKET])

def _parse_verify(v: str):
    v = str(v).strip().lower()
    if v in ("0","false","no","off"): return False
    if v in ("1","true","yes","on"):  return True
    return v

def _s3_client():
    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        use_ssl=(str(S3_USE_SSL).strip().lower() not in ("0","false","no","off")),
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

# ------------------ Assets (download to CACHE_ROOT) ------------------
def _ensure_aio(metrics: Dict[str, Any]) -> str:
    if os.path.exists(AIO_LOCAL_PATH) and os.path.getsize(AIO_LOCAL_PATH) > 0:
        metrics["aio"] = {"source": "local", "size_bytes": os.path.getsize(AIO_LOCAL_PATH), "seconds": 0}
        return AIO_LOCAL_PATH
    _ensure_free_space(CACHE_ROOT, max(20.0, MIN_FREE_GB))
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
    _ensure_free_space(CACHE_ROOT, max(10.0, MIN_FREE_GB))
    os.makedirs(CHROMA_LOCAL_DIR, exist_ok=True)
    if SNAPSHOT_URL:
        tmp = os.path.join(CACHE_ROOT, "chroma-snapshot.tar.gz")
        t0 = time.time()
        size = _download_url(SNAPSHOT_URL, tmp)
        with tarfile.open(tmp, "r:gz") as tf:
            tf.extractall(CHROMA_LOCAL_DIR)
        os.remove(tmp)
        metrics["snapshot"] = {"source": "url", "size_bytes": size, "seconds": time.time() - t0}
        return CHROMA_LOCAL_DIR
    t0 = time.time()
    snapshot_download(
        repo_id=CHROMA_BASE_ID,
        local_dir=CHROMA_LOCAL_DIR,
        local_dir_use_symlinks=True,
        allow_patterns=[
            "model_index.json","*.json","ae.safetensors",
            "vae/*","text_encoder/*","tokenizer/*","*.safetensors"
        ],
        token=HF_TOKEN or None
    )
    metrics["snapshot"] = {"source": "huggingface", "size_bytes": _dir_size_bytes(CHROMA_LOCAL_DIR), "seconds": time.time() - t0}
    return CHROMA_LOCAL_DIR

# ------------------ Diffusers ------------------
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
    ck_m, sp_m = {}, {}
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
    log.info("Chroma pipeline ready.")
    return _PIPE

def generate_images(
    prompt: str,
    negative_prompt: Optional[str] = None,
    steps: int = 6,
    num_images: int = 1,
    width: Optional[int] = None,
    height: Optional[int] = None,
    guidance_scale: Optional[float] = None,
) -> List[Image.Image]:
    with _PIPE_LOCK:
        _build_pipeline({})
        pipe = _PIPE

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

# ------------------ ComfyUI graph parsing (heuristic) ------------------
def parse_comfy_prompt_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    graph = payload.get("prompt") or {}
    prompt_text = None
    negative_text = None
    steps = None
    width = None
    height = None
    guidance = None
    for _, node in graph.items():
        ctype = str(node.get("class_type", "")).lower()
        inputs = node.get("inputs", {}) or {}
        if "textencode" in ctype or "cliptextencode" in ctype:
            t = inputs.get("text")
            if isinstance(t, str):
                if prompt_text is None:
                    prompt_text = t
                elif negative_text is None:
                    negative_text = t
        if "ksampler" in ctype:
            if steps is None and "steps" in inputs:
                try: steps = int(inputs["steps"])
                except Exception: pass
            if "cfg" in inputs and guidance is None:
                try: guidance = float(inputs["cfg"])
                except Exception: pass
        if "emptylatentimage" in ctype:
            if width is None and "width" in inputs:
                try: width = int(inputs["width"])
                except Exception: pass
            if height is None and "height" in inputs:
                try: height = int(inputs["height"])
                except Exception: pass
    if prompt_text is None:
        prompt_text = payload.get("positive") or payload.get("prompt") or ""
    if negative_text is None:
        negative_text = payload.get("negative") or None
    if steps is None:
        steps = int(payload.get("steps", 6))
    return {
        "prompt": prompt_text,
        "negative_prompt": negative_text,
        "steps": steps,
        "width": width,
        "height": height,
        "guidance_scale": guidance,
        "num_images": int(payload.get("num_images", 1)),
        "output_format": str(payload.get("output_format", "PNG")).upper()
    }

# ------------------ Core processing (shared by both modes) ------------------
def process_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    if input_data.get("debug"):
        return {"status": "ok", "paths": _path_report()}

    if input_data.get("warmup"):
        with _PIPE_LOCK:
            _build_pipeline({})
        return {"status": "ok", "warmed": True}

    if input_data.get("comfy"):
        params = parse_comfy_prompt_payload(input_data.get("payload") or {})
        if not params["prompt"]:
            return {"status": "error", "error": "No prompt text in Comfy payload"}
        imgs = generate_images(
            prompt=params["prompt"],
            negative_prompt=params["negative_prompt"],
            steps=params["steps"],
            num_images=params["num_images"],
            width=params["width"],
            height=params["height"],
            guidance_scale=params["guidance_scale"],
        )
        if not _have_s3():
            return {"status": "error", "error": "missing S3 envs"}
        fmt = params["output_format"]
        urls = [_upload_and_presign(im, fmt=fmt) for im in imgs]
        pid = uuid.uuid4().hex
        HISTORY[pid] = {"status": "completed", "outputs": {"result": {"images": [{"url": u} for u in urls]}}}
        return {"status": "ok", "prompt_id": pid}

    prompt = input_data.get("prompt")
    if not prompt or not isinstance(prompt, str):
        with _PIPE_LOCK:
            _build_pipeline({})
        return {"status": "ok", "warmed": True}

    imgs = generate_images(
        prompt=prompt,
        negative_prompt=input_data.get("negative_prompt"),
        steps=int(input_data.get("steps", 6)),
        num_images=int(input_data.get("num_images", 1)),
        width=input_data.get("width"),
        height=input_data.get("height"),
        guidance_scale=input_data.get("guidance_scale"),
    )
    if not _have_s3():
        return {"status": "error", "error": "missing S3 envs"}
    fmt = str(input_data.get("output_format", "PNG")).upper()
    urls = [_upload_and_presign(im, fmt=fmt) for im in imgs]
    return {"status": "ok", "count": len(urls), "results": [{"url": u} for u in urls]}

# ------------------ FastAPI app (Load Balancer + ComfyUI) ------------------
app = FastAPI(title="Chroma Hybrid API", version="1.1.0")

@app.get("/health")
def health():
    return {"status": "ok", "paths": _path_report()}

@app.get("/queue")
def queue_status():
    return {"queue_remaining": 0}

@app.post("/run")
async def lb_run(req: Request):
    body = await req.json()
    input_data = body.get("input") if isinstance(body, dict) and "input" in body else body
    out = process_input(input_data or {})
    return JSONResponse(out)

@app.post("/prompt")
def comfy_prompt(payload: Dict[str, Any] = Body(...)):
    params = parse_comfy_prompt_payload(payload)
    if not params["prompt"]:
        raise HTTPException(status_code=400, detail="No prompt text found in graph or payload.")
    imgs = generate_images(
        prompt=params["prompt"],
        negative_prompt=params["negative_prompt"],
        steps=params["steps"],
        num_images=params["num_images"],
        width=params["width"],
        height=params["height"],
        guidance_scale=params["guidance_scale"],
    )
    if not _have_s3():
        raise HTTPException(status_code=500, detail="Missing S3 envs for output upload.")
    fmt = params["output_format"]
    urls = [_upload_and_presign(im, fmt=fmt) for im in imgs]
    pid = uuid.uuid4().hex
    HISTORY[pid] = {"status": "completed", "outputs": {"result": {"images": [{"url": u} for u in urls]}}}
    return {"prompt_id": pid}

@app.get("/history/{prompt_id}")
def comfy_history(prompt_id: str):
    item = HISTORY.get(prompt_id)
    if not item:
        raise HTTPException(status_code=404, detail="prompt_id not found")
    return {"history": {prompt_id: item}}

# ------------------ Runpod Queue mode handler ------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    data = event.get("input") or {}
    try:
        return process_input(data)
    except Exception as e:
        log.exception("handler error")
        return {"status": "error", "error": str(e)}

# ------------------ Start HTTP server in background (so both modes work) ------------------
def _start_http_server():
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

if ENABLE_HTTP and str(ENABLE_HTTP).strip().lower() not in ("0","false","no","off"):
    t = threading.Thread(target=_start_http_server, daemon=True)
    t.start()

# Queue worker (blocks main thread)
runpod.serverless.start({"handler": handler})
