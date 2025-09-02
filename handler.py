import os, io, time, uuid, stat, tarfile, shutil, logging, threading, urllib.request, base64, inspect
from typing import Any, Dict, List, Optional, Tuple

import runpod
import torch
from PIL import Image
from huggingface_hub import snapshot_download, hf_hub_download
try:
    from huggingface_hub.constants import HF_HUB_CACHE as HUB_CACHE_PATH_CONST
except Exception:
    HUB_CACHE_PATH_CONST = None

import boto3
from botocore.config import Config as BotoConfig
from fastapi import FastAPI, HTTPException, Body, Request
from fastapi.responses import JSONResponse, StreamingResponse  # <-- added StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ------------------ Logging ------------------
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
log = logging.getLogger("chroma")

# ------------------ Runtime config ------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOW_VRAM = str(os.environ.get("LOW_VRAM", "0")).strip().lower() in ("1", "true", "yes", "on")
DTYPE  = torch.bfloat16 if str(os.getenv("DTYPE", "bf16")).lower() in ("bf16","bfloat16") else torch.float16

CACHE_ROOT = os.environ.get("CACHE_ROOT", "/cache")
os.makedirs(CACHE_ROOT, exist_ok=True)

PORT = int(os.getenv("PORT", "8000"))
ENABLE_HTTP = os.getenv("ENABLE_HTTP", "1")

# ------------------ Globals ------------------
_PIPE_LOCK = threading.Lock()
_PIPE = None

HISTORY: Dict[str, Any] = {}
PROGRESS: Dict[str, Any] = {}

def _free_gb(path="/"):
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize / (1024**3)
    except Exception:
        return 0.0

def _dir_size_bytes(root):
    total = 0
    for d, _, files in os.walk(root):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(d, f))
            except Exception:
                pass
    return total

# ------------------ Model paths ------------------
CHROMA_LOCAL_DIR = os.path.join(CACHE_ROOT, "chroma_repo")
CHROMA_REVISION  = os.getenv("CHROMA_REVISION", None)

AIO_FILENAMES  = [
    os.getenv("AIO_FILENAME_PRI", "Chroma-Rapid-AIO-v2.safetensors"),
    os.getenv("AIO_FILENAME_ALT", "Chroma-Rapid-AIO.safetensors"),
    "chroma-rapid-aio.safetensors",
]
AIO_LOCAL_PATH = os.getenv("AIO_LOCAL_PATH", os.path.join(CACHE_ROOT, "chroma_aio.safetensors"))
AIO_URL        = os.getenv("AIO_URL")
SNAPSHOT_URL   = os.getenv("SNAPSHOT_URL")
HF_TOKEN       = os.getenv("HF_TOKEN")

# Hugging Face caches (force to /cache and limit size)
HF_HOME_DEFAULT = os.path.join(CACHE_ROOT, "hf_cache")
os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", HF_HOME_DEFAULT))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.getenv("HUGGINGFACE_HUB_CACHE", HF_HOME_DEFAULT))
os.environ.setdefault("HF_HUB_CACHE", os.getenv("HF_HUB_CACHE", HF_HOME_DEFAULT))
os.environ.setdefault("TRANSFORMERS_CACHE", os.getenv("TRANSFORMERS_CACHE", HF_HOME_DEFAULT))

# ------------------ Device / Mem helpers ------------------
def _apply_memory_saving(pipe):
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass
    try:
        pipe.enable_model_cpu_offload()  # newer diffusers
    except Exception:
        pass

# ------------------ S3 ------------------
def _normalize_endpoint(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    u = url.strip()
    # prepend scheme if missing
    if not (u.startswith("http://") or u.startswith("https://")):
        use_ssl = str(os.getenv("S3_USE_SSL", "1")).strip().lower() not in ("0","false","no","off")
        u = ("https://" if use_ssl else "http://") + u
    return u

S3_ENDPOINT = _normalize_endpoint(os.environ.get("S3_ENDPOINT"))
S3_REGION   = os.environ.get("S3_REGION")
S3_KEY      = os.environ.get("S3_KEY")
S3_SECRET   = os.environ.get("S3_SECRET")
S3_BUCKET   = os.environ.get("S3_BUCKET")
S3_PREFIX   = os.environ.get("S3_PREFIX", "outputs")
S3_PRESIGN_EXPIRES = int(os.environ.get("S3_PRESIGN_EXPIRES", "21600"))
S3_CONTENT_DISPOSITION = os.environ.get("S3_CONTENT_DISPOSITION", "")

def _have_s3():
    return all([S3_ENDPOINT, S3_KEY, S3_SECRET, S3_BUCKET])

def _s3_client():
    cfg = {
        "aws_access_key_id": S3_KEY,
        "aws_secret_access_key": S3_SECRET,
        "endpoint_url": S3_ENDPOINT,
        "region_name": S3_REGION or "us-east-1",
        "config": BotoConfig(signature_version="s3v4", retries={"max_attempts": 3}),
    }
    return boto3.client("s3", **cfg)

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
    if os.path.exists(AIO_LOCAL_PATH) and os.path.getsize(AIO_LOCAL_PATH) > 1024 * 1024:
        metrics["aio"] = {"source": "local", "size_bytes": os.path.getsize(AIO_LOCAL_PATH)}
        return AIO_LOCAL_PATH

    t0 = time.time()
    # Try local filenames in CHROMA_LOCAL_DIR
    for fname in AIO_FILENAMES:
        p = os.path.join(CHROMA_LOCAL_DIR, fname)
        if os.path.exists(p) and os.path.getsize(p) > 1024 * 1024:
            shutil.copy2(p, AIO_LOCAL_PATH)
            metrics["aio"] = {"source": "repo", "fname": fname, "size_bytes": os.path.getsize(AIO_LOCAL_PATH), "seconds": time.time() - t0}
            return AIO_LOCAL_PATH

    # Direct URL
    if AIO_URL:
        try:
            with urllib.request.urlopen(AIO_URL) as r, open(AIO_LOCAL_PATH, "wb") as f:
                shutil.copyfileobj(r, f)
            metrics["aio"] = {"source": "url", "size_bytes": os.path.getsize(AIO_LOCAL_PATH), "seconds": time.time() - t0}
            return AIO_LOCAL_PATH
        except Exception as e:
            log.warning("AIO_URL download failed: %s", e)

    # HF hub download (repo snapshot must exist already)
    if not os.path.isdir(CHROMA_LOCAL_DIR) or not os.listdir(CHROMA_LOCAL_DIR):
        raise RuntimeError("CHROMA_LOCAL_DIR is empty; snapshot failed or missing.")

    for fname in AIO_FILENAMES:
        p = os.path.join(CHROMA_LOCAL_DIR, fname)
        if os.path.exists(p):
            shutil.copy2(p, AIO_LOCAL_PATH)
            metrics["aio"] = {"source": "repo", "fname": fname, "size_bytes": os.path.getsize(AIO_LOCAL_PATH), "seconds": time.time() - t0}
            return AIO_LOCAL_PATH

    raise RuntimeError("AIO tensor not found.")

def _path_report():
    return {
        "cache_root": CACHE_ROOT,
        "hf_home": os.environ.get("HF_HOME"),
        "hf_hub_cache": os.environ.get("HUGGINGFACE_HUB_CACHE"),
        "hf_hub_cache_const": HUB_CACHE_PATH_CONST,
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
        "chroma_local_dir": CHROMA_LOCAL_DIR,
        "aio_local_path": AIO_LOCAL_PATH,
        "free_gb_cache_root": round(_free_gb(CACHE_ROOT), 2),
        "free_gb_rootfs": round(_free_gb("/"), 2),
        "dtype": str(DTYPE),
        "low_vram": LOW_VRAM,
    }

# ------------------ S3 ------------------
def _normalize_endpoint(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    u = url.strip()
    # prepend scheme if missing
    if not (u.startswith("http://") or u.startswith("https://")):
        use_ssl = str(os.getenv("S3_USE_SSL", "1")).strip().lower() not in ("0","false","no","off")
        u = ("https://" if use_ssl else "http://") + u
    return u

# ------------------ Snapshot / Repo ------------------
def _ensure_repo(metrics: Dict[str, Any]) -> str:
    if os.path.isdir(CHROMA_LOCAL_DIR) and os.listdir(CHROMA_LOCAL_DIR):
        metrics["snapshot"] = {
            "source": "cached",
            "size_bytes": _dir_size_bytes(CHROMA_LOCAL_DIR),
        }
        return CHROMA_LOCAL_DIR

    t0 = time.time()
    # Allow both SNAPSHOT_URL and the hub snapshot
    if SNAPSHOT_URL:
        tar_path = os.path.join(CACHE_ROOT, "snapshot.tar")
        with urllib.request.urlopen(SNAPSHOT_URL) as r, open(tar_path, "wb") as f:
            shutil.copyfileobj(r, f)
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(CHROMA_LOCAL_DIR)
        os.remove(tar_path)
        metrics["snapshot"] = {
            "source": "url",
            "size_bytes": _dir_size_bytes(CHROMA_LOCAL_DIR),
            "seconds": time.time() - t0,
        }
        return CHROMA_LOCAL_DIR

    # Fall back to hub snapshot
    repo_id = os.getenv("CHROMA_REPO", "Phr00t/Chroma-Rapid-AIO")
    snapshot_download(
        repo_id=repo_id,
        revision=CHROMA_REVISION,
        local_dir=CHROMA_LOCAL_DIR,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "config.json",
            "model_index.json",
            "vae/**",
            "scheduler/**",
            "safety_checker/**",
            "feature_extractor/**",
            "text_encoder/**",
            "tokenizer/**",
            # NOTE: intentionally exclude transformer/unet weights; AIO provides them
        ],
        token=HF_TOKEN or None,
    )
    metrics["snapshot"] = {
        "source": "huggingface",
        "size_bytes": _dir_size_bytes(CHROMA_LOCAL_DIR),
        "seconds": time.time() - t0,
    }
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

    DiffusionPipeline, FMEDS, CPipe, CTrans, T2D = _import_diffusers()
    _ensure_repo(metrics)
    _ensure_aio(metrics)

    # Build transformer from AIO weights
    if CTrans is not None:
        transformer = CTrans.from_single_file(
            AIO_LOCAL_PATH,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        )
    else:
        # fallback: generic Transformer2DModel
        if T2D is None:
            raise RuntimeError("No available Transformer2DModel to load AIO.")
        transformer = T2D.from_single_file(
            AIO_LOCAL_PATH,
            torch_dtype=DTYPE,
            low_cpu_mem_usage=True,
        )

    # Build pipeline with local snapshot
    if CPipe is not None:
        pipe = CPipe.from_pretrained(
            CHROMA_LOCAL_DIR,
            transformer=transformer,
            torch_dtype=DTYPE,
            local_files_only=True,
            trust_remote_code=True,
        )
    else:
        pipe = DiffusionPipeline.from_pretrained(
            CHROMA_LOCAL_DIR,
            torch_dtype=DTYPE,
            local_files_only=True,
            trust_remote_code=True,
        )
        if hasattr(pipe, "transformer"):
            pipe.transformer = transformer

    # Optionally swap to FlowMatch Euler if present
    if FMEDS is not None and hasattr(pipe, "scheduler"):
        try:
            pipe.scheduler = FMEDS.from_config(pipe.scheduler.config)
        except Exception:
            pass

    _apply_memory_saving(pipe)

    if LOW_VRAM:
        try:
            pipe.enable_sequential_cpu_offload()
        except Exception:
            pass
        log.info("LOW_VRAM on: using sequential CPU offload (skipping pipe.to()).")
    else:
        pipe.to(DEVICE, dtype=DTYPE)

    _PIPE = pipe
    log.info("Chroma pipeline ready (dtype=%s, low_vram=%s).", DTYPE, LOW_VRAM)
    return _PIPE

# ------------------ Inference ------------------
def _round8(x: int) -> int:
    return max(64, int(x // 8) * 8)

def _parse_size(s: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    if not s or not isinstance(s, str): return None, None
    try:
        w, h = s.lower().split("x")
        return _round8(int(w)), _round8(int(h))
    except Exception:
        return None, None

# ------------------ Scheduler / Noise-schedule helpers (added) ------------------
def _set_scheduler(pipe, name: Optional[str]):
    """Map sampler/algorithm name (e.g. euler_a, dpmpp_2m) to a diffusers scheduler."""
    if not name:
        return
    try:
        from diffusers import (
            PNDMScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler,
            HeunDiscreteScheduler, KDPM2DiscreteScheduler, KDPM2AncestralDiscreteScheduler,
            DDIMScheduler, DEISMultistepScheduler, DDPMScheduler,
            DPMSolverMultistepScheduler, DPMSolverSinglestepScheduler, UniPCMultistepScheduler,
        )
        try:
            from diffusers import FlowMatchEulerDiscreteScheduler as FMEDS  # optional
        except Exception:
            FMEDS = None
    except Exception:
        return
    cfg = getattr(pipe.scheduler, "config", None)
    if cfg is None:
        return
    n = str(name).strip().lower()
    def _mk(Cls):
        try:
            return Cls.from_config(cfg) if Cls else None
        except Exception:
            return None
    mapping = {
        "pndm": _mk(PNDMScheduler),
        "plms": _mk(PNDMScheduler),  # approx
        "lms": _mk(LMSDiscreteScheduler),
        "heun": _mk(HeunDiscreteScheduler),
        "euler": _mk(EulerDiscreteScheduler),
        "euler_a": _mk(EulerAncestralDiscreteScheduler),
        "euler_ancestral": _mk(EulerAncestralDiscreteScheduler),
        "ddim": _mk(DDIMScheduler),
        "kdpm2": _mk(KDPM2DiscreteScheduler), "dpm2": _mk(KDPM2DiscreteScheduler),
        "kdpm2_a": _mk(KDPM2AncestralDiscreteScheduler), "kdpm2_ancestral": _mk(KDPM2AncestralDiscreteScheduler), "dpm2_a": _mk(KDPM2AncestralDiscreteScheduler),
        "deis": _mk(DEISMultistepScheduler),
        "ddpm": _mk(DDPMScheduler),
        "dpmpp_2m": _mk(DPMSolverMultistepScheduler),
        "dpmpp_2m_sde": _mk(DPMSolverMultistepScheduler),
        "dpmpp_sde": _mk(DPMSolverSinglestepScheduler),
        "dpmsolver": _mk(DPMSolverSinglestepScheduler),
        "uni_pc": _mk(UniPCMultistepScheduler),
        "flowmatch_euler": _mk(FMEDS),
    }
    new_sched = mapping.get(n)
    if new_sched is not None:
        pipe.scheduler = new_sched

def _apply_noise_schedule(pipe, schedule_name: Optional[str]):
    """Apply noise schedule tweaks like 'karras' or 'exponential' when supported by current scheduler."""
    if not schedule_name:
        return
    s = str(schedule_name).strip().lower()
    sched = getattr(pipe, "scheduler", None)
    cfg = getattr(sched, "config", None)
    if sched is None or cfg is None:
        return
    cfg_dict = dict(cfg.__dict__)
    if s == "karras":
        cfg_dict["use_karras_sigmas"] = True
    elif s == "normal":
        # reset to defaults if flags exist
        if "use_karras_sigmas" in cfg_dict: cfg_dict["use_karras_sigmas"] = False
        if "use_exponential_sigmas" in cfg_dict: cfg_dict["use_exponential_sigmas"] = False
    elif s == "exponential":
        cfg_dict["use_exponential_sigmas"] = True
    # other names are no-ops for diffusers; keep compatibility with callers
    try:
        pipe.scheduler = type(sched).from_config(cfg_dict)
    except Exception:
        pass

def _read_init_image(val: Optional[str]) -> Optional[Image.Image]:
    """Decode init image from URL, data URI, or raw base64. Returns PIL.Image or None."""
    if not val or not isinstance(val, str):
        return None
    try:
        if val.startswith("http://") or val.startswith("https://"):
            with urllib.request.urlopen(val) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB")
        if val.startswith("data:"):
            b64 = val.split(",", 1)[1]
            return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
        # assume raw base64
        return Image.open(io.BytesIO(base64.b64decode(val))).convert("RGB")
    except Exception:
        return None

def generate_images(prompt: str, negative_prompt: Optional[str] = None,
                    steps: int = 6, num_images: int = 1,
                    width: Optional[int] = None, height: Optional[int] = None,
                    guidance_scale: Optional[float] = None,
                    seed: Optional[int] = None,
                    scheduler: Optional[str] = None,
                    sampler_name: Optional[str] = None,
                    denoise: Optional[float] = None,
                    init_image: Optional[Image.Image] = None) -> List[Image.Image]:
    with _PIPE_LOCK:
        _build_pipeline({})
        pipe = _PIPE

    # Apply sampler (algorithm) and optional noise schedule
    _set_scheduler(pipe, sampler_name)
    _apply_noise_schedule(pipe, scheduler)

    W = _round8(width if width else 768)
    H = _round8(height if height else 768)
    N = max(1, int(num_images))
    steps = int(steps)

    base_kwargs = {
        "prompt": prompt,
        "num_inference_steps": steps,
        "width": W,
        "height": H,
    }
    if negative_prompt: base_kwargs["negative_prompt"] = negative_prompt
    if guidance_scale is not None: base_kwargs["guidance_scale"] = float(guidance_scale)

    # img2img support and denoise strength if the pipeline accepts them
    allowed = set(inspect.signature(pipe.__call__).parameters.keys())
    if init_image is not None:
        if "image" in allowed:
            base_kwargs["image"] = init_image
        elif "init_image" in allowed:
            base_kwargs["init_image"] = init_image
    if denoise is not None:
        if "strength" in allowed:
            base_kwargs["strength"] = float(denoise)
        elif "denoise" in allowed:
            base_kwargs["denoise"] = float(denoise)

    def _single(gen_seed: Optional[int]):
        gen = torch.Generator(device=DEVICE) if torch.cuda.is_available() else torch.Generator()
        if gen_seed is not None:
            gen = gen.manual_seed(int(gen_seed))
        out = pipe(**base_kwargs, generator=gen)
        imgs = out.images if hasattr(out, "images") else out
        return imgs if isinstance(imgs, list) else [imgs]

    images: List[Image.Image] = []
    try:
        images = _single(seed)
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        torch.cuda.empty_cache()
        base_kwargs["num_inference_steps"] = min(base_kwargs["num_inference_steps"], 6)
        base_kwargs["width"]  = _round8(int(base_kwargs["width"]  * 0.75)) if base_kwargs["width"]  > 768 else 640
        base_kwargs["height"] = _round8(int(base_kwargs["height"] * 0.75)) if base_kwargs["height"] > 768 else 640
        images = _single(seed)

    while len(images) < N:
        torch.cuda.empty_cache()
        more = _single(None if seed is None else seed + len(images))
        images += more

    return images[:N]

# ------------------ Progress helpers (added) ------------------
def _progress_begin(pid: str, total_steps: int):
    PROGRESS[pid] = {
        "status": "running",
        "step": 0,
        "total": total_steps,
        "percent": 0.0,
        "started": time.time(),
        "updated": time.time(),
    }

def _progress_update(pid: str, step: int, total: int):
    rec = PROGRESS.get(pid) or {}
    rec["status"] = "running"
    rec["step"] = step
    rec["total"] = total
    rec["percent"] = float(step) / float(total) if total else 0.0
    rec["updated"] = time.time()
    PROGRESS[pid] = rec

def _progress_finish(pid: str):
    rec = PROGRESS.get(pid) or {}
    rec["status"] = "completed"
    rec["percent"] = 1.0
    rec["updated"] = time.time()
    PROGRESS[pid] = rec

def _progress_fail(pid: str, err: str):
    PROGRESS[pid] = {
        "status": "error",
        "step": 0,
        "total": 0,
        "percent": 0.0,
        "started": time.time(),
        "updated": time.time(),
        "error": err,
    }

# ------------------ Async job runner with callback (added) ------------------
def _run_job_async(pid: str, params: Dict[str, Any]):
    try:
        with _PIPE_LOCK:
            _build_pipeline({})
            pipe = _PIPE

        # Apply sampler and noise schedule for async path (Comfy)
        _set_scheduler(pipe, params.get("sampler_name") or params.get("scheduler"))
        _apply_noise_schedule(pipe, params.get("scheduler"))

        prompt = params["prompt"]
        negative_prompt = params.get("negative_prompt")
        steps = int(params.get("steps", 6))
        width = params.get("width")
        height = params.get("height")
        guidance_scale = params.get("guidance_scale")
        seed = params.get("seed")
        fmt = str(params.get("output_format", "PNG")).upper()

        W = _round8(width if width else 768)
        H = _round8(height if height else 768)

        kwargs = {
            "prompt": prompt,
            "num_inference_steps": steps,
            "width": W,
            "height": H,
        }
        if negative_prompt:
            kwargs["negative_prompt"] = negative_prompt
        if guidance_scale is not None:
            kwargs["guidance_scale"] = float(guidance_scale)

        _progress_begin(pid, steps)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=DEVICE) if torch.cuda.is_available() else torch.Generator()
            generator = generator.manual_seed(int(seed))

        def _cb(step_idx: int, timestep, latents):
            _progress_update(pid, step_idx + 1, steps)

        # pass denoise/strength if supported in async path
        allowed = set(inspect.signature(pipe.__call__).parameters.keys())
        if params.get("denoise") is not None:
            dn = float(params["denoise"])
            if "strength" in allowed:
                kwargs["strength"] = dn
            elif "denoise" in allowed:
                kwargs["denoise"] = dn
        out = pipe(
            **kwargs,
            generator=generator,
            callback=_cb,
            callback_steps=1,
        )
        images = out.images if hasattr(out, "images") else out
        if not isinstance(images, list):
            images = [images]

        if not _have_s3():
            _progress_fail(pid, "missing S3 envs")
            return
        urls = [_upload_and_presign(im, fmt=fmt) for im in images]
        HISTORY[pid] = {"status": "completed", "outputs": {"result": {"images": [{"url": u} for u in urls]}}}
        _progress_finish(pid)

    except Exception as e:
        log.exception("async job error")
        _progress_fail(pid, str(e))

# ------------------ ComfyUI graph parsing (heuristic) ------------------
def parse_comfy_prompt_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    graph = payload.get("prompt") or {}
    prompt_text = None
    negative_text = None
    steps = None
    width = None
    height = None
    guidance = None
    sampler_name = None
    denoise = None
    for _, node in graph.items():
        ctype = str(node.get("class_type", "")).lower()
        inputs = node.get("inputs", {}) or {}
        if "sampler_name" in inputs and isinstance(inputs.get("sampler_name"), str):
            sampler_name = inputs["sampler_name"]
        if "denoise" in inputs:
            try:
                denoise = float(inputs["denoise"])  # may be 0..1
            except Exception:
                pass
        if "textencode" in ctype or "cliptextencode" in ctype:
            t = inputs.get("text")
            if isinstance(t, str):
                if prompt_text is None:
                    prompt_text = t
                else:
                    # treat another as negative if not already set
                    if negative_text is None:
                        negative_text = t
        if "neg" in ctype and negative_text is None:
            nt = inputs.get("text")
            if isinstance(nt, str):
                negative_text = nt
        if "ksampler" in ctype:
            if steps is None and "steps" in inputs:
                try: steps = int(inputs["steps"])
                except Exception: pass
            if "cfg" in inputs and guidance is None:
                try: guidance = float(inputs["cfg"])
                except Exception: pass
        if "image" in inputs and width is None and height is None:
            # some graphs include size in a node
            try:
                w = inputs.get("width")
                h = inputs.get("height")
                if isinstance(w, int) and isinstance(h, int):
                    width, height = int(w), int(h)
            except Exception:
                pass

    if width is None or height is None:
        w, h = _parse_size(payload.get("size"))
        if w: width = w
        if h: height = h
    if guidance is None and payload.get("cfg") is not None:
        try: guidance = float(payload.get("cfg"))
        except Exception: pass
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
        "output_format": str(payload.get("output_format", "PNG")).upper(),
        "seed": payload.get("seed"),
        "sampler_name": sampler_name if sampler_name is not None else payload.get("sampler_name"),
        "denoise": denoise if denoise is not None else payload.get("denoise"),
    }

# ------------------ Core processing (shared by both modes) ------------------
def process_input(input_data: Dict[str, Any]) -> Dict[str, Any]:
    if input_data.get("debug"):
        return {"status": "ok", "paths": _path_report()}

    if input_data.get("cleanup"):
        removed = 0
        hub_dir = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HUB_CACHE") or os.path.join(CACHE_ROOT, "hf_cache")
        for p in [CHROMA_LOCAL_DIR, AIO_LOCAL_PATH, hub_dir]:
            if not p:
                continue
            if os.path.islink(p) or os.path.isfile(p):
                try:
                    os.remove(p); removed += 1
                except Exception:
                    pass
            elif os.path.isdir(p):
                try:
                    shutil.rmtree(p, ignore_errors=True); removed += 1
                except Exception:
                    pass
        return {"status": "ok", "cleanup_removed": removed, "free_gb": round(_free_gb(CACHE_ROOT), 2)}

    if input_data.get("warmup"):
        with _PIPE_LOCK:
            _build_pipeline({})
        return {"status": "ok", "warmed": True}

    # --- Flexible request keys ---
    prompt = input_data.get("prompt")
    if input_data.get("size") and (not input_data.get("width") or not input_data.get("height")):
        w, h = _parse_size(input_data.get("size"))
        if w: input_data["width"] = w
        if h: input_data["height"] = h
    if "n" in input_data and "num_images" not in input_data:
        input_data["num_images"] = input_data.get("n")
    if "cfg" in input_data and "guidance_scale" not in input_data:
        input_data["guidance_scale"] = input_data.get("cfg")

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
            seed=params.get("seed"),
            sampler_name=params.get("sampler_name"),
            denoise=params.get("denoise"),
        )
        if not _have_s3():
            return {"status": "error", "error": "missing S3 envs"}
        fmt = params["output_format"]
        urls = [_upload_and_presign(im, fmt=fmt) for im in imgs]
        pid = uuid.uuid4().hex
        HISTORY[pid] = {"status": "completed", "outputs": {"result": {"images": [{"url": u} for u in urls]}}}
        return {"status": "ok", "count": len(urls), "results": [{"url": u} for u in urls]}

    if not prompt:
        return {"status": "error", "error": "Missing 'prompt'."}

    if input_data.get("sse"):
        # start async job and stream progress
        prompt_id = uuid.uuid4().hex
        params = dict(input_data)
        params["prompt"] = prompt
        t = threading.Thread(target=_run_job_async, args=(prompt_id, params), daemon=True)
        t.start()

        def event_gen():
            last = None
            while True:
                rec = PROGRESS.get(prompt_id)
                if rec is None:
                    yield f"event: gone\ndata: {prompt_id}\n\n"
                    return
                if rec != last:
                    payload = {
                        "prompt_id": prompt_id,
                        "status": rec.get("status"),
                        "step": rec.get("step"),
                        "total": rec.get("total"),
                        "percent": rec.get("percent"),
                        "updated": rec.get("updated"),
                        "error": rec.get("error"),
                    }
                    import json as _json
                    yield "data: " + _json.dumps(payload) + "\n\n"
                    last = dict(rec)
                if rec.get("status") in ("completed", "error"):
                    return
                time.sleep(0.5)
        return StreamingResponse(event_gen(), media_type="text/event-stream")

    init_img = _read_init_image(input_data.get("init_image") or input_data.get("image") or input_data.get("init_image_url") or input_data.get("image_url"))
    imgs = generate_images(
        prompt=prompt,
        negative_prompt=input_data.get("negative_prompt"),
        steps=int(input_data.get("steps", 6)),
        num_images=int(input_data.get("num_images", 1)),
        width=input_data.get("width"),
        height=input_data.get("height"),
        guidance_scale=input_data.get("guidance_scale"),
        seed=input_data.get("seed"),
        scheduler=input_data.get("scheduler"),
        sampler_name=input_data.get("sampler_name"),
        denoise=input_data.get("denoise"),
        init_image=init_img,
    )
    if not _have_s3():
        return {"status": "error", "error": "missing S3 envs"}
    fmt = str(input_data.get("output_format", "PNG")).upper()
    urls = [_upload_and_presign(im, fmt=fmt) for im in imgs]
    return {"status": "ok", "count": len(urls), "results": [{"url": u} for u in urls]}

# ------------------ FastAPI app (Load Balancer + ComfyUI) ------------------
app = FastAPI(title="Chroma Hybrid API", version="1.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8188", "http://127.0.0.1:8188", "*"],
    allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

@app.get("/health")
def health():
    try:
        with _PIPE_LOCK:
            ready = _PIPE is not None
            if not ready:
                _build_pipeline({})
        return {"status": "ok", "ready": True, "device": str(DEVICE), "dtype": str(DTYPE), "low_vram": LOW_VRAM}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.post("/run")
def http_run(input_data: Dict[str, Any] = Body(...)):
    try:
        out = process_input(input_data)
        return JSONResponse(out)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("http_run error")
        raise HTTPException(status_code=500, detail=str(e))

# ---------- modified: make /prompt async & return prompt_id immediately ----------
@app.post("/prompt")
def comfy_prompt(payload: Dict[str, Any] = Body(...)):
    params = parse_comfy_prompt_payload(payload)
    if not params["prompt"]:
        raise HTTPException(status_code=400, detail="No prompt text found in graph or payload.")
    pid = uuid.uuid4().hex
    HISTORY.pop(pid, None)
    PROGRESS.pop(pid, None)
    t = threading.Thread(target=_run_job_async, args=(pid, params), daemon=True)
    t.start()
    return {"prompt_id": pid}

@app.get("/history/{prompt_id}")
def comfy_history(prompt_id: str):
    rec = HISTORY.get(prompt_id)
    if not rec:
        raise HTTPException(status_code=404, detail="not found")
    return rec

@app.get("/progress/{prompt_id}")
def comfy_progress(prompt_id: str):
    return PROGRESS.get(prompt_id, {"status": "unknown"})

@app.get("/sse/{prompt_id}")
def comfy_sse(prompt_id: str):
    def event_gen():
        last = None
        while True:
            rec = PROGRESS.get(prompt_id)
            if rec is None:
                yield f"event: gone\ndata: {prompt_id}\n\n"
                return
            if rec != last:
                payload = {
                    "prompt_id": prompt_id,
                    "status": rec.get("status"),
                    "step": rec.get("step"),
                    "total": rec.get("total"),
                    "percent": rec.get("percent"),
                    "updated": rec.get("updated"),
                    "error": rec.get("error"),
                }
                import json as _json
                yield "data: " + _json.dumps(payload) + "\n\n"
                last = dict(rec)
            if rec.get("status") in ("completed", "error"):
                return
            time.sleep(0.5)
    return StreamingResponse(event_gen(), media_type="text/event-stream")

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
