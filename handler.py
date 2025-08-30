import io, json, os, time, uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download
from diffusers import ChromaPipeline, ChromaTransformer2DModel, FlowMatchEulerDiscreteScheduler
import runpod

import boto3, botocore

# -------------------- Flags & helpers --------------------
def _as_bool(v: Optional[str], default: bool = False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() not in ("0", "false", "no", "off", "")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

DTYPE = _pick_dtype()

# -------------------- Model config --------------------
# AIO transformer checkpoint (single file)
HF_REPO          = os.environ.get("AIO_REPO", "Phr00t/Chroma-Rapid-AIO")
AIO_FILENAME_PRI = os.environ.get("AIO_FILENAME_PRI", "Chroma-Rapid-AIO-v2.safetensors")
AIO_FILENAME_ALT = os.environ.get("AIO_FILENAME_ALT", "Chroma-Rapid-AIO.safetensors")
AIO_LOCAL_PATH   = os.environ.get("AIO_LOCAL_PATH")  # optional: absolute path to .safetensors

# Chroma base (configs, VAE, text encoder/tokenizer)
CHROMA_BASE_ID    = os.environ.get("CHROMA_BASE_ID", "lodestones/Chroma")
CHROMA_LOCAL_DIR  = os.environ.get("CHROMA_LOCAL_DIR", "/weights/hf/lodestones__Chroma")
HF_OFFLINE_OK     = _as_bool(os.environ.get("HF_HUB_OFFLINE"), default=False)

ENABLE_CPU_OFFLOAD = _as_bool(os.environ.get("ENABLE_CPU_OFFLOAD"), default=True)
ENABLE_VAE_TILING  = _as_bool(os.environ.get("ENABLE_VAE_TILING"), default=True)

PIPE: Optional[ChromaPipeline] = None

# -------------------- MinIO/S3 config --------------------
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")      # e.g. https://s3.us-west-000.backblazeb2.com
S3_ACCESS_KEY   = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY   = os.environ.get("S3_SECRET_KEY")
S3_REGION       = os.environ.get("S3_REGION", "us-east-1")
S3_BUCKET       = os.environ.get("S3_BUCKET")            # required
S3_PREFIX       = os.environ.get("S3_PREFIX", "chroma/outputs/")
S3_PRESIGN_EXP  = int(os.environ.get("S3_PRESIGN_EXPIRES", "3600"))
S3_USE_SSL      = _as_bool(os.environ.get("S3_USE_SSL"), default=True)
S3_VERIFY_SSL   = _as_bool(os.environ.get("S3_VERIFY_SSL"), default=True)
S3_CONTENT_DISP = os.environ.get("S3_CONTENT_DISPOSITION", "attachment")  # or "inline"

S3_CLIENT = None
def _ensure_s3():
    global S3_CLIENT
    if S3_CLIENT is not None:
        return S3_CLIENT
    if not (S3_ENDPOINT_URL and S3_ACCESS_KEY and S3_SECRET_KEY and S3_BUCKET):
        return None
    cfg = botocore.config.Config(signature_version="s3v4", s3={"addressing_style": "path"})
    sess = boto3.session.Session()
    S3_CLIENT = sess.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
        region_name=S3_REGION,
        use_ssl=S3_USE_SSL,
        verify=S3_VERIFY_SSL,
        config=cfg,
    )
    return S3_CLIENT

# -------------------- Local snapshot for Chroma base --------------------
def _ensure_chroma_base_local() -> str:
    """
    Ensure the Chroma base repo (configs + VAE + tokenizer/encoders) exists locally.
    If HF is unreachable but a local copy exists, we proceed offline.
    """
    idx = os.path.join(CHROMA_LOCAL_DIR, "model_index.json")
    if os.path.isdir(CHROMA_LOCAL_DIR) and os.path.exists(idx):
        return CHROMA_LOCAL_DIR

    token = os.environ.get("HF_TOKEN")

    if HF_OFFLINE_OK:
        # Offline requested but no local snapshot → fail early with a clear message
        raise RuntimeError(
            "HF offline mode is enabled (HF_HUB_OFFLINE=1) but CHROMA_LOCAL_DIR has no snapshot. "
            "Pre-seed the folder or disable HF_HUB_OFFLINE for the first run."
        )

    allow = [
        "model_index.json",
        "*.json",
        "ae.safetensors",                 # Flux VAE (naming may vary; keep permissive)
        "vae/*",                          # in case repo changes structure
        "text_encoder/*",
        "tokenizer/*",
        "*.safetensors",                  # catch-all
    ]

    last_err = None
    for backoff in (1, 3, 7):
        try:
            snapshot_download(
                repo_id=CHROMA_BASE_ID,
                token=token,
                local_dir=CHROMA_LOCAL_DIR,
                local_dir_use_symlinks=False,
                allow_patterns=allow,
            )
            return CHROMA_LOCAL_DIR
        except Exception as e:
            last_err = e
            time.sleep(backoff)
    raise RuntimeError(f"Could not snapshot '{CHROMA_BASE_ID}'. Network/token issue? {last_err}")

# -------------------- AIO checkpoint resolution --------------------
def _download_aio_checkpoint() -> str:
    """
    Prefer a local AIO .safetensors if provided, otherwise try HF Hub.
    Honors HF_HUB_OFFLINE by using local cache only when enabled.
    """
    if AIO_LOCAL_PATH and os.path.isfile(AIO_LOCAL_PATH):
        return AIO_LOCAL_PATH

    token = os.environ.get("HF_TOKEN")
    kwargs = {}
    if HF_OFFLINE_OK:
        kwargs["local_files_only"] = True

    try:
        return hf_hub_download(repo_id=HF_REPO, filename=AIO_FILENAME_PRI, token=token, resume_download=True, **kwargs)
    except Exception:
        return hf_hub_download(repo_id=HF_REPO, filename=AIO_FILENAME_ALT, token=token, resume_download=True, **kwargs)

# -------------------- Diffusers pipeline build --------------------
def _build_pipeline() -> ChromaPipeline:
    ckpt = _download_aio_checkpoint()
    transformer = ChromaTransformer2DModel.from_single_file(ckpt, torch_dtype=DTYPE)

    local_model_root = _ensure_chroma_base_local()
    pipe = ChromaPipeline.from_pretrained(
        local_model_root,
        transformer=transformer,
        torch_dtype=DTYPE,
        local_files_only=True,  # enforce offline for the base repo at runtime
    )
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)

    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if ENABLE_CPU_OFFLOAD:
            pipe.enable_model_cpu_offload()
        if ENABLE_VAE_TILING:
            pipe.enable_vae_tiling()
    return pipe

def _ensure_pipe() -> ChromaPipeline:
    global PIPE
    if PIPE is None:
        PIPE = _build_pipeline()
    return PIPE

# -------------------- Parsing helpers --------------------
def _parse_json_maybe(s: Any) -> Any:
    if isinstance(s, str):
        t = s.strip()
        if t and (t.startswith("{") or t.startswith("[")):
            try:
                return json.loads(t)
            except Exception:
                return s
    return s

def _as_list(x: Any) -> Optional[List[str]]:
    if x is None:
        return None
    if isinstance(x, list):
        return [str(i) for i in x]
    return [str(x)]

def _get_first_of(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _i(x: Any, default: Optional[int]):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

def _f(x: Any, default: Optional[float]):
    try:
        return float(x)
    except Exception:
        return default

def _size(d: Dict[str, Any]):
    h = _i(d.get("height"), None)
    w = _i(d.get("width"), None)
    sz = d.get("size") or d.get("resolution")
    if (h is None or w is None) and isinstance(sz, str) and "x" in sz.lower():
        try:
            w_s, h_s = sz.lower().split("x")
            w = w or _i(w_s, None)
            h = h or _i(h_s, None)
        except Exception:
            pass
    def _align(n): return int(round(n / 8) * 8) if n else n
    return _align(h), _align(w)

def _prepare_generators(seed: Union[None, int, List[int]], total_images: int):
    if seed is None:
        return None, None
    if isinstance(seed, list):
        gens = []
        for i in range(total_images):
            s = int(seed[i % len(seed)])
            gens.append(torch.Generator(device=DEVICE).manual_seed(s))
        return gens, seed
    s = int(seed)
    gen = torch.Generator(device=DEVICE).manual_seed(s)
    return gen, s

def _safe_prefix(p: str) -> str:
    if not p:
        return ""
    if not p.endswith("/"):
        p += "/"
    return p.lstrip("/")

def _img_to_bytes(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()

def _upload_bytes_list(img_bytes_list: List[bytes], fmt: str):
    s3 = _ensure_s3()
    if s3 is None:
        return {"uploaded": False, "error": "S3/MinIO not configured", "keys": [], "urls": []}
    ts = datetime.utcnow().strftime("%Y/%m/%d/%H%M%S")
    base = _safe_prefix(S3_PREFIX) + ts
    keys, urls = [], []
    for idx, blob in enumerate(img_bytes_list):
        key = f"{base}/{uuid.uuid4().hex}_{idx}.{fmt.lower()}"
        extra = {"ContentType": f"image/{fmt.lower()}", "ContentDisposition": S3_CONTENT_DISP}
        s3.put_object(Bucket=S3_BUCKET, Key=key, Body=blob, **extra)
        url = s3.generate_presigned_url(
            "get_object", Params={"Bucket": S3_BUCKET, "Key": key}, ExpiresIn=S3_PRESIGN_EXP
        )
        keys.append(key)
        urls.append(url)
    return {"uploaded": True, "keys": keys, "urls": urls}

# -------------------- Core generate --------------------
def generate(params: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    pipe = _ensure_pipe()

    prompt   = _get_first_of(params, ["prompt", "prompts", "text", "texts"])
    negative = _get_first_of(params, ["negative_prompt", "negative_prompts", "neg", "negative"])
    steps    = _i(_get_first_of(params, ["num_inference_steps", "steps"]), 8)
    guidance = _f(_get_first_of(params, ["guidance_scale", "cfg", "cfg_scale"]), 1.0)
    n_images = _i(_get_first_of(params, ["num_images_per_prompt", "num_images", "n", "batch_size"]), 1)
    seed     = _get_first_of(params, ["seed", "seeds"])
    fmt      = (params.get("output_format") or "PNG").upper()

    h, w = _size(params)
    prompt   = _parse_json_maybe(prompt)
    negative = _parse_json_maybe(negative)
    prompt_l   = _as_list(prompt)
    negative_l = _as_list(negative) if negative is not None else None
    if not prompt_l:
        raise ValueError("Missing 'prompt' (or 'prompts'/'text'/'texts').")

    gen, seed_out = _prepare_generators(seed, n_images)
    out = pipe(
        prompt=prompt_l,
        negative_prompt=negative_l,
        height=h,
        width=w,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        num_images_per_prompt=int(n_images),
        generator=gen,
    )

    img_bytes_list = [_img_to_bytes(img, fmt=fmt) for img in out.images]
    s3r = _upload_bytes_list(img_bytes_list, fmt=fmt)

    t1 = time.time()
    return {
        "status": "ok",
        "count": len(img_bytes_list),
        "uploaded": s3r.get("uploaded", False),
        "upload_error": s3r.get("error"),
        "presigned_urls": s3r.get("urls", []),
        "s3_bucket": S3_BUCKET,
        "s3_keys": s3r.get("keys", []),
        "meta": {
            "height": out.images[0].height if out.images else h,
            "width": out.images[0].width if out.images else w,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "num_images": n_images,
            "seed": seed_out,
            "dtype": str(DTYPE).replace("torch.", ""),
            "device": DEVICE,
            "time_sec": round(t1 - t0, 3),
            "format": fmt,
        },
    }

# -------------------- Runpod entry --------------------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flexible payloads:
      { "input": { "prompt": "a car", "steps": 8, "cfg": 1.0, "n": 1, "size": "1024x1024" } }
      { "prompt": "a car", ... } (no input wrapper)
      Or stringified JSON in 'input'/'body'/'data'
    """
    try:
        data = event.get("input", event)
        data = _parse_json_maybe(data) or {}
        if isinstance(data, dict):
            alt = data.get("body") or data.get("data") or data.get("payload")
            if isinstance(alt, (dict, str)):
                alt = _parse_json_maybe(alt)
                if isinstance(alt, dict):
                    data = {**data, **alt}
        return generate(data)
    except Exception as e:
        return {"status": "error", "message": str(e)}

runpod.serverless.start({"handler": handler})
