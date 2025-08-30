import io, json, os, time, uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image
from huggingface_hub import hf_hub_download
from diffusers import ChromaPipeline, ChromaTransformer2DModel, FlowMatchEulerDiscreteScheduler
import runpod

import boto3, botocore

# ---------- Model config ----------
HF_REPO = os.environ.get("AIO_REPO", "Phr00t/Chroma-Rapid-AIO")
AIO_FILENAME_PRI = os.environ.get("AIO_FILENAME_PRI", "Chroma-Rapid-AIO-v2.safetensors")
AIO_FILENAME_ALT = os.environ.get("AIO_FILENAME_ALT", "Chroma-Rapid-AIO.safetensors")
CHROMA_BASE_ID   = os.environ.get("CHROMA_BASE_ID", "lodestones/Chroma")

ENABLE_CPU_OFFLOAD = os.environ.get("ENABLE_CPU_OFFLOAD", "1").lower() not in ("0", "false")
ENABLE_VAE_TILING  = os.environ.get("ENABLE_VAE_TILING", "1").lower() not in ("0", "false")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32
DTYPE = pick_dtype()

PIPE: Optional[ChromaPipeline] = None

# ---------- MinIO/S3 config ----------
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL")      # e.g. https://minio.local:9000
S3_ACCESS_KEY   = os.environ.get("S3_ACCESS_KEY")
S3_SECRET_KEY   = os.environ.get("S3_SECRET_KEY")
S3_REGION       = os.environ.get("S3_REGION", "us-east-1")
S3_BUCKET       = os.environ.get("S3_BUCKET")            # required
S3_PREFIX       = os.environ.get("S3_PREFIX", "chroma/outputs/")
S3_PRESIGN_EXP  = int(os.environ.get("S3_PRESIGN_EXPIRES", "3600"))
S3_USE_SSL      = os.environ.get("S3_USE_SSL", "1").lower() not in ("0", "false")
S3_VERIFY_SSL   = os.environ.get("S3_VERIFY_SSL", "1").lower() not in ("0", "false")
S3_CONTENT_DISP = os.environ.get("S3_CONTENT_DISPOSITION", "attachment")  # or inline

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

# ---------- Helpers ----------
def _download_aio_checkpoint() -> str:
    token = os.environ.get("HF_TOKEN")
    try:
        return hf_hub_download(repo_id=HF_REPO, filename=AIO_FILENAME_PRI, token=token, resume_download=True)
    except Exception:
        return hf_hub_download(repo_id=HF_REPO, filename=AIO_FILENAME_ALT, token=token, resume_download=True)

def _build_pipeline() -> ChromaPipeline:
    ckpt = _download_aio_checkpoint()
    transformer = ChromaTransformer2DModel.from_single_file(ckpt, torch_dtype=DTYPE)
    pipe = ChromaPipeline.from_pretrained(CHROMA_BASE_ID, transformer=transformer, torch_dtype=DTYPE)
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if ENABLE_CPU_OFFLOAD: pipe.enable_model_cpu_offload()
        if ENABLE_VAE_TILING:  pipe.enable_vae_tiling()
    return pipe

def _ensure_pipe() -> ChromaPipeline:
    global PIPE
    if PIPE is None:
        PIPE = _build_pipeline()
    return PIPE

def _parse_json_maybe(s: Any) -> Any:
    if isinstance(s, str):
        t = s.strip()
        if t and (t.startswith("{") or t.startswith("[")):
            try: return json.loads(t)
            except Exception: return s
    return s

def _as_list(x: Any) -> Optional[List[str]]:
    if x is None: return None
    if isinstance(x, list): return [str(i) for i in x]
    return [str(x)]

def _get_first_of(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

def _i(x: Any, default: Optional[int]):  # int
    try: return int(x)
    except Exception:
        try: return int(float(x))
        except Exception: return default

def _f(x: Any, default: Optional[float]):  # float
    try: return float(x)
    except Exception: return default

def _size(d: Dict[str, Any]):
    h = _i(d.get("height"), None)
    w = _i(d.get("width"), None)
    sz = d.get("size") or d.get("resolution")
    if (h is None or w is None) and isinstance(sz, str) and "x" in sz.lower():
        try:
            w_s, h_s = sz.lower().split("x")
            w = w or _i(w_s, None); h = h or _i(h_s, None)
        except Exception: pass
    def _align(n): return int(round(n/8)*8) if n else n
    return _align(h), _align(w)

def _prepare_generators(seed: Union[None, int, List[int]], total_images: int):
    if seed is None: return None, None
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
    if not p: return ""
    if not p.endswith("/"): p += "/"
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
        keys.append(key); urls.append(url)
    return {"uploaded": True, "keys": keys, "urls": urls}

# ---------- Core ----------
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
    if not prompt_l: raise ValueError("Missing 'prompt' (or 'prompts'/'text'/'texts').")

    gen, seed_out = _prepare_generators(seed, n_images)
    out = pipe(
        prompt=prompt_l,
        negative_prompt=negative_l,
        height=h, width=w,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        num_images_per_prompt=int(n_images),
        generator=gen,
    )

    # Upload only (no base64 in response)
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

# ---------- Runpod entry ----------
def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Flexible ComfyUI payloads:
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
