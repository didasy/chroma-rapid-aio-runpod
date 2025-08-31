### Examples

#### Req/Res

```
Request
{
  "input": {
    "prompt": "Ultradetailed photo of a classic sports car at dusk, glossy reflections",
    "negative_prompt": "blurry, low quality",
    "steps": 8,
    "cfg": 1.0,
    "n": 1,
    "size": "1024x1024",
    "seed": 2025,
    "output_format": "PNG"
  }
}
```

```
Response
{
  "status": "ok",
  "count": 1,
  "uploaded": true,
  "presigned_urls": ["https://minio...X-Amz-Signature=..."],
  "s3_bucket": "chroma-art",
  "s3_keys": ["chroma/outputs/2025/08/30/091234/abcd1234_0.png"],
  "meta": { "height": 1024, "width": 1024, "num_inference_steps": 8, "guidance_scale": 1.0, "num_images": 1, "seed": 2025, "dtype": "bfloat16", "device": "cuda", "time_sec": 1.23, "format": "PNG" }
}

```

```
Cheat sheet

{
  "name": "Chroma Hybrid API – Text2Img",
  "modes": {
    "queue_mode": {
      "submit": "POST /run or /runsync (Runpod Serverless)",
      "body": { "input": { "...": "see schema.fields" } }
    },
    "load_balancer_mode": {
      "submit": "POST /run (direct to container/LB)",
      "body": { "...": "either direct fields or {\"input\": {...}}" }
    }
  },
  "schema": {
    "fields": {
      "prompt":        { "type": "string", "required": true, "desc": "Text description to generate." },
      "negative_prompt": { "type": "string", "required": false, "desc": "Things to avoid." },
      "steps":         { "type": "integer", "default": 6, "desc": "Inference steps (6–12 typical)." },
      "guidance_scale":{ "type": "number", "aliases": ["cfg"], "required": false, "desc": "Prompt strength; ~0.8–2.0 common." },
      "num_images":    { "type": "integer", "aliases": ["n"], "default": 1, "desc": "Images to return (generated sequentially)." },
      "width":         { "type": "integer", "required": false, "desc": "Pixels, multiple of 8; default 768." },
      "height":        { "type": "integer", "required": false, "desc": "Pixels, multiple of 8; default 768." },
      "size":          { "type": "string", "required": false, "format": "WxH", "desc": "Alternative to width/height (e.g., \"1024x1024\")." },
      "seed":          { "type": "integer", "required": false, "desc": "Reproducibility; increments for multi-image." },
      "output_format": { "type": "string", "enum": ["PNG","JPG","WEBP"], "default": "PNG" },
      "warmup":        { "type": "boolean", "desc": "Preload weights; returns warmed:true" },
      "cleanup":       { "type": "boolean", "desc": "Wipes caches to reclaim space (dangerous)." },
      "debug":         { "type": "boolean", "desc": "Returns paths and free space info." },
      "comfy":         { "type": "boolean", "desc": "Treat payload as ComfyUI /prompt." },
      "payload":       { "type": "object", "required_if": "comfy=true", "desc": "ComfyUI /prompt body." }
    },
    "notes": [
      "LB mode accepts either direct fields or {\"input\": {...}}.",
      "If width/height omitted, handler defaults to 768x768 and retries smaller on OOM.",
      "S3/B2 presigned URLs are returned for downloads."
    ]
  },
  "responses": {
    "ok_results": {
      "status": "ok",
      "count": 1,
      "results": [{ "url": "https://..." }]
    },
    "ok_warmed": { "status": "ok", "warmed": true },
    "ok_comfy":  { "status": "ok", "prompt_id": "abcdef1234" },
    "error":     { "status": "error", "error": "message" }
  },
  "examples": {
    "queue_mode_runsync": {
      "request": {
        "input": {
          "prompt": "Ultradetailed photo of a classic sports car at dusk, glossy reflections",
          "negative_prompt": "blurry, low quality",
          "steps": 8,
          "cfg": 1,
          "n": 1,
          "size": "1024x1024",
          "seed": 2025,
          "output_format": "PNG"
        }
      }
    },
    "lb_mode_direct": {
      "request": {
        "prompt": "a neon car in the rain, cinematic",
        "steps": 6,
        "num_images": 1,
        "width": 768,
        "height": 768,
        "output_format": "PNG"
      }
    },
    "warmup": { "request": { "input": { "warmup": true } } },
    "debug":  { "request": { "input": { "debug": true } } },
    "comfy_prompt": {
      "request": {
        "input": {
          "comfy": true,
          "payload": {
            "prompt": {
              "1": { "class_type": "CLIPTextEncode", "inputs": { "text": "a neon car in the rain" } },
              "2": { "class_type": "KSampler", "inputs": { "steps": 6, "cfg": 1.0 } },
              "3": { "class_type": "EmptyLatentImage", "inputs": { "width": 768, "height": 768 } }
            },
            "client_id": "local-comfy"
          }
        }
      }
    }
  }
}

```
