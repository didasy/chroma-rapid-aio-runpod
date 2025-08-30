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
