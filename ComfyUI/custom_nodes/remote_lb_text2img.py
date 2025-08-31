# ComfyUI custom node: call your remote backend (Load Balancer)
# Save as: ComfyUI/custom_nodes/remote_lb_text2img.py
# You won’t see live step-by-step progress in the Comfy UI (unless you add a progress endpoint later).

import io, json, time, requests, PIL.Image
from typing import List, Tuple

class RemoteLBText2Img:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_url": ("STRING", {"default": "http://127.0.0.1:3000"}),
                "route": (["/run","/prompt"],),
                "prompt": ("STRING", {"multiline": True}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0}),
                "n": ("INT", {"default": 1, "min": 1, "max": 8}),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
                "height": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1}),
                "output_format": (["PNG","JPG","WEBP"],),
                "timeout_sec": ("INT", {"default": 1800, "min": 30, "max": 7200}),
            },
            "optional": {
                "extra_headers_json": ("STRING", {"default": ""})  # e.g. '{"Authorization":"Bearer XYZ"}'
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "go"
    CATEGORY = "Runpod/Remote"

    def _headers(self, extra_headers_json: str):
        h = {"Content-Type": "application/json"}
        if extra_headers_json.strip():
            try:
                h.update(json.loads(extra_headers_json))
            except Exception:
                pass
        return h

    def _download_urls(self, urls: List[str], timeout: int) -> List[PIL.Image.Image]:
        out = []
        for u in urls:
            r = requests.get(u, timeout=timeout)
            r.raise_for_status()
            out.append(PIL.Image.open(io.BytesIO(r.content)).convert("RGB"))
        return out

    def go(self, base_url: str, route: str, prompt: str, negative_prompt: str,
           steps: int, cfg: float, n: int, width: int, height: int,
           seed: int, output_format: str, timeout_sec: int, extra_headers_json: str = ""):

        base = base_url.rstrip("/")
        headers = self._headers(extra_headers_json)

        if route == "/run":
            body = {
                "prompt": prompt,
                "negative_prompt": negative_prompt or None,
                "steps": int(steps),
                "cfg": float(cfg),
                "n": int(n),
                "width": int(width),
                "height": int(height),
                "seed": int(seed),
                "output_format": output_format
            }
            r = requests.post(f"{base}/run", data=json.dumps(body), headers=headers, timeout=timeout_sec)
            r.raise_for_status()
            js = r.json()
            if js.get("status") != "ok" or "results" not in js:
                raise RuntimeError(f"Backend error: {js}")
            urls = [it["url"] for it in js["results"] if "url" in it]
            images = self._download_urls(urls, timeout_sec)
            return (images,)

        else:  # route == "/prompt"
            payload = {
                "prompt": {
                    "1": {"class_type":"CLIPTextEncode","inputs":{"text": prompt}},
                    "2": {"class_type":"KSampler","inputs":{"steps": int(steps), "cfg": float(cfg)}},
                    "3": {"class_type":"EmptyLatentImage","inputs":{"width": int(width), "height": int(height)}}
                },
                "client_id":"remote-lb"
            }
            r = requests.post(f"{base}/prompt", data=json.dumps(payload), headers=headers, timeout=timeout_sec)
            r.raise_for_status()
            pid = r.json().get("prompt_id")
            if not pid: raise RuntimeError(f"No prompt_id: {r.text}")

            # poll history
            start = time.time()
            while True:
                hr = requests.get(f"{base}/history/{pid}", headers=headers, timeout=30)
                if hr.status_code == 200:
                    h = hr.json()
                    hist = h.get("history",{}).get(pid,{})
                    out = hist.get("outputs",{}).get("result",{}).get("images",[])
                    urls = [x["url"] for x in out if "url" in x]
                    if urls:
                        images = self._download_urls(urls, timeout_sec)
                        return (images,)
                if time.time() - start > timeout_sec:
                    raise TimeoutError("Remote prompt timed out")
                time.sleep(2)

NODE_CLASS_MAPPINGS = {"RemoteLBText2Img": RemoteLBText2Img}
NODE_DISPLAY_NAME_MAPPINGS = {"RemoteLBText2Img": "Remote LB Text2Img"}
