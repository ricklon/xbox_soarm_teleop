# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#     "transformers==4.57.1",
#     "torch>=2.5",
#     "torchvision",
#     "pillow",
#     "einops",
#     "accelerate",
#     "molmo_utils",
# ]
# ///
"""Serve Molmo2-ER pointing over HTTP for the twin's ``locate`` tool (``--pointer molmo``).

Molmo2-ER needs its own transformers version, so it runs in its own process
with its own dependencies (uv reads them from the header above):

    uv run scripts/molmo_server.py                 # http://127.0.0.1:8890
    uv run scripts/molmo_server.py --port 8891 --model allenai/Molmo2-4B

The weights (about 19 GB) come from Hugging Face on first use and run the
model's own code (``trust_remote_code``). On a Mac it runs on the GPU (MPS).

POST /point  {"image_b64", "description", "width", "height"}
          -> {"u", "v", "raw", "seconds"}  (pixels; u and v null if it saw nothing)
GET  /health -> {"ok": true, "model": ...}
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COORDS = re.compile(r"<(?:points|tracks)[^>]*?coords=\"([0-9\t:;, .]+)\"")
TRIPLE = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")
MOLMO1 = re.compile(r"<point[^>]*?x=\"([0-9.]+)\"[^>]*?y=\"([0-9.]+)\"")
PROMPT = "Point to the {description}."


def parse(text: str, width: int, height: int) -> tuple[float, float] | None:
    match = COORDS.search(text)
    if match:
        for chunk in re.split(r"[\t;]", match.group(1).strip()):
            triple = TRIPLE.search(chunk.strip())
            if triple:
                return float(triple.group(2)) / 1000 * width, float(triple.group(3)) / 1000 * height
    match = MOLMO1.search(text)
    if match:
        return float(match.group(1)) / 100 * width, float(match.group(2)) / 100 * height
    return None


class Molmo:
    def __init__(self, model_id: str, device: str | None) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if device is None:
            device = (
                "mps"
                if torch.backends.mps.is_available()
                else "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        self.torch = torch
        self.device = device
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, trust_remote_code=True, dtype=dtype
        ).to(device)
        self.model.eval()
        self.lock = threading.Lock()  # one image at a time on the GPU

    def point(self, image_bytes: bytes, description: str) -> tuple[str, int, int]:
        from PIL import Image

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    dict(type="text", text=PROMPT.format(description=description)),
                    dict(type="image", image=image),
                ],
            }
        ]
        with self.lock:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
            tokens = generated[0, inputs["input_ids"].size(1) :]
            text = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
        return text, image.width, image.height


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="allenai/Molmo2-ER")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--device", default=None, help="mps, cuda or cpu (default: best available)")
    args = parser.parse_args()

    print(f"Loading {args.model}...", flush=True)
    started = time.monotonic()
    molmo = Molmo(args.model, args.device)
    print(f"Loaded on {molmo.device} in {time.monotonic() - started:.0f} s", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, {"ok": True, "model": molmo.model_id, "device": molmo.device})
            else:
                self._send(404, {"ok": False})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/point":
                self._send(404, {"ok": False})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                started = time.monotonic()
                text, width, height = molmo.point(
                    base64.b64decode(body["image_b64"]), str(body["description"])
                )
                point = parse(text, width, height)
                self._send(
                    200,
                    {
                        "u": point[0] if point else None,
                        "v": point[1] if point else None,
                        "raw": text,
                        "seconds": round(time.monotonic() - started, 3),
                    },
                )
            except Exception as exc:  # noqa: BLE001 - told to the caller
                self._send(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, fmt: str, *args) -> None:
            print(f"{self.address_string()} {fmt % args}", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Molmo pointing server: http://{args.host}:{args.port}/point", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
