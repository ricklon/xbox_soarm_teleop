"""Pointing models: "where is the red cube?" in, a pixel out.

The twin's ``locate`` tool asks a pointer for the pixel that shows an object
and turns it into xyz (``twin.projection``). Backends:

- ``oracle``: the true position projected into the image. No model; the best
  any pointer could do, for testing the rest of the pipeline.
- ``fixed``: always the same pixel (tests).
- ``gemini``: Gemini Robotics-ER through the Gemini API (``GEMINI_API_KEY``).
- ``openrouter``: any vision model on OpenRouter (``OPENROUTER_API_KEY``),
  asked for the same JSON as Gemini: general Gemini and Gemma 4 models
  (OpenRouter has no Robotics-ER or Molmo2-ER).
- ``ollama``: a vision model served by a local Ollama (a baseline: general
  vision models are not trained to point).
- ``http``: any server that takes ``{"image_b64", "description", "width",
  "height"}`` and returns ``{"u", "v"}`` in pixels (``scripts/molmo_server.py``
  serves Molmo2-ER this way).

Keys are read from the environment, else from a ``.env`` file: the current
folder's, the project's (see ``.env.example``), then ``~/.env``.
They are never logged or returned: errors pass through :func:`mask_secrets`.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

HTTP_TIMEOUT_S = 120.0
RATE_LIMIT_RETRIES = 4  # HTTP 429: wait as the API asks, then try again
RATE_LIMIT_MAX_WAIT_S = 60.0
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_DEFAULT_MODEL = "gemini-robotics-er-2-preview"
OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_DEFAULT_MODEL = "gemma4:12b"
MOLMO_DEFAULT_URL = "http://127.0.0.1:8890"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_DEFAULT_MODEL = "google/gemma-4-31b-it:free"

POINT_PROMPT = (
    "Point to the {description}. Give exactly one point, on the middle of it. "
    'Answer only with JSON in this format: [{{"point": [y, x], "label": "{description}"}}], '
    "where the point is in [y, x] order, normalized to 0-1000."
)

_SECRET_NAMES = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "HF_TOKEN",
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_ACCOUNT_ID",
)


class PointerError(RuntimeError):
    """The pointing model could not be asked, or its answer made no sense."""


@dataclass(frozen=True)
class PointRequest:
    """What a pointer is asked.

    Attributes:
        image: Encoded image (PNG or JPEG bytes).
        mime: The image's media type.
        width: Image width (pixels).
        height: Image height (pixels).
        description: What to point at, in words ("red cube").
        truth_uv: The pixel the object's true center projects to. Only the
            oracle reads it; real backends must not.
    """

    image: bytes
    mime: str
    width: int
    height: int
    description: str
    truth_uv: tuple[float, float] | None = None


@dataclass(frozen=True)
class Point:
    """A pixel (u right, v down, from the image's top-left corner)."""

    u: float
    v: float
    raw: str = ""
    seconds: float | None = None


class Pointer(Protocol):
    name: str
    last_usage: dict | None
    """Tokens and cost of the last call (``input_tokens``, ``output_tokens``,
    ``cost_usd``; None where not known). None for pointers that are not models."""

    def point(self, request: PointRequest) -> Point | None:
        """The pixel showing ``request.description``, or None if it is not seen."""


def _usage(input_tokens, output_tokens, cost_usd) -> dict:
    def as_int(value):
        return int(value) if isinstance(value, (int, float)) else None

    return {
        "input_tokens": as_int(input_tokens),
        "output_tokens": as_int(output_tokens),
        "cost_usd": float(cost_usd) if isinstance(cost_usd, (int, float)) else None,
    }


def add_usage(total: dict | None, usage: dict | None) -> dict | None:
    """Add one call's usage to a running total. An unknown cost makes the total's cost unknown."""
    if usage is None:
        return total
    if total is None:
        total = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "calls": 0}
    out = dict(total)
    for key in ("input_tokens", "output_tokens"):
        if usage.get(key) is not None and out[key] is not None:
            out[key] += usage[key]
    if usage.get("cost_usd") is None or out["cost_usd"] is None:
        out["cost_usd"] = None
    else:
        out["cost_usd"] += usage["cost_usd"]
    out["calls"] += 1
    return out


# ------------------------------------------------------------------ secrets
# The checkout this package runs from (an editable install), whose .env is read.
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def env_files() -> list[Path]:
    """The ``.env`` files keys are read from, first match wins.

    The current folder's, then the project's (copy ``.env.example`` there),
    then the one in your home folder.
    """
    files: list[Path] = []
    for path in (Path.cwd() / ".env", PROJECT_ROOT / ".env", Path.home() / ".env"):
        if path not in files:
            files.append(path)
    return files


def _read_env_value(path: Path, name: str) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'") or None
    return None


def load_secret(name: str, env_file: str | Path | None = None) -> str | None:
    """A key from the environment, else from a ``.env`` file (``NAME=value`` lines).

    ``env_file`` reads that one file instead of :func:`env_files`. An empty
    value (a template copied but not filled in) counts as not set.
    """
    value = os.environ.get(name)
    if value and value.strip():
        return value.strip()
    for path in [Path(env_file)] if env_file is not None else env_files():
        value = _read_env_value(path, name)
        if value:
            return value
    return None


def mask_secrets(text: str, secrets: tuple[str | None, ...] | None = None) -> str:
    """``text`` with any known key replaced by ``***``."""
    if secrets is None:
        secrets = tuple(load_secret(name) for name in _SECRET_NAMES)
    for secret in secrets:
        if secret and len(secret) >= 6:
            text = text.replace(secret, "***")
    return re.sub(r"(key=)[A-Za-z0-9_\-]{12,}", r"\1***", text)


# ------------------------------------------------------------------ parsing
def _strip_fences(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    return match.group(1) if match else text


def parse_yx_1000(text: str, width: int, height: int) -> Point | None:
    """Parse ``[{"point": [y, x], ...}]`` (0-1000) as Gemini Robotics-ER answers."""
    body = _strip_fences(text).strip()
    candidates: list[Any] = []
    try:
        candidates = [json.loads(body)]
    except json.JSONDecodeError:
        for match in re.finditer(r"\[\s*\{.*?\}\s*\]|\{[^{}]*\"point\"[^{}]*\}", body, re.S):
            try:
                candidates.append(json.loads(match.group(0)))
            except json.JSONDecodeError:
                continue
    for value in candidates:
        items = value if isinstance(value, list) else [value]
        for item in items:
            point = item.get("point") if isinstance(item, dict) else None
            if (
                isinstance(point, (list, tuple))
                and len(point) == 2
                and all(isinstance(c, (int, float)) for c in point)
            ):
                y, x = (float(c) for c in point)
                return Point(x / 1000.0 * width, y / 1000.0 * height, raw=text)
    return None


_MOLMO_COORDS = re.compile(r"<(?:points|tracks)[^>]*?coords=\"([0-9\t:;, .]+)\"")
# Point index, then x and y written with 3 or 4 digits (0-1000), as Molmo2 parses them.
_MOLMO_TRIPLE = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")
_MOLMO1_POINT = re.compile(r"<point[^>]*?x=\"([0-9.]+)\"[^>]*?y=\"([0-9.]+)\"")


def parse_molmo(text: str, width: int, height: int) -> Point | None:
    """Parse a Molmo2 pointing answer (``<points coords="1 x y"/>``, 0-1000).

    Molmo (1) answers ``<point x="52.3" y="41.2">`` in percent; that is read too.
    """
    match = _MOLMO_COORDS.search(text)
    if match:
        coords = match.group(1)
        # Multi-image answers prefix each image's points with its index and a tab.
        last = re.split(r"[\t;]", coords.strip())
        for chunk in last:
            triple = _MOLMO_TRIPLE.search(chunk.strip())
            if triple:
                x, y = float(triple.group(2)), float(triple.group(3))
                return Point(x / 1000.0 * width, y / 1000.0 * height, raw=text)
    match = _MOLMO1_POINT.search(text)
    if match:
        x, y = float(match.group(1)), float(match.group(2))
        return Point(x / 100.0 * width, y / 100.0 * height, raw=text)
    return None


# ------------------------------------------------------------------ HTTP
def _retry_delay(exc: urllib.error.HTTPError, body: str, attempt: int) -> float:
    """How long a rate-limited API asks us to wait (s): its own hint, else backoff."""
    header = exc.headers.get("Retry-After") if exc.headers is not None else None
    if header:
        try:
            return min(float(header), RATE_LIMIT_MAX_WAIT_S)
        except ValueError:
            pass
    match = re.search(r'"retryDelay"\s*:\s*"([0-9.]+)s"', body)
    if match:
        return min(float(match.group(1)), RATE_LIMIT_MAX_WAIT_S)
    return min(5.0 * 2**attempt, RATE_LIMIT_MAX_WAIT_S)


def _post_json(url: str, body: dict, headers: dict[str, str] | None = None, timeout=HTTP_TIMEOUT_S):
    """POST JSON, return the JSON reply. Rate limits (HTTP 429) are waited out a few times."""
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", **(headers or {})}
    )
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            if exc.code == 429 and attempt < RATE_LIMIT_RETRIES:
                time.sleep(_retry_delay(exc, detail, attempt))
                continue
            raise PointerError(mask_secrets(f"HTTP {exc.code} from {url}: {detail[:500]}")) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PointerError(mask_secrets(f"cannot reach {url}: {exc}")) from None
    raise AssertionError("unreachable")


# ------------------------------------------------------------------ backends
@dataclass
class OraclePointer:
    """Points at the true center: the best a pointer could do."""

    name: str = "oracle"
    last_usage: dict | None = None

    def point(self, request: PointRequest) -> Point | None:
        if request.truth_uv is None:
            return None
        u, v = request.truth_uv
        if not (0 <= u < request.width and 0 <= v < request.height):
            return None
        return Point(float(u), float(v), raw="oracle", seconds=0.0)


@dataclass
class FixedPointer:
    """Always answers the same pixel, or what ``answer(request)`` returns (tests)."""

    u: float = 0.0
    v: float = 0.0
    answer: Callable[[PointRequest], Point | None] | None = None
    name: str = "fixed"
    calls: list[PointRequest] = field(default_factory=list)
    last_usage: dict | None = None

    def point(self, request: PointRequest) -> Point | None:
        self.calls.append(request)
        if self.answer is not None:
            return self.answer(request)
        return Point(self.u, self.v, raw="fixed")


@dataclass
class GeminiPointer:
    """Gemini Robotics-ER via the Gemini API. Needs ``GEMINI_API_KEY``."""

    model: str = GEMINI_DEFAULT_MODEL
    key_name: str = "GEMINI_API_KEY"
    thinking_budget: int = 0
    post: Callable[..., dict] = _post_json
    name: str = "gemini"
    last_usage: dict | None = None

    def point(self, request: PointRequest) -> Point | None:
        key = load_secret(self.key_name)
        if not key:
            raise PointerError(f"{self.key_name} is not set (environment or a .env file)")
        body = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": request.mime,
                                "data": base64.b64encode(request.image).decode("ascii"),
                            }
                        },
                        {"text": POINT_PROMPT.format(description=request.description)},
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.5,
                "thinkingConfig": {"thinkingBudget": self.thinking_budget},
            },
        }
        reply = self.post(GEMINI_URL.format(model=self.model), body, headers={"x-goog-api-key": key})
        meta = (reply.get("usageMetadata") or {}) if isinstance(reply, dict) else {}
        thinking = meta.get("thoughtsTokenCount") or 0
        out = (meta.get("candidatesTokenCount") or 0) + thinking if meta else None
        # The Gemini API reports tokens, not money: no price is recorded for it.
        self.last_usage = _usage(meta.get("promptTokenCount"), out, None)
        try:
            text = "".join(part.get("text", "") for part in reply["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError):
            raise PointerError(mask_secrets(f"unexpected Gemini reply: {str(reply)[:300]}")) from None
        return parse_yx_1000(text, request.width, request.height)


@dataclass
class OllamaPointer:
    """A vision model on a local Ollama, asked for the same JSON as Gemini."""

    model: str = OLLAMA_DEFAULT_MODEL
    url: str = OLLAMA_URL
    post: Callable[..., dict] = _post_json
    name: str = "ollama"
    last_usage: dict | None = None

    def point(self, request: PointRequest) -> Point | None:
        reply = self.post(
            f"{self.url.rstrip('/')}/api/chat",
            {
                "model": self.model,
                "stream": False,
                "think": False,  # a thinking model otherwise writes for minutes first
                "options": {"temperature": 0, "num_predict": 200},
                "messages": [
                    {
                        "role": "user",
                        "content": POINT_PROMPT.format(description=request.description),
                        "images": [base64.b64encode(request.image).decode("ascii")],
                    }
                ],
            },
        )
        self.last_usage = _usage(reply.get("prompt_eval_count"), reply.get("eval_count"), 0.0)
        text = (reply.get("message") or {}).get("content", "")
        return parse_yx_1000(text, request.width, request.height)


@dataclass
class OpenRouterPointer:
    """A vision model on OpenRouter (OpenAI-style chat). Needs ``OPENROUTER_API_KEY``."""

    model: str = OPENROUTER_DEFAULT_MODEL
    key_name: str = "OPENROUTER_API_KEY"
    post: Callable[..., dict] = _post_json
    name: str = "openrouter"
    last_usage: dict | None = None

    def point(self, request: PointRequest) -> Point | None:
        key = load_secret(self.key_name)
        if not key:
            raise PointerError(f"{self.key_name} is not set (environment or a .env file)")
        image = base64.b64encode(request.image).decode("ascii")
        reply = self.post(
            OPENROUTER_URL,
            {
                "model": self.model,
                "temperature": 0,
                # Thinking models spend tokens before answering: 300 cut the JSON off.
                "max_tokens": 2048,
                "reasoning": {"effort": "low"},
                "usage": {"include": True},  # the reply then says what the call cost
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": POINT_PROMPT.format(description=request.description),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{request.mime};base64,{image}"},
                            },
                        ],
                    }
                ],
            },
            headers={"Authorization": f"Bearer {key}"},
        )
        usage = (reply.get("usage") or {}) if isinstance(reply, dict) else {}
        self.last_usage = _usage(
            usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("cost")
        )
        try:
            text = reply["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            detail = (reply.get("error") or {}).get("message") if isinstance(reply, dict) else None
            raise PointerError(
                mask_secrets(f"unexpected OpenRouter reply: {detail or str(reply)[:300]}", (key,))
            ) from None
        return parse_yx_1000(text, request.width, request.height)


@dataclass
class HttpPointer:
    """A pointing server of our own (``scripts/molmo_server.py``)."""

    url: str = MOLMO_DEFAULT_URL
    name: str = "molmo"
    post: Callable[..., dict] = _post_json
    last_usage: dict | None = None

    def point(self, request: PointRequest) -> Point | None:
        reply = self.post(
            f"{self.url.rstrip('/')}/point",
            {
                "image_b64": base64.b64encode(request.image).decode("ascii"),
                "mime": request.mime,
                "description": request.description,
                "width": request.width,
                "height": request.height,
            },
        )
        self.last_usage = _usage(None, None, 0.0)  # runs on this computer
        if reply.get("u") is None or reply.get("v") is None:
            return None
        return Point(float(reply["u"]), float(reply["v"]), raw=str(reply.get("raw", ""))[:500])


BACKENDS = ("oracle", "gemini", "molmo", "ollama", "openrouter")


def make_pointer(spec: str | None) -> Pointer | None:
    """A pointer from a command-line spec: ``NAME`` or ``NAME:OPTION``.

    ``oracle``; ``gemini[:model]``; ``ollama[:model]``; ``openrouter[:model]``; ``molmo[:url]``;
    ``http:URL``; ``none`` (or empty) for no pointer.
    """
    if not spec or spec == "none":
        return None
    name, _, option = spec.partition(":")
    if name == "oracle":
        return OraclePointer()
    if name == "gemini":
        return GeminiPointer(model=option or GEMINI_DEFAULT_MODEL)
    if name == "ollama":
        return OllamaPointer(model=option or OLLAMA_DEFAULT_MODEL)
    if name == "openrouter":
        return OpenRouterPointer(model=option or OPENROUTER_DEFAULT_MODEL)
    if name == "molmo":
        return HttpPointer(url=option or MOLMO_DEFAULT_URL, name="molmo")
    if name == "http":
        if not option:
            raise ValueError("http needs a URL: http:http://host:port")
        return HttpPointer(url=option, name="http")
    raise ValueError(f"Unknown pointer '{spec}'. Known: {', '.join(BACKENDS)}, http:URL, none")


def pixel_distance(a: Point, uv) -> float:
    """Pixels between a pointer's answer and an image point."""
    return float(np.hypot(a.u - uv[0], a.v - uv[1]))
