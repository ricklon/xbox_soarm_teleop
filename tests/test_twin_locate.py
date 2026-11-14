"""``locate``: a pointing model's pixel turned into xyz, and scored against the truth."""

import asyncio
import json
import logging
import socket
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from xbox_soarm_teleop.twin import build_twin, pointing
from xbox_soarm_teleop.twin.mcp_server import McpThread, build_mcp_server
from xbox_soarm_teleop.twin.pointing import (
    FixedPointer,
    GeminiPointer,
    HttpPointer,
    OpenRouterPointer,
    OraclePointer,
    Point,
    PointerError,
    PointRequest,
    load_secret,
    make_pointer,
    mask_secrets,
    parse_molmo,
    parse_yx_1000,
)

warnings.filterwarnings("ignore")
logging.getLogger("xbox_soarm_teleop").setLevel(logging.ERROR)

FAKE_KEY = "AIzaTESTKEY0123456789abcdef"


# ------------------------------------------------------------------ parsing
def test_parse_gemini_style_points():
    assert parse_yx_1000('[{"point": [500, 250], "label": "red cube"}]', 640, 480) == Point(
        160.0, 240.0, raw='[{"point": [500, 250], "label": "red cube"}]'
    )
    fenced = 'Here:\n```json\n[{"point": [100, 900], "label": "cube"}]\n```'
    point = parse_yx_1000(fenced, 1000, 1000)
    assert (point.u, point.v) == (900.0, 100.0)
    loose = 'I think it is {"point": [10, 20], "label": "x"} there.'
    assert (parse_yx_1000(loose, 1000, 1000).u, parse_yx_1000(loose, 1000, 1000).v) == (20.0, 10.0)
    assert parse_yx_1000("I cannot see it.", 640, 480) is None
    assert parse_yx_1000("[]", 640, 480) is None


def test_parse_molmo_points():
    point = parse_molmo('<points coords="1 523 412">red cube</points>', 1000, 1000)
    assert (point.u, point.v) == (523.0, 412.0)
    point = parse_molmo('<points alt="red cube" coords="1 1 250 500"/>', 640, 480)
    assert (point.u, point.v) == (160.0, 240.0)
    point = parse_molmo('<point x="50.0" y="25.0" alt="cube">cube</point>', 640, 480)
    assert (point.u, point.v) == (320.0, 120.0)
    assert parse_molmo("There is no cube.", 640, 480) is None


# ------------------------------------------------------------------ secrets
def test_keys_come_from_the_environment_or_dot_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(f'# comment\nexport GEMINI_API_KEY="{FAKE_KEY}"\nOTHER=1\n')
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert load_secret("GEMINI_API_KEY", env) == FAKE_KEY
    assert load_secret("MISSING_KEY", env) is None
    monkeypatch.setenv("GEMINI_API_KEY", "from-env-123456")
    assert load_secret("GEMINI_API_KEY", env) == "from-env-123456"


def _isolate_env_files(monkeypatch, tmp_path):
    """No real .env file is read: work, project and home folders are all temporary."""
    work, project, home = tmp_path / "work", tmp_path / "project", tmp_path / "home"
    for folder in (work, project, home):
        folder.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(pointing, "PROJECT_ROOT", project)
    monkeypatch.setenv("HOME", str(home))
    return work, project, home


def test_project_dot_env_comes_before_the_home_one(tmp_path, monkeypatch):
    work, project, home = _isolate_env_files(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    (home / ".env").write_text("OPENROUTER_API_KEY=from-home-123456\n")
    assert load_secret("OPENROUTER_API_KEY") == "from-home-123456"
    (project / ".env").write_text("OPENROUTER_API_KEY=from-project-123456\n")
    assert load_secret("OPENROUTER_API_KEY") == "from-project-123456"
    (work / ".env").write_text("OPENROUTER_API_KEY=from-work-123456\n")
    assert load_secret("OPENROUTER_API_KEY") == "from-work-123456"
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env-123456")
    assert load_secret("OPENROUTER_API_KEY") == "from-env-123456"


def test_an_empty_value_in_a_dot_env_does_not_hide_a_later_one(tmp_path, monkeypatch):
    work, project, home = _isolate_env_files(monkeypatch, tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    (project / ".env").write_text("GEMINI_API_KEY=\n")  # copied from .env.example, not filled in
    (home / ".env").write_text("GEMINI_API_KEY=from-home-123456\n")
    assert load_secret("GEMINI_API_KEY") == "from-home-123456"


def test_the_template_lists_every_key_with_no_values():
    template = Path(__file__).parent.parent / ".env.example"
    entries = {}
    for line in template.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, _, value = line.partition("=")
            entries[key.strip()] = value.strip()
    for name in (
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "CLOUDFLARE_API_TOKEN",
        "CLOUDFLARE_ACCOUNT_ID",
    ):
        assert name in entries, name
    assert all(value == "" for value in entries.values())  # never a real key in git


def test_secrets_are_masked():
    text = f"HTTP 400 for https://x/y?key={FAKE_KEY}: bad key {FAKE_KEY}"
    masked = mask_secrets(text, (FAKE_KEY,))
    assert FAKE_KEY not in masked and "***" in masked


# ------------------------------------------------------------------ backends
def _request(**kwargs) -> PointRequest:
    base = dict(image=b"\x89PNG", mime="image/png", width=640, height=480, description="red cube")
    return PointRequest(**{**base, **kwargs})


def test_gemini_sends_the_key_in_a_header_and_reads_the_point(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    sent = {}

    def post(url, body, headers=None, timeout=None):
        sent.update(url=url, body=body, headers=headers)
        text = '```json\n[{"point": [500, 500], "label": "red cube"}]\n```'
        return {"candidates": [{"content": {"parts": [{"text": text}]}}]}

    point = GeminiPointer(model="gemini-robotics-er-test", post=post).point(_request())
    assert (point.u, point.v) == (320.0, 240.0)
    assert sent["headers"] == {"x-goog-api-key": FAKE_KEY}
    assert FAKE_KEY not in sent["url"]  # never in the URL, where logs keep it
    assert "gemini-robotics-er-test" in sent["url"]
    parts = sent["body"]["contents"][0]["parts"]
    assert parts[0]["inline_data"]["mime_type"] == "image/png"
    assert "red cube" in parts[1]["text"]


def test_openrouter_sends_a_bearer_key_and_the_image(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    sent = {}

    def post(url, body, headers=None, timeout=None):
        sent.update(url=url, body=body, headers=headers)
        return {"choices": [{"message": {"content": '[{"point": [250, 750], "label": "red cube"}]'}}]}

    point = OpenRouterPointer(model="google/gemma-4-31b-it:free", post=post).point(_request())
    assert (point.u, point.v) == (480.0, 120.0)
    assert sent["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert sent["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in sent["url"]
    assert sent["body"]["model"] == "google/gemma-4-31b-it:free"
    content = sent["body"]["messages"][0]["content"]
    assert content[0]["type"] == "text" and "red cube" in content[0]["text"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_openrouter_errors_say_why_without_the_key(monkeypatch, tmp_path):
    _isolate_env_files(monkeypatch, tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(PointerError, match="OPENROUTER_API_KEY"):
        OpenRouterPointer(post=lambda *a, **k: {}).point(_request())
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    with pytest.raises(PointerError) as info:
        OpenRouterPointer(post=lambda *a, **k: {"error": {"message": f"bad key {FAKE_KEY}"}}).point(
            _request()
        )
    assert FAKE_KEY not in str(info.value)


class _FakeResponse:
    def __init__(self, body: dict):
        self.body = json.dumps(body).encode()

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code: int, body: dict, headers: dict | None = None):
    import email.message
    import io
    import urllib.error

    message = email.message.Message()
    for name, value in (headers or {}).items():
        message[name] = value
    return urllib.error.HTTPError(
        "http://x", code, "err", message, io.BytesIO(json.dumps(body).encode())
    )


def test_rate_limits_are_waited_out(monkeypatch):
    replies = [
        _http_error(429, {"error": {"details": [{"retryDelay": "7s"}]}}),
        _http_error(429, {"error": {}}, {"Retry-After": "3"}),
        _FakeResponse({"ok": 1}),
    ]
    waits = []
    monkeypatch.setattr(pointing.time, "sleep", waits.append)

    def urlopen(request, timeout=None):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(pointing.urllib.request, "urlopen", urlopen)
    assert pointing._post_json("http://x", {}) == {"ok": 1}
    assert waits == [7.0, 3.0]


def test_rate_limits_give_up_in_the_end(monkeypatch):
    monkeypatch.setattr(pointing.time, "sleep", lambda s: None)

    def urlopen(request, timeout=None):
        raise _http_error(429, {"error": {"message": "quota"}})

    monkeypatch.setattr(pointing.urllib.request, "urlopen", urlopen)
    with pytest.raises(PointerError, match="HTTP 429"):
        pointing._post_json("http://x", {})


def test_openrouter_leaves_room_for_reasoning(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    sent = {}

    def post(url, body, headers=None, timeout=None):
        sent.update(body)
        return {"choices": [{"message": {"content": '[{"point": [1, 2]}]'}}]}

    OpenRouterPointer(post=post).point(_request())
    assert sent["max_tokens"] >= 2000
    assert sent["reasoning"] == {"effort": "low"}


def test_gemini_without_a_key_says_so(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    _isolate_env_files(monkeypatch, tmp_path)
    with pytest.raises(PointerError, match="GEMINI_API_KEY"):
        GeminiPointer(post=lambda *a, **k: {}).point(_request())


def test_http_pointer_and_oracle():
    def post(url, body, headers=None, timeout=None):
        assert url.endswith("/point") and body["description"] == "red cube"
        return {"u": 12.5, "v": 30.0, "raw": "<points/>"}

    assert HttpPointer(url="http://x", post=post).point(_request()) == Point(
        12.5, 30.0, raw="<points/>"
    )
    assert OraclePointer().point(_request(truth_uv=(5.0, 6.0))).u == 5.0
    assert OraclePointer().point(_request(truth_uv=(-5.0, 6.0))) is None  # out of the picture


def test_make_pointer_specs():
    assert make_pointer(None) is None and make_pointer("none") is None
    assert make_pointer("oracle").name == "oracle"
    assert make_pointer("gemini:gemini-x").model == "gemini-x"
    assert make_pointer("molmo").url.startswith("http://127.0.0.1")
    assert make_pointer("ollama:gemma4:12b").model == "gemma4:12b"
    assert make_pointer("openrouter:google/gemini-3.5-flash").model == "google/gemini-3.5-flash"
    assert make_pointer("openrouter").name == "openrouter"
    with pytest.raises(ValueError):
        make_pointer("crystal_ball")


# ------------------------------------------------------------------ locate
def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def running():
    """A no-coordinates twin ticking on its own thread, with an MCP server."""
    service = build_twin(scene="pick_place_basic", with_controller=False, hide_positions=True)
    alive = threading.Event()
    alive.set()

    def loop():
        while alive.is_set():
            start = time.monotonic()
            service.tick()
            time.sleep(max(0.0, service.dt - (time.monotonic() - start)))
        service.close()

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    mcp = McpThread(build_mcp_server(service), port=_free_port())
    mcp.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", mcp.port)) == 0:
                break
        time.sleep(0.05)
    yield service, mcp.url
    alive.clear()
    thread.join(timeout=5)


@pytest.mark.parametrize("view", ["overhead", "front", "side"])
def test_locate_with_a_perfect_pointer_finds_the_object(running, view):
    service, _ = running
    service.pointer = OraclePointer()
    before = len(service.estimate_scores())
    result = service.locate("pick_cube", view=view, client_id="t")
    assert result["ok"], result
    truth = service.grasp.position("pick_cube")
    assert np.linalg.norm(np.array(result["xyz_m"]) - truth) < 0.002
    assert result["description"] == "red cube" and result["backend"] == "oracle"
    assert result["image_size"] == [640, 480]
    # The move that puts the jaw center on the estimate, from the arm's own pose.
    jaw = np.array(service.observe()["jaw_center_xyz_m"])
    move = result["jaw_move_m"]
    floor = service.workspace_bounds["z"][0] + 0.015
    assert [move["dx"], move["dy"]] == pytest.approx(
        list(np.array(result["xyz_m"][:2]) - jaw[:2]), abs=1e-3
    )
    assert move["dz"] == pytest.approx(max(result["xyz_m"][2], floor) - jaw[2], abs=1e-3)
    scores = service.estimate_scores()
    assert len(scores) == before + 1
    assert scores[-1]["source"] == f"locate:oracle:{view}"
    assert scores[-1]["error_cm"] < 0.2


def test_locate_scores_a_wrong_pixel_without_saying_so(running):
    service, _ = running
    service.pointer = FixedPointer(u=20.0, v=20.0)
    result = service.locate("pick_cube", view="overhead")
    assert result["ok"] and "error" not in json.dumps(result)
    assert service.estimate_scores()[-1]["error_cm"] > 10.0
    request = service.pointer.calls[-1]
    assert request.image[:4] == b"\x89PNG" and (request.width, request.height) == (640, 480)


def test_locate_reports_misses_and_problems(running, monkeypatch):
    service, _ = running
    service.pointer = None
    assert service.locate("pick_cube")["reason"] == "no_pointer"
    service.pointer = FixedPointer(answer=lambda request: None)
    assert service.locate("pick_cube")["reason"] == "not_found"
    unknown = service.locate("moon")
    assert unknown["reason"] == "invalid_request" and "pick_cube" in unknown["detail"]
    assert service.locate("pick_cube", view="ceiling")["reason"] == "invalid_request"

    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)

    def leaky(request):
        raise PointerError(f"HTTP 403: key {FAKE_KEY} refused")

    service.pointer = FixedPointer(answer=leaky)
    failed = service.locate("pick_cube")
    assert failed["reason"] == "pointer_error" and FAKE_KEY not in failed["detail"]


def test_locate_over_mcp(running):
    service, url = running
    service.pointer = OraclePointer()

    async def go():
        async with streamable_http_client(url) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {tool.name for tool in (await session.list_tools()).tools}
                result = await session.call_tool("locate", {"name": "pick_cube", "view": "front"})
                return tools, json.loads(result.content[0].text)

    tools, result = asyncio.run(go())
    assert "locate" in tools
    assert result["ok"] is True and len(result["xyz_m"]) == 3
    truth = service.grasp.position("pick_cube")
    assert np.linalg.norm(np.array(result["xyz_m"]) - truth) < 0.002


# ------------------------------------------------------------------ usage and cost
def test_gemini_records_its_token_use(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)

    def post(url, body, headers=None, timeout=None):
        return {
            "candidates": [{"content": {"parts": [{"text": '[{"point": [500, 500]}]'}]}}],
            "usageMetadata": {
                "promptTokenCount": 1290,
                "candidatesTokenCount": 21,
                "thoughtsTokenCount": 5,
            },
        }

    pointer = GeminiPointer(post=post)
    pointer.point(_request())
    assert pointer.last_usage == {"input_tokens": 1290, "output_tokens": 26, "cost_usd": None}


def test_openrouter_asks_for_and_records_its_cost(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)
    sent = {}

    def post(url, body, headers=None, timeout=None):
        sent.update(body)
        return {
            "choices": [{"message": {"content": "[]"}}],  # nothing found still costs
            "usage": {"prompt_tokens": 800, "completion_tokens": 40, "cost": 0.000123},
        }

    pointer = OpenRouterPointer(post=post)
    assert pointer.point(_request()) is None
    assert sent["usage"] == {"include": True}
    assert pointer.last_usage == {"input_tokens": 800, "output_tokens": 40, "cost_usd": 0.000123}


def test_local_backends_cost_nothing():
    from xbox_soarm_teleop.twin.pointing import OllamaPointer

    def post(url, body, headers=None, timeout=None):
        return {"message": {"content": "[]"}, "prompt_eval_count": 700, "eval_count": 12}

    pointer = OllamaPointer(post=post)
    pointer.point(_request())
    assert pointer.last_usage == {"input_tokens": 700, "output_tokens": 12, "cost_usd": 0.0}
    molmo = HttpPointer(post=lambda *a, **k: {"u": 1.0, "v": 2.0})
    molmo.point(_request())
    assert molmo.last_usage == {"input_tokens": None, "output_tokens": None, "cost_usd": 0.0}
    oracle = OraclePointer()
    oracle.point(_request(truth_uv=(1.0, 1.0)))
    assert oracle.last_usage is None  # not a model


def test_usage_totals_add_up():
    from xbox_soarm_teleop.twin.pointing import add_usage

    total = add_usage(None, {"input_tokens": 10, "output_tokens": 2, "cost_usd": 0.5})
    total = add_usage(total, {"input_tokens": 5, "output_tokens": None, "cost_usd": None})
    # A call whose cost is unknown makes the total unknown too, but tokens still add.
    assert total == {"input_tokens": 15, "output_tokens": 2, "cost_usd": None, "calls": 2}
