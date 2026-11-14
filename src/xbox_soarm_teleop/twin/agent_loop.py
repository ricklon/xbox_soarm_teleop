"""A lean agent loop: any OpenRouter model drives a twin through its MCP tools.

The harness is kept small and the same for every model, so differences
between runs come from the model:

- the system prompt is the twin's own MCP server instructions;
- the tools are the server's tools, offered as function tools;
- tool calls run one at a time, in the order the model gave them (a reply
  with several calls never sends two motions at once);
- each step's tokens, cost (OpenRouter reports it) and time are recorded.

It connects to the twin over MCP like any agent would, so it works against a
single twin or one arm of a workshop hub (pass that arm's key).

    run = run_agent("http://127.0.0.1:8765/mcp", "z-ai/glm-5.3-flash", "Put the red cube in the box.")
    print(run.stop_reason, run.usage["cost_usd"], run.final_text)

The OpenRouter key comes from ``OPENROUTER_API_KEY`` (environment or a
``.env`` file) and is masked in every error.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx2
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from xbox_soarm_teleop.twin.pointing import (
    OPENROUTER_URL,
    PointerError,
    _post_json,
    load_secret,
    mask_secrets,
)

DEFAULT_MAX_STEPS = 60
DEFAULT_MAX_COST_USD = 0.50
DEFAULT_TIMEOUT_S = 900.0
MAX_TOOL_RESULT_CHARS = 12000  # a camera image's base64 never goes to a text model
TOOL_TIMEOUT_S = 60.0


class AgentError(RuntimeError):
    """The run could not start or the model could not be asked."""


@dataclass
class AgentRun:
    """What happened in one run."""

    model: str
    task: str
    stop_reason: str = ""
    """``answered`` (the model stopped calling tools), ``max_steps``, ``max_cost``,
    ``timeout`` or ``error``."""
    final_text: str = ""
    error: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    time: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "task": self.task,
            "stop_reason": self.stop_reason,
            "final_text": self.final_text,
            "error": self.error,
            "usage": self.usage,
            "time": self.time,
            "tool_calls": self.tool_calls,
            "steps": self.steps,
        }


def mcp_tool_to_function(tool) -> dict[str, Any]:
    """An MCP tool as an OpenAI-style function tool."""
    schema = dict(getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {})
    properties = {k: v for k, v in (schema.get("properties") or {}).items() if k != "ctx"}
    parameters = {"type": "object", "properties": properties}
    required = [name for name in schema.get("required") or [] if name in properties]
    if required:
        parameters["required"] = required
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": (tool.description or "")[:1024],
            "parameters": parameters,
        },
    }


def _result_text(result) -> str:
    """A tool result as text for the model. Pictures are described, not sent."""
    parts = []
    for item in getattr(result, "content", None) or []:
        kind = getattr(item, "type", "")
        if kind == "text":
            parts.append(item.text)
        elif kind == "image":
            parts.append(
                json.dumps(
                    {
                        "image": getattr(item, "mime_type", None) or getattr(item, "mimeType", "image"),
                        "note": "a picture was taken; this text-only harness does not show it to "
                        "the model. Use locate to find objects.",
                    }
                )
            )
    text = "\n".join(parts) or "{}"
    return text[:MAX_TOOL_RESULT_CHARS]


async def _call_tool(session: ClientSession, name: str, arguments: dict) -> tuple[bool, str]:
    try:
        result = await asyncio.wait_for(session.call_tool(name, arguments), TOOL_TIMEOUT_S)
    except asyncio.TimeoutError:
        return False, json.dumps({"ok": False, "reason": "timeout", "detail": f"{name} took too long"})
    except Exception as exc:  # noqa: BLE001 - told to the model
        return False, json.dumps({"ok": False, "reason": "error", "detail": str(exc)[:300]})
    text = _result_text(result)
    if getattr(result, "is_error", False):
        return False, text
    try:
        ok = json.loads(text).get("ok", True) is not False
    except (json.JSONDecodeError, AttributeError):
        ok = True
    return ok, text


async def _run(
    url: str,
    model: str,
    task: str,
    *,
    post: Callable[..., dict],
    arm_key: str | None,
    max_steps: int,
    max_cost_usd: float,
    timeout_s: float,
    on_step: Callable[[dict], None] | None,
) -> AgentRun:
    key = load_secret("OPENROUTER_API_KEY")
    if not key:
        raise AgentError("OPENROUTER_API_KEY is not set (environment or a .env file)")
    run = AgentRun(model=model, task=task)
    totals = {"steps": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    model_s = tools_s = 0.0
    started = time.monotonic()
    headers = {"Authorization": f"Bearer {arm_key}"} if arm_key else None
    async with httpx2.AsyncClient(headers=headers, timeout=60) as http_client:
        async with streamable_http_client(url, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                init = await session.initialize()
                tools = [mcp_tool_to_function(tool) for tool in (await session.list_tools()).tools]
                known = {tool["function"]["name"] for tool in tools}
                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": getattr(init, "instructions", None) or ""},
                    {"role": "user", "content": task},
                ]
                while True:
                    if totals["steps"] >= max_steps:
                        run.stop_reason = "max_steps"
                        break
                    if totals["cost_usd"] is not None and totals["cost_usd"] >= max_cost_usd:
                        run.stop_reason = "max_cost"
                        break
                    if time.monotonic() - started > timeout_s:
                        run.stop_reason = "timeout"
                        break
                    body = {
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "auto",
                        "usage": {"include": True},
                    }
                    asked = time.monotonic()
                    try:
                        reply = await asyncio.to_thread(
                            post, OPENROUTER_URL, body, headers={"Authorization": f"Bearer {key}"}
                        )
                    except PointerError as exc:
                        run.stop_reason, run.error = "error", mask_secrets(str(exc), (key,))
                        break
                    step_s = time.monotonic() - asked
                    model_s += step_s
                    usage = reply.get("usage") or {}
                    totals["steps"] += 1
                    totals["input_tokens"] += int(usage.get("prompt_tokens") or 0)
                    totals["output_tokens"] += int(usage.get("completion_tokens") or 0)
                    if usage.get("cost") is None:
                        totals["cost_usd"] = None
                    elif totals["cost_usd"] is not None:
                        totals["cost_usd"] += float(usage["cost"])
                    try:
                        message = reply["choices"][0]["message"]
                    except (KeyError, IndexError, TypeError):
                        detail = (
                            (reply.get("error") or {}).get("message")
                            if isinstance(reply, dict)
                            else None
                        )
                        run.stop_reason = "error"
                        run.error = mask_secrets(
                            f"unexpected reply: {detail or str(reply)[:300]}", (key,)
                        )
                        break
                    calls = message.get("tool_calls") or []
                    text = (message.get("content") or "").strip()
                    step = {
                        "step": totals["steps"],
                        "model_s": round(step_s, 3),
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                        "cost_usd": usage.get("cost"),
                        "text": text[:2000],
                        "calls": [],
                    }
                    assistant = {"role": "assistant", "content": message.get("content") or ""}
                    if calls:
                        assistant["tool_calls"] = calls
                    messages.append(assistant)
                    if not calls:
                        run.stop_reason, run.final_text = "answered", text
                        run.steps.append(step)
                        if on_step:
                            on_step(step)
                        break
                    for call in calls:  # one at a time, in the model's order
                        function = call.get("function") or {}
                        name = function.get("name", "")
                        began = time.monotonic()
                        try:
                            arguments = json.loads(function.get("arguments") or "{}")
                            if not isinstance(arguments, dict):
                                raise ValueError("arguments must be a JSON object")
                        except (json.JSONDecodeError, ValueError) as exc:
                            ok = False
                            content = json.dumps(
                                {
                                    "ok": False,
                                    "reason": "invalid_request",
                                    "detail": f"arguments are not valid JSON: {exc}",
                                }
                            )
                        else:
                            if name not in known:
                                ok = False
                                content = json.dumps(
                                    {
                                        "ok": False,
                                        "reason": "invalid_request",
                                        "detail": f"no tool named {name!r}",
                                    }
                                )
                            else:
                                ok, content = await _call_tool(session, name, arguments)
                        call_s = time.monotonic() - began
                        tools_s += call_s
                        record = {
                            "step": totals["steps"],
                            "name": name,
                            "ok": ok,
                            "seconds": round(call_s, 3),
                        }
                        run.tool_calls.append(record)
                        step["calls"].append(
                            {**record, "arguments": function.get("arguments", "")[:500]}
                        )
                        messages.append(
                            {"role": "tool", "tool_call_id": call.get("id", ""), "content": content}
                        )
                    run.steps.append(step)
                    if on_step:
                        on_step(step)
    total_s = time.monotonic() - started
    run.usage = totals
    run.time = {
        "total_s": round(total_s, 1),
        "model_s": round(model_s, 1),
        "tools_s": round(tools_s, 1),
        "other_s": round(total_s - model_s - tools_s, 1),
    }
    return run


def run_agent(
    url: str,
    model: str,
    task: str,
    *,
    post: Callable[..., dict] = _post_json,
    arm_key: str | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_cost_usd: float = DEFAULT_MAX_COST_USD,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    on_step: Callable[[dict], None] | None = None,
) -> AgentRun:
    """Run one task with one OpenRouter model against the twin at ``url``.

    Args:
        url: The twin's MCP endpoint.
        model: An OpenRouter model id (``z-ai/glm-5.3-flash``); a leading
            ``openrouter/`` is dropped.
        task: What to do, as the user would say it.
        post: Sends a request to OpenRouter (tests pass a fake).
        arm_key: A hub arm's key, sent as a bearer token to the twin.
        max_steps: Model replies at most.
        max_cost_usd: Stop once the run has cost this much.
        timeout_s: Stop once the run has taken this long.
        on_step: Called with each step as it finishes (for live display).

    Raises:
        AgentError: no OpenRouter key.
    """
    if model.startswith("openrouter/"):
        model = model[len("openrouter/") :]
    return asyncio.run(
        _run(
            url,
            model,
            task,
            post=post,
            arm_key=arm_key,
            max_steps=max_steps,
            max_cost_usd=max_cost_usd,
            timeout_s=timeout_s,
            on_step=on_step,
        )
    )
