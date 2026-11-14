"""Can a text-only agent pick and place without coordinates, using ``locate``?

Starts its own twin (``--no-coordinates``, a pointing model behind
``locate``) on a spare port, points an opencode agent at it through a
throwaway config, asks it to put the object in the goal box, and then
grades the run from the twin's session log, which holds the truth. What the
agent says about its own success is not trusted.

    uv run python scripts/agent_trial.py --model opencode/iceman-free --pointer molmo
    uv run python scripts/agent_trial.py --model opencode/glm-5.3-flash-free \\
        --scene pick_place_easy --pointer molmo --runs 3
    uv run python scripts/agent_trial.py --model openrouter/z-ai/glm-5.3-flash --pointer gemini

An ``openrouter/...`` model gets OPENROUTER_API_KEY from the environment or
.env (never printed).

The throwaway config turns off any other ``soarm-arm`` server in your opencode
setup (such as a workshop hub), so the agent can reach only this twin, and
turns off opencode's shell and file tools. Start ``scripts/molmo_server.py``
first for ``--pointer molmo``.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from xbox_soarm_teleop.twin.pointing import load_secret

PROMPT = """\
You are connected to a simulated robot arm through the soarm-bench MCP tools.
Read the server instructions. Object coordinates are hidden in this twin:
observe will not tell you where objects are. Use locate to find them.

Task: put the {description} ({name}) into the green goal box, then home the
arm. Work step by step with the tools; check the result of every call. When
you are done, say in one line whether it is in the box and how you know.
"""

DESCRIPTIONS = {
    "pick_place_basic": ("pick_cube", "red cube"),
    "pick_place_easy": ("bean_bag", "blue bean bag"),
}


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_port(port: int, timeout_s: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED", "STOPPED"}


def parse_opencode_events(raw: str) -> dict:
    """Read ``opencode run --format json`` output: tokens, cost and a readable transcript.

    Each model step ends with a ``step_finish`` event that carries its tokens
    and cost (USD, as opencode prices the model). Lines that are not JSON are
    skipped.
    """
    usage = {
        "steps": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "cache_read_tokens": 0,
        "cost_usd": 0.0,
    }
    transcript: list[str] = []
    last_text = ""
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        part = event.get("part") or {} if isinstance(event, dict) else {}
        kind = part.get("type")
        if kind == "step-finish":
            tokens = part.get("tokens") or {}
            usage["steps"] += 1
            usage["input_tokens"] += int(tokens.get("input") or 0)
            usage["output_tokens"] += int(tokens.get("output") or 0)
            usage["reasoning_tokens"] += int(tokens.get("reasoning") or 0)
            usage["cache_read_tokens"] += int((tokens.get("cache") or {}).get("read") or 0)
            usage["cost_usd"] += float(part.get("cost") or 0.0)
        elif kind == "text" and part.get("text"):
            last_text = part["text"].strip()
            transcript.append(last_text)
        elif kind == "tool":
            state = part.get("state") or {}
            transcript.append(f"> {part.get('tool')} {json.dumps(state.get('input', {}))}")
    return {"usage": usage, "last_text": last_text, "transcript": "\n".join(transcript)}


def time_split(lines: list[dict]) -> dict:
    """Where a run's wall time went, from the session log's ``wall_s`` stamps.

    ``locate_s`` is time in the pointing model, ``motion_s`` time the arm was
    moving (an action still running at the end counts to the last line), and
    ``other_s`` the rest: mostly the agent's model thinking between calls.
    ``first_tool_s`` is how long before the agent's first tool call.
    """
    if not lines:
        return {}
    session = float(lines[-1].get("wall_s", 0.0))
    requests = [line for line in lines if line["kind"] == "request"]
    first_tool = float(requests[0]["wall_s"]) if requests else None
    locate = sum(
        float(line["data"].get("seconds") or 0.0) for line in lines if line["kind"] == "locate_result"
    )
    started: dict[str, float] = {}
    ended: dict[str, float] = {}
    for line in lines:
        if line["kind"] != "status" or not line.get("action_id"):
            continue
        action = line["action_id"]
        started.setdefault(action, float(line["wall_s"]))
        if line["data"].get("state") in TERMINAL_STATES:
            ended.setdefault(action, float(line["wall_s"]))
    motion = sum(ended.get(action, session) - start for action, start in started.items())
    return {
        "session_s": round(session, 1),
        "first_tool_s": round(first_tool, 1) if first_tool is not None else None,
        "locate_s": round(locate, 1),
        "motion_s": round(motion, 1),
        "other_s": round(session - locate - motion, 1),
        "actions": len(started),
    }


def grade(log_path: Path, name: str) -> dict:
    """What really happened, from the session log."""
    lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    observations = [line for line in lines if line["kind"] == "observation"]
    final = observations[-1]["data"]["scene"]["objects"].get(name, {}) if observations else {}
    requests = [line for line in lines if line["kind"] == "request"]
    estimates = [line["data"] for line in lines if line["kind"] == "estimate"]
    events = [line["data"].get("text", "") for line in lines if line["kind"] == "event"]
    held = any(f"{name} attached" in text for text in events)
    return {
        "in_goal": bool(final.get("in_goal")),
        "ever_held": held,
        "final_xyz_m": final.get("xyz_m"),
        "tool_requests": len(requests),
        "locate_calls": sum(1 for r in requests if r["data"].get("type") == "locate"),
        "locate_errors_cm": [
            e["error_cm"] for e in estimates if e.get("source", "").startswith("locate")
        ],
        "misses": sum(1 for text in events if f"{name} missed" in text),
        "success_checks": [
            {k: line["data"].get(k) for k in ("verdict", "p_in_goal", "correct")}
            for line in lines
            if line["kind"] == "success_check" and line["data"].get("ok", True)
        ],
        "time": time_split(lines),
    }


def run_opencode(args, work: Path, port: int, prompt: str) -> dict:
    """One run with opencode as the harness (what students use)."""
    config = {
        "$schema": "https://opencode.ai/config.json",
        "mcp": {
            "soarm-arm": {"type": "remote", "url": "http://127.0.0.1:9/off", "enabled": False},
            "soarm-bench": {
                "type": "remote",
                "url": f"http://127.0.0.1:{port}/mcp",
                "enabled": True,
            },
        },
        "tools": {"bash": False, "edit": False, "write": False, "patch": False, "webfetch": False},
    }
    (work / "opencode.json").write_text(json.dumps(config))
    # Its own data and state folders: with the OpenCode desktop app running,
    # a command-line opencode sharing them stalls before its first step.
    env = {
        **os.environ,
        "OPENCODE_CONFIG": str(work / "opencode.json"),
        "XDG_DATA_HOME": str(work / "opencode-data"),
        "XDG_STATE_HOME": str(work / "opencode-state"),
    }
    if args.model.startswith("openrouter/") and "OPENROUTER_API_KEY" not in env:
        key = load_secret("OPENROUTER_API_KEY")  # from .env; never printed
        if key:
            env["OPENROUTER_API_KEY"] = key
    started = time.monotonic()
    try:
        done = subprocess.run(
            [
                "opencode",
                "run",
                "-m",
                args.model,
                "--format",
                "json",  # one event per line, with each step's tokens and cost
                "--dir",
                str(work),
                prompt,
            ],
            cwd=work,
            env=env,
            stdin=subprocess.DEVNULL,  # opencode run otherwise waits to read a message from stdin
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        raw, stderr, timed_out = done.stdout, done.stderr, False
    except subprocess.TimeoutExpired as exc:
        raw = (
            (exc.stdout or b"").decode("utf-8", "replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr, timed_out = "", True
    seconds = round(time.monotonic() - started, 1)
    (work / "events.jsonl").write_text(raw)
    parsed = parse_opencode_events(raw)
    (work / "transcript.txt").write_text(parsed["transcript"] + ("\n" + stderr if stderr else ""))
    return {
        "seconds": seconds,
        "timed_out": timed_out,
        "agent_usage": parsed["usage"],  # the agent's own model: tokens and cost
        "agent_said": parsed["last_text"][-200:],
    }


def run_loop(args, work: Path, port: int, prompt: str) -> dict:
    """One run with the lean agent loop as the harness (twin.agent_loop)."""
    from xbox_soarm_teleop.twin.agent_loop import run_agent

    agent = run_agent(
        f"http://127.0.0.1:{port}/mcp",
        args.model,
        prompt,
        max_steps=args.max_steps,
        max_cost_usd=args.max_cost,
        timeout_s=args.timeout,
    )
    (work / "agent_run.json").write_text(json.dumps(agent.to_dict(), indent=1))
    return {
        "seconds": agent.time.get("total_s"),
        "timed_out": agent.stop_reason == "timeout",
        "stop_reason": agent.stop_reason,
        "agent_error": agent.error,
        "agent_usage": agent.usage,
        "agent_time": agent.time,
        "agent_said": agent.final_text[-200:],
    }


def run_once(args, run: int) -> dict:
    name, description = DESCRIPTIONS[args.scene]
    port = free_port()
    work = Path(tempfile.mkdtemp(prefix="agent_trial_"))
    logs = work / "logs"
    twin = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "xbox_soarm_teleop.cli.twin",
            "--headless",
            "--no-controller",
            "--no-coordinates",
            "--scene",
            args.scene,
            "--port",
            str(port),
            "--log-dir",
            str(logs),
            "--max-seconds",
            str(args.timeout + 60),
            *(["--pointer", args.pointer] if args.pointer else []),
            *(["--judge", args.judge] if args.judge else []),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        if not wait_for_port(port):
            return {"run": run, "error": "twin did not start"}
        prompt = PROMPT.format(description=description, name=name)
        harness = run_loop if args.harness == "loop" else run_opencode
        outcome = harness(args, work, port, prompt)
    finally:
        time.sleep(2.0)  # let a released object settle before the twin stops
        twin.send_signal(signal.SIGINT)  # the twin closes its log on Ctrl+C
        try:
            twin.wait(timeout=20)
        except subprocess.TimeoutExpired:
            twin.kill()
    log_files = sorted(logs.glob("*.jsonl"))
    result = {
        "run": run,
        "model": args.model,
        "scene": args.scene,
        "pointer": args.pointer,
        "judge": args.judge,
    }
    result.update(harness=args.harness, work_dir=str(work), **outcome)
    if log_files:
        result.update(grade(log_files[-1], name))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        required=True,
        help="opencode model (opencode/glm-5.3-flash-free), or an OpenRouter model for --harness loop "
        "(z-ai/glm-5.3-flash)",
    )
    parser.add_argument(
        "--harness",
        choices=("opencode", "loop"),
        default="opencode",
        help="opencode: the CLI students use. loop: the lean agent loop (twin.agent_loop), the same "
        "prompt and tools for every model, OpenRouter models only",
    )
    parser.add_argument(
        "--judge", default=None, choices=("clef", "clef-flash"), help="Success judge for check_success"
    )
    parser.add_argument("--max-steps", type=int, default=60, help="Model replies at most (loop)")
    parser.add_argument(
        "--max-cost", type=float, default=0.50, help="Stop a run past this many USD (loop)"
    )
    parser.add_argument("--pointer", default="molmo", help="Pointing model spec for locate")
    parser.add_argument("--scene", default="pick_place_basic", choices=sorted(DESCRIPTIONS))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0, help="Seconds per run")
    parser.add_argument("--out", default="twin_logs/agent_trials.jsonl")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    successes = 0
    for run in range(args.runs):
        result = run_once(args, run)
        successes += bool(result.get("in_goal"))
        print(json.dumps(result), flush=True)
        with out.open("a") as handle:
            handle.write(json.dumps(result) + "\n")
    print(
        f"{args.model} ({args.harness}) {args.scene} pointer={args.pointer}: "
        f"{successes}/{args.runs} in the goal box"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
