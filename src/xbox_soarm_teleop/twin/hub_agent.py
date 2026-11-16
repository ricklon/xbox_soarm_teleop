"""The agent panel's runs: the benchmarked agent loop, started from a student's page.

Each arm can run one agent at a time (``twin.agent_loop.run_agent``) on the
instructor's OpenRouter key, with a model the instructor allows. The run
connects to the arm through the hub's own MCP address with the arm's key, so
it gets exactly the tools, names and instructions any agent gets, and the
arm's own rules (takeover, stops, ownership) apply unchanged.

Limits: each arm has a budget for the agent model's cost (and each run is
capped at what is left of it); a few runs at a time across the hub (parallel
batches ran an OpenRouter account into "in-flight" refusals); a step limit and
a time limit per run. The pointing model and judge the arm's tools use are on
the instructor's keys too and are not counted here.

Stop ends the run before its next step and ends the arm's current action, the
way a person taking over does, so a pick in progress stops at once. A stop
latched on the page ends the run too.
"""

from __future__ import annotations

import itertools
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from xbox_soarm_teleop.twin.agent_loop import AgentError, run_agent
from xbox_soarm_teleop.twin.pointing import load_secret, mask_secrets

DEFAULT_MODELS = ("google/gemma-4-31b-it", "z-ai/glm-5.3-flash", "google/gemini-3.5-flash")
DEFAULT_BUDGET_USD = 0.50  # per arm, for the whole session
PER_RUN_CAP_USD = 0.25
MAX_STEPS = 40
TIMEOUT_S = 600.0
MAX_RUNNING = 2  # across the hub
MAX_TASK_CHARS = 2000
# Added to what the student typed (issue 53: "I wish it could communicate to
# the operator more"; "I don't know what it would say when completed").
HOW_TO_REPORT = """

How to work with the person watching: before each action, say in one short
sentence what you are about to do and why. If a tool answers ok: false, say
what went wrong and what you will do about it (or that it cannot be done);
do not repeat the same call hoping for another answer. When you finish, give
a short summary for the person: what you did, what happened, and anything you
could not do. End with exactly one line, RESULT: done or RESULT: not_done."""


@dataclass
class PanelRun:
    """One run from a page, as the page shows it."""

    run_id: int
    task: str
    model: str
    sees: bool
    started: float = field(default_factory=time.time)
    steps: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    stop_reason: str = ""
    final_text: str = ""
    error: str | None = None
    cost_usd: float = 0.0
    seconds: float = 0.0
    stop: threading.Event = field(default_factory=threading.Event)

    def view(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "model": self.model,
            "sees": self.sees,
            "steps": list(self.steps),
            "done": self.done,
            "stop_reason": self.stop_reason,
            "final_text": self.final_text,
            "verdict": verdict_of(self.final_text) if self.done else None,
            "error": self.error,
            "cost_usd": round(self.cost_usd, 5),
            "seconds": round(self.seconds, 1),
        }


def verdict_of(text: str) -> str | None:
    """The agent's own verdict from its last RESULT line: done, not_done, or None."""
    found = re.findall(r"result:\s*\**\s*(not[ _]done|done)", (text or "").lower())
    if not found:
        return None
    return "not_done" if found[-1].startswith("not") else "done"


class Refused(Exception):
    """A run that cannot start: ``status`` is the HTTP status to answer with."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class AgentDesk:
    """Starts, follows and stops panel runs for a hub's arms.

    Args:
        port: The hub's port: runs connect to 127.0.0.1 on it, whatever address
            students use.
        models: The OpenRouter models a page may choose.
        budget_usd: Each arm's budget for agent model costs.
        post: Sends a request to OpenRouter (tests pass a scripted model).
    """

    def __init__(
        self,
        *,
        port: int,
        models: tuple[str, ...] = DEFAULT_MODELS,
        budget_usd: float = DEFAULT_BUDGET_USD,
        post: Callable[..., dict] | None = None,
        max_running: int = MAX_RUNNING,
    ) -> None:
        self.url = f"http://127.0.0.1:{port}/mcp"
        self.models = tuple(models)
        self.budget_usd = float(budget_usd)
        self.post = post
        self.max_running = max_running
        self.available = bool(load_secret("OPENROUTER_API_KEY")) and bool(self.models)
        self.spent: dict[str, float] = {}
        self.runs: dict[str, PanelRun] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()

    def state(self, room) -> dict[str, Any]:
        """What the page shows: what it may do, what is left, the current or last run."""
        run = self.runs.get(room.token)
        return {
            "available": self.available,
            "models": list(self.models),
            "budget_usd": self.budget_usd,
            "spent_usd": round(self.spent.get(room.token, 0.0), 5),
            "running": run is not None and not run.done,
            "run": run.view() if run is not None else None,
        }

    def start(self, room, task: str, model: str, sees: bool) -> PanelRun:
        task = (task or "").strip()
        if not self.available:
            raise Refused(503, "this hub has no agent: the instructor has not set OPENROUTER_API_KEY")
        if not task or len(task) > MAX_TASK_CHARS:
            raise Refused(400, f"the task must be 1 to {MAX_TASK_CHARS} characters")
        if model not in self.models:
            raise Refused(400, f"model must be one of: {', '.join(self.models)}")
        with self._lock:
            current = self.runs.get(room.token)
            if current is not None and not current.done:
                raise Refused(409, "an agent is already running on this arm: stop it first")
            left = self.budget_usd - self.spent.get(room.token, 0.0)
            if left <= 0:
                raise Refused(402, "this arm's agent budget is used up; ask your instructor")
            running = sum(1 for r in self.runs.values() if not r.done)
            if running >= self.max_running:
                raise Refused(429, "the hub is running as many agents as it can; try again shortly")
            run = PanelRun(next(self._ids), task, model, bool(sees))
            self.runs[room.token] = run
        thread = threading.Thread(
            target=self._run, args=(room, run, min(PER_RUN_CAP_USD, left)), daemon=True
        )
        self._threads[room.token] = thread
        thread.start()
        return run

    def stop(self, room) -> bool:
        """Stop the arm's run: before its next step, and the arm's current action now."""
        run = self.runs.get(room.token)
        if run is None or run.done:
            return False
        run.stop.set()
        room.service.abort_action("the agent was stopped from the page")
        return True

    def close(self) -> None:
        for token, run in list(self.runs.items()):
            run.stop.set()
        for thread in list(self._threads.values()):
            thread.join(timeout=10)

    def _run(self, room, run: PanelRun, cap_usd: float) -> None:
        started = time.monotonic()
        kwargs = {"post": self.post} if self.post is not None else {}
        try:
            result = run_agent(
                self.url,
                run.model,
                run.task + HOW_TO_REPORT,
                arm_key=room.token,
                max_steps=MAX_STEPS,
                max_cost_usd=cap_usd,
                timeout_s=TIMEOUT_S,
                on_step=run.steps.append,
                sees_images=run.sees,
                should_stop=lambda: run.stop.is_set() or room.service.stop_latched_by is not None,
                **kwargs,
            )
            run.stop_reason = result.stop_reason
            run.final_text = result.final_text
            run.error = _explain(result.error)
            run.cost_usd = float((result.usage or {}).get("cost_usd") or 0.0)
        except AgentError as exc:
            run.stop_reason, run.error = "error", _explain(str(exc))
        except Exception as exc:  # noqa: BLE001 - shown on the page
            run.stop_reason, run.error = "error", _explain(f"{type(exc).__name__}: {exc}")
        finally:
            run.seconds = time.monotonic() - started
            with self._lock:
                self.spent[room.token] = self.spent.get(room.token, 0.0) + run.cost_usd
            run.done = True
            _log(room, run)


def _explain(error: str | None) -> str | None:
    """An error as a page shows it: keys masked, and the hub's own account
    problems said to be the hub's, not the agent's."""
    if not error:
        return None
    error = mask_secrets(error)
    if "HTTP 402" in error:
        return "The hub's model account is out of credit (tell your instructor). " + error[:200]
    if "HTTP 429" in error:
        return "The model service is busy (rate limit); try again in a minute. " + error[:200]
    return error[:500]


def _log(room, run: PanelRun) -> None:
    """Record the run in the arm's session log, nested under one key."""
    log = getattr(room.service, "_log", None)
    if log is None:
        return
    log(
        "panel_run",
        run={
            "run_id": run.run_id,
            "task": run.task,
            "model": run.model,
            "sees": run.sees,
            "naming": room.service.naming,
            "steps": len(run.steps),
            "stop_reason": run.stop_reason,
            "cost_usd": round(run.cost_usd, 5),
            "seconds": round(run.seconds, 1),
            "error": run.error,
        },
    )
