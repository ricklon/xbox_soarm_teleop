"""Host one simulated arm per student in a single process.

For a remote workshop: the instructor runs the hub, every student gets a
secret link to their own arm. A student drives it from a web page (which
stands in for the Xbox controller) and can connect an AI agent to the same
arm over MCP. The arbitration is the twin's own: the person at the page
overrides the agent and owns the stop.

One loop thread ticks every room, so the rooms can share the compiled model,
the IK solver and the camera renderer (:class:`~xbox_soarm_teleop.twin.TwinParts`).
A room then costs its simulator state and about 0.2 ms per tick. Camera frames
cost about 4 ms each, so they are rationed: a fixed time budget per tick,
shared round-robin between the rooms that are waiting for one.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from xbox_soarm_teleop.teleoperators.xbox import XboxState
from xbox_soarm_teleop.twin import TwinService, build_parts, build_twin
from xbox_soarm_teleop.twin.cameras import CameraRig

# One-shot buttons a page may press, and the controller button each stands for.
PRESSES = {"stop": "b_button", "release": "start_button", "home": "a_button", "reset": "y_button"}
RENDER_BUDGET_S = 0.008  # of each 20 ms tick


def _unit(value: Any) -> float:
    """A page-supplied axis value as a float in -1..1 (anything else is 0)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        return 0.0
    return max(-1.0, min(1.0, float(value)))


class WebController:
    """A web page standing in for the Xbox controller of one room.

    The page posts what the person wants (``update``); the twin reads it like
    a controller every tick (``read``). Motion is deadman-gated the same way
    as the pad: the arm moves only while the page keeps saying so, and a page
    that goes quiet for ``STALE_S`` (closed tab, lost connection) releases it.

    Posted fields, all optional:

    - ``fwd``, ``left``, ``up``, ``roll``: -1..1, the direction to move.
    - ``hold``: engage without moving (for the gripper).
    - ``grip``: gripper target 0 (open) .. 1 (closed).
    - ``press``: list of one-shot buttons: ``stop``, ``release``, ``home``, ``reset``.
    """

    STALE_S = 0.5
    GONE_S = 3.0

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._wanted: dict[str, Any] = {}
        self._presses: list[str] = []
        self._updated_at = -float("inf")
        self._was_engaged = False
        self._pressed_at = -float("inf")
        self._trigger = 0.0
        self.gripper_feedback: Callable[[], float] = lambda: 0.0
        """Where the gripper is now; set by the hub once the twin exists."""

    def update(self, wanted: dict[str, Any]) -> None:
        """Record the page's latest post. Safe from any thread."""
        presses = wanted.get("press") or []
        with self._lock:
            self._wanted = wanted
            self._updated_at = self._clock()
            if isinstance(presses, list):
                self._presses.extend(p for p in presses if isinstance(p, str) and p in PRESSES)

    def press(self, button: str) -> None:
        """Press a one-shot button from the server side (the instructor's stop).

        Delivered on the next tick even when no page is open.
        """
        if button not in PRESSES:
            raise ValueError(f"Unknown button '{button}'. Known: {', '.join(PRESSES)}")
        with self._lock:
            self._presses.append(button)
            self._pressed_at = self._clock()

    @property
    def is_connected(self) -> bool:
        """A page has posted recently (or a server-side press is being delivered)."""
        now = self._clock()
        return now - self._updated_at <= self.GONE_S or now - self._pressed_at <= 0.2

    def read(self) -> XboxState:
        """The controller state for this tick."""
        with self._lock:
            fresh = self._clock() - self._updated_at <= self.STALE_S
            wanted = self._wanted if fresh else {}
            presses, self._presses = self._presses, []
        fwd, left = _unit(wanted.get("fwd")), _unit(wanted.get("left"))
        up, roll = _unit(wanted.get("up")), _unit(wanted.get("roll"))
        engaged = bool(wanted.get("hold")) or any((fwd, left, up, roll))
        if engaged and not self._was_engaged:
            # Start from where the gripper is, as the twin expects of a trigger:
            # taking over must not open a closed gripper (and drop the cube).
            self._trigger = float(self.gripper_feedback())
        elif engaged and wanted.get("grip") is not None:
            self._trigger = max(0.0, _unit(wanted["grip"]))
        self._was_engaged = engaged
        state = XboxState(
            left_bumper=engaged,
            left_stick_y=-fwd,
            left_stick_x=-left,
            right_stick_y=-up,
            right_stick_x=roll,
            right_trigger=self._trigger,
        )
        for name in set(presses):
            button = PRESSES[name]
            setattr(state, button, True)
            setattr(state, f"{button}_pressed", True)
        return state


@dataclass
class Room:
    """One student's arm."""

    name: str
    token: str
    service: TwinService
    controller: WebController
    log_path: Path | None = None


class Hub:
    """A set of rooms ticked by one loop.

    Args:
        seats: Number of rooms (students).
        scene: Scene for every room.
        world: Lighting/floor preset.
        log_dir: Directory for one session log per room; no logs when None.
        roster_path: JSON file holding the room and admin tokens. Reused when
            it exists, so the links handed to students survive a restart.
    """

    def __init__(
        self,
        seats: int,
        *,
        scene: str = "pick_place_basic",
        world: str | None = None,
        log_dir: str | Path | None = None,
        roster_path: str | Path | None = None,
    ) -> None:
        if seats < 1:
            raise ValueError("seats must be at least 1")
        self.scene = scene
        self.parts = build_parts(scene=scene, world=world)
        # Offscreen GL objects are created on first use, on the loop thread.
        self.cameras = CameraRig(self.parts.model, reflections=False)
        self.admin_token, tokens = self._load_roster(roster_path, seats)
        self.rooms: dict[str, Room] = {}
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for index, token in enumerate(tokens, start=1):
            controller = WebController()
            log_path = None
            if log_dir is not None:
                log_path = Path(log_dir) / f"hub-{stamp}" / f"arm-{index:02d}.jsonl"
            service = build_twin(
                parts=self.parts, controller=controller, cameras=self.cameras, log_path=log_path
            )
            controller.gripper_feedback = lambda service=service: float(service.state.gripper_pos)
            self.rooms[token] = Room(f"Arm {index}", token, service, controller, log_path)
        self.dt = next(iter(self.rooms.values())).service.dt
        self._render_turn = 0
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._closed = False
        self.ticks = 0
        self.late_ticks = 0
        self.loop_hz = 0.0
        """Measured loop rate, updated about once a second while running."""

    # ---------------------------------------------------------------- roster
    @staticmethod
    def _load_roster(roster_path: str | Path | None, seats: int) -> tuple[str, list[str]]:
        admin, tokens = None, []
        path = Path(roster_path) if roster_path is not None else None
        if path is not None and path.exists():
            saved = json.loads(path.read_text())
            admin = saved.get("admin")
            tokens = [seat["token"] for seat in saved.get("rooms", [])][:seats]
        admin = admin or secrets.token_urlsafe(12)
        while len(tokens) < seats:
            tokens.append(secrets.token_urlsafe(12))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            roster = {
                "admin": admin,
                "rooms": [{"name": f"Arm {i}", "token": t} for i, t in enumerate(tokens, start=1)],
            }
            path.write_text(json.dumps(roster, indent=2) + "\n")
            path.chmod(0o600)
        return admin, tokens

    def room(self, token: str) -> Room | None:
        """The room a token opens, or None."""
        return self.rooms.get(token)

    # ------------------------------------------------------------------ loop
    def tick(self) -> None:
        """Advance every room one control period, then draw camera frames.

        Call from one thread only (the hub's loop thread once started).
        """
        rooms = list(self.rooms.values())
        for room in rooms:
            room.service.tick(render=False)
        # Share the render budget round-robin; always draw at least one.
        deadline = time.perf_counter() + RENDER_BUDGET_S
        for offset in range(len(rooms)):
            room = rooms[(self._render_turn + offset) % len(rooms)]
            if room.service.render_waiting:
                room.service.render_pending()
                self._render_turn = (self._render_turn + offset + 1) % len(rooms)
                if time.perf_counter() >= deadline:
                    break
        self.ticks += 1

    def start(self) -> None:
        """Run the loop in a background thread at the control rate."""
        if self._thread is not None:
            return
        self._running.set()
        self._thread = threading.Thread(target=self._run, name="hub-loop", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        next_tick = measured_from = time.monotonic()
        measured_ticks = self.ticks
        while self._running.is_set():
            self.tick()
            if next_tick - measured_from >= 1.0:
                elapsed = time.monotonic() - measured_from
                self.loop_hz = (self.ticks - measured_ticks) / elapsed
                measured_from, measured_ticks = time.monotonic(), self.ticks
            next_tick += self.dt
            now = time.monotonic()
            if next_tick <= now - self.dt:  # stalled: restart the grid, no burst
                self.late_ticks += 1
                next_tick = now
            time.sleep(max(0.0, next_tick - time.monotonic()))
        self._shutdown()

    def _shutdown(self) -> None:
        for room in self.rooms.values():
            room.service.close()
        self.cameras.close()

    def close(self) -> None:
        """Stop the loop and close every room. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        if self._thread is not None:
            self._running.clear()
            self._thread.join(timeout=5.0)
        else:
            self._shutdown()
