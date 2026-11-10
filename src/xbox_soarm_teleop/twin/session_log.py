"""Session JSONL log for the twin (design §10, ``soarm.session_log/v1``).

One JSON object per line. Every line has the same keys:

- ``t``: control time in seconds (``frame * dt``).
- ``wall_s``: wall-clock seconds since the log was opened.
- ``frame``: control tick counter.
- ``kind``: ``header``, ``request``, ``response``, ``status``, ``stop``,
  ``stop_reset``, ``ownership``, ``event``, ``observation`` or ``shutdown``.
- ``client_id`` / ``action_id``: who and which action, or null.
- ``data``: the payload for that kind.

The first line is the header: schema versions, git commit, URDF hash, scene,
limits in force and calibration identity. Observations are written at
``observation_hz`` and at every tick around any other line, so transitions can
be replayed at full rate without logging every idle tick. While nothing in
the observation changes (an idle arm), it is written at ``IDLE_HZ`` instead.

Writing never raises into the control loop: on an I/O error the log disables
itself and keeps the error in :attr:`SessionLog.error`.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from xbox_soarm_teleop.schemas.agent import ACTION_SCHEMA_ID, OBSERVATION_SCHEMA_ID

SESSION_LOG_SCHEMA_ID = "soarm.session_log/v1"
OBSERVATION_HZ = 10.0
FULL_RATE_S = 0.5
IDLE_HZ = 1.0
_OBSERVATION_CLOCK = ("frame", "sim_time_s")
# Constant for the whole run, so recorded once in the header's limits.
_OBSERVATION_OMIT = ("workspace",)


def _default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {k: getattr(value, k) for k in value.__dataclass_fields__}
    if hasattr(value, "tolist"):  # numpy scalars and arrays
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


class SessionLog:
    """Append-only JSONL log of one twin session. Safe to call from any thread.

    Args:
        path: File to create (parent directories are created).
        header: Payload of the header line (see :func:`session_header`).
        dt: Control period (s), used to downsample observations.
        observation_hz: Observation rate when nothing else is happening.
        full_rate_s: After any other line, log every tick for this long.
    """

    def __init__(
        self,
        path: str | Path,
        header: dict[str, Any],
        *,
        dt: float,
        observation_hz: float = OBSERVATION_HZ,
        full_rate_s: float = FULL_RATE_S,
    ) -> None:
        self.path = Path(path)
        self.error: OSError | None = None
        self._lock = threading.Lock()
        self._every = max(1, round(1.0 / (observation_hz * dt)))
        self._full_rate_frames = max(0, round(full_rate_s / dt))
        self._idle_every = max(self._every, round(1.0 / (IDLE_HZ * dt)))
        self._full_rate_until = -1
        self._last_written: dict[str, Any] | None = None
        self._pending: dict[str, Any] | None = None
        self._started = time.monotonic()
        self._closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        with self._lock:
            self._write("header", 0.0, 0, None, None, {"schema": SESSION_LOG_SCHEMA_ID, **header})
            self._flush()

    def event(
        self,
        kind: str,
        *,
        t: float,
        frame: int,
        client_id: str | None = None,
        action_id: str | None = None,
        **data: Any,
    ) -> None:
        """Write one line, preceded by the latest observation if it was skipped."""
        with self._lock:
            self._write_pending()
            self._full_rate_until = frame + self._full_rate_frames
            self._write(kind, t, frame, client_id, action_id, data)
            self._flush()

    def observation(self, observation: dict[str, Any]) -> None:
        """Offer this tick's observation; written at 10 Hz or around transitions."""
        with self._lock:
            frame = observation["frame"]
            self._pending = observation
            if frame <= self._full_rate_until:
                self._write_pending()
            elif frame % self._every == 0:
                if frame % self._idle_every == 0 or self._changed(observation):
                    self._write_pending()

    def flush(self) -> None:
        with self._lock:
            self._flush()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if not self._file.closed:
                try:
                    self._file.close()
                except OSError as exc:
                    self.error = self.error or exc

    # ------------------------------------------------- internals (lock held)
    def _changed(self, observation: dict[str, Any]) -> bool:
        last = self._last_written
        return last is None or any(
            observation[k] != last.get(k) for k in observation if k not in _OBSERVATION_CLOCK
        )

    def _write_pending(self) -> None:
        observation, self._pending = self._pending, None
        if observation is not None:
            self._last_written = observation
            data = {k: v for k, v in observation.items() if k not in _OBSERVATION_OMIT}
            self._write(
                "observation", observation["sim_time_s"], observation["frame"], None, None, data
            )

    def _write(
        self,
        kind: str,
        t: float,
        frame: int,
        client_id: str | None,
        action_id: str | None,
        data: dict[str, Any],
    ) -> None:
        if self.error is not None or self._closed:
            return
        line = {
            "t": round(float(t), 6),
            "wall_s": round(time.monotonic() - self._started, 3),
            "frame": int(frame),
            "kind": kind,
            "client_id": client_id,
            "action_id": action_id,
            "data": data,
        }
        try:
            self._file.write(json.dumps(line, default=_default, separators=(",", ":")) + "\n")
        except (OSError, ValueError) as exc:  # ValueError: file closed under us
            self.error = exc if isinstance(exc, OSError) else OSError(str(exc))

    def _flush(self) -> None:
        if self.error is None and not self._file.closed:
            try:
                self._file.flush()
            except OSError as exc:
                self.error = exc


def read_session_log(path: str | Path) -> list[dict[str, Any]]:
    """Load every line of a session log."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _git_identity(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", "-C", str(root), *args], capture_output=True, text=True, timeout=5, check=True
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain") if commit else None
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def session_header(
    *,
    scene: str,
    world: str,
    urdf_path: str | Path,
    limits: dict[str, Any],
    backend: str = "mujoco",
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the header payload: what was run, and under which limits.

    Args:
        scene: Scene name.
        world: Lighting/floor preset.
        urdf_path: Robot model; recorded by name and SHA-256.
        limits: Limits in force (workspace, rates, timeouts).
        backend: ``mujoco`` for the twin; a hardware backend names itself.
        calibration: Calibration identity from ``audit_calibration`` on
            hardware; None in simulation.
    """
    urdf_path = Path(urdf_path)
    return {
        "schemas": {
            "session_log": SESSION_LOG_SCHEMA_ID,
            "observation": OBSERVATION_SCHEMA_ID,
            "action": ACTION_SCHEMA_ID,
        },
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": _git_identity(urdf_path.parent),
        "backend": backend,
        "urdf": urdf_path.name,
        "urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
        "scene": scene,
        "world": world,
        "limits": limits,
        "calibration": calibration,
    }
