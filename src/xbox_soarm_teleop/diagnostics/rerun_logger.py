"""Rerun visualization logger for teleoperation and simulation loops.

Logs joint angles, end-effector position, and control metadata to a Rerun
session so they can be viewed live (local or gRPC-remote) or replayed from a
saved .rrd file.

Rerun is an *optional* dependency (``pip install rerun-sdk``).  All public
methods silently no-op when Rerun is not installed, so callers need no
try/except blocks.

Usage — local viewer (opens Rerun app automatically)::

    logger = RerunLogger(mode="spawn")
    # ... in the control loop:
    logger.log_frame(frame, t_s, joint_deg_dict, ee_pos_xyz)
    logger.close()

Usage — remote viewer (viewer connects from a student laptop)::

    logger = RerunLogger(mode="serve", addr="0.0.0.0:9876")
    # On the student laptop: rerun --connect rerun+grpc://<pi-ip>:9876

Usage — save to file for later replay::

    logger = RerunLogger(mode="save", rrd_path="session.rrd")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_RERUN_AVAILABLE: bool = True
try:
    import rerun as rr  # type: ignore[import-untyped]
except ImportError:
    _RERUN_AVAILABLE = False


class RerunLogger:
    """Log control-loop data to a Rerun session.

    Args:
        app_id: Rerun application identifier shown in the viewer title bar.
        mode: One of ``"spawn"`` (local viewer), ``"serve"`` (gRPC server for
            remote viewer), ``"connect"`` (connect to an already-running
            viewer), or ``"save"`` (write .rrd file).
        addr: gRPC address string used for ``"serve"`` and ``"connect"`` modes
            (e.g. ``"0.0.0.0:9876"`` or ``"rerun+grpc://192.168.1.5:9876"``).
        rrd_path: Output path for ``"save"`` mode (e.g. ``"session.rrd"``).
        timeline: Name of the frame-number timeline shown in the viewer.
    """

    def __init__(
        self,
        app_id: str = "xbox_soarm_teleop",
        mode: str = "spawn",
        addr: str = "0.0.0.0:9876",
        rrd_path: str | Path = "session.rrd",
        timeline: str = "frame",
    ) -> None:
        self._enabled = _RERUN_AVAILABLE
        self._timeline = timeline

        if not self._enabled:
            print(
                "WARNING: rerun-sdk not installed. "
                "Run `pip install rerun-sdk` to enable Rerun visualization.",
                flush=True,
            )
            return

        rr.init(app_id, spawn=False)

        if mode == "spawn":
            rr.spawn()
            print("Rerun: opened local viewer.", flush=True)

        elif mode == "serve":
            # Serve gRPC stream — remote viewers connect to this address
            _addr = addr if "://" in addr else f"rerun+grpc://{addr}"
            rr.connect_grpc(_addr)
            print(
                f"Rerun: serving gRPC stream on {_addr}. "
                f"Connect with: rerun --connect {_addr}",
                flush=True,
            )

        elif mode == "connect":
            # Connect to an existing Rerun viewer
            _addr = addr if "://" in addr else f"rerun+grpc://{addr}"
            rr.connect_grpc(_addr)
            print(f"Rerun: connected to viewer at {_addr}.", flush=True)

        elif mode == "save":
            rrd_path = Path(rrd_path)
            rr.save(str(rrd_path))
            print(f"Rerun: saving recording to {rrd_path}.", flush=True)

        else:
            raise ValueError(
                f"Unknown Rerun mode {mode!r}. Use 'spawn', 'serve', 'connect', or 'save'."
            )

    # ------------------------------------------------------------------
    # Main logging method
    # ------------------------------------------------------------------

    def log_frame(
        self,
        frame: int,
        t_s: float,
        joint_deg: dict[str, float],
        ee_pos: np.ndarray | None = None,
        gripper_deg: float | None = None,
        mode: str = "",
    ) -> None:
        """Log one control-loop frame to Rerun.

        Args:
            frame: Zero-based frame index (used as the sequence timeline value).
            t_s: Wall time in seconds since loop start (shown as a second timeline).
            joint_deg: Dict of ``{joint_name: angle_deg}`` for arm joints.
            ee_pos: Optional (3,) array of EE position [x, y, z] in metres.
            gripper_deg: Optional gripper position in degrees.
            mode: Control mode string logged as a text annotation.
        """
        if not self._enabled:
            return

        rr.set_time(self._timeline, sequence=frame)
        rr.set_time("wall_time", timestamp=t_s)

        # --- Joint angles ---
        for name, deg in joint_deg.items():
            rr.log(f"joints/{name}", rr.Scalars([deg]))

        # --- Gripper ---
        if gripper_deg is not None:
            rr.log("joints/gripper", rr.Scalars([gripper_deg]))

        # --- EE position ---
        if ee_pos is not None:
            pos = np.asarray(ee_pos, dtype=float).reshape(3)
            rr.log("ee/position", rr.Points3D([pos]))
            rr.log("ee/x", rr.Scalars([float(pos[0])]))
            rr.log("ee/y", rr.Scalars([float(pos[1])]))
            rr.log("ee/z", rr.Scalars([float(pos[2])]))

        # --- Mode annotation (logged sparsely to avoid spam) ---
        if mode and frame % 30 == 0:
            rr.log("control/mode", rr.TextLog(mode))

    def close(self) -> None:
        """Finalise the Rerun session (flush pending data)."""
        if not self._enabled:
            return
        # rr.disconnect() flushes and closes the gRPC connection / file handle
        try:
            rr.disconnect()
        except Exception:
            pass
