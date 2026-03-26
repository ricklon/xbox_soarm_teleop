"""Per-frame control-loop timing for benchmark instrumentation.

Captures timing data for the Chameleon Trovi benchmark harness that compares
edge device performance (Pi 5, BeagleY-AI, Jetson Nano).

Usage
-----
    from xbox_soarm_teleop.diagnostics.benchmark import LoopTimer

    timer = LoopTimer(mode="crane")
    while running:
        timer.start_frame()
        t0 = time.perf_counter()
        state = controller.read()
        controller_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        kinematics.solve(...)
        ik_ms = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        robot.send_action(action)
        servo_ms = (time.perf_counter() - t0) * 1000.0

        timer.record(frame, controller_ms, ik_ms, servo_ms)

    timer.write_csv("benchmark_20260325_143022.csv")
    print(timer.summary())
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path


class LoopTimer:
    """Collect per-frame timing data and write a structured CSV.

    Args:
        mode: Control mode string stored with each row (crane/joint/cartesian).
        rss_every: Sample resident set size every this many frames (0 = disabled).
    """

    _HEADER = [
        "frame",
        "timestamp",
        "controller_read_ms",
        "ik_solve_ms",
        "servo_write_ms",
        "loop_total_ms",
        "mode",
        "rss_mb",
    ]

    def __init__(self, mode: str, rss_every: int = 100) -> None:
        self.mode = mode
        self.rss_every = rss_every
        self._rows: list[dict] = []
        self._t_frame: float = 0.0

    def start_frame(self) -> None:
        """Mark the start of a new control-loop iteration."""
        self._t_frame = time.perf_counter()

    def record(
        self,
        frame: int,
        controller_ms: float,
        ik_ms: float,
        servo_ms: float,
    ) -> None:
        """Record timing for the current frame.

        Args:
            frame: Zero-based frame index.
            controller_ms: Controller read time in milliseconds.
            ik_ms: IK solve time in milliseconds (0 for joint/crane modes).
            servo_ms: Robot send_action time in milliseconds.
        """
        total_ms = (time.perf_counter() - self._t_frame) * 1000.0
        rss = ""
        if self.rss_every > 0 and frame % self.rss_every == 0:
            rss = f"{self._rss_mb():.1f}"
        self._rows.append(
            {
                "frame": frame,
                "timestamp": f"{self._t_frame:.6f}",
                "controller_read_ms": f"{controller_ms:.3f}",
                "ik_solve_ms": f"{ik_ms:.3f}",
                "servo_write_ms": f"{servo_ms:.3f}",
                "loop_total_ms": f"{total_ms:.3f}",
                "mode": self.mode,
                "rss_mb": rss,
            }
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write_csv(self, path: str | Path) -> Path:
        """Write all recorded rows to a CSV file.

        Args:
            path: Output path.

        Returns:
            Resolved path that was written.
        """
        path = Path(path)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._HEADER)
            writer.writeheader()
            writer.writerows(self._rows)
        return path

    def summary(self) -> str:
        """Return a human-readable timing summary string."""
        n = len(self._rows)
        if n == 0:
            return "No benchmark data recorded."

        def _col(key: str) -> list[float]:
            return [float(r[key]) for r in self._rows]

        totals = _col("loop_total_ms")
        controllers = _col("controller_read_ms")
        iks = _col("ik_solve_ms")
        servos = _col("servo_write_ms")

        sorted_totals = sorted(totals)
        p50 = sorted_totals[n // 2]
        p95 = sorted_totals[min(int(n * 0.95), n - 1)]
        p99 = sorted_totals[min(int(n * 0.99), n - 1)]
        mean_hz = 1000.0 / max(statistics.mean(totals), 1e-6)

        lines = [
            f"Benchmark summary ({n} frames, mode={self.mode})",
            f"  loop_total:      mean={statistics.mean(totals):.1f}ms"
            f"  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms"
            f"  max={max(totals):.1f}ms  ({mean_hz:.1f} Hz effective)",
            f"  controller_read: mean={statistics.mean(controllers):.2f}ms"
            f"  max={max(controllers):.2f}ms",
            f"  ik_solve:        mean={statistics.mean(iks):.2f}ms"
            f"  max={max(iks):.2f}ms",
            f"  servo_write:     mean={statistics.mean(servos):.2f}ms"
            f"  max={max(servos):.2f}ms",
        ]
        return "\n".join(lines)

    @property
    def frame_count(self) -> int:
        return len(self._rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rss_mb() -> float:
        """Return current process RSS in MiB (Linux/macOS)."""
        try:
            import resource

            ru = resource.getrusage(resource.RUSAGE_SELF)
            # Linux: ru_maxrss is kibibytes; macOS: bytes
            import sys

            if sys.platform == "darwin":
                return ru.ru_maxrss / (1024 * 1024)
            return ru.ru_maxrss / 1024.0
        except Exception:
            return 0.0

    @staticmethod
    def default_path(prefix: str = "benchmark") -> Path:
        """Generate a timestamped output path in the current directory."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        return Path(f"{prefix}_{ts}.csv")
