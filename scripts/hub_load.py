#!/usr/bin/env python3
"""Load-test soarm-hub: how many pictures a second does each student get?

Starts a hub in its own process, then plays the part of every student's web
page over real HTTP: it posts the controls ten times a second and asks for
480x360 camera pictures at the rate the hub tells the page to use. Prints
what each student received, the upload this needs, and the hub's loop rate.

Use it before a workshop to see what your machine can carry:

    uv run python scripts/hub_load.py --seats 30
    uv run python scripts/hub_load.py --seats 30 --picture-rate 15
    uv run python scripts/hub_load.py --seats 30 --mesh-detail-mm 0   # original meshes

It measures this computer only. A tunnel and the students' own connections
come on top.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx2 as httpx

VIEWS = ("overhead", "front", "side", "wrist")


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _student(client, base: str, token: str, view: str, seconds: float, out: dict) -> None:
    """One student's page: controls every 100 ms, pictures as often as the hub asks."""
    picture_ms = 130.0
    stop_at = time.monotonic() + seconds

    async def controls() -> None:
        nonlocal picture_ms
        while time.monotonic() < stop_at:
            try:
                reply = await client.post(f"{base}/r/{token}/pad", json={})
                picture_ms = float(reply.json().get("picture_ms", picture_ms))
            except Exception:  # noqa: BLE001 - the page retries too
                out["errors"] += 1
            await asyncio.sleep(0.1)

    async def pictures() -> None:
        while time.monotonic() < stop_at:
            asked = time.monotonic()
            try:
                reply = await client.get(
                    f"{base}/r/{token}/camera.jpg", params={"view": view, "w": 480, "h": 360}
                )
                if reply.status_code == 200:
                    out["pictures"] += 1
                    out["bytes"] += len(reply.content)
                    out["waits"].append(time.monotonic() - asked)
                else:
                    out["busy"] += 1
            except Exception:  # noqa: BLE001
                out["errors"] += 1
            await asyncio.sleep(max(0.01, picture_ms / 1000 - (time.monotonic() - asked)))

    await asyncio.gather(controls(), pictures())


async def _run(base: str, roster: dict, seconds: float, warmup: float) -> dict:
    tokens = [room["token"] for room in roster["rooms"]]
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    async with httpx.AsyncClient(timeout=10, limits=limits) as client:
        warm = [{"pictures": 0, "bytes": 0, "busy": 0, "errors": 0, "waits": []} for _ in tokens]
        await asyncio.gather(
            *(_student(client, base, t, VIEWS[i % 4], warmup, warm[i]) for i, t in enumerate(tokens))
        )
        results = [{"pictures": 0, "bytes": 0, "busy": 0, "errors": 0, "waits": []} for _ in tokens]
        started = time.monotonic()
        await asyncio.gather(
            *(
                _student(client, base, t, VIEWS[i % 4], seconds, results[i])
                for i, t in enumerate(tokens)
            )
        )
        elapsed = time.monotonic() - started
        admin = await client.get(f"{base}/admin/{roster['admin']}/rooms")
    waits = sorted(w for r in results for w in r["waits"]) or [0.0]
    pictures = [r["pictures"] / elapsed for r in results]
    total_bytes = sum(r["bytes"] for r in results)
    return {
        "per_student": sum(pictures) / len(pictures),
        "slowest_student": min(pictures),
        "total": sum(pictures),
        "kb_per_picture": total_bytes / max(1, sum(r["pictures"] for r in results)) / 1000,
        "mbit_per_student": total_bytes * 8 / elapsed / len(results) / 1e6,
        "median_wait_ms": 1000 * waits[len(waits) // 2],
        "p95_wait_ms": 1000 * waits[int(len(waits) * 0.95)],
        "busy": sum(r["busy"] for r in results),
        "errors": sum(r["errors"] for r in results),
        "loop_hz": admin.json()["loop_hz"],
    }


def measure(seats: int, mesh_detail_mm: float, picture_rate: float, seconds: float) -> dict:
    """Start a hub with these settings, load it, stop it, and return the figures."""
    port = _free_port()
    with tempfile.TemporaryDirectory() as tmp:
        roster_path = Path(tmp) / "roster.json"
        command = [
            sys.executable, "-m", "xbox_soarm_teleop.cli.hub", "--seats", str(seats),
            "--port", str(port), "--no-log", "--roster", str(roster_path),
            "--mesh-detail-mm", str(mesh_detail_mm), "--picture-rate", str(picture_rate),
        ]  # fmt: skip
        hub = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            base = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if hub.poll() is not None:
                    raise RuntimeError("the hub stopped while starting")
                try:
                    if roster_path.exists() and httpx.get(base + "/", timeout=1).status_code == 200:
                        break
                except Exception:  # noqa: BLE001 - not listening yet
                    pass
                time.sleep(0.2)
            else:
                raise RuntimeError("the hub did not start within 2 minutes")
            roster = json.loads(roster_path.read_text())
            return asyncio.run(_run(base, roster, seconds, warmup=3.0))
        finally:
            hub.terminate()
            try:
                hub.wait(timeout=15)
            except subprocess.TimeoutExpired:
                hub.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seats", type=int, action="append", help="Students; repeat to try several")
    parser.add_argument(
        "--mesh-detail-mm", type=float, action="append", help="Repeatable; 0 = originals"
    )
    parser.add_argument("--picture-rate", type=float, action="append", help="Repeatable")
    parser.add_argument("--seconds", type=float, default=10.0, help="Measured time per row")
    args = parser.parse_args()
    print(
        "students  meshes  asked/s  got/s each  slowest  kB/picture  Mbit/s each  "
        "wait ms (median, 95%)  loop Hz  busy  errors"
    )
    for seats in args.seats or [10]:
        for detail in args.mesh_detail_mm or [2.0]:
            for rate in args.picture_rate or [10.0]:
                r = measure(seats, detail, rate, args.seconds)
                meshes = "full" if detail <= 0 else f"{detail:g} mm"
                print(
                    f"{seats:8d}  {meshes:6s}  {rate:7.0f}  {r['per_student']:10.1f}  "
                    f"{r['slowest_student']:7.1f}  {r['kb_per_picture']:10.1f}  "
                    f"{r['mbit_per_student']:11.2f}  {r['median_wait_ms']:8.0f}, {r['p95_wait_ms']:5.0f}"
                    f"        {r['loop_hz']:7.1f}  {r['busy']:4d}  {r['errors']:6d}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
