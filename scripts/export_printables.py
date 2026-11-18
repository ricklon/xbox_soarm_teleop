"""Writes assets/printables/: each object as STEP, STL and SolidCraft design.

Run from the repo root (needs SolidCraft's command line tool):

    uv run python scripts/export_printables.py
    uv run python scripts/export_printables.py --cli /path/to/solidcraft-cli

Sizes come from the simulation (xbox_soarm_teleop.sim.printables), so rerun
this whenever an object's size changes; tests/test_printables.py checks the
files still match.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from xbox_soarm_teleop.sim.printables import parts, preview_script, solidcraft_script

DEFAULT_CLI = "/Applications/SolidCraft.app/Contents/MacOS/solidcraft-cli"
OUT = Path("assets/printables")


def build(cli: str, part, folder: Path) -> dict:
    """Build one part with SolidCraft; return what it measured."""
    design = folder / "solidcraft" / f"{part.name}.solidcraft"
    design.unlink(missing_ok=True)  # saving over it would leave a .bak beside it
    with tempfile.TemporaryDirectory() as scratch:
        script = Path(scratch) / f"{part.name}.json"
        script.write_text(json.dumps(solidcraft_script(part), indent=1))
        subprocess.run(
            [
                cli, "run", str(script),
                "--out", str(folder / "step" / f"{part.name}.step"),
                "--out", str(folder / "stl" / f"{part.name}.stl"),
                "--save", str(design),
                "--quiet",
            ],
            check=True,
        )  # fmt: skip
    measured = subprocess.run(
        [cli, "eval", "--in", str(folder / "step" / f"{part.name}.step")],
        check=True,
        capture_output=True,
        text=True,
    )
    body = json.loads(measured.stdout)["bodies"][0]
    if not body["closed"] or body["kernel"]["shells"] != 1:
        raise SystemExit(f"{part.name}: SolidCraft did not make one closed solid: {body}")
    return body


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cli", default=DEFAULT_CLI, help="solidcraft-cli to run")
    args = parser.parse_args()
    for sub in ("step", "stl", "solidcraft"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    for part in parts():
        body = build(args.cli, part, OUT)
        volume = body["mass_g"] / body["density_g_cm3"] * 1000.0
        print(
            f"{part.name:16s} {part.outer_mm} volume {volume:9.0f} mm3 (expected {part.volume_mm3():9.0f})"
        )
    with tempfile.TemporaryDirectory() as scratch:
        script = Path(scratch) / "preview.json"
        script.write_text(json.dumps(preview_script(), indent=1))
        subprocess.run(
            [args.cli, "snapshot", str(script), "--out", str(OUT / "preview.png"), "--view", "iso",
             "--width", "1600", "--height", "800", "--no-axes"],
            check=True,
        )  # fmt: skip
    print(f"wrote {OUT}/step, stl, solidcraft and preview.png")


if __name__ == "__main__":
    main()
