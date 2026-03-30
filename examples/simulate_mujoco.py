#!/usr/bin/env python3
"""Compatibility wrapper for the packaged MuJoCo simulation CLI.

Prefer ``uv run simulate-mujoco`` for normal use.
"""

from xbox_soarm_teleop.cli.simulate_mujoco import main

if __name__ == "__main__":
    raise SystemExit(main())
