#!/usr/bin/env python3
"""Compatibility wrapper for the IK smoke test CLI.

Prefer ``uv run ik-smoke`` for normal use.
"""

from xbox_soarm_teleop.cli.ik_smoke import main

if __name__ == "__main__":
    raise SystemExit(main())
