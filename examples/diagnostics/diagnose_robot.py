#!/usr/bin/env python3
"""Compatibility wrapper for the packaged robot-diagnostics CLI.

Prefer ``uv run diagnose-robot`` for normal use.
"""

from xbox_soarm_teleop.cli.diagnose_robot import main

if __name__ == "__main__":
    raise SystemExit(main())
