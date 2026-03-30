#!/usr/bin/env python3
"""Compatibility wrapper for the packaged real-teleop CLI.

Prefer ``uv run teleoperate-real`` for normal use.
"""

from xbox_soarm_teleop.cli.teleoperate_real import main

if __name__ == "__main__":
    raise SystemExit(main())
