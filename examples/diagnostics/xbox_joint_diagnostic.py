#!/usr/bin/env python3
"""Compatibility wrapper for the packaged Xbox joint-diagnostic CLI.

Prefer ``uv run xbox-joint-diagnostic`` for normal use.
"""

from xbox_soarm_teleop.cli.xbox_joint_diagnostic import main

if __name__ == "__main__":
    raise SystemExit(main())
