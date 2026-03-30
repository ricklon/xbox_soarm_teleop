#!/usr/bin/env python3
"""Compatibility wrapper for the packaged joint-ROM CLI.

Prefer ``uv run joint-rom-test`` for normal use.
"""

from xbox_soarm_teleop.cli.joint_rom_test import main

if __name__ == "__main__":
    raise SystemExit(main())
