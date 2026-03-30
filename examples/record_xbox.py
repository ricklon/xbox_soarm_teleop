#!/usr/bin/env python3
"""Compatibility wrapper for the packaged recording CLI.

Prefer ``uv run record-xbox`` for normal use.
"""

from xbox_soarm_teleop.cli.record_xbox import main

if __name__ == "__main__":
    raise SystemExit(main())
