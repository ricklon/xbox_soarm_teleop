#!/usr/bin/env python3
"""Compatibility wrapper for the packaged joint-diagnostic analysis CLI.

Prefer ``uv run analyze-joint-diag`` for normal use.
"""

from xbox_soarm_teleop.cli.analyze_joint_diag import main

if __name__ == "__main__":
    raise SystemExit(main())
