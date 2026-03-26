#!/usr/bin/env python3
"""Compatibility shim — use teleoperate_real.py instead.

This file is kept so existing bookmarks/docs don't break.
"""
import subprocess
import sys

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "examples/teleoperate_real.py"] + sys.argv[1:]))
