#!/usr/bin/env python3
"""Validate reference scores against the released table contract, offline."""
from pathlib import Path
import subprocess
import sys
if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).with_name("make_paper_tables.py")), "--check", "--no-write"]))
