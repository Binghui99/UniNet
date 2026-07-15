#!/usr/bin/env python3
"""Standalone UniNet multi-input -> T-Matrix command.

This root-level entry works from a source checkout before installation. After
``pip install -e .``, the equivalent command is simply ``tmatrix``.
"""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from uninet.standardize import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

