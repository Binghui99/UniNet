#!/usr/bin/env python3
"""Task 1: MFP representation learning plus autoencoder anomaly detection."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uninet.task_runner import main_for_task  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main_for_task("anomaly"))

