#!/usr/bin/env python3
"""Generate deterministic PCAP and T-Matrix fixtures (install package first)."""

from uninet.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["smoke", "--output-dir", "data/smoke"]))

