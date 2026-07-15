#!/usr/bin/env python3
"""Generate tiny deterministic token datasets for all four task CLIs."""

import json
from pathlib import Path
from typing import Any, Dict, List


PAD = 1041


def sample(label: Any, class_index: int, item_index: int, length: int = 64) -> Dict[str, Any]:
    real_length = 48
    base = 40 + class_index * 170
    tokens = [min(1039, base + ((position * 7 + item_index * 3) % 31)) for position in range(real_length)]
    tokens.extend([PAD] * (length - real_length))
    segments = [2] * 8 + [1] * 16 + [0] * (real_length - 24) + [0] * (length - real_length)
    return {
        "input": tokens,
        "true_value": [0] * length,
        "mask_index": [0] * length,
        "segment_label": segments,
        "sequence_label": label,
        "metadata": {
            "session_id": f"synthetic-{class_index}-{item_index}",
            "real_length": real_length,
            "synthetic": True,
        },
    }


def write_dataset(path: Path, classes: List[Any], samples_per_class: int = 8) -> None:
    samples = [
        sample(label, class_index, item_index)
        for class_index, label in enumerate(classes)
        for item_index in range(samples_per_class)
    ]
    payload = {
        "format": "uninet-tmatrix-tokenized",
        "schema_version": "1.0",
        "synthetic": True,
        "purpose": "software smoke test only",
        "samples": samples,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    output = Path("data/smoke/tasks")
    write_dataset(output / "task1_anomaly.json", [0, 1], 10)
    write_dataset(output / "task2_attack.json", ["benign", "ddos", "botnet"], 8)
    write_dataset(
        output / "task3_iot.json",
        ["camera", "thermostat", "smart-plug", "voice-assistant"],
        8,
    )
    write_dataset(output / "task4_website.json", ["site-a", "site-b", "site-c", "unknown"], 8)
    print(f"Wrote four synthetic task datasets to {output}")


if __name__ == "__main__":
    main()

