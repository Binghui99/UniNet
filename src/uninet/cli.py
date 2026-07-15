"""Command-line interface for UniNet preprocessing and smoke data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .synthetic import generate_smoke_pcaps
from .tmatrix import ExtractionConfig, TMatrixExtractor
from .tokenizer import TMatrixTokenizer


def _config_from_args(args: argparse.Namespace) -> ExtractionConfig:
    return ExtractionConfig(
        window_seconds=args.window_seconds,
        flow_timeout_seconds=args.flow_timeout_seconds,
        context_mode=args.context_mode,
        key_ip=args.key_ip,
        internal_networks=args.internal_network or ExtractionConfig().internal_networks,
        max_packets_per_flow=args.max_packets_per_flow,
    )


def convert(args: argparse.Namespace) -> int:
    extractor = TMatrixExtractor(_config_from_args(args))
    matrices = []
    source_stats = []
    for input_path in args.input:
        found, stats = extractor.from_pcap(input_path, label=args.label)
        matrices.extend(found)
        source_stats.append({"path": str(input_path), **vars(stats)})
    if not matrices:
        print("No IP sessions were extracted", file=sys.stderr)
        return 2
    tokenizer = TMatrixTokenizer(bins=args.bins)
    if args.tokenizer_in:
        tokenizer = TMatrixTokenizer.load(args.tokenizer_in)
        samples = [
            tokenizer.transform(matrix, args.max_tokens, args.mask_ratio, args.seed + index)
            for index, matrix in enumerate(matrices)
        ]
    else:
        samples = tokenizer.fit_transform(matrices, args.max_tokens, args.mask_ratio, args.seed)
    payload = {
        "format": "uninet-tmatrix-tokenized",
        "schema_version": "1.0",
        "vocabulary": {"size": 1042, "mask_token": 1040, "pad_token": 1041},
        "extraction": {
            "window_seconds": args.window_seconds,
            "flow_timeout_seconds": args.flow_timeout_seconds,
            "context_mode": args.context_mode,
            "max_tokens": args.max_tokens,
            "mask_ratio": args.mask_ratio,
        },
        "pcap_stats": source_stats,
        "samples": samples,
    }
    if args.format == "jsonl" or (args.format is None and Path(args.output).suffix == ".jsonl"):
        _write_jsonl(Path(args.output), payload)
    else:
        _write_json(Path(args.output), payload)
    if args.raw_output:
        _write_json(Path(args.raw_output), {
            "format": "uninet-tmatrix-raw",
            "schema_version": "1.0",
            "samples": [matrix.to_dict() for matrix in matrices],
        })
    if args.tokenizer_out:
        tokenizer.save(args.tokenizer_out)
    print(f"Wrote {len(samples)} session(s) to {args.output}")
    return 0


def smoke(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    captures = generate_smoke_pcaps(output_dir)
    extractor = TMatrixExtractor(ExtractionConfig(window_seconds=900, context_mode="internal"))
    matrices = []
    manifest: List[Dict[str, object]] = []
    for path, label, description in captures:
        found, stats = extractor.from_pcap(path, label=label)
        matrices.extend(found)
        manifest.append({
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "label": label,
            "description": description,
            "packets": stats.decoded,
            "sessions": len(found),
        })
    tokenizer = TMatrixTokenizer().fit(matrices)
    samples = [tokenizer.transform(matrix, max_tokens=args.max_tokens) for matrix in matrices]
    _write_json(output_dir / "tmatrix-smoke.json", {
        "format": "uninet-tmatrix-tokenized",
        "schema_version": "1.0",
        "synthetic": True,
        "samples": samples,
    })
    _write_json(output_dir / "manifest.json", {"synthetic": True, "captures": manifest})
    tokenizer.save(output_dir / "tokenizer.json")
    print(f"Generated {len(captures)} PCAPs and {len(samples)} T-Matrix sample(s) in {output_dir}")
    return 0


def inspect_dataset(args: argparse.Namespace) -> int:
    path = Path(args.dataset)
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        metadata = records[0].get("metadata", {}) if records else {}
        payload = {**metadata, "samples": [row["sample"] for row in records if row.get("record_type") == "sample"]}
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    samples = payload.get("samples", [])
    labels: Dict[str, int] = {}
    lengths = []
    for sample in samples:
        label = str(sample.get("sequence_label"))
        labels[label] = labels.get(label, 0) + 1
        metadata = sample.get("metadata", {})
        if "real_length" in metadata:
            lengths.append(metadata["real_length"])
    summary = {
        "format": payload.get("format"),
        "schema_version": payload.get("schema_version"),
        "samples": len(samples),
        "labels": labels,
        "token_length_min": min(lengths) if lengths else None,
        "token_length_max": max(lengths) if lengths else None,
    }
    print(json.dumps(summary, indent=2))
    return 0


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {key: value for key, value in payload.items() if key != "samples"}
    rows = [{"record_type": "metadata", "metadata": metadata}]
    rows.extend({"record_type": "sample", "sample": sample} for sample in payload["samples"])
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def _add_extraction_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--window-seconds", type=float, default=900.0)
    parser.add_argument("--flow-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--context-mode", choices=("internal", "src", "dst", "pair", "key-ip"), default="internal")
    parser.add_argument("--key-ip", help="Only include packets involving this IP (with --context-mode key-ip)")
    parser.add_argument("--internal-network", action="append", help="CIDR considered internal; repeat as needed")
    parser.add_argument("--max-packets-per-flow", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uninet", description="UniNet PCAP and T-Matrix tools")
    parser.add_argument("--version", action="version", version="UniNet 0.2.0")
    commands = parser.add_subparsers(dest="command", required=True)

    pcap = commands.add_parser("pcap2tmatrix", help="Convert one or more PCAPs to tokenized T-Matrix JSON")
    pcap.add_argument("input", nargs="+", type=Path)
    pcap.add_argument("-o", "--output", type=Path, required=True)
    pcap.add_argument("--format", choices=("json", "jsonl"), help="Default: infer JSONL from .jsonl, otherwise JSON")
    pcap.add_argument("--raw-output", type=Path, help="Also save human-readable un-tokenized features")
    pcap.add_argument("--tokenizer-in", type=Path, help="Reuse fitted training tokenizer")
    pcap.add_argument("--tokenizer-out", type=Path, help="Save fitted tokenizer boundaries")
    pcap.add_argument("--max-tokens", type=int, default=2000)
    pcap.add_argument("--bins", type=int, default=1040)
    pcap.add_argument("--mask-ratio", type=float, default=0.0)
    pcap.add_argument("--seed", type=int, default=0)
    pcap.add_argument("--label", help="Optional sequence label attached to all input captures")
    _add_extraction_arguments(pcap)
    pcap.set_defaults(func=convert)

    smoke_parser = commands.add_parser("smoke", help="Generate deterministic synthetic PCAP/T-Matrix fixtures")
    smoke_parser.add_argument("--output-dir", type=Path, default=Path("data/smoke"))
    smoke_parser.add_argument("--max-tokens", type=int, default=256)
    smoke_parser.set_defaults(func=smoke)

    inspect_parser = commands.add_parser("inspect", help="Summarize a tokenized T-Matrix JSON dataset")
    inspect_parser.add_argument("dataset", type=Path)
    inspect_parser.set_defaults(func=inspect_dataset)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
