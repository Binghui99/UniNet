"""Multi-input standardization into the canonical UniNet T-Matrix format."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import ipaddress
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import Flow, Packet, Session, TMatrix
from .tmatrix import (
    FLOW_FEATURES,
    ExtractionConfig,
    TMatrixExtractor,
    extract_session_features,
    tcp_flag_code,
)
from .tokenizer import TMatrixTokenizer


PACKET_ALIASES = {
    "timestamp": ("timestamp", "time", "ts", "start_time"),
    "src_ip": ("src_ip", "source_ip", "src", "saddr"),
    "dst_ip": ("dst_ip", "destination_ip", "dst", "daddr"),
    "src_port": ("src_port", "source_port", "sport"),
    "dst_port": ("dst_port", "destination_port", "dport"),
    "protocol": ("protocol", "proto", "ip_protocol"),
    "size": ("size", "length", "packet_size", "frame_len", "bytes"),
    "tcp_flags": ("tcp_flags", "flags", "tcp_flag"),
}

FLOW_ALIASES = {
    **PACKET_ALIASES,
    "duration": ("duration", "flow_duration"),
    "mean_iat": ("mean_iat", "avg_iat", "flow_iat_mean"),
    "std_iat": ("std_iat", "flow_iat_std"),
    "packet_count": ("packet_count", "packets", "tot_pkts"),
    "byte_count": ("byte_count", "flow_bytes", "tot_bytes"),
    "mean_packet_size": ("mean_packet_size", "avg_packet_size", "packet_size_mean"),
    "std_packet_size": ("std_packet_size", "packet_size_std"),
    "tcp_flag_code": ("tcp_flag_code",),
}

PROTOCOLS = {"icmp": 1, "tcp": 6, "udp": 17, "ipv6-icmp": 58, "icmpv6": 58}
FLAG_BITS = {
    "FIN": 0x01,
    "SYN": 0x02,
    "RST": 0x04,
    "PSH": 0x08,
    "ACK": 0x10,
    "URG": 0x20,
    "ECE": 0x40,
    "CWR": 0x80,
    "NS": 0x100,
}


class InputError(ValueError):
    """Raised when an adapter cannot safely interpret an input record."""


def _value(
    row: Mapping[str, Any], aliases: Iterable[str], default: Any = None, required: bool = False
) -> Any:
    lowered = {str(key).strip().lower(): value for key, value in row.items()}
    for alias in aliases:
        if alias in lowered and lowered[alias] not in (None, ""):
            return lowered[alias]
    if required:
        raise InputError(f"missing required field; accepted names: {', '.join(aliases)}")
    return default


def _number(value: Any, name: str, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise InputError(f"{name} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise InputError(f"{name} must be finite, got {value!r}")
    return result


def _integer(value: Any, name: str, default: int = 0) -> int:
    return int(_number(value, name, float(default)))


def _timestamp(value: Any) -> float:
    if isinstance(value, (int, float)):
        return _number(value, "timestamp")
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        try:
            parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise InputError(f"timestamp must be Unix seconds or ISO-8601, got {value!r}") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.timestamp()


def _protocol(value: Any) -> int:
    if isinstance(value, str) and not value.strip().isdigit():
        key = value.strip().lower()
        if key not in PROTOCOLS:
            raise InputError(f"unknown protocol {value!r}; use an IP protocol number")
        return PROTOCOLS[key]
    return _integer(value, "protocol")


def _flags(value: Any) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    try:
        return int(text, 0)
    except ValueError:
        parts = text.upper().replace("+", "|").replace(",", "|").split("|")
        unknown = [part.strip() for part in parts if part.strip() not in FLAG_BITS]
        if unknown:
            raise InputError(f"unknown TCP flags: {', '.join(unknown)}")
        return sum(FLAG_BITS[part.strip()] for part in parts if part.strip())


def packet_from_record(row: Mapping[str, Any]) -> Packet:
    src_ip = str(_value(row, PACKET_ALIASES["src_ip"], required=True))
    dst_ip = str(_value(row, PACKET_ALIASES["dst_ip"], required=True))
    try:
        ipaddress.ip_address(src_ip)
        ipaddress.ip_address(dst_ip)
    except ValueError as exc:
        raise InputError(f"invalid IP address in {src_ip!r} -> {dst_ip!r}") from exc
    return Packet(
        timestamp=_timestamp(_value(row, PACKET_ALIASES["timestamp"], required=True)),
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=_integer(_value(row, PACKET_ALIASES["src_port"], -1), "src_port", -1),
        dst_port=_integer(_value(row, PACKET_ALIASES["dst_port"], -1), "dst_port", -1),
        protocol=_protocol(_value(row, PACKET_ALIASES["protocol"], required=True)),
        size=_integer(_value(row, PACKET_ALIASES["size"], required=True), "size"),
        tcp_flags=_flags(_value(row, PACKET_ALIASES["tcp_flags"], 0)),
    )


def _read_records(path: Path, delimiter: Optional[str] = None) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        chosen = delimiter or ("\t" if suffix == ".tsv" else ",")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return [dict(row) for row in csv.DictReader(stream, delimiter=chosen)]
    if suffix == ".jsonl":
        records = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict) and value.get("record_type") in {"metadata", "sample"}:
                raise InputError("tokenized T-Matrix JSONL is already standardized")
            if not isinstance(value, dict):
                raise InputError(f"JSONL line {line_number} must be an object")
            records.append(value)
        return records
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("records", "packets", "flows"):
                if isinstance(payload.get(key), list):
                    return payload[key]
        raise InputError("JSON table must be a list or contain records/packets/flows")
    raise InputError(f"table input must be CSV, TSV, JSON, or JSONL, got {path.suffix}")


def _label(row: Mapping[str, Any], column: str, override: Any) -> Any:
    if override is not None:
        return override
    return _value(row, (column.lower(),), None)


def _label_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def packet_records_to_matrices(
    records: Sequence[Mapping[str, Any]],
    extractor: TMatrixExtractor,
    source: str,
    label_column: str = "label",
    label_override: Any = None,
) -> List[TMatrix]:
    grouped: Dict[str, Tuple[Any, List[Packet]]] = {}
    for index, row in enumerate(records, 1):
        try:
            packet = packet_from_record(row)
        except InputError as exc:
            raise InputError(f"{source}: packet record {index}: {exc}") from exc
        label = _label(row, label_column, label_override)
        grouped.setdefault(_label_key(label), (label, []))[1].append(packet)
    matrices = []
    for label, packets in grouped.values():
        matrices.extend(extractor.from_packets(packets, source=source, label=label))
    return matrices


def _flow_values(row: Mapping[str, Any]) -> Dict[str, float]:
    packet_count = _number(_value(row, FLOW_ALIASES["packet_count"], 0), "packet_count")
    byte_count = _number(_value(row, FLOW_ALIASES["byte_count"], 0), "byte_count")
    mean_size_default = byte_count / packet_count if packet_count else 0.0
    flags = _flags(_value(row, FLOW_ALIASES["tcp_flags"], 0))
    values = {
        "duration": _number(_value(row, FLOW_ALIASES["duration"], 0), "duration"),
        "mean_iat": _number(_value(row, FLOW_ALIASES["mean_iat"], 0), "mean_iat"),
        "std_iat": _number(_value(row, FLOW_ALIASES["std_iat"], 0), "std_iat"),
        "packet_count": packet_count,
        "byte_count": byte_count,
        "mean_packet_size": _number(
            _value(row, FLOW_ALIASES["mean_packet_size"], mean_size_default),
            "mean_packet_size",
        ),
        "std_packet_size": _number(
            _value(row, FLOW_ALIASES["std_packet_size"], 0), "std_packet_size"
        ),
        "tcp_flag_code": _number(
            _value(row, FLOW_ALIASES["tcp_flag_code"], tcp_flag_code(flags)),
            "tcp_flag_code",
        ),
    }
    return {name: values[name] for name in FLOW_FEATURES}


def flow_records_to_matrices(
    records: Sequence[Mapping[str, Any]],
    extractor: TMatrixExtractor,
    source: str,
    label_column: str = "label",
    label_override: Any = None,
) -> List[TMatrix]:
    grouped: Dict[Tuple[str, int, str], List[Tuple[float, Flow, Dict[str, float], Any]]] = defaultdict(list)
    for index, row in enumerate(records, 1):
        try:
            packet = packet_from_record({**row, "size": _value(row, PACKET_ALIASES["size"], 0)})
            context = extractor.context_for(packet)
            if context is None:
                continue
            label = _label(row, label_column, label_override)
            window = math.floor(packet.timestamp / extractor.config.window_seconds)
            flow = Flow((packet.src_ip, packet.src_port, packet.dst_ip, packet.dst_port, packet.protocol))
            values = _flow_values(row)
        except InputError as exc:
            raise InputError(f"{source}: flow record {index}: {exc}") from exc
        grouped[(context, window, _label_key(label))].append(
            (packet.timestamp, flow, values, label)
        )

    matrices = []
    for (context, window, _), rows in sorted(grouped.items()):
        rows.sort(key=lambda item: item[0])
        start = window * extractor.config.window_seconds
        flows = [item[1] for item in rows]
        flow_features = [item[2] for item in rows]
        session = Session(
            session_id=f"{context}@{start:.6f}",
            context_ip=context,
            window_start=start,
            window_end=start + extractor.config.window_seconds,
            flows=flows,
            label=rows[0][3],
        )
        matrices.append(
            TMatrix(
                session=session,
                session_features=extract_session_features(session, flow_features),
                flow_features=flow_features,
                packet_features=[[] for _ in flows],
                source=source,
            )
        )
    return matrices


def _load_raw_tmatrix(path: Path) -> List[TMatrix]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "uninet-tmatrix-raw":
        raise InputError("T-Matrix input must use format='uninet-tmatrix-raw'")
    return [TMatrix.from_dict(sample) for sample in payload.get("samples", [])]


def detect_input_kind(path: Path) -> str:
    if path.suffix.lower() in {".pcap", ".cap"}:
        return "pcap"
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("format") == "uninet-tmatrix-raw":
            return "tmatrix"
    records = _read_records(path)
    if not records:
        raise InputError(f"cannot infer empty input {path}; pass --input-kind")
    keys = {str(key).lower() for key in records[0]}
    flow_markers = {alias for name in FLOW_FEATURES for alias in FLOW_ALIASES[name]}
    return "flow" if keys & flow_markers else "packet"


def _parse_label(value: Optional[str]) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_tokenized(path: Path, payload: Dict[str, Any]) -> None:
    if path.suffix.lower() != ".jsonl":
        _write_json(path, payload)
        return
    metadata = {key: value for key, value in payload.items() if key != "samples"}
    rows = [json.dumps({"record_type": "metadata", "metadata": metadata})]
    rows.extend(
        json.dumps({"record_type": "sample", "sample": sample})
        for sample in payload["samples"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmatrix.py",
        description="Standardize PCAP, packet tables, flow tables, or raw T-Matrix into UniNet input",
    )
    parser.add_argument("input", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--input-kind", choices=("auto", "pcap", "packet", "flow", "tmatrix"), default="auto"
    )
    parser.add_argument(
        "--representation", choices=("raw", "tokenized", "both"), default="tokenized"
    )
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--tokenizer-in", type=Path)
    parser.add_argument("--tokenizer-out", type=Path)
    parser.add_argument("--label", help="Override all labels; accepts JSON values such as 0")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--delimiter", help="CSV delimiter; inferred for .csv/.tsv by default")
    parser.add_argument("--window-seconds", type=float, default=900.0)
    parser.add_argument("--flow-timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--context-mode", choices=("internal", "src", "dst", "pair", "key-ip"), default="internal"
    )
    parser.add_argument("--key-ip")
    parser.add_argument("--internal-network", action="append")
    parser.add_argument("--max-packets-per-flow", type=int)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--bins", type=int, default=1040)
    parser.add_argument("--mask-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = ExtractionConfig(
        window_seconds=args.window_seconds,
        flow_timeout_seconds=args.flow_timeout_seconds,
        context_mode=args.context_mode,
        key_ip=args.key_ip,
        internal_networks=args.internal_network or ExtractionConfig().internal_networks,
        max_packets_per_flow=args.max_packets_per_flow,
    )
    extractor = TMatrixExtractor(config)
    label_override = _parse_label(args.label)
    matrices: List[TMatrix] = []
    inputs = []
    try:
        for path in args.input:
            kind = detect_input_kind(path) if args.input_kind == "auto" else args.input_kind
            if kind == "pcap":
                found, stats = extractor.from_pcap(path, label=label_override)
                summary = {"path": str(path), "kind": kind, **vars(stats)}
            elif kind == "tmatrix":
                found = _load_raw_tmatrix(path)
                summary = {"path": str(path), "kind": kind, "records": len(found)}
            else:
                records = _read_records(path, args.delimiter)
                converter = packet_records_to_matrices if kind == "packet" else flow_records_to_matrices
                found = converter(
                    records,
                    extractor,
                    source=str(path),
                    label_column=args.label_column,
                    label_override=label_override,
                )
                summary = {"path": str(path), "kind": kind, "records": len(records)}
            matrices.extend(found)
            summary["samples"] = len(found)
            inputs.append(summary)
    except (InputError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"tmatrix: error: {exc}", file=sys.stderr)
        return 2

    if not matrices:
        print("tmatrix: error: no T-Matrix samples were produced", file=sys.stderr)
        return 2

    raw_payload = {
        "format": "uninet-tmatrix-raw",
        "schema_version": "1.0",
        "inputs": inputs,
        "samples": [matrix.to_dict() for matrix in matrices],
    }
    if args.representation in {"raw", "both"}:
        raw_path = args.output if args.representation == "raw" else (
            args.raw_output or args.output.with_name(args.output.stem + ".raw.json")
        )
        _write_json(raw_path, raw_payload)

    if args.representation in {"tokenized", "both"}:
        tokenizer = (
            TMatrixTokenizer.load(args.tokenizer_in)
            if args.tokenizer_in
            else TMatrixTokenizer(args.bins).fit(matrices)
        )
        samples = [
            tokenizer.transform(matrix, args.max_tokens, args.mask_ratio, args.seed + index)
            for index, matrix in enumerate(matrices)
        ]
        tokenized_payload = {
            "format": "uninet-tmatrix-tokenized",
            "schema_version": "1.0",
            "inputs": inputs,
            "vocabulary": {"size": 1042, "mask_token": 1040, "pad_token": 1041},
            "samples": samples,
        }
        _write_tokenized(args.output, tokenized_payload)
        if args.tokenizer_out:
            tokenizer.save(args.tokenizer_out)
    print(f"Standardized {len(matrices)} sample(s) from {len(args.input)} input file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
