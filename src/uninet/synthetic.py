"""Deterministic synthetic PCAP fixtures for installation and CI smoke tests."""

from __future__ import annotations

import ipaddress
import struct
from pathlib import Path
from typing import Iterable, List, Tuple, Union


def internet_checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b"\x00"
    total = sum(struct.unpack(f"!{len(data) // 2}H", data))
    total = (total >> 16) + (total & 0xFFFF)
    total += total >> 16
    return (~total) & 0xFFFF


def ipv4_packet(src: str, dst: str, protocol: int, payload: bytes, ident: int = 0) -> bytes:
    src_bytes = ipaddress.ip_address(src).packed
    dst_bytes = ipaddress.ip_address(dst).packed
    header = struct.pack(
        "!BBHHHBBH4s4s",
        0x45,
        0,
        20 + len(payload),
        ident,
        0x4000,
        64,
        protocol,
        0,
        src_bytes,
        dst_bytes,
    )
    checksum = internet_checksum(header)
    return header[:10] + struct.pack("!H", checksum) + header[12:] + payload


def tcp_segment(src_port: int, dst_port: int, flags: int, payload: bytes = b"") -> bytes:
    return struct.pack("!HHIIBBHHH", src_port, dst_port, 1, 0, 0x50, flags, 64240, 0, 0) + payload


def udp_datagram(src_port: int, dst_port: int, payload: bytes = b"") -> bytes:
    return struct.pack("!HHHH", src_port, dst_port, 8 + len(payload), 0) + payload


def ethernet(payload: bytes, ethertype: int = 0x0800) -> bytes:
    return bytes.fromhex("00112233445566778899aabb") + struct.pack("!H", ethertype) + payload


def write_pcap(path: Union[str, Path], packets: Iterable[Tuple[float, bytes]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for timestamp, frame in packets:
            seconds = int(timestamp)
            micros = round((timestamp - seconds) * 1_000_000)
            stream.write(struct.pack("<IIII", seconds, micros, len(frame), len(frame)))
            stream.write(frame)


def _frame(src: str, dst: str, proto: int, transport: bytes, ident: int) -> bytes:
    return ethernet(ipv4_packet(src, dst, proto, transport, ident))


def benign_browse_packets(base: float = 1_700_000_000.0) -> List[Tuple[float, bytes]]:
    host, dns, web = "10.0.0.10", "8.8.8.8", "93.184.216.34"
    rows = [
        (0.00, _frame(host, dns, 17, udp_datagram(53000, 53, b"query"), 1)),
        (0.03, _frame(dns, host, 17, udp_datagram(53, 53000, b"answer"), 2)),
        (0.10, _frame(host, web, 6, tcp_segment(51000, 443, 0x02), 3)),
        (0.13, _frame(web, host, 6, tcp_segment(443, 51000, 0x12), 4)),
        (0.15, _frame(host, web, 6, tcp_segment(51000, 443, 0x10), 5)),
        (0.20, _frame(host, web, 6, tcp_segment(51000, 443, 0x18, b"encrypted-request"), 6)),
        (0.28, _frame(web, host, 6, tcp_segment(443, 51000, 0x18, b"encrypted-response" * 3), 7)),
        (0.35, _frame(host, web, 6, tcp_segment(51000, 443, 0x11), 8)),
    ]
    return [(base + offset, frame) for offset, frame in rows]


def dns_burst_packets(base: float = 1_700_001_000.0) -> List[Tuple[float, bytes]]:
    host, dns = "10.0.0.20", "1.1.1.1"
    rows = []
    for index in range(12):
        rows.append((index * 0.01, _frame(host, dns, 17, udp_datagram(54000 + index, 53, b"x" * 20), index)))
    return [(base + offset, frame) for offset, frame in rows]


def syn_scan_packets(base: float = 1_700_002_000.0) -> List[Tuple[float, bytes]]:
    host, target = "10.0.0.30", "203.0.113.8"
    rows = []
    for index, port in enumerate((21, 22, 23, 25, 53, 80, 110, 139, 443, 445, 3306, 8080)):
        rows.append((index * 0.005, _frame(host, target, 6, tcp_segment(55000 + index, port, 0x02), index)))
    return [(base + offset, frame) for offset, frame in rows]


def generate_smoke_pcaps(output_dir: Union[str, Path]) -> List[Tuple[Path, int, str]]:
    output_dir = Path(output_dir)
    captures = [
        ("benign_browse.pcap", benign_browse_packets(), 0, "benign"),
        ("dns_burst.pcap", dns_burst_packets(), 1, "synthetic_dns_burst"),
        ("syn_scan.pcap", syn_scan_packets(), 1, "synthetic_syn_scan"),
    ]
    result = []
    for filename, packets, label, description in captures:
        path = output_dir / filename
        write_pcap(path, packets)
        result.append((path, label, description))
    return result
