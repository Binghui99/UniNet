"""Small, dependency-free classic-PCAP reader for common traffic captures.

The reader intentionally supports a conservative set of link/network protocols:
Ethernet (including VLAN tags), Linux cooked capture v1, raw IPv4/IPv6, TCP, and
UDP. Unsupported or malformed packets are skipped and counted instead of being
silently mis-decoded.
"""

from __future__ import annotations

import ipaddress
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterator, List, Optional, Tuple, Union

from .schema import Packet


class PcapError(ValueError):
    pass


@dataclass
class PcapStats:
    records: int = 0
    decoded: int = 0
    non_ip: int = 0
    unsupported: int = 0
    malformed: int = 0


MAGIC = {
    b"\xd4\xc3\xb2\xa1": ("<", 1_000_000.0),
    b"\xa1\xb2\xc3\xd4": (">", 1_000_000.0),
    b"\x4d\x3c\xb2\xa1": ("<", 1_000_000_000.0),
    b"\xa1\xb2\x3c\x4d": (">", 1_000_000_000.0),
}


class ClassicPcapReader:
    """Stream decoded packet metadata from a classic ``.pcap`` file."""

    def __init__(self, path: Union[str, Path], max_record_bytes: int = 16 * 1024 * 1024):
        self.path = Path(path)
        self.max_record_bytes = max_record_bytes
        self.stats = PcapStats()

    def __iter__(self) -> Iterator[Packet]:
        with self.path.open("rb") as stream:
            yield from self._read(stream)

    def _read(self, stream: BinaryIO) -> Iterator[Packet]:
        magic = stream.read(4)
        if magic == b"\x0a\x0d\x0d\x0a":
            raise PcapError("PCAPNG is not supported by the built-in reader; convert it with "
                            "editcap -F pcap input.pcapng output.pcap")
        if magic not in MAGIC:
            raise PcapError("Not a supported classic PCAP file")
        endian, resolution = MAGIC[magic]
        header = stream.read(20)
        if len(header) != 20:
            raise PcapError("Truncated PCAP global header")
        _major, _minor, _tz, _sigfigs, _snaplen, linktype = struct.unpack(
            endian + "HHIIII", header
        )
        if linktype not in {1, 101, 113}:
            raise PcapError(f"Unsupported PCAP link type {linktype}; supported: Ethernet, RAW, SLL")

        record_struct = struct.Struct(endian + "IIII")
        while True:
            raw_header = stream.read(16)
            if not raw_header:
                break
            if len(raw_header) != 16:
                self.stats.malformed += 1
                break
            sec, fraction, captured_len, original_len = record_struct.unpack(raw_header)
            if captured_len > self.max_record_bytes:
                raise PcapError(
                    f"PCAP record declares {captured_len} bytes; safety limit is "
                    f"{self.max_record_bytes} bytes"
                )
            frame = stream.read(captured_len)
            self.stats.records += 1
            if len(frame) != captured_len:
                self.stats.malformed += 1
                break
            try:
                decoded = decode_frame(frame, linktype, original_len or captured_len)
            except (IndexError, struct.error, ValueError):
                self.stats.malformed += 1
                continue
            if decoded is None:
                self.stats.non_ip += 1
                continue
            src, dst, src_port, dst_port, protocol, tcp_flags = decoded
            if protocol not in {6, 17}:
                self.stats.unsupported += 1
            packet = Packet(
                timestamp=sec + fraction / resolution,
                src_ip=src,
                dst_ip=dst,
                src_port=src_port,
                dst_port=dst_port,
                protocol=protocol,
                size=original_len or captured_len,
                tcp_flags=tcp_flags,
            )
            self.stats.decoded += 1
            yield packet


def decode_frame(
    frame: bytes, linktype: int, original_len: int
) -> Optional[Tuple[str, str, int, int, int, int]]:
    del original_len
    if linktype == 1:
        if len(frame) < 14:
            raise ValueError("short Ethernet frame")
        ethertype = struct.unpack("!H", frame[12:14])[0]
        offset = 14
        while ethertype in {0x8100, 0x88A8, 0x9100}:
            if len(frame) < offset + 4:
                raise ValueError("short VLAN header")
            ethertype = struct.unpack("!H", frame[offset + 2 : offset + 4])[0]
            offset += 4
    elif linktype == 113:
        if len(frame) < 16:
            raise ValueError("short Linux SLL header")
        ethertype = struct.unpack("!H", frame[14:16])[0]
        offset = 16
    else:  # DLT_RAW
        if not frame:
            raise ValueError("empty raw packet")
        version = frame[0] >> 4
        ethertype = 0x0800 if version == 4 else 0x86DD if version == 6 else 0
        offset = 0

    payload = frame[offset:]
    if ethertype == 0x0800:
        network = _decode_ipv4(payload)
    elif ethertype == 0x86DD:
        network = _decode_ipv6(payload)
    else:
        return None
    if network is None:
        return None
    src, dst, protocol, transport = network
    src_port, dst_port, flags = _decode_transport(protocol, transport)
    return src, dst, src_port, dst_port, protocol, flags


def _decode_ipv4(data: bytes) -> Optional[Tuple[str, str, int, bytes]]:
    if len(data) < 20 or data[0] >> 4 != 4:
        raise ValueError("invalid IPv4 header")
    ihl = (data[0] & 0x0F) * 4
    if ihl < 20 or len(data) < ihl:
        raise ValueError("invalid IPv4 IHL")
    fragment = struct.unpack("!H", data[6:8])[0]
    if fragment & 0x1FFF:  # non-initial fragment has no transport header
        return None
    protocol = data[9]
    src = str(ipaddress.ip_address(data[12:16]))
    dst = str(ipaddress.ip_address(data[16:20]))
    return src, dst, protocol, data[ihl:]


def _decode_ipv6(data: bytes) -> Optional[Tuple[str, str, int, bytes]]:
    if len(data) < 40 or data[0] >> 4 != 6:
        raise ValueError("invalid IPv6 header")
    next_header = data[6]
    offset = 40
    while next_header in {0, 43, 44, 51, 60}:
        if len(data) < offset + 8:
            raise ValueError("short IPv6 extension header")
        following = data[offset]
        if next_header == 44:
            fragment = struct.unpack("!H", data[offset + 2 : offset + 4])[0]
            if fragment & 0xFFF8:
                return None
            length = 8
        elif next_header == 51:
            length = (data[offset + 1] + 2) * 4
        else:
            length = (data[offset + 1] + 1) * 8
        next_header = following
        offset += length
    src = str(ipaddress.ip_address(data[8:24]))
    dst = str(ipaddress.ip_address(data[24:40]))
    return src, dst, next_header, data[offset:]


def _decode_transport(protocol: int, data: bytes) -> Tuple[int, int, int]:
    if protocol == 6:
        if len(data) < 20:
            return -1, -1, 0
        src_port, dst_port = struct.unpack("!HH", data[:4])
        flags = data[13] | ((data[12] & 0x01) << 8)
        return src_port, dst_port, flags
    if protocol == 17:
        if len(data) < 8:
            return -1, -1, 0
        src_port, dst_port = struct.unpack("!HH", data[:4])
        return src_port, dst_port, 0
    return -1, -1, 0


def read_pcap(path: Union[str, Path]) -> Tuple[List[Packet], PcapStats]:
    reader = ClassicPcapReader(path)
    packets = list(reader)
    return packets, reader.stats
