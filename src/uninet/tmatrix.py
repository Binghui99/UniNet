"""PCAP to multi-granular T-Matrix feature extraction."""

from __future__ import annotations

import ipaddress
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .pcap import PcapStats, read_pcap
from .schema import Feature, Flow, Packet, Session, TMatrix


DEFAULT_INTERNAL_NETWORKS = (
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "127.0.0.0/8",
    "fc00::/7",
    "fe80::/10",
    "::1/128",
)

SESSION_FEATURES = (
    "flow_count",
    "unique_src_ips",
    "unique_dst_ips",
    "unique_service_ports",
    "mean_flow_duration",
    "std_flow_duration",
    "mean_flow_bytes",
    "std_flow_bytes",
)
FLOW_FEATURES = (
    "duration",
    "mean_iat",
    "std_iat",
    "packet_count",
    "byte_count",
    "mean_packet_size",
    "std_packet_size",
    "tcp_flag_code",
)
PACKET_FEATURES = (
    "src_port",
    "dst_port",
    "protocol",
    "direction",
    "iat",
    "packet_size",
)

CONTINUOUS_FEATURES = {
    "flow_count",
    "unique_src_ips",
    "unique_dst_ips",
    "unique_service_ports",
    "mean_flow_duration",
    "std_flow_duration",
    "mean_flow_bytes",
    "std_flow_bytes",
    "duration",
    "mean_iat",
    "std_iat",
    "packet_count",
    "byte_count",
    "mean_packet_size",
    "std_packet_size",
    "iat",
    "packet_size",
}


@dataclass
class ExtractionConfig:
    """Controls how packets are grouped into sessions and bidirectional flows."""

    window_seconds: float = 900.0
    flow_timeout_seconds: float = 60.0
    context_mode: str = "internal"
    key_ip: Optional[str] = None
    internal_networks: Sequence[str] = field(default_factory=lambda: DEFAULT_INTERNAL_NETWORKS)
    max_packets_per_flow: Optional[int] = None

    def validate(self) -> None:
        if self.window_seconds <= 0 or self.flow_timeout_seconds <= 0:
            raise ValueError("window and flow timeout must be positive")
        if self.context_mode not in {"internal", "src", "dst", "pair", "key-ip"}:
            raise ValueError("context_mode must be internal, src, dst, pair, or key-ip")
        if self.context_mode == "key-ip" and not self.key_ip:
            raise ValueError("key_ip is required when context_mode is key-ip")
        if self.max_packets_per_flow is not None and self.max_packets_per_flow <= 0:
            raise ValueError("max_packets_per_flow must be positive")


class TMatrixExtractor:
    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self.config.validate()
        self._networks = tuple(ipaddress.ip_network(value) for value in self.config.internal_networks)

    def from_pcap(
        self, path: Union[str, Path], label: Optional[object] = None
    ) -> Tuple[List[TMatrix], PcapStats]:
        packets, stats = read_pcap(path)
        matrices = self.from_packets(packets, source=str(path), label=label)
        return matrices, stats

    def from_packets(
        self, packets: Iterable[Packet], source: str = "", label: Optional[object] = None
    ) -> List[TMatrix]:
        grouped: Dict[Tuple[str, int], List[Packet]] = defaultdict(list)
        for packet in sorted(packets, key=lambda item: item.timestamp):
            context = self._context(packet)
            if context is None:
                continue
            window_index = math.floor(packet.timestamp / self.config.window_seconds)
            packet.direction = 0 if packet.src_ip == context else 1
            grouped[(context, window_index)].append(packet)

        matrices = []
        for (context, window_index), session_packets in sorted(grouped.items()):
            start = window_index * self.config.window_seconds
            end = start + self.config.window_seconds
            flows = self._make_flows(session_packets)
            session_id = f"{context}@{start:.6f}"
            session = Session(session_id, context, start, end, flows, label)
            matrices.append(self._make_matrix(session, source))
        return matrices

    def _context(self, packet: Packet) -> Optional[str]:
        mode = self.config.context_mode
        if mode == "src":
            return packet.src_ip
        if mode == "dst":
            return packet.dst_ip
        if mode == "pair":
            return "<->".join(sorted((packet.src_ip, packet.dst_ip)))
        if mode == "key-ip":
            if self.config.key_ip in {packet.src_ip, packet.dst_ip}:
                return self.config.key_ip
            return None
        src_internal = self._is_internal(packet.src_ip)
        dst_internal = self._is_internal(packet.dst_ip)
        if src_internal and not dst_internal:
            return packet.src_ip
        if dst_internal and not src_internal:
            return packet.dst_ip
        if src_internal and dst_internal:
            return packet.src_ip
        # Public-to-public traces still remain usable; source is the observation context.
        return packet.src_ip

    def context_for(self, packet: Packet) -> Optional[str]:
        """Return the configured session context for an adapter-provided record."""

        return self._context(packet)

    def _is_internal(self, value: str) -> bool:
        address = ipaddress.ip_address(value)
        return any(address in network for network in self._networks if address.version == network.version)

    def _make_flows(self, packets: List[Packet]) -> List[Flow]:
        active: Dict[Tuple[Tuple[str, int], Tuple[str, int], int], Flow] = {}
        last_seen: Dict[Tuple[Tuple[str, int], Tuple[str, int], int], float] = {}
        completed: List[Flow] = []
        for packet in packets:
            endpoints = sorted(((packet.src_ip, packet.src_port), (packet.dst_ip, packet.dst_port)))
            canonical = (endpoints[0], endpoints[1], packet.protocol)
            flow = active.get(canonical)
            if flow is not None and packet.timestamp - last_seen[canonical] > self.config.flow_timeout_seconds:
                completed.append(flow)
                flow = None
            if flow is None:
                key = (packet.src_ip, packet.src_port, packet.dst_ip, packet.dst_port, packet.protocol)
                flow = Flow(key=key)
                active[canonical] = flow
            if self.config.max_packets_per_flow is None or len(flow.packets) < self.config.max_packets_per_flow:
                packet.iat = max(0.0, packet.timestamp - flow.packets[-1].timestamp) if flow.packets else 0.0
                flow.packets.append(packet)
            last_seen[canonical] = packet.timestamp
        completed.extend(active.values())
        return sorted(completed, key=lambda flow: flow.packets[0].timestamp if flow.packets else 0.0)

    def _make_matrix(self, session: Session, source: str) -> TMatrix:
        flow_features = [extract_flow_features(flow) for flow in session.flows]
        packet_features = [
            [extract_packet_features(packet) for packet in flow.packets] for flow in session.flows
        ]
        return TMatrix(
            session=session,
            session_features=extract_session_features(session, flow_features),
            flow_features=flow_features,
            packet_features=packet_features,
            source=source,
        )


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: Sequence[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def tcp_flag_code(mask: int) -> int:
    """Map TCP flags to the compact codes described in the UniNet paper."""

    flag_order = (0x10, 0x02, 0x01, 0x08, 0x20, 0x04, 0x40, 0x80, 0x100)
    active = [index + 1 for index, bit in enumerate(flag_order) if mask & bit]
    if len(active) == 1:
        return active[0]
    common = {0x12: 10, 0x18: 11, 0x30: 12, 0x11: 13, 0x14: 14}
    return common.get(mask & 0x1FF, 15 if active else 0)


def port_representation(port: int) -> int:
    if 1 <= port <= 1024:
        return port
    if port == 8080:
        return 1025
    if port == 3306:
        return 1026
    return 1027


def extract_packet_features(packet: Packet) -> Dict[str, float]:
    return {
        "src_port": float(port_representation(packet.src_port)),
        "dst_port": float(port_representation(packet.dst_port)),
        "protocol": float(packet.protocol),
        "direction": float(packet.direction),
        "iat": packet.iat,
        "packet_size": float(packet.size),
    }


def extract_flow_features(flow: Flow) -> Dict[str, float]:
    packets = flow.packets
    sizes = [float(packet.size) for packet in packets]
    iats = [packet.iat for packet in packets[1:]]
    duration = max(0.0, packets[-1].timestamp - packets[0].timestamp) if packets else 0.0
    flag_mask = 0
    for packet in packets:
        flag_mask |= packet.tcp_flags
    return {
        "duration": duration,
        "mean_iat": _mean(iats),
        "std_iat": _std(iats),
        "packet_count": float(len(packets)),
        "byte_count": float(sum(sizes)),
        "mean_packet_size": _mean(sizes),
        "std_packet_size": _std(sizes),
        "tcp_flag_code": float(tcp_flag_code(flag_mask)),
    }


def extract_session_features(
    session: Session, flow_features: Sequence[Dict[str, float]]
) -> Dict[str, float]:
    packets = [packet for flow in session.flows for packet in flow.packets]
    durations = [features["duration"] for features in flow_features]
    sizes = [features["byte_count"] for features in flow_features]
    endpoints = [flow.key for flow in session.flows]
    src_ips = {packet.src_ip for packet in packets} or {key[0] for key in endpoints}
    dst_ips = {packet.dst_ip for packet in packets} or {key[2] for key in endpoints}
    observed_ports = [
        port
        for packet in packets
        for port in (packet.src_port, packet.dst_port)
    ] or [port for key in endpoints for port in (key[1], key[3])]
    service_ports = {
        port
        for port in observed_ports
        if 1 <= port <= 1024 or port in {3306, 8080}
    }
    return {
        "flow_count": float(len(session.flows)),
        "unique_src_ips": float(len(src_ips)),
        "unique_dst_ips": float(len(dst_ips)),
        "unique_service_ports": float(len(service_ports)),
        "mean_flow_duration": _mean(durations),
        "std_flow_duration": _std(durations),
        "mean_flow_bytes": _mean(sizes),
        "std_flow_bytes": _std(sizes),
    }


def flatten_matrix(matrix: TMatrix) -> List[Feature]:
    features: List[Feature] = []
    for name in SESSION_FEATURES:
        features.append(Feature(name, matrix.session_features[name], 2, "continuous"))
    for index, flow_values in enumerate(matrix.flow_features):
        for name in FLOW_FEATURES:
            kind = "categorical" if name == "tcp_flag_code" else "continuous"
            features.append(Feature(f"flow[{index}].{name}", flow_values[name], 1, kind))
        for packet_index, packet_values in enumerate(matrix.packet_features[index]):
            for name in PACKET_FEATURES:
                kind = "continuous" if name in {"iat", "packet_size"} else "categorical"
                features.append(
                    Feature(f"flow[{index}].packet[{packet_index}].{name}", packet_values[name], 0, kind)
                )
    return features
