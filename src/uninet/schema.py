"""Typed, JSON-serializable objects used by the T-Matrix pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Packet:
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int
    size: int
    tcp_flags: int = 0
    direction: int = 0
    iat: float = 0.0


@dataclass
class Flow:
    key: Tuple[str, int, str, int, int]
    packets: List[Packet] = field(default_factory=list)


@dataclass
class Session:
    session_id: str
    context_ip: str
    window_start: float
    window_end: float
    flows: List[Flow] = field(default_factory=list)
    label: Optional[Any] = None


@dataclass
class TMatrix:
    """One session represented at session, flow, and packet granularities."""

    session: Session
    session_features: Dict[str, float]
    flow_features: List[Dict[str, float]]
    packet_features: List[List[Dict[str, float]]]
    source: str = ""
    schema_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["session"]["flows"] = [
            {"key": list(flow.key), "packet_count": len(flow.packets)}
            for flow in self.session.flows
        ]
        return result


@dataclass(frozen=True)
class Feature:
    name: str
    value: float
    segment: int
    kind: str

