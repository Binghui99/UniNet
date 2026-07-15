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

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TMatrix":
        """Restore a raw T-Matrix document without inventing packet records."""

        session_payload = payload["session"]
        flows = [
            Flow(key=tuple(flow["key"]))
            for flow in session_payload.get("flows", [])
        ]
        session = Session(
            session_id=session_payload["session_id"],
            context_ip=session_payload["context_ip"],
            window_start=float(session_payload["window_start"]),
            window_end=float(session_payload["window_end"]),
            flows=flows,
            label=session_payload.get("label"),
        )
        return cls(
            session=session,
            session_features={key: float(value) for key, value in payload["session_features"].items()},
            flow_features=[
                {key: float(value) for key, value in row.items()}
                for row in payload["flow_features"]
            ],
            packet_features=[
                [{key: float(value) for key, value in row.items()} for row in flow]
                for flow in payload["packet_features"]
            ],
            source=payload.get("source", ""),
            schema_version=payload.get("schema_version", "1.0"),
        )


@dataclass(frozen=True)
class Feature:
    name: str
    value: float
    segment: int
    kind: str
