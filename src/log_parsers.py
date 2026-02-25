from __future__ import annotations

from typing import Any

from src.schema import DEFAULT_FEATURE_ROW


def _base_row() -> dict[str, Any]:
    return DEFAULT_FEATURE_ROW.copy()


def parse_zeek_conn_log(record: dict[str, Any]) -> dict[str, Any]:
    """Map a simplified Zeek conn.log-like record to model feature schema."""
    row = _base_row()
    proto = str(record.get("proto", "tcp")).lower()
    row["protocol_type"] = proto if proto in {"tcp", "udp", "icmp"} else "tcp"
    row["duration"] = float(record.get("duration", 0) or 0)
    row["src_bytes"] = float(record.get("orig_bytes", 0) or 0)
    row["dst_bytes"] = float(record.get("resp_bytes", 0) or 0)
    row["count"] = float(record.get("conn_count", 1) or 1)
    row["srv_count"] = float(record.get("service_count", 1) or 1)
    row["same_srv_rate"] = float(record.get("same_srv_rate", 1.0) or 1.0)
    row["serror_rate"] = float(record.get("serror_rate", 0.0) or 0.0)
    return row


def parse_suricata_eve_flow(record: dict[str, Any]) -> dict[str, Any]:
    """Map a simplified Suricata eve flow-like record to model feature schema."""
    row = _base_row()
    proto = str(record.get("proto", "TCP")).lower()
    row["protocol_type"] = proto if proto in {"tcp", "udp", "icmp"} else "tcp"
    flow = record.get("flow", {}) or {}
    row["duration"] = float(flow.get("age", 0) or 0)
    row["src_bytes"] = float(flow.get("bytes_toserver", 0) or 0)
    row["dst_bytes"] = float(flow.get("bytes_toclient", 0) or 0)
    row["count"] = float(flow.get("pkts_toserver", 1) or 1)
    row["srv_count"] = float(flow.get("pkts_toclient", 1) or 1)
    row["same_srv_rate"] = float(record.get("same_srv_rate", 1.0) or 1.0)
    row["serror_rate"] = float(record.get("serror_rate", 0.0) or 0.0)
    return row
