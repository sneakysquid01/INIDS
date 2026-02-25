from src.log_parsers import parse_zeek_conn_log, parse_suricata_eve_flow


def test_parse_zeek_conn_log_basic_mapping():
    out = parse_zeek_conn_log(
        {
            "proto": "tcp",
            "duration": 12.5,
            "orig_bytes": 100,
            "resp_bytes": 50,
            "conn_count": 4,
            "service_count": 3,
        }
    )
    assert out["protocol_type"] == "tcp"
    assert out["duration"] == 12.5
    assert out["src_bytes"] == 100.0
    assert out["dst_bytes"] == 50.0


def test_parse_suricata_flow_basic_mapping():
    out = parse_suricata_eve_flow(
        {
            "proto": "UDP",
            "flow": {
                "age": 5,
                "bytes_toserver": 200,
                "bytes_toclient": 25,
                "pkts_toserver": 2,
                "pkts_toclient": 1,
            },
        }
    )
    assert out["protocol_type"] == "udp"
    assert out["duration"] == 5.0
    assert out["src_bytes"] == 200.0
    assert out["dst_bytes"] == 25.0
