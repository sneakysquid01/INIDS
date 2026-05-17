"""D-03: Packet capture sanitization security regression tests.

Verifies that sanitize_packet_output() clamps numeric fields and sanitizes
string fields on Packet objects. Prevents injection via crafted packet data.
"""
import time
import pytest
from src.packet_capture.packet_stream import (
    Packet,
    sanitize_packet_output,
    PORT_MAX,
    PORT_MIN,
    PACKET_LEN_MAX,
    VLAN_ID_MAX,
)


def _make_clean_packet(**kwargs) -> Packet:
    defaults = dict(
        timestamp=time.time(),
        src_ip="1.2.3.4",
        dst_ip="5.6.7.8",
        src_port=12345,
        dst_port=80,
        protocol="tcp",
        packet_len=1500,
    )
    defaults.update(kwargs)
    return Packet(**defaults)


# ---------------------------------------------------------------------------
# D-03-1: Numeric field clamping
# ---------------------------------------------------------------------------

class TestNumericFieldClamping:
    def test_src_port_clamped_high(self):
        pkt = _make_clean_packet(src_port=99999)
        result = sanitize_packet_output(pkt)
        assert result.src_port == PORT_MAX

    def test_dst_port_clamped_high(self):
        pkt = _make_clean_packet(dst_port=99999)
        result = sanitize_packet_output(pkt)
        assert result.dst_port == PORT_MAX

    def test_src_port_clamped_low(self):
        pkt = _make_clean_packet(src_port=-5)
        result = sanitize_packet_output(pkt)
        assert result.src_port == PORT_MIN

    def test_packet_len_clamped_high(self):
        pkt = _make_clean_packet(packet_len=999999)
        result = sanitize_packet_output(pkt)
        assert result.packet_len == PACKET_LEN_MAX

    def test_packet_len_clamped_low(self):
        pkt = _make_clean_packet(packet_len=-100)
        result = sanitize_packet_output(pkt)
        assert result.packet_len == 0

    def test_vlan_id_clamped_high(self):
        pkt = _make_clean_packet(vlan_id=9999)
        result = sanitize_packet_output(pkt)
        assert result.vlan_id == VLAN_ID_MAX

    def test_vlan_id_clamped_low(self):
        pkt = _make_clean_packet(vlan_id=-1)
        result = sanitize_packet_output(pkt)
        assert result.vlan_id == 0

    def test_vlan_id_none_preserved(self):
        pkt = _make_clean_packet(vlan_id=None)
        result = sanitize_packet_output(pkt)
        assert result.vlan_id is None

    def test_timestamp_future_clamped(self):
        pkt = _make_clean_packet(timestamp=time.time() + 86400 * 365)
        result = sanitize_packet_output(pkt)
        assert result.timestamp <= time.time() + 86400 + 1  # within tolerance

    def test_timestamp_negative_clamped_to_zero(self):
        pkt = _make_clean_packet(timestamp=-1.0)
        result = sanitize_packet_output(pkt)
        assert result.timestamp == 0.0

    def test_valid_port_passes_through(self):
        pkt = _make_clean_packet(src_port=443, dst_port=8080)
        result = sanitize_packet_output(pkt)
        assert result.src_port == 443
        assert result.dst_port == 8080


# ---------------------------------------------------------------------------
# D-03-2: IP address sanitization
# ---------------------------------------------------------------------------

class TestIpSanitization:
    def test_valid_ipv4_preserved(self):
        pkt = _make_clean_packet(src_ip="192.168.1.1")
        result = sanitize_packet_output(pkt)
        assert result.src_ip == "192.168.1.1"

    def test_valid_ipv6_preserved(self):
        pkt = _make_clean_packet(src_ip="::1", dst_ip="2001:db8::1")
        result = sanitize_packet_output(pkt)
        assert result.src_ip == "::1"

    def test_invalid_ip_stripped(self):
        pkt = _make_clean_packet(src_ip="not-an-ip")
        result = sanitize_packet_output(pkt)
        assert result.src_ip == ""

    def test_injection_in_ip_stripped(self):
        pkt = _make_clean_packet(src_ip="1.2.3.4; rm -rf /")
        result = sanitize_packet_output(pkt)
        assert result.src_ip == ""

    def test_empty_ip_stays_empty(self):
        pkt = _make_clean_packet(src_ip="")
        result = sanitize_packet_output(pkt)
        assert result.src_ip == ""


# ---------------------------------------------------------------------------
# D-03-3: Protocol sanitization
# ---------------------------------------------------------------------------

class TestProtocolSanitization:
    def test_tcp_preserved(self):
        pkt = _make_clean_packet(protocol="tcp")
        result = sanitize_packet_output(pkt)
        assert result.protocol == "tcp"

    def test_udp_preserved(self):
        pkt = _make_clean_packet(protocol="UDP")
        result = sanitize_packet_output(pkt)
        assert result.protocol == "udp"

    def test_unknown_protocol_sanitized(self):
        pkt = _make_clean_packet(protocol="<script>alert(1)</script>")
        result = sanitize_packet_output(pkt)
        assert result.protocol == "unknown"

    def test_empty_protocol_sanitized(self):
        pkt = _make_clean_packet(protocol="")
        result = sanitize_packet_output(pkt)
        assert result.protocol == "unknown"


# ---------------------------------------------------------------------------
# D-03-4: MAC address sanitization
# ---------------------------------------------------------------------------

class TestMacSanitization:
    def test_valid_mac_preserved(self):
        pkt = _make_clean_packet(src_mac="aa:bb:cc:dd:ee:ff")
        result = sanitize_packet_output(pkt)
        assert result.src_mac == "aa:bb:cc:dd:ee:ff"

    def test_invalid_mac_stripped(self):
        pkt = _make_clean_packet(src_mac="not-a-mac; DROP TABLE")
        result = sanitize_packet_output(pkt)
        assert result.src_mac is None

    def test_none_mac_preserved(self):
        pkt = _make_clean_packet(src_mac=None)
        result = sanitize_packet_output(pkt)
        assert result.src_mac is None


# ---------------------------------------------------------------------------
# D-03-5: packet_data bytes safety
# ---------------------------------------------------------------------------

class TestPacketData:
    def test_valid_bytes_preserved(self):
        pkt = _make_clean_packet(packet_data=b"\x00\x01\x02\x03")
        result = sanitize_packet_output(pkt)
        assert result.packet_data == b"\x00\x01\x02\x03"

    def test_non_bytes_replaced_with_empty(self):
        pkt = _make_clean_packet(packet_data=None)
        # Force non-bytes by bypassing type check
        pkt.packet_data = None  # type: ignore
        result = sanitize_packet_output(pkt)
        assert result.packet_data == b""


# ---------------------------------------------------------------------------
# D-03-6: Immutability — original packet not mutated
# ---------------------------------------------------------------------------

class TestOriginalNotMutated:
    def test_original_unchanged_after_sanitize(self):
        pkt = _make_clean_packet(src_port=99999, protocol="<evil>")
        orig_src_port = pkt.src_port
        orig_protocol = pkt.protocol
        sanitize_packet_output(pkt)
        assert pkt.src_port == orig_src_port
        assert pkt.protocol == orig_protocol
