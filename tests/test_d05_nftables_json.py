"""D-05: nftables JSON handle parsing security regression tests.

Verifies that NftablesFirewallAdapter uses nft -j JSON output mode for
unblock() and list_rules() — not fragile text parsing.
Rollback boundary: revert firewall_adapters.py if this breaks.
"""
import json
import subprocess
from unittest.mock import MagicMock
import pytest
from src.firewall_adapters import NftablesFirewallAdapter


def _make_adapter(responses: dict[tuple, tuple]) -> NftablesFirewallAdapter:
    """Create adapter with a mock run_cmd that returns pre-defined responses."""
    def run_cmd(args, **kwargs):
        key = tuple(args)
        result = MagicMock()
        if key in responses:
            rc, stdout = responses[key]
        else:
            rc, stdout = 1, ""
        result.returncode = rc
        result.stdout = stdout
        result.stderr = ""
        return result
    return NftablesFirewallAdapter(run_cmd=run_cmd)


# ---------------------------------------------------------------------------
# Helpers: build nft JSON responses
# ---------------------------------------------------------------------------

def _build_nft_json(rules: list[dict]) -> str:
    """Build a minimal nft -j list chain JSON response."""
    items = [{"metainfo": {"version": "1.0.3", "release_name": "Topsy Turvy"}}]
    for r in rules:
        items.append({"rule": r})
    return json.dumps({"nftables": items})


def _drop_rule(handle: int, ip: str, protocol: str = "ip") -> dict:
    return {
        "family": "inet",
        "table": "filter",
        "chain": "input",
        "handle": handle,
        "expr": [
            {
                "match": {
                    "op": "==",
                    "left": {"payload": {"protocol": protocol, "field": "saddr"}},
                    "right": ip,
                }
            },
            {"drop": None},
        ],
    }


def _accept_rule(handle: int) -> dict:
    return {"family": "inet", "table": "filter", "chain": "input", "handle": handle, "expr": [{"accept": None}]}


# ---------------------------------------------------------------------------
# D-05-1: JSON mode used for unblock
# ---------------------------------------------------------------------------

class TestUnblockUsesJson:
    def test_unblock_sends_j_flag(self):
        """unblock() must invoke nft with -j flag."""
        calls = []

        def run_cmd(args, **kwargs):
            calls.append(list(args))
            result = MagicMock()
            result.returncode = 0
            result.stdout = _build_nft_json([])
            result.stderr = ""
            return result

        adapter = NftablesFirewallAdapter(run_cmd=run_cmd)
        adapter.unblock("1.2.3.4")
        list_calls = [c for c in calls if "list" in c]
        assert any("-j" in c for c in list_calls), (
            "unblock() must pass -j flag for JSON mode"
        )

    def test_unblock_returns_true_when_no_rules(self):
        """Unblocking an IP that has no rules returns True (no-op success)."""
        nft_json = _build_nft_json([])
        adapter = _make_adapter({
            ("nft", "-j", "list", "chain", "inet", "filter", "input"): (0, nft_json),
        })
        assert adapter.unblock("1.2.3.4") is True

    def test_unblock_deletes_correct_handle(self):
        """unblock() extracts the handle from JSON and deletes the right rule."""
        target = "1.2.3.4"
        rule = _drop_rule(handle=42, ip=target)
        nft_json = _build_nft_json([rule])
        deleted_handles = []

        def run_cmd(args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = nft_json
            result.stderr = ""
            if "delete" in args:
                deleted_handles.append(args[-1])
            return result

        adapter = NftablesFirewallAdapter(run_cmd=run_cmd)
        ok = adapter.unblock(target)
        assert ok is True
        assert "42" in deleted_handles

    def test_unblock_multiple_handles(self):
        """Multiple rules for same IP: all handles must be deleted."""
        target = "5.6.7.8"
        rules = [_drop_rule(7, target), _drop_rule(13, target)]
        nft_json = _build_nft_json(rules)
        deleted_handles = []

        def run_cmd(args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = nft_json
            result.stderr = ""
            if "delete" in args:
                deleted_handles.append(args[-1])
            return result

        adapter = NftablesFirewallAdapter(run_cmd=run_cmd)
        ok = adapter.unblock(target)
        assert ok is True
        assert "7" in deleted_handles
        assert "13" in deleted_handles

    def test_unblock_does_not_delete_other_ips(self):
        """Other IPs in the table are not touched."""
        rules = [_drop_rule(1, "1.1.1.1"), _drop_rule(2, "9.9.9.9")]
        nft_json = _build_nft_json(rules)
        deleted_handles = []

        def run_cmd(args, **kwargs):
            result = MagicMock()
            result.returncode = 0
            result.stdout = nft_json
            result.stderr = ""
            if "delete" in args:
                deleted_handles.append(args[-1])
            return result

        adapter = NftablesFirewallAdapter(run_cmd=run_cmd)
        adapter.unblock("1.1.1.1")
        assert "2" not in deleted_handles  # 9.9.9.9's handle must not be deleted

    def test_unblock_returns_false_on_delete_failure(self):
        """If any delete command fails, unblock() returns False."""
        target = "3.3.3.3"
        nft_json = _build_nft_json([_drop_rule(99, target)])
        call_count = [0]

        def run_cmd(args, **kwargs):
            result = MagicMock()
            result.stderr = ""
            if "list" in args:
                result.returncode = 0
                result.stdout = nft_json
            else:
                result.returncode = 1
                result.stdout = ""
            return result

        adapter = NftablesFirewallAdapter(run_cmd=run_cmd)
        ok = adapter.unblock(target)
        assert ok is False


# ---------------------------------------------------------------------------
# D-05-2: JSON mode used for list_rules
# ---------------------------------------------------------------------------

class TestListRulesUsesJson:
    def test_list_rules_sends_j_flag(self):
        """list_rules() must invoke nft with -j flag."""
        calls = []

        def run_cmd(args, **kwargs):
            calls.append(list(args))
            result = MagicMock()
            result.returncode = 0
            result.stdout = _build_nft_json([])
            result.stderr = ""
            return result

        adapter = NftablesFirewallAdapter(run_cmd=run_cmd)
        adapter.list_rules()
        assert any("-j" in c for c in calls), "list_rules() must pass -j flag"

    def test_list_rules_returns_blocked_ips(self):
        """list_rules() returns the IPs with active drop rules."""
        nft_json = _build_nft_json([_drop_rule(1, "1.2.3.4"), _drop_rule(2, "5.6.7.8")])
        adapter = _make_adapter({
            ("nft", "-j", "list", "chain", "inet", "filter", "input"): (0, nft_json),
        })
        rules = adapter.list_rules()
        assert "1.2.3.4" in rules
        assert "5.6.7.8" in rules

    def test_list_rules_excludes_accept_rules(self):
        """Accept rules must not appear in list_rules() output."""
        nft_json = _build_nft_json([_drop_rule(1, "1.2.3.4"), _accept_rule(2)])
        adapter = _make_adapter({
            ("nft", "-j", "list", "chain", "inet", "filter", "input"): (0, nft_json),
        })
        rules = adapter.list_rules()
        assert len(rules) == 1
        assert "1.2.3.4" in rules

    def test_list_rules_empty_on_nft_failure(self):
        """If nft fails, list_rules() returns an empty list."""
        adapter = _make_adapter({
            ("nft", "-j", "list", "chain", "inet", "filter", "input"): (1, ""),
        })
        assert adapter.list_rules() == []

    def test_list_rules_empty_on_invalid_json(self):
        """Malformed JSON from nft must not crash; returns empty list."""
        adapter = _make_adapter({
            ("nft", "-j", "list", "chain", "inet", "filter", "input"): (0, "INVALID JSON {{"),
        })
        assert adapter.list_rules() == []


# ---------------------------------------------------------------------------
# D-05-3: _parse_handles_json unit tests
# ---------------------------------------------------------------------------

class TestParseHandlesJson:
    def setup_method(self):
        self.adapter = NftablesFirewallAdapter()

    def test_finds_handle_for_target(self):
        nft_json = _build_nft_json([_drop_rule(55, "10.0.0.1")])
        handles = self.adapter._parse_handles_json("10.0.0.1", nft_json)
        assert handles == [55]

    def test_does_not_find_handle_for_other_ip(self):
        nft_json = _build_nft_json([_drop_rule(55, "10.0.0.1")])
        handles = self.adapter._parse_handles_json("10.0.0.2", nft_json)
        assert handles == []

    def test_invalid_json_returns_empty(self):
        handles = self.adapter._parse_handles_json("1.2.3.4", "not json")
        assert handles == []

    def test_empty_json_returns_empty(self):
        handles = self.adapter._parse_handles_json("1.2.3.4", _build_nft_json([]))
        assert handles == []

    def test_multiple_rules_for_same_ip(self):
        nft_json = _build_nft_json([_drop_rule(10, "2.2.2.2"), _drop_rule(20, "2.2.2.2")])
        handles = self.adapter._parse_handles_json("2.2.2.2", nft_json)
        assert sorted(handles) == [10, 20]

    def test_non_drop_rules_excluded(self):
        nft_json = _build_nft_json([_accept_rule(99)])
        handles = self.adapter._parse_handles_json("1.2.3.4", nft_json)
        assert handles == []
