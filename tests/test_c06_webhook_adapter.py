"""C-06 security regression tests — Webhook Adapter + Reconcile (Step 21).

Validation checkpoints per PLAN.md C-06 spec:
1. All adapters expose supports_rule_query class attribute
2. MockFirewallAdapter.supports_rule_query is True
3. UfwFirewallAdapter.supports_rule_query is True
4. NftablesFirewallAdapter.supports_rule_query is True
5. WebhookFirewallAdapter.supports_rule_query is False
6. WebhookFirewallAdapter.list_rules() raises NotImplementedError
7. WebhookFirewallAdapter no longer has a blocked_targets field
8. WebhookFirewallAdapter.block() does not mutate local state after call
9. WebhookFirewallAdapter.unblock() does not mutate local state after call
10. reconcile() returns early (skipped=True) when adapter.supports_rule_query is False
11. reconcile() compares rules normally when adapter.supports_rule_query is True
12. UfwFirewallAdapter.block() treats "Skipping" output as success (G-PREV-2)
13. UfwFirewallAdapter.block() treats "already exists" output as success (G-PREV-2)
14. WebhookFirewallAdapter logs a WARNING on instantiation (G-PREV-3)
15. Security regression: reconcile() with stateless adapter never calls list_rules()
"""
from __future__ import annotations

import logging
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.firewall_adapters import (
    FirewallAdapter,
    MockFirewallAdapter,
    NftablesFirewallAdapter,
    UfwFirewallAdapter,
    WebhookFirewallAdapter,
)
from src.ips.action_executor import ActionExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ufw_result(returncode: int, stdout: str = "", stderr: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def _make_executor(adapter: FirewallAdapter, ops_store=None) -> ActionExecutor:
    return ActionExecutor(adapter=adapter, adapter_name="test", ops_store=ops_store)


# ---------------------------------------------------------------------------
# 1–5: supports_rule_query attribute presence and values
# ---------------------------------------------------------------------------

def test_mock_adapter_supports_rule_query_true():
    assert MockFirewallAdapter().supports_rule_query is True


def test_ufw_adapter_supports_rule_query_true():
    assert UfwFirewallAdapter().supports_rule_query is True


def test_nftables_adapter_supports_rule_query_true():
    assert NftablesFirewallAdapter().supports_rule_query is True


def test_webhook_adapter_supports_rule_query_false():
    assert WebhookFirewallAdapter().supports_rule_query is False


def test_firewall_adapter_abc_default_supports_rule_query():
    assert FirewallAdapter.supports_rule_query is True


# ---------------------------------------------------------------------------
# 6: WebhookFirewallAdapter.list_rules() raises NotImplementedError
# ---------------------------------------------------------------------------

def test_webhook_list_rules_raises_not_implemented():
    adapter = WebhookFirewallAdapter()
    with pytest.raises(NotImplementedError):
        adapter.list_rules()


# ---------------------------------------------------------------------------
# 7–9: blocked_targets removed; no local state mutation
# ---------------------------------------------------------------------------

def test_webhook_adapter_has_no_blocked_targets_field():
    adapter = WebhookFirewallAdapter()
    assert not hasattr(adapter, "blocked_targets"), \
        "blocked_targets must be removed from WebhookFirewallAdapter"


def test_webhook_block_does_not_mutate_local_state():
    """block() must not accumulate state — the adapter is stateless."""
    adapter = WebhookFirewallAdapter(webhook_url="https://example.invalid")
    # Patch _post to succeed
    adapter._post = MagicMock(return_value=True)
    adapter.block("1.2.3.4", ttl_seconds=60)
    assert not hasattr(adapter, "blocked_targets")


def test_webhook_unblock_does_not_mutate_local_state():
    adapter = WebhookFirewallAdapter(webhook_url="https://example.invalid")
    adapter._post = MagicMock(return_value=True)
    adapter.unblock("1.2.3.4")
    assert not hasattr(adapter, "blocked_targets")


# ---------------------------------------------------------------------------
# 10–11: reconcile() gated on supports_rule_query
# ---------------------------------------------------------------------------

def test_reconcile_skips_stateless_adapter():
    """reconcile() must return early without querying list_rules() for webhook adapter."""
    ops_store = MagicMock()
    ops_store.list_active_blocks.return_value = [
        {"id": 1, "target": "10.0.0.1", "executed": True, "dry_run": False}
    ]
    adapter = WebhookFirewallAdapter()
    executor = _make_executor(adapter, ops_store=ops_store)
    result = executor.reconcile()
    assert result.get("skipped") is True
    assert result["missing_in_firewall"] == 0
    assert result["orphan_firewall_rules"] == 0


def test_reconcile_skipped_does_not_call_list_rules():
    """Security regression: reconcile() must never call list_rules() on a stateless adapter."""
    ops_store = MagicMock()
    ops_store.list_active_blocks.return_value = []
    adapter = WebhookFirewallAdapter()
    adapter.list_rules = MagicMock()  # should never be called
    executor = _make_executor(adapter, ops_store=ops_store)
    executor.reconcile()
    adapter.list_rules.assert_not_called()


def test_reconcile_runs_for_queryable_adapter():
    """reconcile() must perform comparison for adapters with supports_rule_query=True."""
    ops_store = MagicMock()
    ops_store.list_active_blocks.return_value = [
        {"id": 1, "target": "10.0.0.1", "executed": True, "dry_run": False}
    ]
    adapter = MockFirewallAdapter()
    # DB has 10.0.0.1 as active; firewall has nothing → should show missing
    executor = _make_executor(adapter, ops_store=ops_store)
    result = executor.reconcile()
    assert result.get("skipped") is not True
    assert result["missing_in_firewall"] == 1


def test_reconcile_no_missing_when_db_and_fw_agree():
    ops_store = MagicMock()
    ops_store.list_active_blocks.return_value = [
        {"id": 1, "target": "10.0.0.1", "executed": True, "dry_run": False}
    ]
    adapter = MockFirewallAdapter()
    adapter.block("10.0.0.1")
    executor = _make_executor(adapter, ops_store=ops_store)
    result = executor.reconcile()
    assert result["missing_in_firewall"] == 0
    assert result["orphan_firewall_rules"] == 0


# ---------------------------------------------------------------------------
# 12–13: UFW block idempotency (G-PREV-2)
# ---------------------------------------------------------------------------

def test_ufw_block_idempotent_skipping_in_stderr():
    """UFW exit-nonzero + 'Skipping' in output must be treated as success."""
    def mock_run(args, **kwargs):
        return _make_ufw_result(returncode=1, stderr="Skipping adding existing rule")

    adapter = UfwFirewallAdapter(run_cmd=mock_run)
    assert adapter.block("1.2.3.4") is True


def test_ufw_block_idempotent_skipping_in_stdout():
    def mock_run(args, **kwargs):
        return _make_ufw_result(returncode=1, stdout="Skipping adding existing rule\n")

    adapter = UfwFirewallAdapter(run_cmd=mock_run)
    assert adapter.block("1.2.3.4") is True


def test_ufw_block_idempotent_already_exists_in_stderr():
    def mock_run(args, **kwargs):
        return _make_ufw_result(returncode=1, stderr="Rule already exists")

    adapter = UfwFirewallAdapter(run_cmd=mock_run)
    assert adapter.block("1.2.3.4") is True


def test_ufw_block_real_failure_returns_false():
    """A genuine UFW failure (not duplicate) must still return False."""
    def mock_run(args, **kwargs):
        return _make_ufw_result(returncode=1, stderr="ERROR: Invalid IP address")

    adapter = UfwFirewallAdapter(run_cmd=mock_run)
    assert adapter.block("1.2.3.4") is False


def test_ufw_block_success_returns_true():
    def mock_run(args, **kwargs):
        return _make_ufw_result(returncode=0, stdout="Rule added\n")

    adapter = UfwFirewallAdapter(run_cmd=mock_run)
    assert adapter.block("1.2.3.4") is True


def test_ufw_block_invalid_ip_returns_false():
    adapter = UfwFirewallAdapter(run_cmd=MagicMock())
    assert adapter.block("not-an-ip") is False


# ---------------------------------------------------------------------------
# 14: WebhookFirewallAdapter logs WARNING on instantiation (G-PREV-3)
# ---------------------------------------------------------------------------

def test_webhook_adapter_logs_warning_on_init(caplog):
    with caplog.at_level(logging.WARNING, logger="src.firewall_adapters"):
        WebhookFirewallAdapter()
    assert any("best-effort" in r.message or "NOT guaranteed" in r.message for r in caplog.records), \
        "WebhookFirewallAdapter must log a non-persistence warning on instantiation"


# ---------------------------------------------------------------------------
# 15: Security regression — reconcile() with no ops_store still safe
# ---------------------------------------------------------------------------

def test_reconcile_without_ops_store_returns_zeros():
    adapter = MockFirewallAdapter()
    executor = _make_executor(adapter, ops_store=None)
    result = executor.reconcile()
    assert result == {"db_active": 0, "firewall_rules": 0, "missing_in_firewall": 0, "orphan_firewall_rules": 0}


def test_reconcile_stateless_adapter_without_ops_store():
    adapter = WebhookFirewallAdapter()
    executor = _make_executor(adapter, ops_store=None)
    result = executor.reconcile()
    # ops_store is None check fires first — no skipped key expected here
    assert result["missing_in_firewall"] == 0
