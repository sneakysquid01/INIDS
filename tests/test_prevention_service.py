from src.firewall_adapters import MockFirewallAdapter
from src.prevention_service import PreventionService


def test_monitor_mode_does_not_block():
    svc = PreventionService()
    action = svc.evaluate("Attack", 99.0, source="1.1.1.1")
    assert action is None


def test_auto_block_mode_blocks_high_confidence_attack_dry_run():
    svc = PreventionService()
    svc.set_policy(mode="auto_block", confidence_block_threshold=90, block_ttl_seconds=60, dry_run=True)
    action = svc.evaluate("Attack", 95.0, source="1.1.1.1")
    assert action is not None
    assert action.action == "block"
    assert action.target == "1.1.1.1"
    assert action.dry_run is True
    assert action.executed is False


def test_auto_block_executes_with_mock_adapter_when_not_dry_run():
    adapter = MockFirewallAdapter()
    svc = PreventionService(adapter=adapter)
    svc.set_policy(mode="auto_block", confidence_block_threshold=90, block_ttl_seconds=60, dry_run=False)
    action = svc.evaluate("Attack", 95.0, source="9.9.9.9")

    assert action is not None
    assert action.executed is True
    assert adapter.blocked_targets.get("9.9.9.9") == 60
