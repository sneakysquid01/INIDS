from src.prevention_service import PreventionService


def test_monitor_mode_does_not_block():
    svc = PreventionService()
    action = svc.evaluate("Attack", 99.0, source="1.1.1.1")
    assert action is None


def test_auto_block_mode_blocks_high_confidence_attack():
    svc = PreventionService()
    svc.set_policy(mode="auto_block", confidence_block_threshold=90, block_ttl_seconds=60)
    action = svc.evaluate("Attack", 95.0, source="1.1.1.1")
    assert action is not None
    assert action.action == "block"
    assert action.target == "1.1.1.1"
