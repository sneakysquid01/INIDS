from src.ops_store import OpsStore


def test_ops_store_roundtrip(tmp_path):
    store = OpsStore(str(tmp_path / "ops.db"))

    store.save_alert(
        {
            "id": "al_1",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "severity": "high",
            "prediction": "Attack",
            "confidence": 99.1,
            "profile": "balanced",
            "reason": "model_prediction",
        }
    )
    store.save_action(
        {
            "action": "block",
            "target": "1.1.1.1",
            "reason": "attack_confidence_99.1",
            "expires_at": "2026-01-01T00:05:00+00:00",
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    )
    store.add_audit("policy_update", "mode=auto_block", "2026-01-01T00:00:00+00:00")

    assert len(store.list_alerts(limit=10)) == 1
    assert len(store.list_actions(limit=10)) == 1
    assert len(store.list_audits(limit=10)) == 1


def test_cleanup_expired_actions(tmp_path):
    store = OpsStore(str(tmp_path / "ops.db"))

    store.save_action(
        {
            "action": "block",
            "target": "1.1.1.1",
            "reason": "expired",
            "expires_at": "2020-01-01T00:00:00+00:00",
            "created_at": "2019-01-01T00:00:00+00:00",
        }
    )
    store.save_action(
        {
            "action": "block",
            "target": "2.2.2.2",
            "reason": "active",
            "expires_at": "2999-01-01T00:00:00+00:00",
            "created_at": "2019-01-01T00:00:00+00:00",
        }
    )

    removed = store.cleanup_expired_actions(now_iso="2021-01-01T00:00:00+00:00")
    assert removed == 1
    assert len(store.list_actions(limit=10)) == 1
