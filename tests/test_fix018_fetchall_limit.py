"""FIX-018: _fetchall enforces max_rows cap — no unbounded SELECT results."""
import pytest


def make_store(tmp_path):
    from src.ops_store import OpsStore
    db = str(tmp_path / "test.db")
    return OpsStore(db)


class TestFetchallLimit:
    def test_unbounded_select_gets_limit_injected(self, tmp_path):
        store = make_store(tmp_path)
        # Insert 5 alerts
        import uuid, time
        for i in range(5):
            store.save_alert({
                "id": str(uuid.uuid4()),
                "timestamp": "2025-01-01T00:00:00Z",
                "severity": "low",
                "prediction": "normal",
                "confidence": 0.5,
                "profile": "balanced",
                "reason": "test",
                "source_ip": f"1.2.3.{i}",
                "attack_type": "",
                "risk_score": 0.0,
            })
        # Query without LIMIT — must still work and return all 5 (under default cap of 1000)
        rows = store._fetchall("SELECT * FROM alerts ORDER BY id")
        assert len(rows) == 5

    def test_max_rows_default_caps_results(self, tmp_path):
        store = make_store(tmp_path)
        import uuid
        for i in range(10):
            store.save_alert({
                "id": str(uuid.uuid4()),
                "timestamp": "2025-01-01T00:00:00Z",
                "severity": "low",
                "prediction": "normal",
                "confidence": 0.5,
                "profile": "balanced",
                "reason": "test",
                "source_ip": f"10.0.0.{i}",
                "attack_type": "",
                "risk_score": 0.0,
            })
        rows = store._fetchall("SELECT * FROM alerts", max_rows=3)
        assert len(rows) <= 3

    def test_max_rows_over_hard_cap_raises(self, tmp_path):
        store = make_store(tmp_path)
        with pytest.raises(ValueError, match="hard cap"):
            store._fetchall("SELECT * FROM alerts", max_rows=10001)

    def test_query_with_existing_limit_not_double_limited(self, tmp_path):
        store = make_store(tmp_path)
        import uuid
        for i in range(5):
            store.save_alert({
                "id": str(uuid.uuid4()),
                "timestamp": "2025-01-01T00:00:00Z",
                "severity": "low",
                "prediction": "normal",
                "confidence": 0.5,
                "profile": "balanced",
                "reason": "test",
                "source_ip": f"192.168.0.{i}",
                "attack_type": "",
                "risk_score": 0.0,
            })
        # Query already has LIMIT — should not error or produce double LIMIT
        rows = store._fetchall("SELECT * FROM alerts ORDER BY id LIMIT 2")
        assert len(rows) <= 2

    def test_pragma_not_affected_by_limit_injection(self, tmp_path):
        store = make_store(tmp_path)
        # PRAGMA is not a SELECT — must not have LIMIT appended
        cols = store._fetchall("PRAGMA table_info(alerts)")
        assert isinstance(cols, list)
        assert any(c.get("name") == "id" for c in cols)

    def test_hard_cap_constant_is_ten_thousand(self, tmp_path):
        store = make_store(tmp_path)
        assert store._FETCHALL_HARD_MAX == 10000
