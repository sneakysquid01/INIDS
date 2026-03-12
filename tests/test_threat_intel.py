"""Phase C: Threat Intelligence tests.

Covers:
  - ThreatIntelCache upsert / lookup / purge_expired
  - ThreatIntelManager CSV + JSON feed loading
  - ThreatIntelManager stats, feed_summary
  - TIEngine evaluate: match, no-match, expired indicator
  - TIEngine is_ready() tracks cache size
  - _load_ti_feeds() from a temp feed directory (CSV + JSON)
  - Flask API: GET /api/threat-intel/stats
  - Flask API: POST /api/threat-intel/lookup (hit + miss)
  - TI engine participates in /api/detect flow when a feed is loaded
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# ThreatIntelCache
# ---------------------------------------------------------------------------

class TestThreatIntelCache:
    def _make_indicator(self, ip="1.2.3.4", *, ttl=86400.0, last_seen=None):
        from src.threat_intel.feed_manager import TIIndicator
        return TIIndicator(
            indicator_type="ip",
            value=ip,
            source="test",
            severity="high",
            ttl_seconds=ttl,
            last_seen=last_seen if last_seen is not None else time.time(),
        )

    def test_upsert_and_lookup(self):
        from src.threat_intel.feed_manager import ThreatIntelCache
        cache = ThreatIntelCache()
        ind = self._make_indicator("10.0.0.1")
        cache.upsert(ind)
        result = cache.lookup_ip("10.0.0.1")
        assert result is not None
        assert result.value == "10.0.0.1"

    def test_lookup_case_insensitive(self):
        from src.threat_intel.feed_manager import ThreatIntelCache
        cache = ThreatIntelCache()
        ind = self._make_indicator("10.0.0.2")
        cache.upsert(ind)
        assert cache.lookup_ip("10.0.0.2") is not None

    def test_size(self):
        from src.threat_intel.feed_manager import ThreatIntelCache
        cache = ThreatIntelCache()
        assert cache.size() == 0
        cache.upsert(self._make_indicator("1.1.1.1"))
        assert cache.size() == 1

    def test_purge_expired(self):
        from src.threat_intel.feed_manager import ThreatIntelCache
        cache = ThreatIntelCache()
        # Indicator with very short TTL already expired
        old_ind = self._make_indicator("2.2.2.2", ttl=1.0, last_seen=time.time() - 5)
        fresh_ind = self._make_indicator("3.3.3.3", ttl=86400.0)
        cache.upsert(old_ind)
        cache.upsert(fresh_ind)
        assert cache.size() == 2
        removed = cache.purge_expired()
        assert removed == 1
        assert cache.size() == 1
        assert cache.lookup_ip("3.3.3.3") is not None

    def test_purge_fresh_not_removed(self):
        from src.threat_intel.feed_manager import ThreatIntelCache
        cache = ThreatIntelCache()
        cache.upsert(self._make_indicator("4.4.4.4", ttl=86400.0))
        assert cache.purge_expired() == 0


# ---------------------------------------------------------------------------
# ThreatIntelManager feed loading
# ---------------------------------------------------------------------------

class TestThreatIntelManager:
    CSV_DATA = textwrap.dedent("""\
        indicator,severity
        5.5.5.5,high
        6.6.6.6,medium
        7.7.7.7,low
    """)

    JSON_DATA = json.dumps([
        {"value": "8.8.8.8", "severity": "critical"},
        {"ip": "9.9.9.9", "severity": "high"},
    ])

    def test_load_csv_feed(self):
        from src.threat_intel.feed_manager import ThreatIntelManager
        mgr = ThreatIntelManager()
        n = mgr.load_csv_feed(self.CSV_DATA, source="csv_test")
        assert n == 3
        assert mgr.lookup_ip("5.5.5.5") is not None
        assert mgr.lookup_ip("7.7.7.7") is not None

    def test_load_json_feed(self):
        from src.threat_intel.feed_manager import ThreatIntelManager
        mgr = ThreatIntelManager()
        n = mgr.load_json_feed(self.JSON_DATA, source="json_test")
        assert n == 2
        assert mgr.lookup_ip("8.8.8.8") is not None

    def test_stats(self):
        from src.threat_intel.feed_manager import ThreatIntelManager
        mgr = ThreatIntelManager()
        mgr.load_csv_feed(self.CSV_DATA, source="s1")
        stats = mgr.stats()
        assert stats["total_indicators"] == 3
        assert stats["feeds_loaded"] == 1

    def test_feed_summary(self):
        from src.threat_intel.feed_manager import ThreatIntelManager
        mgr = ThreatIntelManager()
        mgr.load_csv_feed(self.CSV_DATA, source="my_feed")
        summary = mgr.feed_summary()
        assert len(summary) == 1
        assert summary[0]["source"] == "my_feed"
        assert summary[0]["indicators_loaded"] == 3

    def test_severity_preserved(self):
        from src.threat_intel.feed_manager import ThreatIntelManager
        mgr = ThreatIntelManager()
        mgr.load_csv_feed(self.CSV_DATA, source="sev_test")
        ind = mgr.lookup_ip("5.5.5.5")
        assert ind.severity == "high"

    def test_lookup_miss_returns_none(self):
        from src.threat_intel.feed_manager import ThreatIntelManager
        mgr = ThreatIntelManager()
        assert mgr.lookup_ip("1.2.3.4") is None

    def test_load_json_feed_skips_non_object_items(self):
        from src.threat_intel.feed_manager import ThreatIntelManager
        mgr = ThreatIntelManager()
        mixed = json.dumps([{"ip": "1.2.3.4"}, "bad-item", 123, None])
        n = mgr.load_json_feed(mixed, source="mixed")
        assert n == 1
        assert mgr.lookup_ip("1.2.3.4") is not None


# ---------------------------------------------------------------------------
# TIEngine evaluate
# ---------------------------------------------------------------------------

class TestTIEngine:
    def _make_engine_with(self, ips: list[str], ttl=86400.0, last_seen=None):
        from src.threat_intel.feed_manager import ThreatIntelManager, TIIndicator
        from src.threat_intel.ti_engine import TIEngine
        mgr = ThreatIntelManager()
        now = time.time()
        for ip in ips:
            mgr.add_indicator(TIIndicator(
                indicator_type="ip",
                value=ip,
                source="unit_test",
                severity="high",
                ttl_seconds=ttl,
                first_seen=now,
                last_seen=last_seen if last_seen is not None else now,
            ))
        return TIEngine(mgr)

    def test_is_ready_false_when_empty(self):
        from src.threat_intel.feed_manager import ThreatIntelManager
        from src.threat_intel.ti_engine import TIEngine
        mgr = ThreatIntelManager()
        engine = TIEngine(mgr)
        assert engine.is_ready() is False

    def test_is_ready_true_after_load(self):
        engine = self._make_engine_with(["192.0.2.1"])
        assert engine.is_ready() is True

    def test_attack_verdict_on_match(self):
        engine = self._make_engine_with(["192.0.2.2"])
        result = engine.evaluate({"source_ip": "192.0.2.2"})
        assert result.verdict == "attack"
        assert result.attack_type == "threat_intel_match"
        assert result.confidence == 90.0

    def test_normal_verdict_on_miss(self):
        engine = self._make_engine_with(["10.0.0.1"])
        result = engine.evaluate({"source_ip": "192.0.2.99"})
        assert result.verdict == "normal"

    def test_normal_verdict_on_expired_indicator(self):
        # last_seen far in the past, ttl=1s → expired
        engine = self._make_engine_with(["10.1.2.3"], ttl=1.0, last_seen=time.time() - 100)
        result = engine.evaluate({"source_ip": "10.1.2.3"})
        assert result.verdict == "normal"

    def test_engine_id_and_type(self):
        from src.threat_intel.feed_manager import ThreatIntelManager
        from src.threat_intel.ti_engine import TIEngine
        engine = TIEngine(ThreatIntelManager())
        assert engine.engine_id == "threat_intel"
        assert engine.engine_type == "ti"

    def test_metadata_populated_on_match(self):
        engine = self._make_engine_with(["172.16.0.5"])
        result = engine.evaluate({"source_ip": "172.16.0.5"})
        assert "ti_source" in result.metadata
        assert result.metadata["ti_source"] == "unit_test"


# ---------------------------------------------------------------------------
# _load_ti_feeds() from temp directory
# ---------------------------------------------------------------------------

class TestLoadTiFeeds:
    def test_loads_csv_feed_from_dir(self, tmp_path):
        feed_file = tmp_path / "test_feed.csv"
        feed_file.write_text("indicator,severity\n11.22.33.44,high\n55.66.77.88,medium\n", encoding="utf-8")

        # Override setting and reset ti_manager to a fresh instance
        import importlib
        import web_app.app as app_mod
        original_mgr = app_mod.ti_manager
        original_setting = app_mod.SETTINGS.ti_feed_dir
        try:
            app_mod.ti_manager = type(original_mgr)()
            # Temporarily patch SETTINGS.ti_feed_dir via the object
            import src.settings as settings_mod
            new_settings = settings_mod.Settings(
                **{**app_mod.SETTINGS.__dict__, "ti_feed_dir": str(tmp_path)}
            )
            app_mod.SETTINGS = new_settings

            count = app_mod._load_ti_feeds()
            assert count == 2
            assert app_mod.ti_manager.lookup_ip("11.22.33.44") is not None
        finally:
            app_mod.ti_manager = original_mgr
            app_mod.SETTINGS = settings_mod.load_settings()

    def test_empty_dir_returns_zero(self, tmp_path):
        import web_app.app as app_mod
        import src.settings as settings_mod
        original_settings = app_mod.SETTINGS
        try:
            new_settings = settings_mod.Settings(
                **{**app_mod.SETTINGS.__dict__, "ti_feed_dir": str(tmp_path)}
            )
            app_mod.SETTINGS = new_settings
            count = app_mod._load_ti_feeds()
            assert count == 0
        finally:
            app_mod.SETTINGS = original_settings

    def test_nonexistent_dir_returns_zero(self):
        import web_app.app as app_mod
        import src.settings as settings_mod
        original_settings = app_mod.SETTINGS
        try:
            new_settings = settings_mod.Settings(
                **{**app_mod.SETTINGS.__dict__, "ti_feed_dir": "/nonexistent_path_xyz"}
            )
            app_mod.SETTINGS = new_settings
            count = app_mod._load_ti_feeds()
            assert count == 0
        finally:
            app_mod.SETTINGS = original_settings


# ---------------------------------------------------------------------------
# Flask API: /api/threat-intel/stats and /api/threat-intel/lookup
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ti_app_client(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("appc")
    os.environ["INIDS_OPS_DB_PATH"] = str(tmp_path / "test_ops.db")
    os.environ["INIDS_PIPELINE_ENABLED"] = "false"
    from web_app.app import app as flask_app
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as client:
        yield client


def _analyst_hdr():
    import base64
    return {"Authorization": f"Basic {base64.b64encode(b'analyst:secret').decode()}"}


def _admin_hdr():
    import base64
    return {"Authorization": f"Basic {base64.b64encode(b'admin:secret').decode()}"}


class TestTIStatsAPI:
    def test_stats_returns_200(self, ti_app_client):
        resp = ti_app_client.get("/api/threat-intel/stats", headers=_analyst_hdr())
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "stats" in data
        assert "engine_ready" in data
        assert "engine_enabled" in data

    def test_stats_structure(self, ti_app_client):
        resp = ti_app_client.get("/api/threat-intel/stats", headers=_analyst_hdr())
        data = json.loads(resp.data)
        assert "total_indicators" in data["stats"]
        assert "feeds_loaded" in data["stats"]


class TestTILookupAPI:
    def test_miss_returns_found_false(self, ti_app_client):
        resp = ti_app_client.post(
            "/api/threat-intel/lookup",
            json={"ip": "1.2.3.4"},
            headers=_analyst_hdr(),
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["found"] is False
        assert data["ip"] == "1.2.3.4"

    def test_missing_ip_returns_400(self, ti_app_client):
        resp = ti_app_client.post(
            "/api/threat-intel/lookup",
            json={},
            headers=_analyst_hdr(),
        )
        assert resp.status_code == 400

    def test_hit_after_manual_load(self, ti_app_client):
        """Load an indicator directly into ti_manager and verify API returns it."""
        import web_app.app as app_mod
        from src.threat_intel.feed_manager import TIIndicator
        app_mod.ti_manager.add_indicator(TIIndicator(
            indicator_type="ip",
            value="198.51.100.1",
            source="api_test",
            severity="critical",
            ttl_seconds=86400.0,
            last_seen=time.time(),
        ))
        resp = ti_app_client.post(
            "/api/threat-intel/lookup",
            json={"ip": "198.51.100.1"},
            headers=_analyst_hdr(),
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["found"] is True
        assert data["indicator"]["severity"] == "critical"
