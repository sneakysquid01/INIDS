from __future__ import annotations

import json


def test_add_ratio_features_missing_columns_returns_unchanged_shape():
    import pandas as pd
    from src.feature_engineering import add_ratio_features

    df = pd.DataFrame([{"src_bytes": 10.0}])
    out = add_ratio_features(df)
    assert list(out.columns) == ["src_bytes"]


def test_add_log_transforms_skips_missing_columns():
    import pandas as pd
    from src.feature_engineering import add_log_transforms

    df = pd.DataFrame([{"foo": 1.0}])
    out = add_log_transforms(df)
    assert list(out.columns) == ["foo"]


def test_add_entropy_features_no_rate_columns_noop():
    import pandas as pd
    from src.feature_engineering import add_entropy_features

    df = pd.DataFrame([{"src_bytes": 1.0, "dst_bytes": 2.0}])
    out = add_entropy_features(df)
    assert "rate_entropy" not in out.columns


def test_add_entropy_features_adds_rate_entropy_when_rates_present():
    import pandas as pd
    from src.feature_engineering import add_entropy_features

    df = pd.DataFrame([
        {"serror_rate": 0.2, "rerror_rate": 0.1, "same_srv_rate": 0.7},
    ])
    out = add_entropy_features(df)
    assert "rate_entropy" in out.columns


def test_enrich_single_row_complete_features_adds_derived_columns():
    from src.feature_engineering import enrich_single_row

    features = {
        "src_bytes": 100.0,
        "dst_bytes": 200.0,
        "serror_rate": 0.2,
        "rerror_rate": 0.1,
        "dst_host_count": 10.0,
        "dst_host_srv_count": 4.0,
        "count": 7.0,
        "srv_count": 3.0,
        "duration": 2.0,
        "same_srv_rate": 0.8,
    }
    out = enrich_single_row(features)
    assert "byte_ratio" in out
    assert "error_rate_delta" in out
    assert "service_diversity" in out
    assert "conn_intensity" in out
    assert "log_src_bytes" in out


def test_enrich_single_row_missing_columns_does_not_raise():
    from src.feature_engineering import enrich_single_row

    out = enrich_single_row({"src_bytes": 100.0})
    assert out["src_bytes"] == 100.0


def test_stream_processor_uses_enriched_features_for_evaluation(monkeypatch):
    from src.pipeline.stream_processor import StreamProcessor

    captured = {}

    class FakeRedis:
        def xgroup_create(self, *args, **kwargs):
            return True

        def xack(self, *args, **kwargs):
            return 1

    class FakeRegistry:
        def evaluate_all(self, features):
            captured["features"] = features
            return []

    class FakeAggregator:
        def aggregate(self, results):
            class _Agg:
                verdict = "normal"
                confidence = 100.0
                attack_type = "normal"
                severity = "low"
                engine_results = []

            return _Agg()

    def fake_enrich(features):
        enriched = dict(features)
        enriched["byte_ratio"] = 0.5
        return enriched

    import src.pipeline.stream_processor as sp_mod

    monkeypatch.setattr(sp_mod, "enrich_single_row", fake_enrich)

    proc = StreamProcessor(
        FakeRedis(),
        FakeRegistry(),
        FakeAggregator(),
        result_callback=lambda a, f: None,
    )
    proc._process_message("1-0", {"payload": json.dumps({"src_bytes": 10, "dst_bytes": 20})})

    assert "byte_ratio" in captured["features"]


def test_stream_processor_enrichment_failure_falls_back_to_raw(monkeypatch):
    from src.pipeline.stream_processor import StreamProcessor

    captured = {}

    class FakeRedis:
        def xgroup_create(self, *args, **kwargs):
            return True

        def xack(self, *args, **kwargs):
            return 1

    class FakeRegistry:
        def evaluate_all(self, features):
            captured["features"] = features
            return []

    class FakeAggregator:
        def aggregate(self, results):
            class _Agg:
                verdict = "normal"
                confidence = 100.0
                attack_type = "normal"
                severity = "low"
                engine_results = []

            return _Agg()

    def fail_enrich(_features):
        raise RuntimeError("boom")

    import src.pipeline.stream_processor as sp_mod

    monkeypatch.setattr(sp_mod, "enrich_single_row", fail_enrich)

    proc = StreamProcessor(
        FakeRedis(),
        FakeRegistry(),
        FakeAggregator(),
        result_callback=lambda a, f: None,
    )
    proc._process_message("1-0", {"payload": json.dumps({"src_bytes": 10, "dst_bytes": 20})})

    assert captured["features"]["src_bytes"] == 10
    assert "byte_ratio" not in captured["features"]


def test_api_detect_enrichment_failure_falls_back_to_raw(monkeypatch, tmp_path):
    import web_app.app as app_mod

    app_mod.app.config["TESTING"] = True

    def fail_enrich(_features):
        raise RuntimeError("enrich failed")

    class FakeAgg:
        def __init__(self):
            self.verdict = "normal"

        def to_dict(self):
            return {"verdict": "normal", "engine_results": []}

    monkeypatch.setattr(app_mod, "enrich_single_row", fail_enrich)
    monkeypatch.setattr(app_mod.engine_registry, "evaluate_all", lambda f: [])
    monkeypatch.setattr(app_mod.engine_aggregator, "aggregate", lambda r: FakeAgg())

    from src.ops_store import OpsStore
    _analyst_key = "fe-test-analyst-key"
    monkeypatch.setenv("INIDS_ANALYST_API_KEY", _analyst_key)
    store = OpsStore(str(tmp_path / "fe_test.db"))
    monkeypatch.setattr(app_mod, "ops_store", store)
    app_mod.app.ops_store = store

    with app_mod.app.test_client() as client:
        resp = client.post(
            "/api/detect",
            json={"source": "10.0.0.1", "features": {"src_bytes": 10, "dst_bytes": 20}},
            headers={"X-API-Key": _analyst_key},
        )

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["verdict"] == "normal"


def test_api_detect_uses_enriched_features(monkeypatch, tmp_path):
    import web_app.app as app_mod

    app_mod.app.config["TESTING"] = True
    seen = {}

    def enrich_ok(features):
        enriched = dict(features)
        enriched["byte_ratio"] = 0.5
        return enriched

    class FakeAgg:
        def __init__(self):
            self.verdict = "normal"

        def to_dict(self):
            return {"verdict": "normal", "engine_results": []}

    def eval_all(features):
        seen["features"] = features
        return []

    monkeypatch.setattr(app_mod, "enrich_single_row", enrich_ok)
    monkeypatch.setattr(app_mod.engine_registry, "evaluate_all", eval_all)
    monkeypatch.setattr(app_mod.engine_aggregator, "aggregate", lambda r: FakeAgg())

    from src.ops_store import OpsStore
    _analyst_key = "fe-test2-analyst-key"
    monkeypatch.setenv("INIDS_ANALYST_API_KEY", _analyst_key)
    store = OpsStore(str(tmp_path / "fe_test2.db"))
    monkeypatch.setattr(app_mod, "ops_store", store)
    app_mod.app.ops_store = store

    with app_mod.app.test_client() as client:
        resp = client.post(
            "/api/detect",
            json={"source": "10.0.0.2", "features": {"src_bytes": 10, "dst_bytes": 20}},
            headers={"X-API-Key": _analyst_key},
        )

    assert resp.status_code == 200
    assert "byte_ratio" in seen["features"]
