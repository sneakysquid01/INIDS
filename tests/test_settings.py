from src.settings import load_settings


def test_load_settings_defaults(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("FLASK_DEBUG", raising=False)
    monkeypatch.delenv("OPS_DB_PATH", raising=False)
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)
    monkeypatch.delenv("FIREWALL_ADAPTER", raising=False)

    s = load_settings()
    assert s.port == 5000
    assert s.host == "0.0.0.0"
    assert s.debug is False
    assert s.ops_db_path == "data/inids_ops.db"

    assert s.firewall_adapter == "mock"
