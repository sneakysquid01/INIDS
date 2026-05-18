"""FIX-001: _read_file_secret helper — _FILE path read, fallback to plain env,
both absent returns empty string, SECRET_KEY absence triggers RuntimeError."""
import os
import pytest


def test_secret_from_file(tmp_path, monkeypatch):
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("s3cr3t-from-file", encoding="utf-8")
    monkeypatch.setenv("MY_SECRET_FILE", str(secret_file))
    monkeypatch.delenv("MY_SECRET", raising=False)

    from src.settings import _read_file_secret
    assert _read_file_secret("MY_SECRET") == "s3cr3t-from-file"


def test_secret_file_missing_falls_back_to_plain_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_SECRET_FILE", str(tmp_path / "nonexistent.txt"))
    monkeypatch.setenv("MY_SECRET", "plain-env-value")

    from src.settings import _read_file_secret
    assert _read_file_secret("MY_SECRET") == "plain-env-value"


def test_secret_file_wins_over_plain_env(tmp_path, monkeypatch):
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("file-wins", encoding="utf-8")
    monkeypatch.setenv("MY_SECRET_FILE", str(secret_file))
    monkeypatch.setenv("MY_SECRET", "plain-loses")

    from src.settings import _read_file_secret
    assert _read_file_secret("MY_SECRET") == "file-wins"


def test_both_absent_returns_empty(monkeypatch):
    monkeypatch.delenv("MY_SECRET_FILE", raising=False)
    monkeypatch.delenv("MY_SECRET", raising=False)

    from src.settings import _read_file_secret
    assert _read_file_secret("MY_SECRET") == ""


def test_fallback_key_used_when_primary_absent(monkeypatch):
    monkeypatch.delenv("MY_SECRET_FILE", raising=False)
    monkeypatch.delenv("MY_SECRET", raising=False)
    monkeypatch.setenv("FLASK_MY_SECRET", "fallback-value")

    from src.settings import _read_file_secret
    assert _read_file_secret("MY_SECRET", "FLASK_MY_SECRET") == "fallback-value"


def test_load_settings_raises_when_secret_key_absent(monkeypatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("SECRET_KEY_FILE", raising=False)
    monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)
    # Also suppress .env loading by pointing at non-existent dir
    monkeypatch.setenv("INIDS_SETTINGS_NO_DOTENV", "1")

    import importlib
    import src.settings as _mod
    # Patch _load_dotenv to no-op so .env on disk doesn't interfere
    monkeypatch.setattr(_mod, "_load_dotenv", lambda: None)

    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        _mod.load_settings()


def test_internal_cidrs_parsed(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("INIDS_INTERNAL_CIDRS", "10.0.0.0/8,172.16.0.0/12")

    import importlib
    import src.settings as _mod
    monkeypatch.setattr(_mod, "_load_dotenv", lambda: None)

    s = _mod.load_settings()
    assert "10.0.0.0/8" in s.internal_cidrs
    assert "172.16.0.0/12" in s.internal_cidrs


def test_internal_cidrs_invalid_entries_skipped(monkeypatch):
    monkeypatch.setenv("SECRET_KEY", "test-secret-key")
    monkeypatch.setenv("INIDS_INTERNAL_CIDRS", "10.0.0.0/8,NOT_A_CIDR,192.168.0.0/16")

    import importlib
    import src.settings as _mod
    monkeypatch.setattr(_mod, "_load_dotenv", lambda: None)

    s = _mod.load_settings()
    assert "10.0.0.0/8" in s.internal_cidrs
    assert "192.168.0.0/16" in s.internal_cidrs
    assert "NOT_A_CIDR" not in s.internal_cidrs
