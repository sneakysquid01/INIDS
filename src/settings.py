from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    ops_db_path: str = "data/inids_ops.db"
    flask_secret_key: str = ""
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    firewall_adapter: str = "mock"
    require_api_keys: bool = False
    require_secret_key: bool = False


def _safe_int(env_key: str, default: int) -> int:
    raw = os.getenv(env_key, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def load_settings() -> Settings:
    port = _safe_int("PORT", 5000)
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    host = os.getenv("HOST", "0.0.0.0")
    ops_db_path = os.getenv("OPS_DB_PATH", "data/inids_ops.db")
    secret = os.getenv("SECRET_KEY", os.getenv("FLASK_SECRET_KEY", "")).strip()
    require_secret_key = os.getenv("INIDS_REQUIRE_SECRET_KEY", "0") == "1"
    require_api_keys = os.getenv("INIDS_REQUIRE_API_KEYS", "0") == "1"
    if require_secret_key and not secret:
        raise ValueError("SECRET_KEY environment variable is required when INIDS_REQUIRE_SECRET_KEY=1")
    if not secret:
        # Backward-compatible dev fallback. Use INIDS_REQUIRE_SECRET_KEY=1 in production.
        secret = "dev-inids-secret"
    rate_reqs = _safe_int("RATE_LIMIT_REQUESTS", 120)
    rate_window = _safe_int("RATE_LIMIT_WINDOW_SECONDS", 60)
    firewall_adapter = os.getenv("FIREWALL_ADAPTER", "mock").strip().lower()
    return Settings(
        host=host,
        port=port,
        debug=debug,
        ops_db_path=ops_db_path,
        flask_secret_key=secret,
        rate_limit_requests=rate_reqs,
        rate_limit_window_seconds=rate_window,
        firewall_adapter=firewall_adapter,
        require_api_keys=require_api_keys,
        require_secret_key=require_secret_key,
    )
