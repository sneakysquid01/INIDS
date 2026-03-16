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
    redis_url: str = ""
    pipeline_enabled: bool = False
    pipeline_batch_size: int = 50
    pipeline_stream_key: str = "inids:flows"
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    firewall_adapter: str = "mock"
    firewall_webhook_url: str = ""
    require_api_keys: bool = False
    require_secret_key: bool = False
    json_logging: bool = False
    ti_feed_dir: str = ""
    ti_refresh_interval_seconds: int = 3600


def _safe_int(env_key: str, default: int) -> int:
    raw = os.getenv(env_key, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _safe_bool(env_key: str, default: bool) -> bool:
    raw = os.getenv(env_key, "")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    port = _safe_int("PORT", 5000)
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    host = os.getenv("HOST", "0.0.0.0")
    ops_db_path = os.getenv("OPS_DB_PATH", "data/inids_ops.db")
    secret = os.getenv("SECRET_KEY", os.getenv("FLASK_SECRET_KEY", "")).strip()
    redis_url = os.getenv("REDIS_URL", "").strip()
    pipeline_enabled = _safe_bool("INIDS_PIPELINE_ENABLED", False)
    pipeline_batch_size = _safe_int("INIDS_PIPELINE_BATCH_SIZE", 50)
    pipeline_stream_key = os.getenv("INIDS_PIPELINE_STREAM_KEY", "inids:flows").strip() or "inids:flows"
    require_secret_key = os.getenv("INIDS_REQUIRE_SECRET_KEY", "0") == "1"
    require_api_keys = os.getenv("INIDS_REQUIRE_API_KEYS", "0") == "1"
    json_logging = _safe_bool("INIDS_JSON_LOGGING", False)
    if require_secret_key and not secret:
        raise ValueError("SECRET_KEY environment variable is required when INIDS_REQUIRE_SECRET_KEY=1")
    if not secret:
        # Backward-compatible dev fallback. Use INIDS_REQUIRE_SECRET_KEY=1 in production.
        # In test/dev environments without explicit secret requirement, use dev secret
        secret = "dev-inids-secret"
    rate_reqs = _safe_int("RATE_LIMIT_REQUESTS", 120)
    rate_window = _safe_int("RATE_LIMIT_WINDOW_SECONDS", 60)
    firewall_adapter = os.getenv("FIREWALL_ADAPTER", "mock").strip().lower()
    firewall_webhook_url = os.getenv("FIREWALL_WEBHOOK_URL", "").strip()
    ti_feed_dir = os.getenv("INIDS_TI_FEED_DIR", "").strip()
    ti_refresh_interval_seconds = _safe_int("INIDS_TI_REFRESH_INTERVAL", 3600)
    return Settings(
        host=host,
        port=port,
        debug=debug,
        ops_db_path=ops_db_path,
        flask_secret_key=secret,
        redis_url=redis_url,
        pipeline_enabled=pipeline_enabled,
        pipeline_batch_size=max(1, pipeline_batch_size),
        pipeline_stream_key=pipeline_stream_key,
        rate_limit_requests=rate_reqs,
        rate_limit_window_seconds=rate_window,
        firewall_adapter=firewall_adapter,
        firewall_webhook_url=firewall_webhook_url,
        require_api_keys=require_api_keys,
        require_secret_key=require_secret_key,
        json_logging=json_logging,
        ti_feed_dir=ti_feed_dir,
        ti_refresh_interval_seconds=max(60, ti_refresh_interval_seconds),
    )
