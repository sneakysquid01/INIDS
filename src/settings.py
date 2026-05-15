from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load simple KEY=VALUE pairs from the project .env file if present."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        return


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    ops_db_path: str = "data/inids_ops.db"
    flask_secret_key: str = ""
    redis_url: str = ""
    pipeline_enabled: bool = True
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
    honeypot_ips: str = ""
    honeypot_ports: str = ""
    honeypot_enabled: bool = True
    elasticsearch_enabled: bool = False
    elasticsearch_hosts: str = "localhost"
    elasticsearch_port: int = 9200
    elasticsearch_use_ssl: bool = False
    elasticsearch_verify_certs: bool = False
    elasticsearch_username: str = ""
    elasticsearch_password: str = ""
    adapter_call_timeout_s: float = 3.0
    adapter_cb_failure_threshold: int = 3
    adapter_cb_open_duration_s: float = 60.0


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
    _load_dotenv()
    port = _safe_int("PORT", 5000)
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    host = os.getenv("HOST", "0.0.0.0")
    ops_db_path = os.getenv("OPS_DB_PATH", os.getenv("INIDS_OPS_DB_PATH", "data/inids_ops.db"))
    secret = os.getenv("SECRET_KEY", os.getenv("FLASK_SECRET_KEY", "")).strip()
    redis_url = os.getenv("REDIS_URL", "").strip()
    pipeline_enabled = _safe_bool("INIDS_PIPELINE_ENABLED", True)
    pipeline_batch_size = _safe_int("INIDS_PIPELINE_BATCH_SIZE", 50)
    pipeline_stream_key = os.getenv("INIDS_PIPELINE_STREAM_KEY", "inids:flows").strip() or "inids:flows"
    require_secret_key = os.getenv("INIDS_REQUIRE_SECRET_KEY", "0") == "1"
    require_api_keys = os.getenv("INIDS_REQUIRE_API_KEYS", "0") == "1"
    json_logging = _safe_bool("INIDS_JSON_LOGGING", False)
    honeypot_ips = os.getenv("INIDS_HONEYPOT_IPS", "").strip()
    honeypot_ports = os.getenv("INIDS_HONEYPOT_PORTS", "").strip()
    honeypot_enabled = _safe_bool("INIDS_HONEYPOT_ENABLED", True)
    elasticsearch_enabled = _safe_bool("ELASTICSEARCH_ENABLED", False)
    if not secret:
        raise RuntimeError("SECRET_KEY environment variable is required for security")
    rate_reqs = _safe_int("RATE_LIMIT_REQUESTS", 120)
    rate_window = _safe_int("RATE_LIMIT_WINDOW_SECONDS", 60)
    firewall_adapter = os.getenv("FIREWALL_ADAPTER", "mock").strip().lower()
    firewall_webhook_url = os.getenv("FIREWALL_WEBHOOK_URL", "").strip()
    ti_feed_dir = os.getenv("INIDS_TI_FEED_DIR", "").strip()
    ti_refresh_interval_seconds = _safe_int("INIDS_TI_REFRESH_INTERVAL", 3600)
    elasticsearch_hosts = os.getenv("ELASTICSEARCH_HOSTS", "localhost").strip()
    elasticsearch_port = _safe_int("ELASTICSEARCH_PORT", 9200)
    elasticsearch_use_ssl = _safe_bool("ELASTICSEARCH_USE_SSL", False)
    elasticsearch_verify_certs = _safe_bool("ELASTICSEARCH_VERIFY_CERTS", False)
    elasticsearch_username = os.getenv("ELASTICSEARCH_USERNAME", "").strip()
    elasticsearch_password = os.getenv("ELASTICSEARCH_PASSWORD", "").strip()
    adapter_call_timeout_s = float(os.getenv("ADAPTER_CALL_TIMEOUT_S", "3.0"))
    adapter_cb_failure_threshold = _safe_int("ADAPTER_CB_FAILURE_THRESHOLD", 3)
    adapter_cb_open_duration_s = float(os.getenv("ADAPTER_CB_OPEN_DURATION_S", "60.0"))
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
        honeypot_ips=honeypot_ips,
        honeypot_ports=honeypot_ports,
        honeypot_enabled=honeypot_enabled,
        elasticsearch_enabled=elasticsearch_enabled,
        elasticsearch_hosts=elasticsearch_hosts,
        elasticsearch_port=elasticsearch_port,
        elasticsearch_use_ssl=elasticsearch_use_ssl,
        elasticsearch_verify_certs=elasticsearch_verify_certs,
        elasticsearch_username=elasticsearch_username,
        elasticsearch_password=elasticsearch_password,
        adapter_call_timeout_s=adapter_call_timeout_s,
        adapter_cb_failure_threshold=adapter_cb_failure_threshold,
        adapter_cb_open_duration_s=adapter_cb_open_duration_s,
    )
