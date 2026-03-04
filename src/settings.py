from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    ops_db_path: str = "data/inids_ops.db"
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
    flask_secret_key: str = ""
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    rate_limit_max_keys: int = 10000
    firewall_adapter: str = "mock"
    admin_api_key: str = ""
    sensor_api_key: str = ""
    viewer_api_key: str = ""
    enable_ips_scheduler: bool = False
    scheduler_interval_seconds: int = 15
    scheduler_reconcile_every: int = 20


def _require_env(names: tuple[str, ...], label: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError(f"Missing required environment variable for {label}: one of {', '.join(names)}")
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
    flask_secret_key: str = "dev-inids-secret"
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    firewall_adapter: str = "mock"
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs


def load_settings() -> Settings:
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    host = os.getenv("HOST", "0.0.0.0")
    ops_db_path = os.getenv("OPS_DB_PATH", "data/inids_ops.db")
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
    secret = _require_env(("SECRET_KEY", "FLASK_SECRET_KEY"), "Flask secret key")
    rate_reqs = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
    rate_window = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    rate_max_keys = int(os.getenv("RATE_LIMIT_MAX_KEYS", "10000"))
    firewall_adapter = os.getenv("FIREWALL_ADAPTER", "mock").strip().lower()
    admin_api_key = _require_env(("INIDS_ADMIN_API_KEY",), "admin API key")
    sensor_api_key = _require_env(("INIDS_SENSOR_API_KEY", "INIDS_ANALYST_API_KEY"), "sensor API key")
    viewer_api_key = _require_env(("INIDS_VIEWER_API_KEY",), "viewer API key")
    enable_scheduler = os.getenv("INIDS_ENABLE_IPS_SCHEDULER", "0") == "1"
    scheduler_interval_seconds = int(os.getenv("INIDS_SCHEDULER_INTERVAL_SECONDS", "15"))
    scheduler_reconcile_every = int(os.getenv("INIDS_SCHEDULER_RECONCILE_EVERY", "20"))
=======
=======
>>>>>>> theirs
=======
>>>>>>> theirs
    secret = os.getenv("FLASK_SECRET_KEY", "dev-inids-secret")
    rate_reqs = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
    rate_window = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
    firewall_adapter = os.getenv("FIREWALL_ADAPTER", "mock").strip().lower()
<<<<<<< ours
<<<<<<< ours
>>>>>>> theirs
=======
>>>>>>> theirs
=======
>>>>>>> theirs
    return Settings(
        host=host,
        port=port,
        debug=debug,
        ops_db_path=ops_db_path,
        flask_secret_key=secret,
        rate_limit_requests=rate_reqs,
        rate_limit_window_seconds=rate_window,
<<<<<<< ours
<<<<<<< ours
<<<<<<< ours
        rate_limit_max_keys=rate_max_keys,
        firewall_adapter=firewall_adapter,
        admin_api_key=admin_api_key,
        sensor_api_key=sensor_api_key,
        viewer_api_key=viewer_api_key,
        enable_ips_scheduler=enable_scheduler,
        scheduler_interval_seconds=scheduler_interval_seconds,
        scheduler_reconcile_every=scheduler_reconcile_every,
=======
        firewall_adapter=firewall_adapter,
>>>>>>> theirs
=======
        firewall_adapter=firewall_adapter,
>>>>>>> theirs
=======
        firewall_adapter=firewall_adapter,
>>>>>>> theirs
    )
