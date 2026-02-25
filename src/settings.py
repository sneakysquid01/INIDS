from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    ops_db_path: str = "data/inids_ops.db"
    flask_secret_key: str = "dev-inids-secret"
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60
    firewall_adapter: str = "mock"


def load_settings() -> Settings:
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    host = os.getenv("HOST", "0.0.0.0")
    ops_db_path = os.getenv("OPS_DB_PATH", "data/inids_ops.db")
    secret = os.getenv("FLASK_SECRET_KEY", "dev-inids-secret")
    rate_reqs = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
    rate_window = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
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
    )
