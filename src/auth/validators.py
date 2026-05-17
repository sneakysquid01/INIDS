from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_REQUIRED_SECRETS = ("SECRET_KEY", "INIDS_ADMIN_API_KEY")

_PLACEHOLDER_VALUES: frozenset[str] = frozenset(
    {
        "changeme",
        "placeholder",
        "secret",
        "password",
        "test",
        "example",
        "default",
        "replace_me",
        "your_key_here",
        "your_secret_here",
        "insert_key_here",
        "notset",
        "none",
        "null",
        "",
    }
)


def _is_placeholder(value: str) -> bool:
    return value.strip().lower() in _PLACEHOLDER_VALUES


def validate_config_at_startup() -> None:
    """Check required secrets and fail-closed on missing or placeholder values.

    INIDS_JWT_PUBLIC_KEY absence is a WARNING (ephemeral key in use for tests).
    """
    missing = []
    placeholders = []

    for key in _REQUIRED_SECRETS:
        value = os.environ.get(key, "").strip()
        if not value or _is_placeholder(value):
            if not value:
                missing.append(key)
            else:
                placeholders.append(key)

    jwt_pub = os.environ.get("INIDS_JWT_PUBLIC_KEY", "").strip()
    if not jwt_pub or _is_placeholder(jwt_pub):
        logger.warning(
            "INIDS_JWT_PUBLIC_KEY is not set (or is a placeholder). "
            "RS256 tokens use an ephemeral key and will be invalid after restart. "
            "Set INIDS_JWT_PUBLIC_KEY for production."
        )

    if missing:
        raise RuntimeError(
            f"Required secrets are missing: {', '.join(missing)}. "
            "Set these environment variables before starting the service."
        )

    if placeholders:
        raise RuntimeError(
            f"Placeholder values detected for secrets: {', '.join(placeholders)}. "
            "Replace with real credentials before deploying."
        )

    logger.info(
        "Startup config validation passed (%d secrets checked)",
        len(_REQUIRED_SECRETS),
    )
