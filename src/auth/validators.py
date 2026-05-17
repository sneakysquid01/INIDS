from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Always required, regardless of auth mode.
_BASE_REQUIRED_SECRETS = ("SECRET_KEY", "INIDS_ADMIN_API_KEY")

# Required when INIDS_AUTH_COMPAT=false (RS256-only mode). See G-AUTH-3.
_RS256_REQUIRED_SECRETS = ("INIDS_JWT_PUBLIC_KEY",)

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


def _compat_mode() -> bool:
    return os.environ.get("INIDS_AUTH_COMPAT", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_placeholder(value: str) -> bool:
    return value.strip().lower() in _PLACEHOLDER_VALUES


def validate_config_at_startup() -> None:
    """Check required secrets and fail-closed on missing or placeholder values.

    During INIDS_AUTH_COMPAT=true (transition window):
    - BASE_REQUIRED_SECRETS enforced (fail-closed).
    - INIDS_JWT_PUBLIC_KEY absence is a WARNING only (ephemeral key in use).

    When INIDS_AUTH_COMPAT=false (RS256-only mode, Phase C exit):
    - All secrets including INIDS_JWT_PUBLIC_KEY are required (fail-closed).
    """
    compat = _compat_mode()
    secrets_to_check = list(_BASE_REQUIRED_SECRETS)
    if not compat:
        secrets_to_check.extend(_RS256_REQUIRED_SECRETS)

    missing = []
    placeholders = []

    for key in secrets_to_check:
        value = os.environ.get(key, "").strip()
        if not value or _is_placeholder(value):
            if not value:
                missing.append(key)
            else:
                placeholders.append(key)

    if compat:
        jwt_pub = os.environ.get("INIDS_JWT_PUBLIC_KEY", "").strip()
        if not jwt_pub or _is_placeholder(jwt_pub):
            logger.warning(
                "INIDS_JWT_PUBLIC_KEY is not set (or is a placeholder). "
                "RS256 tokens use an ephemeral key and will be invalid after restart. "
                "Set INIDS_JWT_PUBLIC_KEY before disabling INIDS_AUTH_COMPAT."
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
        "Startup config validation passed (%d secrets checked, compat=%s)",
        len(secrets_to_check),
        compat,
    )
