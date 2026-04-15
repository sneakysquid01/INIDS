"""Redis-backed hot-reloadable configuration manager.

Inspired by WatchAD's Redis config approach. Allows configuration changes
to be applied without restart, enabling dynamic threshold tuning,
honeypot configuration changes, and policy updates at runtime.
"""
from __future__ import annotations

import json
import logging
from typing import Any
import redis

logger = logging.getLogger(__name__)


class RedisConfigManager:
    """Manages hot-reloadable configuration stored in Redis.
    
    Configuration keys follow pattern: inids:config:{section}:{key}
    
    Sections:
    - policy: block_threshold, temp_block_threshold, rate_limit_threshold, etc.
    - honeypot: ips, ports, enabled
    - system: learning_period_end, etc.
    
    All config changes are visible immediately across all instances.
    """

    # Redis key prefixes
    CONFIG_PREFIX = "inids:config"
    POLICY_PREFIX = f"{CONFIG_PREFIX}:policy"
    HONEYPOT_PREFIX = f"{CONFIG_PREFIX}:honeypot"
    SYSTEM_PREFIX = f"{CONFIG_PREFIX}:system"

    def __init__(self, redis_client: redis.Redis | None = None) -> None:
        """Initialize config manager with optional Redis client.
        
        If redis_client is None, manager operates in disabled mode
        (configs read-only from initial values).
        """
        self._redis = redis_client
        self._enabled = redis_client is not None
        if redis_client is None:
            logger.info("RedisConfigManager disabled: Redis client not provided")
        else:
            logger.info("RedisConfigManager enabled: Redis-backed config active")

    # ------------------------------------------------------------------
    # Policy Configuration
    # ------------------------------------------------------------------

    def get_policy_threshold(self, key: str, default: float) -> float:
        """Get policy threshold from Redis or return default."""
        if not self._enabled:
            return default
        try:
            value = self._redis.get(f"{self.POLICY_PREFIX}:{key}")
            if value:
                return float(value)
        except Exception as e:
            logger.warning("Error reading policy config %s: %s", key, e)
        return default

    def set_policy_threshold(self, key: str, value: float) -> bool:
        """Set policy threshold in Redis."""
        if not self._enabled:
            logger.warning("Cannot set policy config: Redis not enabled")
            return False
        try:
            self._redis.set(f"{self.POLICY_PREFIX}:{key}", str(value))
            logger.info("Updated policy config %s = %s", key, value)
            return True
        except Exception as e:
            logger.error("Error setting policy config %s: %s", key, e)
            return False

    def get_all_policy_config(self) -> dict[str, float]:
        """Get all policy configuration from Redis."""
        if not self._enabled:
            return {}
        try:
            pattern = f"{self.POLICY_PREFIX}:*"
            keys = self._redis.keys(pattern)
            config = {}
            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode()
                field_name = key.replace(f"{self.POLICY_PREFIX}:", "")
                value = self._redis.get(key)
                if value:
                    try:
                        config[field_name] = float(value)
                    except ValueError:
                        pass
            return config
        except Exception as e:
            logger.warning("Error reading all policy config: %s", e)
            return {}

    def set_all_policy_config(self, config: dict[str, float]) -> bool:
        """Set entire policy configuration in Redis."""
        if not self._enabled:
            logger.warning("Cannot set policy config: Redis not enabled")
            return False
        try:
            pipe = self._redis.pipeline()
            # Clear old config
            for key in self._redis.keys(f"{self.POLICY_PREFIX}:*"):
                pipe.delete(key)
            # Set new config
            for key, value in config.items():
                pipe.set(f"{self.POLICY_PREFIX}:{key}", str(value))
            pipe.execute()
            logger.info("Updated full policy config with %d keys", len(config))
            return True
        except Exception as e:
            logger.error("Error setting policy config: %s", e)
            return False

    # ------------------------------------------------------------------
    # Honeypot Configuration
    # ------------------------------------------------------------------

    def get_honeypot_ips(self, default_ips: list[str] | None = None) -> list[str]:
        """Get honeypot IPs from Redis or return default."""
        if not self._enabled:
            return default_ips or []
        try:
            value = self._redis.get(f"{self.HONEYPOT_PREFIX}:ips")
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning("Error reading honeypot IPs: %s", e)
        return default_ips or []

    def set_honeypot_ips(self, ips: list[str]) -> bool:
        """Set honeypot IPs in Redis."""
        if not self._enabled:
            logger.warning("Cannot set honeypot config: Redis not enabled")
            return False
        try:
            self._redis.set(f"{self.HONEYPOT_PREFIX}:ips", json.dumps(ips))
            logger.info("Updated honeypot IPs: %s", ips)
            return True
        except Exception as e:
            logger.error("Error setting honeypot IPs: %s", e)
            return False

    def get_honeypot_ports(self, default_ports: list[int] | None = None) -> list[int]:
        """Get honeypot ports from Redis or return default."""
        if not self._enabled:
            return default_ports or []
        try:
            value = self._redis.get(f"{self.HONEYPOT_PREFIX}:ports")
            if value:
                return json.loads(value)
        except Exception as e:
            logger.warning("Error reading honeypot ports: %s", e)
        return default_ports or []

    def set_honeypot_ports(self, ports: list[int]) -> bool:
        """Set honeypot ports in Redis."""
        if not self._enabled:
            logger.warning("Cannot set honeypot config: Redis not enabled")
            return False
        try:
            self._redis.set(f"{self.HONEYPOT_PREFIX}:ports", json.dumps(ports))
            logger.info("Updated honeypot ports: %s", ports)
            return True
        except Exception as e:
            logger.error("Error setting honeypot ports: %s", e)
            return False

    # ------------------------------------------------------------------
    # System Configuration
    # ------------------------------------------------------------------

    def get_system_value(self, key: str, default: Any = None) -> Any:
        """Get system configuration value from Redis."""
        if not self._enabled:
            return default
        try:
            value = self._redis.get(f"{self.SYSTEM_PREFIX}:{key}")
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value.decode() if isinstance(value, bytes) else str(value)
        except Exception as e:
            logger.warning("Error reading system config %s: %s", key, e)
        return default

    def set_system_value(self, key: str, value: Any) -> bool:
        """Set system configuration value in Redis."""
        if not self._enabled:
            logger.warning("Cannot set system config: Redis not enabled")
            return False
        try:
            if isinstance(value, (dict, list)):
                json_value = json.dumps(value)
            else:
                json_value = str(value)
            self._redis.set(f"{self.SYSTEM_PREFIX}:{key}", json_value)
            logger.info("Updated system config %s", key)
            return True
        except Exception as e:
            logger.error("Error setting system config %s: %s", key, e)
            return False

    # ------------------------------------------------------------------
    # Generic Configuration Access
    # ------------------------------------------------------------------

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Generic get for any section."""
        if not self._enabled:
            return default
        try:
            redis_key = f"{self.CONFIG_PREFIX}:{section}:{key}"
            value = self._redis.get(redis_key)
            if value is None:
                return default
            try:
                # Try to parse as JSON first (for dicts, lists, etc.)
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # Fall back to string
                return value.decode() if isinstance(value, bytes) else value
        except Exception as e:
            logger.warning("Error reading config %s:%s: %s", section, key, e)
            return default

    def set(self, section: str, key: str, value: Any) -> bool:
        """Generic set for any section."""
        if not self._enabled:
            logger.warning("Cannot set config: Redis not enabled")
            return False
        try:
            redis_key = f"{self.CONFIG_PREFIX}:{section}:{key}"
            if isinstance(value, (dict, list)):
                json_value = json.dumps(value)
            else:
                json_value = str(value)
            self._redis.set(redis_key, json_value)
            logger.info("Updated config %s:%s", section, key)
            return True
        except Exception as e:
            logger.error("Error setting config %s:%s: %s", section, key, e)
            return False

    # ------------------------------------------------------------------
    # Health and Status
    # ------------------------------------------------------------------

    def is_enabled(self) -> bool:
        """Return True if Redis backend is available."""
        return self._enabled

    def health_check(self) -> bool:
        """Check if Redis is reachable."""
        if not self._enabled:
            return False
        try:
            self._redis.ping()
            return True
        except Exception:
            return False
