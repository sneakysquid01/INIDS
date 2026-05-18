"""
FIX-010 regression: Redis service in docker-compose.yml.
"""
from pathlib import Path

COMPOSE = (
    Path(__file__).parent.parent / "deploy" / "compose" / "docker-compose.yml"
).read_text(encoding="utf-8")


def test_redis_service_defined():
    assert "redis:" in COMPOSE


def test_redis_image_is_alpine():
    assert "redis:7-alpine" in COMPOSE


def test_redis_appendonly_enabled():
    assert "--appendonly" in COMPOSE


def test_redis_healthcheck_present():
    assert "redis-cli" in COMPOSE and "ping" in COMPOSE


def test_redis_volume_declared():
    assert "inids-redis:" in COMPOSE


def test_inids_web_has_redis_url():
    assert "REDIS_URL=redis://redis:6379/0" in COMPOSE


def test_inids_web_depends_on_redis():
    assert "depends_on" in COMPOSE
    assert "service_healthy" in COMPOSE
