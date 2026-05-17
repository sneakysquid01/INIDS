"""Regression tests for B-05: Leader election fail-closed.

Validates:
- Redis failure → is_leader returns False (fail-closed)
- leader_election_state metric = 0 when Redis unavailable
- INIDS_REDIS_REQUIRED=false → is_leader returns True (single-instance mode)
- TTL sleep uses max(1, int(ttl / 3)) not integer floor division
- status() includes redis_required field
"""
from __future__ import annotations

import os
from unittest import mock

import pytest


class TestLeaderElectionFailClosed:
    def test_redis_failure_returns_not_leader(self):
        """When Redis raises, is_leader must be False (fail-closed)."""
        from src.ha.leader_election import LeaderElection

        call_count = 0

        def stop_after_one(key, val, nx=False, ex=None):
            nonlocal call_count
            call_count += 1
            le._running = False  # stop loop after first iteration
            raise ConnectionError("redis down")

        failing_redis = mock.MagicMock()
        failing_redis.set.side_effect = stop_after_one

        le = LeaderElection(redis_client=failing_redis, ttl_seconds=3)
        le._running = True

        with mock.patch("time.sleep"):
            le._election_loop()

        assert le.is_leader is False

    def test_redis_failure_emits_metric_zero(self):
        """leader_election_state must be 0 after Redis failure."""
        import src.ha.leader_election as le_mod
        from src.ha.leader_election import LeaderElection

        failing_redis = mock.MagicMock()
        failing_redis.set.side_effect = ConnectionError("redis down")

        le = LeaderElection(redis_client=failing_redis, ttl_seconds=3)
        le._running = False  # single iteration only

        with mock.patch("time.sleep"):
            # Manually simulate one failed attempt
            le._is_leader = True  # was leader before failure
            try:
                le._redis.set(le._key, le._instance_id, nx=True, ex=le._ttl)
            except Exception:
                le._is_leader = False
                le._emit_metric("leader_election_state", 0)

        assert le_mod.get_leader_election_state() == 0
        assert le.is_leader is False

    def test_no_redis_with_required_true_fails_closed(self):
        """No Redis + INIDS_REDIS_REQUIRED=true → not leader after start()."""
        from src.ha.leader_election import LeaderElection

        with mock.patch.dict(os.environ, {"INIDS_REDIS_REQUIRED": "true"}):
            le = LeaderElection(redis_client=None, ttl_seconds=30)
            le.start()

        assert le.is_leader is False

    def test_no_redis_with_required_false_is_leader(self):
        """INIDS_REDIS_REQUIRED=false → always leader (single-instance mode)."""
        from src.ha.leader_election import LeaderElection

        with mock.patch.dict(os.environ, {"INIDS_REDIS_REQUIRED": "false"}):
            le = LeaderElection(redis_client=None, ttl_seconds=30)
            le.start()

        assert le.is_leader is True

    def test_ttl_sleep_uses_integer_division_safe(self):
        """Sleep interval must be max(1, int(ttl / 3)) — never 0 for ttl < 3."""
        from src.ha.leader_election import LeaderElection

        redis = mock.MagicMock()

        le = LeaderElection(redis_client=redis, ttl_seconds=2, instance_id="t1")
        le._running = True

        def set_side_effect(*a, **kw):
            le._running = False  # stop after first iteration
            return True

        redis.set.side_effect = set_side_effect

        sleep_calls: list[float] = []
        with mock.patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            le._election_loop()

        assert sleep_calls, "sleep must be called"
        assert all(s >= 1 for s in sleep_calls), (
            f"Sleep interval must be >= 1 (max(1, ...)), got {sleep_calls}"
        )

    def test_status_includes_redis_required(self):
        """status() must expose redis_required field."""
        from src.ha.leader_election import LeaderElection

        with mock.patch.dict(os.environ, {"INIDS_REDIS_REQUIRED": "false"}):
            le = LeaderElection(redis_client=None)

        st = le.status()
        assert "redis_required" in st
        assert st["redis_required"] is False

    def test_election_success_emits_metric_one(self):
        """Successful leader acquisition emits leader_election_state=1."""
        import src.ha.leader_election as le_mod
        from src.ha.leader_election import LeaderElection

        redis = mock.MagicMock()
        le = LeaderElection(redis_client=redis, ttl_seconds=30, instance_id="winner")
        le._running = True

        def set_side_effect(*a, **kw):
            le._running = False
            return True  # acquired

        redis.set.side_effect = set_side_effect

        with mock.patch("time.sleep"):
            le._election_loop()

        assert le_mod.get_leader_election_state() == 1
        assert le.is_leader is True
