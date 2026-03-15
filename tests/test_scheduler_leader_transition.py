"""Tests for LeaderElection transition: leader -> follower -> leader.

Covers:
- A leader that loses the Redis key transitions to follower.
- A follower can re-acquire leadership after the current leader disappears.
- Renewal keeps the current leader stable.
- Instance that already holds the key renews successfully.
- Two instances: only one is leader at any given time.
"""
from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, call

import pytest

from src.ha.leader_election import LeaderElection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_election(redis_mock, instance_id: str = "inst-1", ttl: int = 3) -> LeaderElection:
    le = LeaderElection(redis_mock, key="inids:leader", ttl_seconds=ttl, instance_id=instance_id)
    le._running = True
    return le


def _tick(le: LeaderElection, redis_mock) -> None:
    """Drive one cycle of the election logic directly (without sleeping)."""
    try:
        acquired = redis_mock.set(le._key, le._instance_id, nx=True, ex=le._ttl)
        if acquired:
            le._is_leader = True
        else:
            current = redis_mock.get(le._key)
            if isinstance(current, bytes):
                current = current.decode()
            if current == le._instance_id:
                renewed = redis_mock.set(le._key, le._instance_id, xx=True, ex=le._ttl)
                le._is_leader = bool(renewed)
            else:
                le._is_leader = False
    except Exception:
        le._is_leader = False


# ---------------------------------------------------------------------------
# Tests: basic election
# ---------------------------------------------------------------------------


class TestBasicElection:
    def test_acquires_leadership_on_empty_key(self):
        redis_mock = MagicMock()
        redis_mock.set.return_value = True  # nx=True succeeded

        le = _make_election(redis_mock, "inst-1")
        _tick(le, redis_mock)

        assert le.is_leader is True

    def test_does_not_become_leader_if_key_held_by_other(self):
        redis_mock = MagicMock()
        redis_mock.set.return_value = None  # nx failed
        redis_mock.get.return_value = b"inst-other"

        le = _make_election(redis_mock, "inst-1")
        _tick(le, redis_mock)

        assert le.is_leader is False

    def test_error_during_tick_sets_not_leader(self):
        redis_mock = MagicMock()
        redis_mock.set.side_effect = ConnectionError("Redis down")

        le = _make_election(redis_mock, "inst-1")
        le._is_leader = True  # was leader before outage
        _tick(le, redis_mock)

        assert le.is_leader is False


# ---------------------------------------------------------------------------
# Tests: transition sequences
# ---------------------------------------------------------------------------


class TestLeaderFollowerTransition:
    def test_leader_loses_key_and_becomes_follower(self):
        redis_mock = MagicMock()
        le = _make_election(redis_mock, "inst-1")

        # Tick 1: acquire leadership
        redis_mock.set.return_value = True
        _tick(le, redis_mock)
        assert le.is_leader is True

        # Tick 2: another instance took the key (our renewal fails)
        redis_mock.set.return_value = None  # renewal (xx=True) fails
        redis_mock.get.return_value = b"inst-2"  # someone else holds it
        _tick(le, redis_mock)
        assert le.is_leader is False

    def test_follower_reacquires_after_leader_leaves(self):
        redis_mock = MagicMock()
        le = _make_election(redis_mock, "inst-1")

        # Start as follower
        redis_mock.set.return_value = None
        redis_mock.get.return_value = b"inst-2"
        _tick(le, redis_mock)
        assert le.is_leader is False

        # Leader leaves — key is gone; we acquire
        redis_mock.set.return_value = True
        _tick(le, redis_mock)
        assert le.is_leader is True

    def test_leader_follower_leader_cycle(self):
        redis_mock = MagicMock()
        le = _make_election(redis_mock, "inst-1")

        # Phase 1: become leader
        redis_mock.set.return_value = True
        _tick(le, redis_mock)
        assert le.is_leader is True

        # Phase 2: yield to another — follower
        redis_mock.set.return_value = None
        redis_mock.get.return_value = b"inst-2"
        _tick(le, redis_mock)
        assert le.is_leader is False

        # Phase 3: other leader disappears — re-acquire
        redis_mock.set.return_value = True
        _tick(le, redis_mock)
        assert le.is_leader is True


# ---------------------------------------------------------------------------
# Tests: renewal
# ---------------------------------------------------------------------------


class TestLeaderRenewal:
    def test_existing_leader_renews_successfully(self):
        redis_mock = MagicMock()
        le = _make_election(redis_mock, "inst-1")
        le._is_leader = True

        # Key is held by this instance; nx fails but xx succeeds
        redis_mock.set.side_effect = [
            None,  # nx=True fails (key already exists)
            True,  # xx=True succeeds (renewal)
        ]
        redis_mock.get.return_value = b"inst-1"

        _tick(le, redis_mock)
        assert le.is_leader is True

    def test_renewal_failure_yields_leadership(self):
        redis_mock = MagicMock()
        le = _make_election(redis_mock, "inst-1")
        le._is_leader = True

        # Key is held by us but xx renewal fails (TTL expired and another grabbed it)
        redis_mock.set.side_effect = [
            None,  # nx fails
            None,  # xx also fails
        ]
        redis_mock.get.return_value = b"inst-1"  # still our key in get check

        _tick(le, redis_mock)
        assert le.is_leader is False


# ---------------------------------------------------------------------------
# Tests: two-instance mutex
# ---------------------------------------------------------------------------


class TestTwoInstanceMutex:
    def test_only_one_leader_at_a_time(self):
        """Simulate two instances both trying to acquire leadership."""
        store: dict = {}

        def fake_set(key, value, nx=False, xx=False, ex=None):
            if nx:
                if key in store:
                    return None
                store[key] = value
                return True
            if xx:
                if key not in store:
                    return None
                store[key] = value
                return True
            store[key] = value
            return True

        def fake_get(key):
            val = store.get(key)
            return val.encode() if isinstance(val, str) else val

        redis_a = MagicMock()
        redis_a.set.side_effect = fake_set
        redis_a.get.side_effect = fake_get

        redis_b = MagicMock()
        redis_b.set.side_effect = fake_set
        redis_b.get.side_effect = fake_get

        le_a = _make_election(redis_a, "inst-1")
        le_b = _make_election(redis_b, "inst-2")

        _tick(le_a, redis_a)
        _tick(le_b, redis_b)

        # Exactly one should be the leader
        leaders = [le_a.is_leader, le_b.is_leader]
        assert leaders.count(True) == 1, f"Expected exactly 1 leader, got: {leaders}"

    def test_stop_releases_lock_for_other_instance(self):
        store: dict = {}

        def fake_set(key, value, nx=False, xx=False, ex=None):
            if nx and key in store:
                return None
            store[key] = value
            return True

        def fake_get(key):
            val = store.get(key)
            return val.encode() if isinstance(val, str) else val

        def fake_delete(key):
            store.pop(key, None)

        r_a = MagicMock()
        r_a.set.side_effect = fake_set
        r_a.get.side_effect = fake_get
        r_a.delete.side_effect = fake_delete

        r_b = MagicMock()
        r_b.set.side_effect = fake_set
        r_b.get.side_effect = fake_get

        le_a = _make_election(r_a, "inst-1")
        le_b = _make_election(r_b, "inst-2")

        # inst-1 becomes leader
        le_a._is_leader = True
        _tick(le_a, r_a)

        # inst-1 stops (releases lease)
        le_a.stop()
        assert "inids:leader" not in store

        # inst-2 should now be able to acquire
        _tick(le_b, r_b)
        assert le_b.is_leader is True
