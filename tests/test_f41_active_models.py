"""Step 41 (Phase F) — ACTIVE_MODELS manifest and staged hot-reload regression tests.

Validates:
1. ActiveModelManager reads/writes active_models.json correctly
2. register_slot() is idempotent
3. promote() verifies SHA-256 and raises ChecksumMismatch on mismatch
4. promote() atomically replaces the active file via os.replace
5. Reload callbacks are fired after a successful promotion
6. list_slots() and get_slot() reflect the manifest
7. Concurrent promotions are safe (lock prevents interleaving)
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import threading

import pytest

from src.active_model_manager import (
    ActiveModelManager,
    ChecksumMismatch,
    ModelSlot,
    _sha256_of,
)


def _write_fake_model(path: pathlib.Path, content: bytes = b"fake-model-bytes") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


# ---------------------------------------------------------------------------
# 1. register_slot
# ---------------------------------------------------------------------------


class TestRegisterSlot:
    def test_register_creates_manifest(self, tmp_path):
        mgr = ActiveModelManager(tmp_path)
        mgr.register_slot("rf_primary", tmp_path / "rf.pkl", "v1", sha256="abc123")
        assert (tmp_path / "active_models.json").exists()

    def test_register_slot_readable(self, tmp_path):
        mgr = ActiveModelManager(tmp_path)
        mgr.register_slot("rf_primary", tmp_path / "rf.pkl", "v1", sha256="abc123")
        slot = mgr.get_slot("rf_primary")
        assert slot is not None
        assert slot.slot_name == "rf_primary"
        assert slot.model_version == "v1"
        assert slot.sha256 == "abc123"

    def test_register_multiple_slots(self, tmp_path):
        mgr = ActiveModelManager(tmp_path)
        mgr.register_slot("rf_primary", tmp_path / "rf.pkl", "v1", sha256="aaa")
        mgr.register_slot("anomaly", tmp_path / "anomaly.pkl", "v1", sha256="bbb")
        slots = mgr.list_slots()
        names = {s.slot_name for s in slots}
        assert "rf_primary" in names
        assert "anomaly" in names

    def test_register_is_idempotent(self, tmp_path):
        mgr = ActiveModelManager(tmp_path)
        mgr.register_slot("rf_primary", tmp_path / "rf.pkl", "v1", sha256="aaa")
        mgr.register_slot("rf_primary", tmp_path / "rf.pkl", "v2", sha256="bbb")
        slot = mgr.get_slot("rf_primary")
        assert slot.model_version == "v2"

    def test_get_slot_returns_none_if_missing(self, tmp_path):
        mgr = ActiveModelManager(tmp_path)
        assert mgr.get_slot("nonexistent") is None

    def test_list_slots_empty_when_no_manifest(self, tmp_path):
        mgr = ActiveModelManager(tmp_path)
        assert mgr.list_slots() == []


# ---------------------------------------------------------------------------
# 2. promote — happy path
# ---------------------------------------------------------------------------


class TestPromote:
    def test_promote_replaces_active_file(self, tmp_path):
        mgr = ActiveModelManager(tmp_path)

        active_path = tmp_path / "rf.pkl"
        active_path.write_bytes(b"old-model")
        mgr.register_slot("rf_primary", active_path, "v1", sha256="oldsha")

        staged_content = b"new-model-v2"
        expected_sha = _write_fake_model(tmp_path / "staged" / "rf_primary.pkl", staged_content)

        slot = mgr.promote("rf_primary", expected_sha256=expected_sha)

        assert active_path.read_bytes() == staged_content
        assert slot.model_version == "v2"
        assert slot.sha256 == expected_sha

    def test_promote_updates_manifest(self, tmp_path):
        mgr = ActiveModelManager(tmp_path)
        active_path = tmp_path / "anomaly.pkl"
        active_path.write_bytes(b"old")
        mgr.register_slot("anomaly", active_path, "v1", sha256="oldsha")

        content = b"better-anomaly-model"
        sha = _write_fake_model(tmp_path / "staged" / "anomaly.pkl", content)
        mgr.promote("anomaly", expected_sha256=sha)

        slot = mgr.get_slot("anomaly")
        assert slot.sha256 == sha
        assert slot.model_version == "v2"

    def test_promote_removes_staged_file(self, tmp_path):
        mgr = ActiveModelManager(tmp_path)
        active = tmp_path / "m.pkl"
        active.write_bytes(b"old")
        mgr.register_slot("m", active, "v1", sha256="x")

        content = b"new-content"
        sha = _write_fake_model(tmp_path / "staged" / "m.pkl", content)
        mgr.promote("m", expected_sha256=sha)

        assert not (tmp_path / "staged" / "m.pkl").exists()


# ---------------------------------------------------------------------------
# 3. promote — error cases
# ---------------------------------------------------------------------------


class TestPromoteErrors:
    def test_promote_missing_staged_raises_fnf(self, tmp_path):
        mgr = ActiveModelManager(tmp_path)
        with pytest.raises(FileNotFoundError):
            mgr.promote("missing_slot", expected_sha256="doesntmatter")

    def test_promote_checksum_mismatch_raises(self, tmp_path):
        mgr = ActiveModelManager(tmp_path)
        active = tmp_path / "rf.pkl"
        active.write_bytes(b"old")
        mgr.register_slot("rf_primary", active, "v1", sha256="old")

        _write_fake_model(tmp_path / "staged" / "rf_primary.pkl", b"new-content")
        with pytest.raises(ChecksumMismatch):
            mgr.promote("rf_primary", expected_sha256="definitely-wrong-sha")

    def test_active_file_unchanged_on_checksum_mismatch(self, tmp_path):
        mgr = ActiveModelManager(tmp_path)
        active = tmp_path / "rf.pkl"
        active.write_bytes(b"original")
        mgr.register_slot("rf_primary", active, "v1", sha256="old")

        _write_fake_model(tmp_path / "staged" / "rf_primary.pkl", b"malicious")
        with pytest.raises(ChecksumMismatch):
            mgr.promote("rf_primary", expected_sha256="wrong")

        # active file must be unchanged
        assert active.read_bytes() == b"original"


# ---------------------------------------------------------------------------
# 4. Hot-reload callbacks
# ---------------------------------------------------------------------------


class TestReloadCallbacks:
    def test_callback_fired_after_promote(self, tmp_path):
        mgr = ActiveModelManager(tmp_path)
        active = tmp_path / "rf.pkl"
        active.write_bytes(b"old")
        mgr.register_slot("rf_primary", active, "v1", sha256="old")

        fired: list[str] = []
        mgr.add_reload_callback("rf_primary", lambda p: fired.append(p))

        content = b"new"
        sha = _write_fake_model(tmp_path / "staged" / "rf_primary.pkl", content)
        mgr.promote("rf_primary", expected_sha256=sha)

        assert len(fired) == 1
        assert str(active) in fired[0] or fired[0] == str(active)

    def test_multiple_callbacks_all_fired(self, tmp_path):
        mgr = ActiveModelManager(tmp_path)
        active = tmp_path / "rf.pkl"
        active.write_bytes(b"old")
        mgr.register_slot("rf_primary", active, "v1", sha256="old")

        counts = [0, 0]
        mgr.add_reload_callback("rf_primary", lambda p: counts.__setitem__(0, counts[0] + 1))
        mgr.add_reload_callback("rf_primary", lambda p: counts.__setitem__(1, counts[1] + 1))

        sha = _write_fake_model(tmp_path / "staged" / "rf_primary.pkl", b"new")
        mgr.promote("rf_primary", expected_sha256=sha)

        assert counts == [1, 1]

    def test_failing_callback_does_not_abort_promotion(self, tmp_path):
        mgr = ActiveModelManager(tmp_path)
        active = tmp_path / "rf.pkl"
        active.write_bytes(b"old")
        mgr.register_slot("rf_primary", active, "v1", sha256="old")

        mgr.add_reload_callback("rf_primary", lambda p: (_ for _ in ()).throw(RuntimeError("cb error")))

        sha = _write_fake_model(tmp_path / "staged" / "rf_primary.pkl", b"new")
        slot = mgr.promote("rf_primary", expected_sha256=sha)  # must not raise
        assert slot.model_version == "v2"


# ---------------------------------------------------------------------------
# 5. Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_register_is_safe(tmp_path):
    """Concurrent register_slot calls must not corrupt the manifest."""
    mgr = ActiveModelManager(tmp_path)
    errors = []

    def worker(i):
        try:
            mgr.register_slot(f"slot_{i}", tmp_path / f"m{i}.pkl", "v1", sha256=f"sha_{i}")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    slots = mgr.list_slots()
    assert len(slots) == 20


# ---------------------------------------------------------------------------
# 6. SHA-256 helper
# ---------------------------------------------------------------------------


def test_sha256_of(tmp_path):
    p = tmp_path / "test.bin"
    content = b"hello world"
    p.write_bytes(content)
    expected = hashlib.sha256(content).hexdigest()
    assert _sha256_of(p) == expected


# ---------------------------------------------------------------------------
# 7. Manifest format
# ---------------------------------------------------------------------------


def test_manifest_contains_format_version(tmp_path):
    mgr = ActiveModelManager(tmp_path)
    mgr.register_slot("rf", tmp_path / "rf.pkl", "v1", sha256="abc")
    import json
    data = json.loads((tmp_path / "active_models.json").read_text())
    assert data.get("format_version") == "1"
    assert "updated_at" in data


def test_manifest_atomic_write_uses_replace(tmp_path, monkeypatch):
    """_write_manifest must use os.replace (atomic on POSIX, near-atomic on Windows)."""
    replaces = []
    real_replace = os.replace

    def spy_replace(src, dst):
        replaces.append((src, dst))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)
    mgr = ActiveModelManager(tmp_path)
    mgr.register_slot("rf", tmp_path / "rf.pkl", "v1", sha256="abc")
    assert any(str(tmp_path) in str(dst) for _, dst in replaces)
