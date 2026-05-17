"""Step 44 (Phase F) — ContextVar greenlet isolation regression tests.

Validates that two concurrent requests carrying different X-Correlation-IDs
never observe each other's correlation ID — the ContextVar must isolate
per-request state even under concurrent load.

Depends on: Step 22 (sanitized correlation IDs, C-07) and correlation_tracing module.
"""
from __future__ import annotations

import threading
from contextvars import copy_context

from src.correlation_tracing import (
    generate_correlation_id,
    get_correlation_id,
    sanitize_correlation_id,
    set_correlation_id,
    _correlation_id_context,
)


# ---------------------------------------------------------------------------
# 1. ContextVar is thread-isolated by default
# ---------------------------------------------------------------------------


def test_contextvar_isolated_across_threads():
    """Two threads setting the same ContextVar must not see each other's value."""
    barrier = threading.Barrier(2)
    results = {}

    def thread_fn(tid, my_id):
        token = _correlation_id_context.set(my_id)
        barrier.wait()  # both threads pause here — maximum concurrency
        results[tid] = _correlation_id_context.get()
        _correlation_id_context.reset(token)

    t1 = threading.Thread(target=thread_fn, args=(1, "id-for-thread-1"))
    t2 = threading.Thread(target=thread_fn, args=(2, "id-for-thread-2"))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert results[1] == "id-for-thread-1", "Thread 1 saw wrong correlation ID"
    assert results[2] == "id-for-thread-2", "Thread 2 saw wrong correlation ID"


# ---------------------------------------------------------------------------
# 2. set_correlation_id / get_correlation_id round-trip per thread
# ---------------------------------------------------------------------------


def test_set_get_roundtrip_is_thread_safe():
    """set_correlation_id/get_correlation_id must return the calling thread's value."""
    errors = []

    def worker(my_id):
        set_correlation_id(my_id)
        got = get_correlation_id()
        if got != my_id:
            errors.append(f"worker({my_id}): got {got!r}")

    threads = [threading.Thread(target=worker, args=(f"req-{i}",)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Correlation ID leakage detected: {errors}"


# ---------------------------------------------------------------------------
# 3. Flask request context isolation
# ---------------------------------------------------------------------------


def test_flask_request_isolation(monkeypatch, tmp_path):
    """Two Flask requests with different X-Correlation-IDs must not mix."""
    import web_app.app as app_module
    from src.ops_store import OpsStore

    monkeypatch.setenv("INIDS_ADMIN_API_KEY", "test-admin-f44")
    store = OpsStore(str(tmp_path / "ops.db"))
    monkeypatch.setattr(app_module, "ops_store", store)
    app_module.app.ops_store = store

    results = {}

    def make_request(cid_in, result_key):
        with app_module.app.test_client() as client:
            resp = client.get(
                "/api/health",
                headers={"X-Correlation-ID": cid_in},
            )
            results[result_key] = resp.headers.get("X-Correlation-ID", "")

    t1 = threading.Thread(target=make_request, args=("cid-request-alpha", "alpha"))
    t2 = threading.Thread(target=make_request, args=("cid-request-beta", "beta"))
    t1.start(); t2.start()
    t1.join(); t2.join()

    assert results.get("alpha") == "cid-request-alpha", (
        f"Request alpha saw wrong correlation ID: {results.get('alpha')!r}"
    )
    assert results.get("beta") == "cid-request-beta", (
        f"Request beta saw wrong correlation ID: {results.get('beta')!r}"
    )
    assert results.get("alpha") != results.get("beta"), (
        "Two concurrent requests returned the same correlation ID — ContextVar leaking"
    )


# ---------------------------------------------------------------------------
# 4. sanitize_correlation_id is idempotent and stateless
# ---------------------------------------------------------------------------


def test_sanitize_is_stateless():
    """sanitize_correlation_id must not share state across calls."""
    ids_a = [sanitize_correlation_id(f"id-{i}") for i in range(50)]
    ids_b = [sanitize_correlation_id(f"id-{i}") for i in range(50)]
    assert ids_a == ids_b, "sanitize_correlation_id is not idempotent"


# ---------------------------------------------------------------------------
# 5. copy_context properly scopes ContextVar writes
# ---------------------------------------------------------------------------


def test_copy_context_scopes_writes():
    """Writes inside copy_context().run() must not escape to the outer context."""
    outer_id = "outer-id-stays"
    _correlation_id_context.set(outer_id)

    inner_id = "inner-id-must-not-escape"
    ctx = copy_context()
    ctx.run(_correlation_id_context.set, inner_id)

    assert _correlation_id_context.get() == outer_id, (
        "ContextVar write inside copy_context leaked to outer context"
    )
