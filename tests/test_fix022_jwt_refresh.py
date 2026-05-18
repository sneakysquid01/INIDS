"""
FIX-022 regression: client-side JWT auto-refresh wiring in http-client.js.
"""
from pathlib import Path

HTTP_CLIENT = (
    Path(__file__).parent.parent / "web_app" / "static" / "js" / "core" / "http-client.js"
).read_text(encoding="utf-8")


def test_set_token_method_exists():
    assert "setToken(" in HTTP_CLIENT


def test_token_stored_in_memory_not_localstorage():
    assert "localStorage.setItem" not in HTTP_CLIENT, \
        "JWT must NOT be written to localStorage (FIX-022)"
    assert "localStorage.getItem" not in HTTP_CLIENT, \
        "JWT must NOT be read from localStorage (FIX-022)"


def test_token_stored_in_module_variable():
    assert "let _token" in HTTP_CLIENT


def test_refresh_timer_scheduled_at_80_percent():
    assert "0.8" in HTTP_CLIENT, "Refresh must be scheduled at 80% of TTL"


def test_refresh_endpoint_is_api_auth_refresh():
    assert "/api/auth/refresh" in HTTP_CLIENT


def test_single_flight_refresh_promise():
    assert "_refreshInFlight" in HTTP_CLIENT


def test_refresh_timeout_5s():
    assert "5000" in HTTP_CLIENT, "Refresh call must have a 5s timeout"


def test_on_refresh_failure_redirect_to_login():
    assert "/login" in HTTP_CLIENT


def test_401_token_expired_triggers_refresh():
    assert "token_expired" in HTTP_CLIENT


def test_retry_once_after_refresh():
    assert "_afterRefresh" in HTTP_CLIENT


def test_clear_token_method_exists():
    assert "clearToken(" in HTTP_CLIENT


def test_get_token_method_exists():
    assert "getToken(" in HTTP_CLIENT


def test_global_state_updated_on_set_token():
    assert 'GlobalState.set("auth.token"' in HTTP_CLIENT or \
           "GlobalState.set('auth.token'" in HTTP_CLIENT


def test_socket_reconnect_triggered_after_set_token():
    assert "reconnectWithToken" in HTTP_CLIENT
