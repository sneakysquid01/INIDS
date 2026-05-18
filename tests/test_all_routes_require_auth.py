"""
Regression tests for PLAN.md Phase A Step 3 (A-03): Auth All Undecorated Routes.
F-AUTH-REMOVE: Updated to check for @require_roles (new decorator) instead of
@require_role/@require_auth (legacy decorators deleted in Phase F).
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Routes that are intentionally public — must NOT require auth
PUBLIC_API_ROUTES = {
    "/health",
    "/api/health",
    "/api/health/live",
    "/api/health/ready",
    "/api/auth/login",
    "/api/auth/refresh",
    "/api/auth/validate",
    "/api/auth/status",
}

# Critical routes the plan explicitly called out as requiring auth
CRITICAL_PROTECTED_ROUTES = [
    ("/api/alerts", "GET"),
    ("/api/fp-suppressions", "GET"),
    ("/api/policy", "GET"),
    ("/api/detect", "POST"),
    ("/api/engines", "GET"),
    ("/api/incidents", "GET"),
    ("/api/modules/auto-blocking", "GET"),
    ("/api/modules/false-positive", "GET"),
    ("/api/modules/policy-tuning", "GET"),
    ("/api/modules/forensic-timeline", "GET"),
    ("/api/modules/behavioral-profiling", "GET"),
    ("/api/modules/drift-monitor", "GET"),
    ("/api/modules/anomaly-learning", "GET"),
    ("/api/modules/network-topology", "GET"),
    ("/api/dashboard/metrics", "GET"),
    ("/api/perception/pulse", "GET"),
    ("/", "GET"),
]


def test_no_rfc1918_in_threat_intel_source():
    """Verify load_threat_intel no longer loads RFC-1918 mock indicators (cross-check with A-02)."""
    app_py = os.path.join(ROOT, "web_app", "app.py")
    with open(app_py, encoding="utf-8") as f:
        source = f.read()
    assert "mock_indicators" not in source


def test_require_role_decorates_fp_suppression_in_source():
    """FP suppression endpoint must have require_roles in source (pre-app-start check)."""
    app_py = os.path.join(ROOT, "web_app", "app.py")
    with open(app_py, encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if '"/api/fp-suppressions"' in line and "@app.route" in line:
            surrounding = "".join(lines[i:i+4])
            assert "@require_roles" in surrounding, (
                f"/api/fp-suppressions must have @require_roles immediately after @app.route, "
                f"found: {surrounding!r}"
            )


def test_module_policy_tuning_is_admin_in_source():
    """Policy tuning module API must require admin role."""
    app_py = os.path.join(ROOT, "web_app", "app.py")
    with open(app_py, encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if '"/api/modules/policy-tuning"' in line and "@app.route" in line:
            surrounding = "".join(lines[i:i+4])
            assert 'require_roles("admin")' in surrounding, (
                f"/api/modules/policy-tuning must require admin role, found: {surrounding!r}"
            )


def test_public_routes_not_decorated_with_require_roles():
    """Public routes must NOT have require_roles (that would block health checks)."""
    app_py = os.path.join(ROOT, "web_app", "app.py")
    with open(app_py, encoding="utf-8") as f:
        lines = f.readlines()

    for public_path in ["/health", "/api/health", "/api/auth/login", "/api/auth/refresh"]:
        for i, line in enumerate(lines):
            if f'"{public_path}"' in line and "@app.route" in line:
                next_lines = "".join(lines[i+1:i+3])
                assert "@require_roles" not in next_lines, (
                    f"Public route {public_path} should not have @require_roles. "
                    f"Found: {next_lines!r}"
                )
                break


def test_dashboard_route_requires_auth_in_source():
    """/dashboard must have require_roles decorator (searches app.py and blueprints)."""
    blueprints_dir = os.path.join(ROOT, "web_app", "blueprints")
    source_files = [os.path.join(ROOT, "web_app", "app.py")]
    if os.path.isdir(blueprints_dir):
        for fn in os.listdir(blueprints_dir):
            if fn.endswith(".py"):
                source_files.append(os.path.join(blueprints_dir, fn))

    for source_file in source_files:
        with open(source_file, encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if '"/dashboard"' in line and ".route(" in line and "main" not in line:
                surrounding = "".join(lines[i:i+4])
                assert "@require_roles" in surrounding, (
                    f"/dashboard must have @require_roles. Found: {surrounding!r}"
                )
                return
    pytest.fail("/dashboard route not found in app.py or any blueprint")


def test_api_predict_requires_auth_in_source():
    """/api/predict must have require_roles decorator (checks app.py and blueprints)."""
    blueprints_dir = os.path.join(ROOT, "web_app", "blueprints")
    source_files = [os.path.join(ROOT, "web_app", "app.py")]
    if os.path.isdir(blueprints_dir):
        for fn in os.listdir(blueprints_dir):
            if fn.endswith(".py"):
                source_files.append(os.path.join(blueprints_dir, fn))

    route_pattern = '"/api/predict"'
    for source_file in source_files:
        with open(source_file, encoding="utf-8") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if route_pattern in line and ".route" in line:
                surrounding = "".join(lines[i:i+4])
                assert "@require_roles" in surrounding, (
                    f"/api/predict in {source_file} missing @require_roles. Found: {surrounding!r}"
                )
                return
    pytest.fail("/api/predict route not found in app.py or any blueprint")
