"""Step 39: Verify app.py route extraction into blueprints — no @app.route remains."""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_PY = ROOT / "web_app" / "app.py"
BLUEPRINTS_DIR = ROOT / "web_app" / "blueprints"

EXPECTED_BLUEPRINTS = [
    "health",
    "auth",
    "ingest",
    "observability",
    "detection",
    "pages",
    "dashboard",
    "prevention",
    "intel",
    "system",
    "modules",
]


def _app_lines():
    return APP_PY.read_text(encoding="utf-8").splitlines()


def test_no_app_route_decorators_in_app_py():
    """All HTTP routes must live in blueprints — zero @app.route in app.py."""
    violations = [
        (i + 1, line)
        for i, line in enumerate(_app_lines())
        if re.match(r"\s*@app\.route\s*\(", line)
    ]
    assert violations == [], (
        f"Found @app.route in app.py at lines: {[ln for ln, _ in violations]}"
    )


def test_all_blueprints_registered_in_app_py():
    """All 11 blueprints must be imported and registered in app.py."""
    text = APP_PY.read_text(encoding="utf-8")
    for name in EXPECTED_BLUEPRINTS:
        bp_var = f"{name}_bp"
        assert f"import {bp_var}" in text or f"from web_app.blueprints.{name} import {bp_var}" in text, (
            f"Blueprint '{bp_var}' not imported in app.py"
        )
        assert f"app.register_blueprint({bp_var})" in text, (
            f"Blueprint '{bp_var}' not registered in app.py"
        )


def test_all_blueprint_files_exist():
    """Every expected blueprint module file must exist."""
    for name in EXPECTED_BLUEPRINTS:
        bp_file = BLUEPRINTS_DIR / f"{name}.py"
        assert bp_file.exists(), f"Blueprint file missing: {bp_file}"


def test_blueprints_define_blueprint_objects():
    """Each blueprint file must define a Blueprint object named <name>_bp."""
    for name in EXPECTED_BLUEPRINTS:
        bp_file = BLUEPRINTS_DIR / f"{name}.py"
        text = bp_file.read_text(encoding="utf-8")
        bp_var = f"{name}_bp"
        assert f"{bp_var} = Blueprint(" in text, (
            f"Blueprint object '{bp_var}' not found in {bp_file.name}"
        )


def test_blueprints_have_no_circular_app_import_at_module_level():
    """Blueprint files must not import web_app.app at module level (only inside functions)."""
    for name in EXPECTED_BLUEPRINTS:
        bp_file = BLUEPRINTS_DIR / f"{name}.py"
        lines = bp_file.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, start=1):
            stripped = line.strip()
            if "import web_app.app" in stripped or "from web_app.app" in stripped:
                # Must be indented (inside a function)
                assert line.startswith(" ") or line.startswith("\t"), (
                    f"{bp_file.name}:{i} — top-level import of web_app.app creates circular import: {line!r}"
                )


def test_app_py_line_count_within_target():
    """app.py should be substantially reduced vs original 3964 lines."""
    lines = _app_lines()
    assert len(lines) < 2500, (
        f"app.py has {len(lines)} lines — expected < 2500 after blueprint extraction"
    )


def test_broadcaster_thread_in_app_py():
    """Background broadcaster thread must remain in app.py (uses socketio directly)."""
    text = APP_PY.read_text(encoding="utf-8")
    assert "_start_module_update_broadcaster" in text
    assert "_module_update_broadcaster" in text
    assert "_update_thread" in text


def test_websocket_handlers_in_app_py():
    """@socketio.on handlers must remain in app.py (cannot live in blueprints)."""
    text = APP_PY.read_text(encoding="utf-8")
    assert "@socketio.on('connect')" in text
    assert "@socketio.on('disconnect')" in text
    assert "handle_subscribe_module" in text


def test_error_handlers_in_app_py():
    """Error handlers must remain in app.py."""
    text = APP_PY.read_text(encoding="utf-8")
    assert "@app.errorhandler(404)" in text
    assert "@app.errorhandler(Exception)" in text


def test_blueprint_routes_use_require_roles():
    """Spot-check that blueprint routes still carry @require_roles auth decoration."""
    for name in ["detection", "prevention", "intel", "system"]:
        bp_file = BLUEPRINTS_DIR / f"{name}.py"
        text = bp_file.read_text(encoding="utf-8")
        assert "@require_roles(" in text, (
            f"{bp_file.name} has no @require_roles decorators — auth may be missing"
        )
