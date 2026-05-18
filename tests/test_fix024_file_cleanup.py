"""
FIX-024 regression: validate_phase_*.py moved to tools/validate/; root global_state.js deleted.
"""
from pathlib import Path

ROOT = Path(__file__).parent.parent
TOOLS_VALIDATE = ROOT / "tools" / "validate"


def test_root_has_no_validate_phase_scripts():
    stragglers = list(ROOT.glob("validate_phase_*.py"))
    assert stragglers == [], f"Root still has validate scripts: {stragglers}"


def test_root_has_no_test_scripts():
    # Root-level test_*.py (ad-hoc, not in tests/) should be gone
    stragglers = [
        p for p in ROOT.glob("test_*.py")
        if p.parent == ROOT
    ]
    assert stragglers == [], f"Root still has ad-hoc test scripts: {stragglers}"


def test_tools_validate_dir_exists():
    assert TOOLS_VALIDATE.is_dir(), "tools/validate/ directory must exist"


def test_validate_scripts_moved_to_tools():
    moved = list(TOOLS_VALIDATE.glob("validate_phase_*.py"))
    assert len(moved) >= 5, f"Expected ≥5 validate scripts in tools/validate/, found {len(moved)}"


def test_root_global_state_js_deleted():
    assert not (ROOT / "global_state.js").exists(), \
        "Root global_state.js (unreferenced) must be deleted"


def test_web_core_global_state_js_preserved():
    """The real module (hyphenated) used by ES modules must still exist."""
    assert (ROOT / "web_app" / "static" / "js" / "core" / "global-state.js").exists(), \
        "web_app/static/js/core/global-state.js must NOT be deleted"
