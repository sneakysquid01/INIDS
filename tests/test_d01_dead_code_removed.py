"""D-01: Dead code removal regression tests.

Verifies that production_hardening.py is deleted and no production code
imports it. Also verifies no dead rate-limiter shim remains importable.
"""
import ast
import os
import pathlib
import importlib


SRC_ROOT = pathlib.Path(__file__).parent.parent / "src"
PRODUCTION_HARDENING_PATH = SRC_ROOT / "production_hardening.py"


def _iter_py_files(root: pathlib.Path):
    """Yield all .py files under root, excluding __pycache__ and venv."""
    for p in root.rglob("*.py"):
        parts = p.parts
        if "__pycache__" in parts or "venv" in parts:
            continue
        yield p


def _extract_imports(source: str) -> list[str]:
    """Return all module names imported in source (best-effort AST parse)."""
    names: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return names
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


# ---------------------------------------------------------------------------
# D-01-1: File must not exist
# ---------------------------------------------------------------------------

class TestProductionHardeningDeleted:
    def test_file_does_not_exist(self):
        assert not PRODUCTION_HARDENING_PATH.exists(), (
            "production_hardening.py still present — D-01 incomplete"
        )

    def test_not_importable(self):
        """Importing the module must fail with ModuleNotFoundError."""
        try:
            import src.production_hardening  # noqa: F401
            assert False, "src.production_hardening should not be importable"
        except ModuleNotFoundError:
            pass

    def test_no_production_code_imports_it(self):
        """No .py file in src/ or web_app/ references production_hardening."""
        project_root = pathlib.Path(__file__).parent.parent
        bad_files: list[str] = []
        for root_dir in (project_root / "src", project_root / "web_app"):
            if not root_dir.exists():
                continue
            for py_file in _iter_py_files(root_dir):
                source = py_file.read_text(encoding="utf-8", errors="replace")
                imports = _extract_imports(source)
                for imp in imports:
                    if "production_hardening" in imp:
                        bad_files.append(str(py_file))
                        break
                # Also catch string references that might slip past the AST
                if "production_hardening" in source and str(py_file) not in bad_files:
                    # Allow comments but flag actual import statements
                    for line in source.splitlines():
                        stripped = line.strip()
                        if stripped.startswith("#"):
                            continue
                        if "production_hardening" in stripped:
                            bad_files.append(str(py_file))
                            break
        assert not bad_files, (
            f"production_hardening still imported by: {bad_files}"
        )


# ---------------------------------------------------------------------------
# D-01-2: Dead rate-limiter shim — SecurityHardeningManager.enforce_rate_limit
# is gone; the unified rate limiter is the only active limiter
# ---------------------------------------------------------------------------

class TestDeadRateLimiterShimGone:
    def test_security_hardening_manager_not_importable(self):
        """SecurityHardeningManager from production_hardening must not be importable."""
        try:
            from src.production_hardening import SecurityHardeningManager  # noqa: F401
            assert False, "SecurityHardeningManager should not be importable"
        except (ModuleNotFoundError, ImportError):
            pass

    def test_unified_rate_limiter_importable(self):
        """Replacement: UnifiedRateLimiter must be importable from src.rate_limiter."""
        from src.rate_limiter import UnifiedRateLimiter  # noqa: F401
        assert UnifiedRateLimiter is not None


# ---------------------------------------------------------------------------
# D-01-3: No grep-detectable import of dead security classes in src/
# ---------------------------------------------------------------------------

class TestNoDeadSecurityImports:
    def test_no_security_hardening_manager_import_in_src(self):
        """src/ code must not import SecurityHardeningManager."""
        project_root = pathlib.Path(__file__).parent.parent
        bad: list[str] = []
        for py_file in _iter_py_files(project_root / "src"):
            source = py_file.read_text(encoding="utf-8", errors="replace")
            if "SecurityHardeningManager" in source:
                bad.append(str(py_file))
        assert not bad, f"SecurityHardeningManager still referenced in: {bad}"

    def test_no_security_hardening_manager_import_in_web_app(self):
        """web_app/ code must not import SecurityHardeningManager."""
        project_root = pathlib.Path(__file__).parent.parent
        bad: list[str] = []
        for py_file in _iter_py_files(project_root / "web_app"):
            source = py_file.read_text(encoding="utf-8", errors="replace")
            if "SecurityHardeningManager" in source:
                bad.append(str(py_file))
        assert not bad, f"SecurityHardeningManager still referenced in: {bad}"
