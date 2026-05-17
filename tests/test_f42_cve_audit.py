"""Step 42 (Phase F) — CVE audit security regression tests.

Validates:
1. Security-critical packages meet minimum safe versions (from Phase F PLAN)
2. pip-audit is available for CI use
3. requirements.txt uses hash-pinned entries (--require-hashes compatible)
4. CI workflow exists and includes pip-audit step
"""
from __future__ import annotations

import importlib.metadata
import pathlib
import subprocess
import sys

import pytest
from packaging.version import Version

ROOT = pathlib.Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Minimum safe versions specified in Phase F PLAN + CVE fixes discovered
# ---------------------------------------------------------------------------

MINIMUM_VERSIONS = {
    "cryptography": "42.0.8",
    "numpy": "1.26.4",
    "Werkzeug": "3.0.6",
}


@pytest.mark.parametrize("package,minimum", list(MINIMUM_VERSIONS.items()))
def test_package_meets_minimum_version(package, minimum):
    """Critical packages must meet Phase F minimum safe versions."""
    try:
        installed = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        pytest.skip(f"{package} not installed in this environment")
    assert Version(installed) >= Version(minimum), (
        f"{package} {installed} < minimum {minimum} — upgrade required (CVE risk)"
    )


# ---------------------------------------------------------------------------
# pip-audit available
# ---------------------------------------------------------------------------


def test_pip_audit_installed():
    """pip-audit must be importable — required for CI dependency scan."""
    result = subprocess.run(
        [sys.executable, "-m", "pip_audit", "--version"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        "pip-audit not installed — CI vulnerability scan will fail. "
        "Run: pip install pip-audit"
    )


# ---------------------------------------------------------------------------
# requirements.txt uses hash pinning
# ---------------------------------------------------------------------------


def test_requirements_txt_has_hashes():
    """requirements.txt must contain --hash= entries (A-07 / Step 42 constraint)."""
    req_file = ROOT / "requirements.txt"
    assert req_file.exists(), "requirements.txt not found"
    content = req_file.read_text(encoding="utf-8")
    assert "--hash=sha256:" in content, (
        "requirements.txt does not contain sha256 hashes — "
        "hash-pinning was required by A-07 and must be preserved"
    )


# ---------------------------------------------------------------------------
# CI workflow includes pip-audit
# ---------------------------------------------------------------------------


def test_ci_workflow_includes_pip_audit():
    """security.yml CI workflow must include a pip-audit step."""
    workflow = ROOT / ".github" / "workflows" / "security.yml"
    assert workflow.exists(), ".github/workflows/security.yml not found"
    content = workflow.read_text(encoding="utf-8")
    assert "pip-audit" in content, (
        "security.yml does not reference pip-audit — "
        "Step 42 requires pip-audit in CI to catch newly published CVEs"
    )
    assert "schedule" in content, (
        "security.yml must run on a schedule to catch newly published CVEs"
    )
