"""
FIX-011 regression: CI coverage gate expanded to ops_store, middleware, web_app.
"""
from pathlib import Path

WORKFLOW = (
    Path(__file__).parent.parent / ".github" / "workflows" / "security.yml"
).read_text(encoding="utf-8")

PYPROJECT = (
    Path(__file__).parent.parent / "pyproject.toml"
).read_text(encoding="utf-8")


def test_workflow_covers_ops_store():
    assert "src/ops_store" in WORKFLOW


def test_workflow_covers_middleware():
    assert "src/middleware" in WORKFLOW


def test_workflow_covers_web_app():
    assert "web_app" in WORKFLOW


def test_workflow_still_covers_auth():
    assert "src/auth" in WORKFLOW


def test_workflow_coverage_report_xml():
    assert "cov-report=xml" in WORKFLOW or "--cov-report=xml" in WORKFLOW


def test_pyproject_coverage_sources_expanded():
    assert "src/ops_store" in PYPROJECT
    assert "src/middleware" in PYPROJECT
    assert "web_app" in PYPROJECT


def test_pyproject_fail_under_50():
    assert "fail_under = 50" in PYPROJECT
