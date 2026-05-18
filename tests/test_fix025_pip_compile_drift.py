"""
FIX-025 regression: pip-compile drift check wired into CI.
"""
from pathlib import Path

WORKFLOW = (
    Path(__file__).parent.parent / ".github" / "workflows" / "security.yml"
).read_text(encoding="utf-8")

REQ_IN = Path(__file__).parent.parent / "requirements.in"
REQ_TXT = Path(__file__).parent.parent / "requirements.txt"


def test_pip_tools_install_step_in_ci():
    assert "pip-tools" in WORKFLOW


def test_pip_compile_step_in_ci():
    assert "pip-compile" in WORKFLOW


def test_diff_check_in_ci():
    assert "diff" in WORKFLOW and "requirements.lock.check" in WORKFLOW


def test_requirements_in_exists():
    assert REQ_IN.exists(), "requirements.in must exist as the pip-compile source"


def test_requirements_in_has_flask():
    assert "Flask" in REQ_IN.read_text() or "flask" in REQ_IN.read_text()


def test_requirements_in_has_flask_compress():
    assert "flask-compress" in REQ_IN.read_text()


def test_requirements_txt_is_hash_pinned():
    content = REQ_TXT.read_text()
    assert "--hash=sha256:" in content


def test_pip_compile_drift_job_exists():
    assert "pip-compile-drift" in WORKFLOW
