import subprocess
import sys


def test_train_cli_dry_run_baseline():
    result = subprocess.run(
        [sys.executable, "-m", "src.train_cli", "--suite", "baseline", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Dry run enabled" in result.stderr or "Dry run enabled" in result.stdout


def test_train_cli_dry_run_full_with_multiclass_flag():
    result = subprocess.run(
        [sys.executable, "-m", "src.train_cli", "--suite", "full", "--include-multiclass", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    output = f"{result.stdout}\n{result.stderr}"
    assert "Selected binary models" in output
