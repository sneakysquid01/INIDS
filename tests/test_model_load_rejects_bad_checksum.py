"""Regression tests for A-06: Model integrity verification.

Verifies that:
- load_model_with_verification raises SecurityError on checksum mismatch (strict)
- warn mode logs and continues on mismatch
- disabled mode skips the check entirely
- AnomalyEngine propagates SecurityError from strict mode
- Correct checksum allows model to load
- load_checksum_manifest parses the manifest file correctly
"""
from __future__ import annotations

import hashlib
import importlib
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_temp_pkl(directory: Path, name: str = "model.pkl") -> Path:
    """Write a minimal joblib-serializable file and return its path."""
    import joblib
    path = directory / name
    joblib.dump({"dummy": True}, path)
    return path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# load_model_with_verification — strict mode (default)
# ---------------------------------------------------------------------------

class TestStrictMode:
    def test_correct_checksum_loads_successfully(self, tmp_path):
        from src.detection.ml_utils import load_model_with_verification

        pkl = _write_temp_pkl(tmp_path)
        good_hash = _sha256(pkl)

        with mock.patch.dict(os.environ, {"INIDS_MODEL_VERIFY": "strict"}):
            import src.detection.ml_utils as ml_utils
            ml_utils._VERIFY_MODE = None  # reset cached mode
            result = load_model_with_verification(pkl, good_hash)

        assert result == {"dummy": True}

    def test_bad_checksum_raises_security_error(self, tmp_path):
        from src.detection.ml_utils import load_model_with_verification, SecurityError

        pkl = _write_temp_pkl(tmp_path)
        wrong_hash = "a" * 64  # all-'a' hex digest — will never match real file

        with mock.patch.dict(os.environ, {"INIDS_MODEL_VERIFY": "strict"}):
            import src.detection.ml_utils as ml_utils
            ml_utils._VERIFY_MODE = None
            with pytest.raises(SecurityError, match="Model integrity check FAILED"):
                load_model_with_verification(pkl, wrong_hash)

    def test_security_error_is_not_swallowed(self, tmp_path):
        """SecurityError must not be a generic Exception subclass that callers ignore."""
        from src.detection.ml_utils import SecurityError

        assert issubclass(SecurityError, RuntimeError)


# ---------------------------------------------------------------------------
# Phase F: warn mode removed — now treated as strict
# ---------------------------------------------------------------------------

class TestWarnModeRemoved:
    def test_warn_mode_raises_security_error(self, tmp_path):
        """Phase F: warn mode removed; unknown modes default to strict behaviour."""
        from src.detection.ml_utils import load_model_with_verification, SecurityError

        pkl = _write_temp_pkl(tmp_path)
        wrong_hash = "b" * 64

        with mock.patch.dict(os.environ, {"INIDS_MODEL_VERIFY": "warn"}):
            import src.detection.ml_utils as ml_utils
            ml_utils._VERIFY_MODE = None
            with pytest.raises(SecurityError):
                load_model_with_verification(pkl, wrong_hash)

    def test_unknown_mode_raises_security_error(self, tmp_path):
        """Any unrecognised INIDS_MODEL_VERIFY value falls through to strict."""
        from src.detection.ml_utils import load_model_with_verification, SecurityError

        pkl = _write_temp_pkl(tmp_path)
        wrong_hash = "c" * 64

        with mock.patch.dict(os.environ, {"INIDS_MODEL_VERIFY": "banana"}):
            import src.detection.ml_utils as ml_utils
            ml_utils._VERIFY_MODE = None
            with pytest.raises(SecurityError):
                load_model_with_verification(pkl, wrong_hash)


# ---------------------------------------------------------------------------
# load_model_with_verification — disabled mode
# ---------------------------------------------------------------------------

class TestDisabledMode:
    def test_disabled_mode_skips_check(self, tmp_path):
        from src.detection.ml_utils import load_model_with_verification

        pkl = _write_temp_pkl(tmp_path)
        wrong_hash = "d" * 64  # definitely wrong

        with mock.patch.dict(os.environ, {"INIDS_MODEL_VERIFY": "disabled"}):
            import src.detection.ml_utils as ml_utils
            ml_utils._VERIFY_MODE = None
            # Must load without error even with wrong hash
            result = load_model_with_verification(pkl, wrong_hash)

        assert result == {"dummy": True}


# ---------------------------------------------------------------------------
# load_checksum_manifest
# ---------------------------------------------------------------------------

class TestChecksumManifest:
    def test_parses_manifest_correctly(self, tmp_path):
        from src.detection.ml_utils import load_checksum_manifest

        manifest = tmp_path / "checksums.sha256"
        manifest.write_text(
            "aabbcc112233445566778899aabbcc112233445566778899aabbcc112233445566  model.pkl\n"
            "# comment line\n"
            "ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00  other.pkl\n",
            encoding="utf-8",
        )
        result = load_checksum_manifest(tmp_path)
        assert result["model.pkl"] == "aabbcc112233445566778899aabbcc112233445566778899aabbcc112233445566"
        assert result["other.pkl"] == "ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00"
        assert len(result) == 2

    def test_missing_manifest_raises_file_not_found(self, tmp_path):
        from src.detection.ml_utils import load_checksum_manifest

        with pytest.raises(FileNotFoundError, match="checksums.sha256"):
            load_checksum_manifest(tmp_path)


# ---------------------------------------------------------------------------
# AnomalyEngine — integration
# ---------------------------------------------------------------------------

class TestAnomalyEngineVerification:
    def test_anomaly_engine_propagates_security_error(self, tmp_path):
        """AnomalyEngine must not swallow SecurityError from strict mode."""
        from src.detection.ml_utils import SecurityError

        pkl = _write_temp_pkl(tmp_path, "anomaly_model.pkl")
        wrong_hash = _sha256(pkl)[:32] + "00" * 16  # corrupted hash

        manifest = tmp_path / "checksums.sha256"
        manifest.write_text(f"{wrong_hash}  anomaly_model.pkl\n", encoding="utf-8")

        with mock.patch.dict(os.environ, {"INIDS_MODEL_VERIFY": "strict"}):
            import src.detection.ml_utils as ml_utils
            ml_utils._VERIFY_MODE = None
            with pytest.raises(SecurityError):
                from src.detection.engines.anomaly_engine import AnomalyEngine
                AnomalyEngine(model_path=pkl)

    def test_anomaly_engine_loads_with_correct_checksum(self, tmp_path):
        """AnomalyEngine loads cleanly when checksum matches."""
        pkl = _write_temp_pkl(tmp_path, "anomaly_model.pkl")
        good_hash = _sha256(pkl)

        manifest = tmp_path / "checksums.sha256"
        manifest.write_text(f"{good_hash}  anomaly_model.pkl\n", encoding="utf-8")

        with mock.patch.dict(os.environ, {"INIDS_MODEL_VERIFY": "strict"}):
            import src.detection.ml_utils as ml_utils
            ml_utils._VERIFY_MODE = None
            from src.detection.engines.anomaly_engine import AnomalyEngine
            engine = AnomalyEngine(model_path=pkl)

        # Model was set (loaded object is a dict in this test, not a real model)
        assert engine._get_model() is not None

    def test_raw_joblib_load_not_called_directly(self):
        """anomaly_engine.py must not call joblib.load() directly in __init__."""
        import ast
        engine_path = Path(__file__).parent.parent / "src" / "detection" / "engines" / "anomaly_engine.py"
        source = engine_path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        # Walk the __init__ method body
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    if isinstance(func, ast.Attribute) and func.attr == "load":
                        if isinstance(func.value, ast.Name) and func.value.id == "joblib":
                            pytest.fail(
                                "AnomalyEngine.__init__ calls joblib.load() directly; "
                                "use load_model_with_verification() instead"
                            )
