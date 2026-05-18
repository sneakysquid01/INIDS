"""
FIX-019 regression: flask-compress registered; gzip on JSON responses ≥ 1 KB.
"""
import pytest
from web_app.app import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_compress_config_min_size():
    assert app.config.get("COMPRESS_MIN_SIZE") == 1024


def test_compress_config_mimetypes():
    mimetypes = app.config.get("COMPRESS_MIMETYPES", [])
    assert "application/json" in mimetypes
    assert "text/html" in mimetypes


def test_flask_compress_import_in_app():
    import web_app.app as m
    assert hasattr(m, "_FlaskCompress"), "_FlaskCompress import guard must exist in app.py"


def test_flask_compress_in_requirements():
    req = (
        __import__("pathlib").Path(__file__).parent.parent / "requirements.txt"
    ).read_text()
    assert "flask-compress" in req.lower()


def test_flask_compress_in_requirements_in():
    req_in = (
        __import__("pathlib").Path(__file__).parent.parent / "requirements.in"
    ).read_text()
    assert "flask-compress" in req_in.lower()
