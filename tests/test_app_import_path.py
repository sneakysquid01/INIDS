import runpy
from pathlib import Path


def test_web_app_script_imports_without_module_error():
    app_path = Path(__file__).resolve().parents[1] / "web_app" / "app.py"
    runpy.run_path(str(app_path), run_name="not_main")
