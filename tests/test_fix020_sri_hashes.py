"""
FIX-020 regression: SRI integrity hashes on CDN assets in base.html.
"""
import re
from pathlib import Path

BASE_HTML = Path(__file__).parent.parent / "web_app" / "templates" / "base.html"


def _html():
    return BASE_HTML.read_text(encoding="utf-8")


def test_chartjs_has_integrity():
    assert 'cdn.jsdelivr.net/npm/chart.js' in _html()
    assert 'integrity="sha384-' in _html()


def test_chartjs_integrity_exact():
    html = _html()
    assert 'sha384-9MhbyIRcBVQiiC7FSd7T38oJNj2Zh+EfxS7/vjhBi4OOT78NlHSnzM31EZRWR1LZ' in html


def test_socketio_has_integrity():
    html = _html()
    assert 'cdn.socket.io/4.5.4/socket.io.min.js' in html
    assert 'sha384-/KNQL8Nu5gCHLqwqfQjA689Hhoqgi2S84SNUxC3roTe4EhJ9AfLkp8QiQcU8AMzI' in html


def test_bootstrap_icons_has_integrity():
    html = _html()
    assert 'bootstrap-icons@1.11.0' in html
    assert 'sha384-QuGBSgV5Im3DzL2z+8Ko9/hqNy/N0O7zwvXAtfd1MvPKWa/UbeLV65cfm4BV5Wgq' in html


def test_sri_tags_have_crossorigin():
    html = _html()
    assert html.count('integrity="sha384-') >= 3, "Expected at least 3 SRI-hashed resources"
    # Each SRI hash must have crossorigin somewhere in the same tag block.
    # Tags may span multiple lines, so find each <script/link … > element.
    tags = re.findall(r'<(?:script|link)[^>]*integrity="sha384-[^"]*"[^>]*>', html, re.DOTALL)
    assert len(tags) >= 3, f"Could not find ≥3 SRI-hashed tags; found {len(tags)}"
    for tag in tags:
        assert 'crossorigin="anonymous"' in tag, \
            f"Missing crossorigin on SRI tag: {tag[:120]}"


def test_tailwind_cdn_comment_present():
    html = _html()
    assert 'dynamic JIT' in html or 'cdn.tailwindcss.com' in html


def test_no_sri_on_local_static():
    html = _html()
    # Local static files should NOT have integrity attributes
    local_lines = [ln for ln in html.splitlines() if "url_for('static'" in ln]
    for line in local_lines:
        assert 'integrity=' not in line, f"Unexpected integrity on local file: {line.strip()}"


def test_check_cdn_integrity_script_exists():
    script = Path(__file__).parent.parent / "scripts" / "check_cdn_integrity.py"
    assert script.exists(), "scripts/check_cdn_integrity.py must exist for CI"
