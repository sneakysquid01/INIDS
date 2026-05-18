"""E-02: CSP script-src must not contain 'unsafe-inline'."""
import ast
import re
from pathlib import Path

import pytest


def _csp_header_value() -> str:
    src = Path("src/middleware.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "Content-Security-Policy" in node.value or "script-src" in node.value:
                if "script-src" in node.value:
                    return node.value
    raise AssertionError("Could not find CSP string in src/middleware.py")


def _script_src_directive(csp: str) -> str:
    m = re.search(r"script-src\s+([^;]+)", csp)
    assert m, f"No script-src directive found in: {csp!r}"
    return m.group(1)


class TestCSPNoUnsafeInline:
    def test_csp_string_found(self):
        val = _csp_header_value()
        assert "script-src" in val

    def test_no_unsafe_inline_in_script_src(self):
        csp = _csp_header_value()
        directive = _script_src_directive(csp)
        assert "'unsafe-inline'" not in directive, (
            f"'unsafe-inline' must not appear in script-src, got: {directive!r}"
        )

    def test_unsafe_inline_not_anywhere_in_script_src(self):
        csp = _csp_header_value()
        directive = _script_src_directive(csp)
        assert "unsafe-inline" not in directive

    def test_self_still_present_in_script_src(self):
        csp = _csp_header_value()
        directive = _script_src_directive(csp)
        assert "'self'" in directive

    def test_cdn_allowlist_preserved(self):
        csp = _csp_header_value()
        directive = _script_src_directive(csp)
        assert "cdn.jsdelivr.net" in directive
        assert "cdn.socket.io" in directive
        assert "cdn.tailwindcss.com" in directive

    def test_style_src_no_unsafe_inline(self):
        """FIX-008: 'unsafe-inline' must NOT appear in style-src."""
        csp = _csp_header_value()
        m = re.search(r"style-src\s+([^;]+)", csp)
        assert m, "No style-src directive found"
        assert "'unsafe-inline'" not in m.group(1), (
            "FIX-008: 'unsafe-inline' must be removed from style-src"
        )

    def test_no_inline_scripts_in_base_html(self):
        src = Path("web_app/templates/base.html").read_text(encoding="utf-8")
        # Find all <script> tags that are not src= references
        inline_blocks = re.findall(
            r"<script(?![^>]*\bsrc\s*=)[^>]*>(.+?)</script>",
            src,
            flags=re.DOTALL,
        )
        # Filter out empty / whitespace-only blocks
        non_empty = [b for b in inline_blocks if b.strip()]
        assert non_empty == [], (
            f"base.html still has {len(non_empty)} inline <script> block(s): "
            + str([b[:80] for b in non_empty])
        )

    def test_no_inline_scripts_in_predict_html(self):
        src = Path("web_app/templates/predict.html").read_text(encoding="utf-8")
        blocks = re.findall(
            r"<script(?![^>]*\bsrc\s*=)[^>]*>(.+?)</script>", src, flags=re.DOTALL
        )
        non_empty = [b for b in blocks if b.strip()]
        assert non_empty == [], (
            f"predict.html still has {len(non_empty)} inline script(s)"
        )

    def test_no_inline_scripts_in_realtime_html(self):
        src = Path("web_app/templates/realtime.html").read_text(encoding="utf-8")
        blocks = re.findall(
            r"<script(?![^>]*\bsrc\s*=)[^>]*>(.+?)</script>", src, flags=re.DOTALL
        )
        non_empty = [b for b in blocks if b.strip()]
        assert non_empty == [], (
            f"realtime.html still has {len(non_empty)} inline script(s)"
        )

    def test_middleware_csp_applied_via_after_request(self):
        """SecurityHeadersMiddleware.HEADERS dict must contain the CSP key."""
        src = Path("src/middleware.py").read_text(encoding="utf-8")
        assert "Content-Security-Policy" in src
        assert "SecurityHeadersMiddleware" in src
