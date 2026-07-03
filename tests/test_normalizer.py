"""Unit tests for ``axiom.compiler.normalizer`` rewrites."""

from __future__ import annotations

from axiom.compiler.normalizer import normalize_ax_source
from axiom.compiler import normalizer_rewrites as nr


def test_normalize_shorthand_and_comments():
    src = "score += bonus; // note\nratio /= total;"
    out, meta = normalize_ax_source(src)
    assert out == "score = score + bonus;\nratio = ratio / total;"
    assert meta["stripped_line_comments"] is True
    assert meta["normalized_shorthand_assignment"] is True


def test_normalize_three_arg_extrema():
    src = "capped = max(a, b, c);\nfloored = min(low, mid, high);"
    out, meta = normalize_ax_source(src)
    assert out == "capped = max(max(a, b), c);\nfloored = min(min(low, mid), high);"
    assert meta["normalized_three_arg_max"] is True
    assert meta["normalized_three_arg_min"] is True


def test_normalize_clip_call():
    src = "bounded = clip(score, 0.0, 1.0);"
    out, meta = nr.rewrite_clip_calls(src)
    assert out == "bounded = max(0.0, min(score, 1.0));"
    assert meta["normalized_clip_call"] is True


def test_normalize_else_if():
    src = "if (x < 0.0) { y = -x; } else if (x < 1.0) { y = x; } else { y = 1.0; }"
    out, meta = nr.rewrite_else_if(src)
    assert "else if" not in out
    assert "else { if (" in out
    assert meta["normalized_else_if"] is True


def test_normalize_logical_and():
    src = "if (a > 0.0 && b > 0.0) { y = a + b; }"
    out, meta = nr.rewrite_logical_operators(src)
    assert "&&" not in out
    assert "if (a > 0.0)" in out
    assert "if (b > 0.0)" in out
    assert meta["normalized_logical_and"] is True


def test_normalize_logical_or():
    src = "if (a > 0.0 || b > 0.0) { y = 1.0; }"
    out, meta = nr.rewrite_logical_operators(src)
    assert "||" not in out
    assert "else {" in out
    assert meta["normalized_logical_or"] is True


def test_normalize_inline_ternary():
    src = "y = good if x > 0.0 else bad;"
    out, meta = nr.rewrite_inline_ternaries(src)
    assert "if (x > 0.0)" in out
    assert "y = good;" in out
    assert "y = bad;" in out
    assert meta["normalized_inline_ternary"] is True


def test_normalize_conservative_tokens():
    src = "x := 2.;\nflag == 1.0;"
    out, meta = normalize_ax_source(src)
    assert ":=" not in out
    assert "2.0" in out
    assert meta["normalized_colon_eq"] is True
    assert meta["normalized_trailing_dot_float"] is True
