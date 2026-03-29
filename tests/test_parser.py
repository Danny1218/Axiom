import pytest
from lark import UnexpectedCharacters, UnexpectedToken
from lark.tree import Tree

from compiler.parser import parse_ax, parse_ax_file, reset_parser


@pytest.fixture(autouse=True)
def _fresh_parser():
    reset_parser()
    yield
    reset_parser()


def test_parse_assign_and_arithmetic():
    t = parse_ax("a = 1 + 2 * 3;")
    assert t.data == "start"
    a = t.children[0]
    assert a.data == "assign_stmt"
    assert str(a.children[0]) == "a"


def test_parse_if_else_braces():
    t = parse_ax("if (1 < 2) { a = 1; } else { a = 2; }")
    iff = t.children[0]
    assert iff.data == "if_stmt"
    assert len(iff.children) == 3
    assert iff.children[2].data == "else_block"


def test_parse_if_without_else():
    t = parse_ax("if (0 == 0) { b = 1; }")
    assert len(t.children[0].children) == 2


def test_parse_comparisons_eq_ne():
    t = parse_ax("if (a == b) { c = 1; } if (a != b) { c = 2; }")
    assert len(t.children) == 2


def test_parse_expr_stmt():
    t = parse_ax("foo + bar;")
    assert t.children[0].data == "expr_stmt"


def test_parse_unary_minus():
    t = parse_ax("x = -5;")
    outer = t.children[0].children[1]
    assert outer.data == "atom"
    inner = outer.children[0]
    assert inner.data == "atom" and str(inner.children[0]) == "5"


def test_parse_parens_override_precedence():
    t = parse_ax("x = (1 + 2) * 3;")
    expr = t.children[0].children[1]
    assert expr.data == "product"


def test_parse_ax_file(tmp_path):
    p = tmp_path / "t.ax"
    p.write_text("k = 42;", encoding="utf-8")
    tree = parse_ax_file(p)
    assert isinstance(tree, Tree)


def test_parse_rejects_invalid_syntax():
    with pytest.raises((UnexpectedToken, UnexpectedCharacters)):
        parse_ax("if (1) {")  # unclosed / incomplete


def test_parse_while_loop():
    t = parse_ax("while (i > 0) { i = i - 1; }")
    assert t.children[0].data == "while_stmt"


def test_parse_while_body_multiple_stmts():
    t = parse_ax("while (1) { a = 1; b = 2; }")
    trees = [c for c in t.children[0].children if hasattr(c, "data")]
    inner = next(x for x in trees if x.data == "inner")
    assert len(inner.children) == 2
