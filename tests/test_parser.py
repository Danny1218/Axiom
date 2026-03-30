import pytest
from lark import UnexpectedCharacters, UnexpectedToken
from lark.tree import Tree

from axiom.compiler.ir import ast_to_ir, parse_program
from axiom.compiler.parser import parse_ax, parse_ax_file, parse_ax_program, reset_parser


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
    if outer.data == "postfix_expr" and len(outer.children) == 1:
        outer = outer.children[0]
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


def test_parse_array_literal_and_index():
    t = parse_ax("a = [1.0, 2.0]; b = a[1];")
    kinds = [c.data for c in t.children if hasattr(c, "data")]
    assert kinds.count("assign_stmt") == 2


def test_parse_function_def_return_and_call():
    src = "def add(a, b) { return a + b; } x = add(1, 2);"
    t = parse_ax(src)
    assert t.children[0].data == "function_def"
    assert t.children[1].data == "assign_stmt"
    funcs, main = parse_program(t)
    assert "add" in funcs and funcs["add"].params == ("a", "b")
    assert main[0][0] == "OP_ASSIGN" and main[0][1] == "x"
    ir = ast_to_ir(t)
    assert ir[-1][0] == "OP_ASSIGN" and ir[-1][1] == "x"
    assert ir[-1][2] == [("OP_LOAD", "_inline_add_0_ret")]


def test_parse_ax_program_helper():
    funcs, main = parse_ax_program("def z() { return 1; } k = 1;")
    assert "z" in funcs and funcs["z"].params == ()
    assert len(main) == 1 and main[0][0] == "OP_ASSIGN"


def test_parse_builtin_sum_mean_dot_calls():
    t = parse_ax("x = [1.0, 2.0, 3.0]; y = sum(x); z = dot(x, [2.0, 2.0, 2.0]); w = mean(x);")
    ir = ast_to_ir(t)
    assert ("OP_REDUCE_SUM",) in ir[1][2]
    assert ("OP_DOT",) in ir[2][2]
    assert ("OP_REDUCE_MEAN",) in ir[3][2]
