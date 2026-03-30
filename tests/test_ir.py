import networkx as nx

from axiom.compiler.ir import ast_to_ir, ir_to_digraph
from axiom.compiler.parser import parse_ax, reset_parser


def test_ir_assign_add_mul():
    reset_parser()
    ir = ast_to_ir(parse_ax("x = 1 + 2; y = x * 3;"))
    assert ir[0][0] == "OP_ASSIGN" and ir[0][1] == "x"
    assert ir[0][2] == [("OP_CONST", 1), ("OP_CONST", 2), ("OP_ADD",)]
    assert ir[1][2] == [("OP_LOAD", "x"), ("OP_CONST", 3), ("OP_MUL",)]


def test_ir_conditional_structure():
    reset_parser()
    ir = ast_to_ir(parse_ax("if (a > b) { x = 1; } else { x = 2; }"))
    op, cond, then_ir, else_ir = ir[0]
    assert op == "OP_CONDITIONAL"
    assert cond[-1] == ("OP_CMP_GT",)
    assert then_ir[0][0] == "OP_ASSIGN"
    assert else_ir[0][0] == "OP_ASSIGN"


def test_ir_cmp_ops():
    reset_parser()
    for src, last in [
        ("if (a < b) { c = 1; }", "OP_CMP_LT"),
        ("if (a == b) { c = 1; }", "OP_CMP_EQ"),
        ("if (a != b) { c = 1; }", "OP_CMP_NE"),
    ]:
        ir = ast_to_ir(parse_ax(src))
        assert ir[0][1][-1] == (last,)


def test_ir_expr_stmt():
    reset_parser()
    ir = ast_to_ir(parse_ax("1 + 2;"))
    assert ir[0][0] == "OP_EXPR_STMT"


def test_ir_sub_div_neg():
    reset_parser()
    ir = ast_to_ir(parse_ax("x = 4 - 2 / 1; y = -3;"))
    xexpr = ir[0][2]
    assert ("OP_SUB",) in xexpr and ("OP_DIV",) in xexpr
    yexpr = ir[1][2]
    assert yexpr[-1] == ("OP_NEG",)


def test_ir_to_digraph_chain():
    reset_parser()
    ir = ast_to_ir(parse_ax("a = 1;"))
    G = ir_to_digraph(ir)
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() == 1
    assert G.nodes[0]["op"] == "OP_ASSIGN"


def test_ir_nested_conditional_blocks():
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1) { a = 1; b = 2; }"))
    _, _, then_ir, else_ir = ir[0]
    assert else_ir == []
    assert len(then_ir) == 2


def test_ir_while_emits_op_loop():
    reset_parser()
    ir = ast_to_ir(parse_ax("while (n > 0) { n = n - 1; }"))
    assert len(ir) == 1 and ir[0][0] == "OP_LOOP"
    assert len(ir[0][2]) >= 1


def test_ir_builtin_reduction_opcodes():
    reset_parser()
    ir = ast_to_ir(parse_ax("a = sum([1.0, 2.0]); b = mean([1.0, 2.0]); c = dot([1.0, 2.0], [3.0, 4.0]);"))
    assert ir[0][2][-1] == ("OP_REDUCE_SUM",)
    assert ir[1][2][-1] == ("OP_REDUCE_MEAN",)
    assert ir[2][2][-1] == ("OP_DOT",)
