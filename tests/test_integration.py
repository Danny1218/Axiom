from pathlib import Path

from axiom.compiler.ir import ast_to_ir, ir_to_digraph
from axiom.compiler.parser import parse_ax_file, reset_parser

ROOT = Path(__file__).resolve().parents[1]


def test_test_ax_parses_and_ir_matches_expectations():
    reset_parser()
    path = ROOT / "test.ax"
    tree = parse_ax_file(path)
    ir = ast_to_ir(tree)
    assert [x[0] for x in ir] == ["OP_ASSIGN", "OP_ASSIGN", "OP_CONDITIONAL", "OP_EXPR_STMT"]
    cond = ir[2][1]
    assert cond[-1] == ("OP_CMP_GT",)
    G = ir_to_digraph(ir)
    assert G.number_of_nodes() == 4
