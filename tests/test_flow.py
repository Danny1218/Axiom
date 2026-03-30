import torch

from axiom.compiler.flow import wire_execution_graph
from axiom.compiler.ir import ast_to_ir, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.engine.supernet import LatentSupernet


def test_wire_execution_graph_matches_build_and_has_abi():
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { a = 1; } else { a = 2; }"))
    assert ir[0][0] == "OP_CONDITIONAL"
    sn = LatentSupernet(6, ("then_ex", "else_ex"), rank=2)
    sn.set_masks({"then_ex": 1.0, "else_ex": 1.0})
    g = wire_execution_graph(ir, sn, [("then_ex", "else_ex")])
    assert g.abi == extract_global_abi(ir, max_vars=6)
    x = torch.randn(4, 6, requires_grad=True)
    y, _, _ = g(x)
    y.sum().backward()
    assert x.grad is not None
