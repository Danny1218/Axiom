from pathlib import Path

import torch

from compiler.flow import wire_execution_graph
from compiler.ir import ast_to_ir
from compiler.parser import parse_ax_file, reset_parser
from engine.supernet import LatentSupernet

ROOT = Path(__file__).resolve().parents[1]


def test_test_ax_topology_forward_backward():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(ROOT / "test.ax"))
    conds = sum(1 for op in ir if op[0] == "OP_CONDITIONAL")
    assert conds == 1
    dim = 8
    sn = LatentSupernet(dim, ("branch_then", "branch_else"), rank=3)
    sn.set_masks({"branch_then": 1.0, "branch_else": 1.0})
    g = wire_execution_graph(ir, sn, [("branch_then", "branch_else")])
    assert any(g.node_kind(n) == "conditional" for n in g.topo_names)
    x = torch.randn(5, dim, requires_grad=True)
    y, _ = g(x)
    assert y.shape == (5, dim)
    y.pow(2).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
