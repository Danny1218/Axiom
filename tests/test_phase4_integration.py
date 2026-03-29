from pathlib import Path

import torch

from compiler.flow import wire_execution_graph
from compiler.ir import ast_to_ir
from compiler.parser import parse_ax_file, reset_parser
from engine.supernet import LatentSupernet

ROOT = Path(__file__).resolve().parents[1]


def test_loop_ax_file_and_graph_forward():
    reset_parser()
    path = ROOT / "loop.ax"
    ir = ast_to_ir(parse_ax_file(path))
    assert any(op[0] == "OP_LOOP" for op in ir)
    sn = LatentSupernet(7, ("p", "q"), rank=2)
    sn.set_masks({"p": 1.0, "q": 1.0})
    g = wire_execution_graph(ir, sn, [], loop_max_unroll=4, loop_num_basis=6)
    x = torch.randn(3, 7, requires_grad=True)
    y = g(x)
    assert y.shape == (3, 7)
    y.pow(2).sum().backward()
    assert x.grad is not None
