"""Phase 8: gradients through IR interpreter → loop snapshots → KAN → trunk input."""

import torch

from compiler.flow import wire_execution_graph
from compiler.ir import ast_to_ir
from compiler.parser import parse_ax, reset_parser
from engine.supernet import LatentSupernet


def test_execution_graph_grad_through_loop_ir_to_trunk_input():
    reset_parser()
    ax = """
while (x > 0) {
  x = x - 1;
  y = y * 2;
}
"""
    ir = ast_to_ir(parse_ax(ax))
    sn = LatentSupernet(4, ("e0", "e1"), rank=2)
    g = wire_execution_graph(ir, sn, [], loop_max_unroll=8, loop_num_basis=4)
    x = torch.randn(2, 4, requires_grad=True)
    y, _ = g(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.count_nonzero(x.grad) > 0


def test_run_loop_snapshots_matches_float_semantics_countdown():
    from engine.interpreter import run_loop_snapshots, make_seed_map

    h = torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0])
    cond = [("OP_LOAD", "i"), ("OP_CONST", 0.0), ("OP_CMP_GT",)]
    body = [("OP_ASSIGN", "i", [("OP_LOAD", "i"), ("OP_CONST", 1.0), ("OP_SUB",)])]
    seed = make_seed_map(cond, body, 5)
    mat, _m = run_loop_snapshots(h, cond, body, dim=5, max_unroll=10, seed_map=seed)
    assert mat.shape == (1, 3, 5)
    assert mat[0, 0, 0].item() == 2.0 and mat[0, -1, 0].item() == 0.0


def test_snapshot_stack_preserves_device_dtype():
    from engine.interpreter import run_loop_snapshots, make_seed_map

    h = torch.tensor([2.0, 0.0, 0.0], device="cpu", dtype=torch.float64)
    cond = [("OP_LOAD", "i"), ("OP_CONST", 0.0), ("OP_CMP_GT",)]
    body = [("OP_ASSIGN", "i", [("OP_LOAD", "i"), ("OP_CONST", 1.0), ("OP_SUB",)])]
    seed = make_seed_map(cond, body, 3)
    mat, _m = run_loop_snapshots(h, cond, body, dim=3, max_unroll=5, seed_map=seed)
    assert mat.shape[0] == 1
    assert mat.device == h.device and mat.dtype == h.dtype
