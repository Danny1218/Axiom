"""SIMT padding: finished rows must not drift in LiquidKAN when batched with longer rows."""

import torch

from compiler.flow import wire_execution_graph
from compiler.ir import ast_to_ir
from compiler.parser import parse_ax, reset_parser
from engine.supernet import LatentSupernet


def test_batched_row0_matches_single_when_copadded_with_longer_row():
    reset_parser()
    ax = """
while (x > 0) {
  x = x - 1;
}
"""
    ir = ast_to_ir(parse_ax(ax))
    torch.manual_seed(42)
    sn = LatentSupernet(4, ("e0", "e1"), rank=2)
    g = wire_execution_graph(ir, sn, [], loop_max_unroll=16, loop_num_basis=4)

    h_single = torch.zeros(1, 4)
    h_single[0, 0] = 2.0

    h_batch = torch.zeros(2, 4)
    h_batch[0, 0] = 2.0
    h_batch[1, 0] = 10.0

    out_single = g(h_single)
    out_batch = g(h_batch)

    assert torch.allclose(out_single[0], out_batch[0], atol=1e-5, rtol=1e-5)


def test_run_loop_snapshots_mask_shape_matches_timesteps():
    from engine.interpreter import make_seed_map, run_loop_snapshots

    h = torch.zeros(3, 5)
    h[:, 0] = torch.tensor([2.0, 5.0, 1.0])
    cond = [("OP_LOAD", "x"), ("OP_CONST", 0.0), ("OP_CMP_GT",)]
    body = [("OP_ASSIGN", "x", [("OP_LOAD", "x"), ("OP_CONST", 1.0), ("OP_SUB",)])]
    seed = make_seed_map(cond, body, 5)
    seq, m = run_loop_snapshots(h, cond, body, dim=5, max_unroll=12, seed_map=seed)
    assert m.shape == (3, seq.shape[1])
    assert m.dtype == torch.bool
