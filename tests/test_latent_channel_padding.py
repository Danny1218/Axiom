"""Latent channel padding: IR snapshot width must match trunk / KAN dim (Phase 15)."""

import torch

from axiom.compiler.flow import wire_execution_graph
from axiom.compiler.ir import ast_to_ir, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.engine.interpreter import run_loop_snapshots
from axiom.engine.supernet import LatentSupernet
from axiom.engine.topology import build_execution_graph_from_ir


def test_interpreted_loop_forward_supernet_wider_than_script_vars():
    reset_parser()
    ir = ast_to_ir(parse_ax("while (x > 0) { x = x - 1; y = y * 2; }"))
    sn = LatentSupernet(8, ("e1", "e2"), rank=2)
    sn.set_masks({"e1": 1.0, "e2": 1.0})
    g = wire_execution_graph(ir, sn, [], loop_max_unroll=4, loop_num_basis=4)
    x = torch.randn(3, 8, dtype=torch.float32)
    out, _, _ = g(x)
    assert out.shape == (3, 8)


def test_run_loop_snapshots_trunk_dim_pads_narrow_feature_axis():
    """When ``dim`` (snapshot layout width) is smaller than ``trunk_dim``, pad last axis."""
    reset_parser()
    ir = ast_to_ir(parse_ax("while (x > 0) { x = x - 1; y = y * 2; }"))
    loop = ir[0]
    cond, body = loop[1], loop[2]
    abi = extract_global_abi(ir, max_vars=8)
    seed = {k: abi[k] for k in ("x", "y") if k in abi}
    B, trunk = 2, 8
    h = torch.randn(B, trunk, dtype=torch.float32)
    seq, m = run_loop_snapshots(
        h,
        cond,
        body,
        dim=2,
        max_unroll=3,
        seed_map=seed,
        trunk_dim=trunk,
    )
    assert seq.shape == (B, 3, trunk)
    assert m.shape == (B, 3)


def test_build_graph_abi_fewer_names_than_dim():
    """ABI may list only script vars; graph dim still full trunk."""
    reset_parser()
    ir = ast_to_ir(parse_ax("while (x > 0) { x = x - 1; y = y * 2; }"))
    abi = extract_global_abi(ir, max_vars=8)
    sn = LatentSupernet(8, ("e1", "e2"), rank=2)
    g = build_execution_graph_from_ir(ir, sn, [], global_abi=abi)
    assert len(g.abi) <= 8
    x = torch.randn(1, 8)
    y, _, _ = g(x)
    assert y.shape == (1, 8)
