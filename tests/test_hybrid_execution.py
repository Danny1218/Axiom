"""Phase 18: root-level IR runs in InterpretedBlock; conditionals blend symbolic branches + LoRA."""

import pytest
import torch
import torch.nn as nn

from compiler.flow import wire_execution_graph
from compiler.ir import ast_to_ir
from compiler.parser import parse_ax, reset_parser
from engine.block_executor import InterpretedBlock
from engine.supernet import LatentSupernet


def test_symbolic_assign_and_hybrid_conditional():
    reset_parser()
    src = """
a = 5;
if (a > 0) {
  b = 10;
} else {
  b = -10;
}
"""
    ir = ast_to_ir(parse_ax(src))
    dim = 2
    sn = LatentSupernet(dim, ("then_0", "else_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")], mutation_entropy_norm_threshold=0.99)
    for b in g.conditional_blocks():
        nn.init.zeros_(b.router.proj.weight)
        # Strong skew so Sinkhorn-balanced weights still favor then (symbolic b=10 vs -10).
        b.router.proj.bias.data = torch.tensor([24.0, -24.0], dtype=b.router.proj.bias.dtype)
    x = torch.zeros(1, dim)
    out, _, _ = g(x)
    ac, bc = g.abi["a"], g.abi["b"]
    assert out[0, ac].item() == pytest.approx(5.0, abs=1e-3)
    assert out[0, bc].item() > 0.0


def test_hybrid_graph_compiles_aot_eager():
    """Full-graph friendly path without MSVC/inductor (Windows CI)."""
    import torch._dynamo.config as dynamo_config

    dynamo_config.capture_dynamic_output_shape_ops = True
    reset_parser()
    ir = ast_to_ir(
        parse_ax(
            """
a = 1;
if (1 > 0) { b = 2; } else { b = 3; }
"""
        )
    )
    dim = 4
    sn = LatentSupernet(dim, ("then_0", "else_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")], mutation_entropy_norm_threshold=0.99)
    x = torch.randn(2, dim)
    fn = torch.compile(g, backend="aot_eager", fullgraph=True)
    out, _, _ = fn(x)
    assert out.shape == x.shape


def test_interpreted_block_then_else_ir_in_isolation():
    reset_parser()
    ir = ast_to_ir(
        parse_ax(
            """
if (1 > 0) { b = 10; } else { b = -10; }
"""
        )
    )
    cond = ir[0]
    _, _, then_ir, else_ir = cond
    abi = {"b": 0}
    h = torch.zeros(1, 2)
    t = InterpretedBlock(list(then_ir), abi, max_unroll=4)(h)
    e = InterpretedBlock(list(else_ir), abi, max_unroll=4)(h)
    assert t[0, 0].item() == pytest.approx(10.0)
    assert e[0, 0].item() == pytest.approx(-10.0)


def test_interpreted_stmt_sets_constant_without_trunk_seed():
    reset_parser()
    ir = ast_to_ir(parse_ax("k = 7;"))
    dim = 3
    sn = LatentSupernet(dim, ("p", "q"), rank=2)
    g = wire_execution_graph(ir, sn, [])
    x = torch.zeros(1, dim)
    out, _, _ = g(x)
    assert g.abi["k"] < dim
    assert out[0, g.abi["k"]].item() == 7.0
