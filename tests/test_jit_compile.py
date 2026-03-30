"""Phase 10: functional forward (out, shadows) is traceable; torch.compile matches eager."""

import torch

from compiler.flow import wire_execution_graph
from compiler.ir import ast_to_ir
from compiler.parser import parse_ax, reset_parser
from engine.supernet import LatentSupernet


def _assert_shadow_dicts_close(a: dict, b: dict, *, atol: float = 1e-5, rtol: float = 1e-5) -> None:
    assert set(a.keys()) == set(b.keys())
    for k in a:
        assert torch.allclose(a[k], b[k], atol=atol, rtol=rtol)


def test_compile_aot_eager_matches_eager_on_mixed_ir_graph():
    reset_parser()
    ax = """
x = 1;
if (x > 0) {
  y = 1;
}
i = 2;
while (i > 0) {
  i = i - 1;
}
"""
    ir = ast_to_ir(parse_ax(ax))
    torch.manual_seed(0)
    sn = LatentSupernet(5, ("then_0", "else_0", "latent_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")], loop_max_unroll=8, loop_num_basis=4)
    x = torch.randn(3, 5)

    out_e, sh_e = g(x)
    compiled = torch.compile(g, backend="aot_eager")
    out_j, sh_j = compiled(x)

    assert torch.allclose(out_e, out_j, atol=1e-5, rtol=1e-5)
    _assert_shadow_dicts_close(sh_e, sh_j)


def test_conditional_sinkhorn_returns_tuple_no_mutation():
    sn = LatentSupernet(4, ("t", "e"), rank=2)
    sn.set_masks({"t": 1.0, "e": 1.0})
    from engine.topology import ConditionalSinkhornBlock

    blk = ConditionalSinkhornBlock(sn, "t", "e", num_iters=4)
    h = torch.randn(2, 4)
    o, shadows = blk(h)
    assert o.shape == h.shape
    assert isinstance(shadows, dict)
