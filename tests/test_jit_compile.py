"""Phase 10/11: functional forward + signals; torch.compile matches eager (fullgraph)."""

import torch
import torch._dynamo.config as dynamo_config

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

    dynamo_config.capture_dynamic_output_shape_ops = True
    out_e, sh_e, sig_e = g(x)
    compiled = torch.compile(g, backend="aot_eager", fullgraph=True)
    out_j, sh_j, sig_j = compiled(x)

    assert torch.allclose(out_e, out_j, atol=1e-5, rtol=1e-5)
    _assert_shadow_dicts_close(sh_e, sh_j)
    assert set(sig_e.keys()) == set(sig_j.keys())
    for k in sig_e:
        assert torch.allclose(sig_e[k], sig_j[k], atol=1e-5, rtol=1e-5)


def test_compile_aot_eager_fullgraph_conditional_only_matches_eager():
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { a = 1; } else { a = 2; }"))
    torch.manual_seed(1)
    sn = LatentSupernet(5, ("then_0", "else_0", "latent_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")])
    x = torch.randn(2, 5)
    dynamo_config.capture_dynamic_output_shape_ops = True
    out_e, sh_e, sig_e = g(x)
    compiled = torch.compile(g, backend="aot_eager", fullgraph=True)
    out_j, sh_j, sig_j = compiled(x)
    assert torch.allclose(out_e, out_j, atol=1e-5, rtol=1e-5)
    _assert_shadow_dicts_close(sh_e, sh_j)
    for k in sig_e:
        assert torch.allclose(sig_e[k], sig_j[k], atol=1e-5, rtol=1e-5)


def test_conditional_sinkhorn_returns_tuple_no_mutation():
    sn = LatentSupernet(4, ("t", "e"), rank=2)
    sn.set_masks({"t": 1.0, "e": 1.0})
    from engine.topology import ConditionalSinkhornBlock

    blk = ConditionalSinkhornBlock(sn, "t", "e", num_iters=4)
    h = torch.randn(2, 4)
    o, shadows, sig = blk(h)
    assert o.shape == h.shape
    assert isinstance(shadows, dict)
    assert isinstance(sig, dict) and "cond" in sig
