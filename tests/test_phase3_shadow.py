import torch

from axiom.compiler.flow import wire_execution_graph
from axiom.compiler.ir import ast_to_ir
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.engine.supernet import LatentSupernet
from axiom.engine.topology import ConditionalSinkhornBlock


def test_supernet_shadow_delta_detached_from_main_grad():
    sn = LatentSupernet(4, ("u",), rank=2)
    sn.set_adapter_mask("u", 1.0)
    sn.is_shadow[0] = True
    x = torch.randn(3, 4, requires_grad=True)
    y = sn(x)
    y.sum().backward()
    assert sn.adapters["u"].U.grad is None


def test_conditional_shadow_no_grad_through_main_but_local_ok():
    torch.manual_seed(0)
    sn = LatentSupernet(5, ("t", "e", "spare"), rank=2)
    sn.set_masks({"t": 1.0, "e": 1.0})
    sn.is_shadow[sn._name_to_idx["t"]] = True
    blk = ConditionalSinkhornBlock(sn, "t", "e", num_iters=20, mutation_entropy_norm_threshold=1.01)
    h1 = torch.randn(4, 5, requires_grad=True)
    out, _, _ = blk(h1)
    out.sum().backward()
    g_ut = sn.adapters["t"].U.grad
    assert g_ut is None or not g_ut.any()
    assert sn.adapters["e"].U.grad is not None

    sn.zero_grad(set_to_none=True)
    h2 = torch.randn(4, 5, requires_grad=True)
    _, shadows, _ = blk(h2)
    y_t = shadows["t"]
    (y_t.pow(2).mean()).backward()
    assert sn.adapters["t"].U.grad is not None


def test_execution_graph_forward_returns_shadow_dict():
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { a = 1; } else { a = 2; }"))
    sn = LatentSupernet(5, ("then_ex", "else_ex", "latent"), rank=2)
    sn.set_masks({"then_ex": 1.0, "else_ex": 1.0})
    sn.is_shadow[sn._name_to_idx["then_ex"]] = True
    g = wire_execution_graph(ir, sn, [("then_ex", "else_ex")], mutation_entropy_norm_threshold=1.01)
    x = torch.randn(2, 5)
    _, loc, sig = g(x)
    assert "then_ex" in loc and loc["then_ex"].shape[0] == 2
    assert "cond_0" in sig and sig["cond_0"].shape == ()
