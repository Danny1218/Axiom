"""Phase 12: fixed unroll interpreter + global fullgraph=True for graphs with OP_LOOP."""

import torch
import torch._dynamo.config as dynamo_config

from axiom.compiler.flow import wire_execution_graph
from axiom.compiler.ir import ast_to_ir
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.engine.supernet import LatentSupernet


def test_compile_fullgraph_while_loop_matches_eager():
    reset_parser()
    ax = """
x = 1;
while (x > 0) {
  x = x - 1;
}
"""
    ir = ast_to_ir(parse_ax(ax))
    torch.manual_seed(0)
    sn = LatentSupernet(5, ("e0", "e1"), rank=2)
    g = wire_execution_graph(ir, sn, [], loop_max_unroll=8, loop_num_basis=4)
    x = torch.randn(2, 5)
    dynamo_config.capture_dynamic_output_shape_ops = True
    out_e, sh_e, sig_e = g(x)
    compiled = torch.compile(g, backend="aot_eager", fullgraph=True)
    out_j, sh_j, sig_j = compiled(x)
    assert torch.allclose(out_e, out_j, atol=1e-5, rtol=1e-5)
    assert sh_e.keys() == sh_j.keys()
    assert sig_e.keys() == sig_j.keys()


def test_compile_fullgraph_mixed_cond_and_loop_matches_eager():
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
    sn = LatentSupernet(5, ("then_0", "else_0", "latent_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")], loop_max_unroll=8, loop_num_basis=4)
    x = torch.randn(3, 5)
    dynamo_config.capture_dynamic_output_shape_ops = True
    out_e, sh_e, sig_e = g(x)
    compiled = torch.compile(g, backend="aot_eager", fullgraph=True)
    out_j, sh_j, sig_j = compiled(x)
    assert torch.allclose(out_e, out_j, atol=1e-5, rtol=1e-5)
    assert set(sh_e.keys()) == set(sh_j.keys())
    for k in sh_e:
        assert torch.allclose(sh_e[k], sh_j[k], atol=1e-5, rtol=1e-5)
    for k in sig_e:
        assert torch.allclose(sig_e[k], sig_j[k], atol=1e-5, rtol=1e-5)


def test_evolutionary_trainer_compile_fullgraph_with_loop_one_epoch():
    torch._dynamo.reset()
    reset_parser()
    ax = """
j = 2;
while (j > 0) {
  j = j - 1;
}
"""
    ir = ast_to_ir(parse_ax(ax))
    sn = LatentSupernet(5, ("a", "b"), rank=2)
    g = wire_execution_graph(ir, sn, [], loop_max_unroll=6, loop_num_basis=4)
    from axiom.engine.dataloader import LiquidSequenceLoader

    seq = torch.randn(40)
    loader = LiquidSequenceLoader(seq, feature_dim=5, batch_size=8, baseline_var=0.02, shuffle=False)
    from axiom.engine.trainer import EvolutionaryTrainer

    tr = EvolutionaryTrainer(g, lr=1e-2, compile_graph=True)
    loss = tr.train_epoch(loader, meta_compiler=None)
    assert loss >= 0.0


def test_run_loop_snapshots_fixed_timesteps_equals_max_unroll():
    from axiom.engine.interpreter import make_seed_map, run_loop_snapshots

    h = torch.zeros(1, 4)
    h[0, 0] = 1.0
    cond = [("OP_LOAD", "i"), ("OP_CONST", 0.0), ("OP_CMP_GT",)]
    body = [("OP_ASSIGN", "i", [("OP_LOAD", "i"), ("OP_CONST", 1.0), ("OP_SUB",)])]
    seed = make_seed_map(cond, body, 4)
    seq, m = run_loop_snapshots(h, cond, body, dim=4, max_unroll=7, seed_map=seed)
    assert seq.shape == (1, 7, 4) and m.shape == (1, 7)
