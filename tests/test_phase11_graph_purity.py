"""Phase 11: pure router entropy tensors, signal bubbling, meta-compiler outside compile, fullgraph."""

import torch
import torch.nn as nn

from compiler.flow import wire_execution_graph
from compiler.ir import ast_to_ir
from compiler.parser import parse_ax, reset_parser
from engine.meta_compiler import MetaCompiler
from engine.supernet import LatentSupernet
from engine.topology import build_execution_graph_from_ir
from engine.trainer import EvolutionaryTrainer


def test_execution_graph_three_tuple_and_signal_keys():
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { a = 1; } else { a = 2; }"))
    sn = LatentSupernet(5, ("then_0", "else_0", "latent_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = build_execution_graph_from_ir(ir, sn, [("then_0", "else_0")])
    x = torch.randn(2, 5)
    out, shadows, sig = g(x)
    assert out.shape == x.shape
    assert isinstance(shadows, dict)
    assert "cond_0" in sig and sig["cond_0"].shape == ()
    th = g.block_mutation_thresholds()
    assert th == {"cond_0": g.node_modules["cond_0"].router.mutation_entropy_norm_threshold}


def test_trainer_fullgraph_meta_uses_bubbled_signals():
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { a = 1; } else { a = 2; }"))
    sn = LatentSupernet(5, ("then_0", "else_0", "latent_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")], mutation_entropy_norm_threshold=0.01)
    for b in g.conditional_blocks():
        nn.init.zeros_(b.router.proj.weight)
        nn.init.zeros_(b.router.proj.bias)
    from engine.dataloader import LiquidSequenceLoader

    seq = torch.randn(48)
    loader = LiquidSequenceLoader(seq, feature_dim=5, batch_size=8, baseline_var=0.02, shuffle=False)
    tr = EvolutionaryTrainer(g, lr=1e-2, compile_graph=True)
    loss = tr.train_epoch(loader, meta_compiler=MetaCompiler(sn))
    assert loss >= 0.0
    assert sn.adapter_mask[sn._name_to_idx["latent_0"]] >= 0.5


def test_router_forward_no_last_mutation_attribute():
    from engine.router import SinkhornRouter

    r = SinkhornRouter(3, 2)
    assert not hasattr(r, "last_mutation_signal")
