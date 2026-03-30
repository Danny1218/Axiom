import torch
import torch.nn as nn

from compiler.flow import wire_execution_graph
from compiler.ir import ast_to_ir
from compiler.parser import parse_ax, reset_parser
from engine.dataloader import LiquidSequenceLoader
from engine.meta_compiler import MetaCompiler
from engine.supernet import LatentSupernet
from engine.trainer import EvolutionaryTrainer


def test_meta_unmask_keeps_same_optimizer():
    """Unmasking does not require a new Adam instance; parameter set is unchanged."""
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { a = 1; } else { a = 2; }"))
    sn = LatentSupernet(5, ("then_0", "else_0", "latent_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")], mutation_entropy_norm_threshold=0.01)
    for b in g.conditional_blocks():
        nn.init.zeros_(b.router.proj.weight)
        nn.init.zeros_(b.router.proj.bias)
    seq = torch.randn(64)
    loader = LiquidSequenceLoader(seq, feature_dim=5, batch_size=16, baseline_var=0.02, shuffle=False)
    tr = EvolutionaryTrainer(g, lr=1e-2)
    oid_before = id(tr.optimizer)
    tr.train_epoch(loader, meta_compiler=MetaCompiler(sn))
    assert sn.adapter_mask[sn._name_to_idx["latent_0"]] >= 0.5
    assert id(tr.optimizer) == oid_before


def test_evolutionary_trainer_target_col_kwarg():
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { a = 1; } else { a = 2; }"))
    sn = LatentSupernet(5, ("then_0", "else_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")], mutation_entropy_norm_threshold=0.99)
    tr = EvolutionaryTrainer(g, target_col=g.abi["a"])
    assert tr.target_col == g.abi["a"]


def test_trainer_compile_graph_runs_one_epoch():
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { a = 1; } else { a = 2; }"))
    sn = LatentSupernet(5, ("then_0", "else_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")], mutation_entropy_norm_threshold=0.99)
    seq = torch.randn(32)
    loader = LiquidSequenceLoader(seq, feature_dim=5, batch_size=8, baseline_var=0.02, shuffle=False)
    tr = EvolutionaryTrainer(g, lr=1e-2, compile_graph=True)
    loss = tr.train_epoch(loader, meta_compiler=None)
    assert loss >= 0.0
