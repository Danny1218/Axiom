"""
Regression: shadow experts must receive gradients during EvolutionaryTrainer.train_epoch
via localized MSE added to the main loss (main path still uses detached shadow deltas).
"""

import torch
import torch.nn.functional as F

from compiler.flow import wire_execution_graph
from compiler.ir import ast_to_ir
from compiler.parser import parse_ax, reset_parser
from engine.dataloader import LiquidSequenceLoader
from engine.supernet import LatentSupernet
from engine.trainer import EvolutionaryTrainer


def test_shadow_expert_weights_update_after_train_epoch():
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { a = 1; } else { a = 2; }"))
    sn = LatentSupernet(5, ("then_ex", "else_ex"), rank=2)
    sn.set_masks({"then_ex": 1.0, "else_ex": 1.0})
    sn.is_shadow[sn._name_to_idx["then_ex"]] = True
    g = wire_execution_graph(ir, sn, [("then_ex", "else_ex")], mutation_entropy_norm_threshold=1.01)
    U_before = sn.adapters["then_ex"].U.detach().clone()
    seq = torch.cumsum(torch.randn(80) * 0.02, dim=0)
    loader = LiquidSequenceLoader(seq, feature_dim=5, batch_size=16, baseline_var=0.03, shuffle=False)
    tr = EvolutionaryTrainer(g, lr=5e-2, shadow_fitness_epochs=5)
    tr.train_epoch(loader, meta_compiler=None)
    U_after = sn.adapters["then_ex"].U.detach()
    assert (U_after - U_before).abs().max().item() > 1e-7


def test_shadow_local_mse_flows_grad_to_adapter_one_step():
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { a = 1; } else { a = 2; }"))
    sn = LatentSupernet(5, ("then_ex", "else_ex"), rank=2)
    sn.set_masks({"then_ex": 1.0, "else_ex": 1.0})
    sn.is_shadow[sn._name_to_idx["then_ex"]] = True
    g = wire_execution_graph(ir, sn, [("then_ex", "else_ex")], mutation_entropy_norm_threshold=1.01)
    x = torch.randn(4, 5)
    y = torch.randn(4, 5)
    g.zero_grad(set_to_none=True)
    out, locs = g(x)
    main = F.mse_loss(out, y)
    assert "then_ex" in locs
    shadow = F.mse_loss(locs["then_ex"], y)
    (main + shadow).backward()
    assert sn.adapters["then_ex"].U.grad is not None
