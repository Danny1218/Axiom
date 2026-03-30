"""E2E: EvolutionaryTrainer + AxiomDataset + ABI-mapped tabular rows."""

import torch.nn as nn
from torch.utils.data import DataLoader

from compiler.flow import wire_execution_graph
from compiler.ir import ast_to_ir
from compiler.parser import parse_ax, reset_parser
from engine.dataloader import AxiomDataset
from engine.supernet import LatentSupernet
from engine.trainer import EvolutionaryTrainer


def test_train_epoch_axiom_dataset_assign_script():
    """``x = a + b`` in the then-branch so Sinkhorn + adapters run (trunk is frozen on the graph)."""
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { x = a + b; } else { x = 0; }"))
    sn = LatentSupernet(16, ("then_e", "else_e"), rank=2)
    sn.set_masks({"then_e": 1.0, "else_e": 1.0})
    g = wire_execution_graph(ir, sn, [("then_e", "else_e")], mutation_entropy_norm_threshold=0.99)
    for b in g.conditional_blocks():
        nn.init.zeros_(b.router.proj.weight)
        nn.init.zeros_(b.router.proj.bias)
    data = [
        {"a": 1.0, "b": 2.0, "target": 3.0},
        {"a": 4.0, "b": 5.0, "target": 9.0},
    ]
    ds = AxiomDataset(
        data, g.abi, trunk_dim=16, target_key="target", broadcast_target=True
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    tr = EvolutionaryTrainer(g, lr=1e-2, compile_graph=False)
    loss = tr.train_epoch(loader, meta_compiler=None)
    assert loss >= 0.0
