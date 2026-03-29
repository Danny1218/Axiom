import json
from pathlib import Path

import torch

from compiler.flow import wire_execution_graph
from compiler.ir import ast_to_ir
from compiler.parser import parse_ax, reset_parser
from compiler.serializer import execution_topology_to_dict, load_state_dict, save_execution_bundle
from engine.dataloader import LiquidSequenceLoader
from engine.supernet import LatentSupernet
from engine.trainer import EvolutionaryTrainer


def test_save_and_load_state_dict(tmp_path):
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { a = 1; } else { a = 2; }"))
    sn = LatentSupernet(5, ("then_0", "else_0", "latent_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")])
    prefix = tmp_path / "b"
    save_execution_bundle(g, prefix, ir=ir)
    assert Path(str(prefix) + ".pt").exists()
    jpath = Path(str(prefix) + "_topology.json")
    assert jpath.exists()
    data = json.loads(jpath.read_text(encoding="utf-8"))
    assert "nodes" in data and "edges" in data and "ir" in data
    sd = load_state_dict(str(prefix) + ".pt")
    assert any(k.startswith("supernet.") for k in sd)


def test_topology_dict_serializable():
    reset_parser()
    ir = ast_to_ir(parse_ax("x = 1;"))
    sn = LatentSupernet(4, ("a", "b"), rank=2)
    g = wire_execution_graph(ir, sn, [])
    d = execution_topology_to_dict(g)
    json.dumps(d)


def test_trainer_runs_consecutive_epochs(tmp_path):
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { a = 1; } else { a = 2; }"))
    sn = LatentSupernet(5, ("then_0", "else_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")])
    seq = torch.cumsum(torch.randn(64) * 0.02, dim=0)
    loader = LiquidSequenceLoader(seq, feature_dim=5, batch_size=16, baseline_var=0.03, shuffle=True)
    tr = EvolutionaryTrainer(g, lr=5e-2, shadow_fitness_epochs=3)
    loss0 = tr.train_epoch(loader, meta_compiler=None)
    assert loss0 >= 0.0
    loss1 = tr.train_epoch(loader, meta_compiler=None)
    assert loss1 >= 0.0
