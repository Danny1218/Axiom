"""Phase 24: sequence.ax + sine loop pipeline (Liquid-KAN via OP_LOOP)."""

import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from axiom.compiler.flow import wire_execution_graph
from axiom.compiler.ir import ast_to_ir, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.engine.dataloader import AxiomDataset
from axiom.engine.inference import AxiomRunner
from axiom.engine.supernet import LatentSupernet
from axiom.engine.topology import build_execution_graph_from_ir
from axiom.engine.trainer import EvolutionaryTrainer


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_sequence_ax_parses_and_contains_loop():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_repo_root() / "examples" / "sequence.ax"))
    assert any(x[0] == "OP_LOOP" for x in ir)
    assert not any(x[0] == "OP_CONDITIONAL" for x in ir)


def test_sequence_abi_has_x_step_y_pred():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_repo_root() / "examples" / "sequence.ax"))
    abi = extract_global_abi(ir, max_vars=32)
    assert abi.get("x") is not None
    assert abi.get("step") is not None
    assert abi.get("y_pred") is not None
    assert "target" not in abi


def test_loop_graph_forward_tensor_shapes():
    """Liquid loop + post-loop stmt: forward accepts (B, D) and returns (B, D)."""
    reset_parser()
    root = _repo_root()
    ir = ast_to_ir(parse_ax_file(root / "examples" / "sequence.ax"))
    dim = 16
    sn = LatentSupernet(dim, ("latent_0", "latent_1"), rank=2)
    g = build_execution_graph_from_ir(ir, sn, [], loop_max_unroll=10, loop_num_basis=4)
    x = torch.randn(5, dim)
    out, _, _ = g(x)
    assert out.shape == x.shape


def test_run_sequence_smoke_low_mse_not_required():
    """Tiny dataset + few epochs: pipeline runs; MSE is finite."""
    reset_parser()
    root = _repo_root()
    ir = ast_to_ir(parse_ax_file(root / "examples" / "sequence.ax"))
    dim = 16
    sn = LatentSupernet(dim, ("latent_0", "latent_1"), rank=2)
    g = wire_execution_graph(
        ir,
        sn,
        [],
        mutation_entropy_norm_threshold=0.99,
        loop_max_unroll=10,
        loop_num_basis=4,
    )
    abi = g.abi
    rows = []
    rng = __import__("random").Random(42)
    for _ in range(24):
        xv = rng.uniform(0.0, math.pi)
        rows.append({"x": xv, "y_pred": 0.0, "target": math.sin(xv)})
    ds = AxiomDataset(rows[:20], abi, trunk_dim=dim, target_key="target")
    loader = DataLoader(ds, batch_size=8, shuffle=False)
    tr = EvolutionaryTrainer(g, lr=5e-2, compile_graph=False, target_col=abi["y_pred"])
    for _ in range(2):
        tr.train_epoch(loader, meta_compiler=None, device=torch.device("cpu"))
    g.eval()
    runner = AxiomRunner(g)
    preds = runner.predict_dict_batch(rows[20:], device=torch.device("cpu"))
    mse = sum(
        (float(p["y_pred"]) - float(r["target"])) ** 2 for p, r in zip(preds, rows[20:])
    ) / max(len(preds), 1)
    assert math.isfinite(mse)
