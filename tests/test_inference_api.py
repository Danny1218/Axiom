"""Phase 13: inference runtime, dict → trunk tensor, no_grad, batching."""

from pathlib import Path

import pytest
import torch

from axiom.cli import main as cli_main
from axiom.compiler.deserializer import load_execution_bundle
from axiom.compiler.flow import wire_execution_graph
from axiom.compiler.ir import ast_to_ir
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.compiler.serializer import save_execution_bundle
from axiom.engine.dataloader import LiquidSequenceLoader
from axiom.engine.inference import AxiomRunner, _batch_inputs_to_tensor, _inputs_to_tensor
from axiom.engine.supernet import LatentSupernet
from axiom.engine.trainer import EvolutionaryTrainer


def test_inputs_to_tensor_broadcast_single_key_legacy_empty_abi():
    dev = torch.device("cpu")
    t = _inputs_to_tensor({"x": 3.5}, {}, 4, device=dev, dtype=torch.float32)
    assert t.shape == (1, 4)
    assert torch.allclose(t, torch.full((1, 4), 3.5))


def test_inputs_to_tensor_abi_column_order_not_key_order():
    dev = torch.device("cpu")
    abi = {"a": 1, "z": 0}
    t = _inputs_to_tensor({"z": 5.0, "a": 1.0}, abi, 8, device=dev, dtype=torch.float32)
    assert t[0, 0] == 5.0 and t[0, 1] == 1.0
    assert t[0, 2:].abs().max() == 0


def test_inputs_to_tensor_too_many_keys_raises_legacy_empty_abi():
    dev = torch.device("cpu")
    with pytest.raises(ValueError, match="too many input keys"):
        _inputs_to_tensor({"a": 1.0, "b": 2.0, "c": 3.0}, {}, 2, device=dev, dtype=torch.float32)


def test_batch_inputs_broadcast_legacy_empty_abi():
    dev = torch.device("cpu")
    t = _batch_inputs_to_tensor([{"x": 1.0}, {"x": 2.0}], {}, 3, device=dev, dtype=torch.float32)
    assert t.shape == (2, 3)
    assert torch.allclose(t[0], torch.full((3,), 1.0))
    assert torch.allclose(t[1], torch.full((3,), 2.0))


def test_batch_inputs_multi_key_with_abi():
    dev = torch.device("cpu")
    abi = {"a": 0, "b": 1}
    t = _batch_inputs_to_tensor(
        [{"a": 1.0, "b": -1.0}, {"a": 0.0, "b": 2.0}],
        abi,
        4,
        device=dev,
        dtype=torch.float32,
    )
    assert t[0, 0] == 1.0 and t[0, 1] == -1.0
    assert t[1, 0] == 0.0 and t[1, 1] == 2.0


def test_inference_train_serialize_predict_batch(tmp_path):
    reset_parser()
    ax_src = """
if (1 > 0) {
  z = 1;
}
while (a > b) {
  a = a - 1;
}
"""
    ir = ast_to_ir(parse_ax(ax_src))
    dim = 8
    torch.manual_seed(42)
    sn = LatentSupernet(dim, ("then_0", "else_0", "latent_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(
        ir,
        sn,
        [("then_0", "else_0")],
        loop_max_unroll=4,
        loop_num_basis=4,
        mutation_entropy_norm_threshold=0.99,
    )
    seq = torch.randn(32)
    loader = LiquidSequenceLoader(seq, feature_dim=dim, batch_size=8, baseline_var=0.02, shuffle=False)
    tr = EvolutionaryTrainer(g, lr=1e-2, compile_graph=True)
    tr.train_epoch(loader, meta_compiler=None)

    prefix = tmp_path / "inf_bundle"
    save_execution_bundle(g.cpu(), prefix, ir=ir)

    loaded = load_execution_bundle(prefix)
    runner = AxiomRunner(loaded)
    out = runner.predict_batch([{"a": 1.0, "b": -1.5}, {"a": 2.0, "b": 0.5}])
    assert out.shape == (2, dim)
    assert torch.isfinite(out).all()
    assert not out.requires_grad


def test_axiom_runner_predict_no_grad(tmp_path):
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { x = 1; } else { x = 0; }"))
    sn = LatentSupernet(4, ("then_0", "else_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")], mutation_entropy_norm_threshold=0.99)
    prefix = tmp_path / "b"
    save_execution_bundle(g, prefix, ir=ir)
    runner = AxiomRunner(load_execution_bundle(prefix))
    y = runner.predict({"x": 2.0})
    assert y.shape == (1, 4)
    assert not y.requires_grad


def test_predict_dict_matches_predict_tensor(tmp_path):
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { a = 1; } else { a = 2; }"))
    sn = LatentSupernet(6, ("then_0", "else_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")], mutation_entropy_norm_threshold=0.99)
    prefix = tmp_path / "dict_bundle"
    save_execution_bundle(g, prefix, ir=ir)
    runner = AxiomRunner(load_execution_bundle(prefix))
    t = runner.predict({"a": 3.0, "b": 9.0})
    d = runner.predict_dict({"a": 3.0, "b": 9.0})
    abi = runner.graph.abi
    for name, col in abi.items():
        assert d[name] == pytest.approx(t[0, col].item())


def test_predict_dict_batch_matches_predict_batch(tmp_path):
    reset_parser()
    ir = ast_to_ir(parse_ax("if (1 > 0) { z = 1; } else { z = 2; }"))
    sn = LatentSupernet(5, ("then_0", "else_0"), rank=2)
    sn.set_masks({"then_0": 1.0, "else_0": 1.0})
    g = wire_execution_graph(ir, sn, [("then_0", "else_0")], mutation_entropy_norm_threshold=0.99)
    prefix = tmp_path / "batch_dict"
    save_execution_bundle(g, prefix, ir=ir)
    runner = AxiomRunner(load_execution_bundle(prefix))
    rows = [{"z": 0.5}, {"z": -1.0}]
    mat = runner.predict_batch(rows)
    dec = runner.predict_dict_batch(rows)
    assert len(dec) == 2
    abi = runner.graph.abi
    for b in range(2):
        for name, col in abi.items():
            assert dec[b][name] == pytest.approx(mat[b, col].item())


def test_main_inference_mode(tmp_path):
    reset_parser()
    ax = Path(__file__).resolve().parents[1] / "train.ax"
    out = tmp_path / "bundle"
    cli_main(
        [
            "train",
            str(ax),
            "--mode",
            "train",
            "--epochs",
            "1",
            "--batch",
            "8",
            "--dim",
            "8",
            "--out",
            str(out),
            "--seed",
            "0",
        ]
    )
    cli_main(["train", "--mode", "inference", "--out", str(out)])
