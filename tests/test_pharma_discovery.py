"""Phase 46: drug_discovery.ax + batched viability (hinge-style penalties)."""

from pathlib import Path

import torch

from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi, extract_neural_node_specs
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.inference import _batch_inputs_to_tensor


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_drug_discovery_ax_three_liquid_neurals():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "drug_discovery.ax"))
    aw = extract_abi_widths(ir, max_vars=128)
    spec = extract_neural_node_specs(ir, aw)
    assert len(spec) == 3
    assert all(arch == "liquid" for _, arch in spec.values())
    abi = extract_global_abi(ir, max_vars=128)
    assert "viability_score" in abi
    assert "carbon_angle" in abi and "drug_polarity" in abi


def test_drug_discovery_batch_forward_and_abi():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "drug_discovery.ax"))
    abi = extract_global_abi(ir, max_vars=128)
    aw = extract_abi_widths(ir, max_vars=128)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    dim = max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)
    rows = [
        {"target_polarity": 10.0, "target_size": 50.0, "ambient_temp": 98.6},
        {"target_polarity": -20.0, "target_size": 20.0, "ambient_temp": 98.6},
    ]
    h = _batch_inputs_to_tensor(
        rows, b.abi, dim, device=torch.device("cpu"), dtype=torch.float32, abi_widths=aw
    )
    assert h.shape[0] == 2
    out = b(h)
    c = int(b.abi["viability_score"])
    assert out.shape[0] == 2
    assert out[:, c].shape == (2,)


def test_viability_improves_over_few_adam_steps():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "drug_discovery.ax"))
    abi = extract_global_abi(ir, max_vars=128)
    aw = extract_abi_widths(ir, max_vars=128)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    dim = max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)
    col = int(b.abi["viability_score"])
    torch.manual_seed(0)
    data = [
        {"target_polarity": 5.0, "target_size": 40.0, "ambient_temp": 98.6},
        {"target_polarity": -15.0, "target_size": 30.0, "ambient_temp": 98.6},
    ]
    opt = torch.optim.Adam(b.parameters(), lr=0.5)
    device, dtype = torch.device("cpu"), torch.float32

    def mean_via() -> float:
        h = _batch_inputs_to_tensor(
            data, b.abi, dim, device=device, dtype=dtype, abi_widths=aw
        )
        with torch.no_grad():
            return float(b(h)[:, col].mean().item())

    v0 = mean_via()
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        h = _batch_inputs_to_tensor(
            data, b.abi, dim, device=device, dtype=dtype, abi_widths=aw
        )
        b.train()
        out = b(h)
        (-out[:, col].mean()).backward()
        opt.step()
    v1 = mean_via()
    assert v1 > v0
