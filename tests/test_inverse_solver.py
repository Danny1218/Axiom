"""Phase 47: neural inverse for y = x^3 + sin(x)*exp(x/10)."""

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi, extract_neural_node_specs
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.inference import _batch_inputs_to_tensor


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _y_py(x: float) -> float:
    return (x**3) + (math.sin(x) * math.exp(x / 10.0))


def test_inverse_solver_ax_liquid_neural_and_computed_y():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "inverse_solver.ax"))
    aw = extract_abi_widths(ir, max_vars=64)
    spec = extract_neural_node_specs(ir, aw)
    assert len(spec) == 1
    assert list(spec.values())[0][1] == "liquid"
    abi = extract_global_abi(ir, max_vars=64)
    assert set(abi.keys()) >= {"target_y", "guess_x", "computed_y", "features"}


def test_batched_forward_matches_scalar_formula_no_shape_blowup():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "inverse_solver.ax"))
    abi = extract_global_abi(ir, max_vars=64)
    aw = extract_abi_widths(ir, max_vars=64)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    dim = max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)
    rows = [{"target_y": _y_py(1.0)}, {"target_y": _y_py(-2.0)}, {"target_y": _y_py(0.5)}]
    h = _batch_inputs_to_tensor(
        rows, b.abi, dim, device=torch.device("cpu"), dtype=torch.float32, abi_widths=aw
    )
    with torch.no_grad():
        out = b(h)
    assert out.shape[0] == 3
    col = int(b.abi["computed_y"])
    for i, x in enumerate([1.0, -2.0, 0.5]):
        gx = float(out[i, b.abi["guess_x"]].item())
        cy = float(out[i, col].item())
        assert abs(cy - _y_py(gx)) < 1e-4


def test_training_reduces_mse_small_batch():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "inverse_solver.ax"))
    abi = extract_global_abi(ir, max_vars=64)
    aw = extract_abi_widths(ir, max_vars=64)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    dim = max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)
    col = int(block.abi["computed_y"])
    torch.manual_seed(0)
    data = []
    targets = []
    for _ in range(32):
        x = (torch.rand(1).item() * 10.0) - 5.0
        y = _y_py(float(x))
        data.append({"target_y": float(y)})
        targets.append(float(y))
    tgt = torch.tensor(targets, dtype=torch.float32)
    opt = torch.optim.Adam(block.parameters(), lr=0.05)
    device, dtype = torch.device("cpu"), torch.float32

    def mse() -> float:
        h = _batch_inputs_to_tensor(
            data, block.abi, dim, device=device, dtype=dtype, abi_widths=aw
        )
        with torch.no_grad():
            return float(F.mse_loss(block(h)[:, col], tgt).item())

    e0 = mse()
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        h = _batch_inputs_to_tensor(
            data, block.abi, dim, device=device, dtype=dtype, abi_widths=aw
        )
        block.train()
        out = block(h)
        F.mse_loss(out[:, col], tgt).backward()
        opt.step()
    e1 = mse()
    assert e1 < e0
