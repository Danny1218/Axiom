"""Phase 48: navier_stokes.ax — vortex-stretching loop + kinetic energy maximization."""

from pathlib import Path

import torch

from axiom.api import AxiomModel
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi, extract_neural_node_specs
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.inference import _batch_inputs_to_tensor


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _has_op_loop(stmts: list) -> bool:
    for st in stmts:
        op = st[0]
        if op == "OP_LOOP":
            return True
        if op == "OP_CONDITIONAL" and (_has_op_loop(st[2]) or _has_op_loop(st[3])):
            return True
    return False


def test_navier_stokes_ax_three_liquid_neurals_and_loop():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "navier_stokes.ax"))
    assert _has_op_loop(ir)
    aw = extract_abi_widths(ir, max_vars=128)
    spec = extract_neural_node_specs(ir, aw)
    assert len(spec) == 3
    assert all(arch == "liquid" for _, arch in spec.values())
    abi = extract_global_abi(ir, max_vars=128)
    assert "kinetic_energy" in abi and "random_seed" in abi


def test_navier_stokes_batch_forward_abi():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "navier_stokes.ax"))
    abi = extract_global_abi(ir, max_vars=128)
    aw = extract_abi_widths(ir, max_vars=128)
    b = InterpretedBlock(ir, abi, abi_widths=aw, max_unroll=20)
    dim = max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)
    rows = [{"random_seed": 0.2}, {"random_seed": -0.7}]
    h = _batch_inputs_to_tensor(
        rows, b.abi, dim, device=torch.device("cpu"), dtype=torch.float32, abi_widths=aw
    )
    out = b(h)
    c = int(b.abi["kinetic_energy"])
    assert out.shape[0] == 2
    assert out[:, c].shape == (2,)


def test_max_unroll_changes_final_kinetic_energy():
    """Fewer static unroll steps = fewer Euler steps → different post-loop state."""
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "navier_stokes.ax"))
    abi = extract_global_abi(ir, max_vars=128)
    aw = extract_abi_widths(ir, max_vars=128)
    dim = max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)
    col = int(abi["kinetic_energy"])
    rows = [{"random_seed": 0.1}, {"random_seed": 0.9}]
    h = _batch_inputs_to_tensor(
        rows, abi, dim, device=torch.device("cpu"), dtype=torch.float32, abi_widths=aw
    )
    torch.manual_seed(7)
    b_short = InterpretedBlock(ir, abi, abi_widths=aw, max_unroll=8)
    torch.manual_seed(7)
    b_full = InterpretedBlock(ir, abi, abi_widths=aw, max_unroll=20)
    with torch.no_grad():
        k8 = b_short(h)[:, col].clone()
        k20 = b_full(h)[:, col].clone()
    assert not torch.allclose(k8, k20)


def test_mean_kinetic_energy_rises_under_adam():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "navier_stokes.ax"))
    abi = extract_global_abi(ir, max_vars=128)
    aw = extract_abi_widths(ir, max_vars=128)
    b = InterpretedBlock(ir, abi, abi_widths=aw, max_unroll=20)
    dim = max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)
    col = int(b.abi["kinetic_energy"])
    torch.manual_seed(0)
    data = [{"random_seed": float(i) * 0.1 - 0.5} for i in range(16)]
    opt = torch.optim.Adam(b.parameters(), lr=0.005)
    device, dtype = torch.device("cpu"), torch.float32

    def mean_ke() -> float:
        h = _batch_inputs_to_tensor(
            data, b.abi, dim, device=device, dtype=dtype, abi_widths=aw
        )
        with torch.no_grad():
            return float(b(h)[:, col].mean().item())

    e0 = mean_ke()
    for _ in range(12):
        opt.zero_grad(set_to_none=True)
        h = _batch_inputs_to_tensor(
            data, b.abi, dim, device=device, dtype=dtype, abi_widths=aw
        )
        b.train()
        ke = b(h)[:, col]
        (-ke.mean()).backward()
        torch.nn.utils.clip_grad_norm_(b.parameters(), 5.0)
        opt.step()
    e1 = mean_ke()
    assert e1 > e0


def test_explain_reports_kinetic_energy():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "navier_stokes.ax"))
    abi = extract_global_abi(ir, max_vars=128)
    aw = extract_abi_widths(ir, max_vars=128)
    b = InterpretedBlock(ir, abi, abi_widths=aw, max_unroll=20)
    model = AxiomModel(b)
    trace = model.explain({"random_seed": 0.5})
    assert "kinetic_energy" in trace
    assert isinstance(trace["kinetic_energy"], (int, float))
    assert trace["kinetic_energy"] == trace["kinetic_energy"]  # not NaN
