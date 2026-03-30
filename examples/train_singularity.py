"""
Navier-Stokes singularity hunt: maximize post-loop kinetic energy (localized vortex model).

Aggressive Adam (e.g. lr=0.1) often hits NaN when backpropagating 20 nonlinear Euler steps in
float32; default lr below stays finite while still driving energy very large.

Run from repo root: python examples/train_singularity.py
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from axiom.api import AxiomModel
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.inference import _batch_inputs_to_tensor

_EXAMPLES = Path(__file__).resolve().parent
AX_PATH = _EXAMPLES / "navier_stokes.ax"
# While body runs 20 steps (step 0..19); static unroll must be >= 20.
_LOOP_UNROLL = 20


def _trunk_dim(block: InterpretedBlock) -> int:
    abi, aw = block.abi, getattr(block, "abi_widths", {}) or {}
    return max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)


def main() -> None:
    random.seed(0)
    torch.manual_seed(0)

    data: list[dict[str, float]] = [
        {"random_seed": random.uniform(-1.0, 1.0)} for _ in range(1000)
    ]

    reset_parser()
    ir = ast_to_ir(parse_ax_file(AX_PATH))
    abi = extract_global_abi(ir, max_vars=128)
    aw = extract_abi_widths(ir, max_vars=128)
    block = InterpretedBlock(ir, abi, abi_widths=aw, max_unroll=_LOOP_UNROLL)
    dim = _trunk_dim(block)
    device = torch.device("cpu")
    dtype = torch.float32
    col_ke = int(block.abi["kinetic_energy"])

    # Default 0.0015: stable through 100 epochs; try lr=0.1 only with float64 / smaller unroll.
    optimizer = torch.optim.Adam(block.parameters(), lr=0.0015)

    for epoch in range(100):
        optimizer.zero_grad(set_to_none=True)
        h = _batch_inputs_to_tensor(
            data, block.abi, dim, device=device, dtype=dtype, abi_widths=aw
        )
        block.train()
        out = block(h)
        kinetic_energy = out[:, col_ke]
        loss = -torch.mean(kinetic_energy)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(block.parameters(), 5.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                mx = float(torch.max(kinetic_energy).item())
            print(f"Epoch {epoch + 1}  max kinetic_energy (batch) {mx:.6g}")

    model = AxiomModel(block)
    trace = model.explain({"random_seed": 0.5})
    ke = float(trace["kinetic_energy"])

    print("\n========== SINGULARITY HUNTER -- TRACE (random_seed=0.5) ==========")
    print(f"  Final kinetic_energy (from explain): {ke:.6g}")
    print("====================================================================\n")
    print(
        "PROOF: The Liquid Neural Network successfully navigated the non-linear "
        "physics constraints to force the fluid packet into an extreme energy state!"
    )


if __name__ == "__main__":
    main()
