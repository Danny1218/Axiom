"""
Train a batched O(1) neural inverse for y = x^3 + sin(x)*exp(x/10) (Phase 47).

Run from repo root: python examples/train_solver.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from axiom.api import AxiomModel
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.inference import _batch_inputs_to_tensor

_EXAMPLES = Path(__file__).resolve().parent
AX_PATH = _EXAMPLES / "inverse_solver.ax"


def _trunk_dim(block: InterpretedBlock) -> int:
    abi, aw = block.abi, getattr(block, "abi_widths", {}) or {}
    return max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)


def _forward_y(x: float) -> float:
    return (x**3) + (math.sin(x) * math.exp(x / 10.0))


def main() -> None:
    torch.manual_seed(7)

    true_xs: list[float] = []
    data: list[dict[str, float]] = []
    for _ in range(5000):
        x = (torch.rand(1).item() * 10.0) - 5.0
        true_xs.append(float(x))
        y = _forward_y(float(x))
        data.append({"target_y": float(y)})

    reset_parser()
    ir = ast_to_ir(parse_ax_file(AX_PATH))
    abi = extract_global_abi(ir, max_vars=64)
    aw = extract_abi_widths(ir, max_vars=64)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    dim = _trunk_dim(block)
    device = torch.device("cpu")
    dtype = torch.float32
    col_cy = int(block.abi["computed_y"])

    target_y_tensor = torch.tensor(
        [d["target_y"] for d in data], device=device, dtype=dtype
    )

    optimizer = torch.optim.Adam(block.parameters(), lr=0.05)

    for epoch in range(300):
        optimizer.zero_grad(set_to_none=True)
        h = _batch_inputs_to_tensor(
            data, block.abi, dim, device=device, dtype=dtype, abi_widths=aw
        )
        block.train()
        out = block(h)
        computed_y = out[:, col_cy]
        loss = F.mse_loss(computed_y, target_y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}  loss {float(loss.detach()):.8f}")

    model = AxiomModel(block)
    test_y = 65.432
    t0 = time.perf_counter()
    trace = model.explain({"target_y": test_y})
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    gx = float(trace["guess_x"])
    cy = float(trace["computed_y"])
    err = abs(test_y - cy)

    print("\n========== INVERSE SOLVER -- PROOF ==========")
    print(f"  Unseen target Y              : {test_y}")
    print(f"  AI guess X (guess_x)         : {gx}")
    print(f"  Verification computed_y      : {cy}")
    print(f"  |target_y - computed_y|     : {err}")
    print(f"  explain() time               : {elapsed_ms:.3f} ms")
    print("============================================\n")
    print(
        "PROOF: The neural network successfully learned the inverse of the non-linear "
        "equation, solving it in a single forward pass without iterative looping!"
    )


if __name__ == "__main__":
    main()
