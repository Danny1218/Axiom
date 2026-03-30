"""
Automated drug-discovery demo: batched viability optimization (Phase 46).

Run from repo root: python examples/train_pharma.py
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
AX_PATH = _EXAMPLES / "drug_discovery.ax"
REPORT_PATH = _EXAMPLES / "drug_report.html"


def _trunk_dim(block: InterpretedBlock) -> int:
    abi, aw = block.abi, getattr(block, "abi_widths", {}) or {}
    return max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)


def main() -> None:
    random.seed(42)
    torch.manual_seed(42)

    data: list[dict[str, float]] = []
    for _ in range(100):
        data.append(
            {
                "target_polarity": random.uniform(-50.0, 50.0),
                "target_size": random.uniform(10.0, 100.0),
                "ambient_temp": 98.6,
            }
        )

    reset_parser()
    ir = ast_to_ir(parse_ax_file(AX_PATH))
    abi = extract_global_abi(ir, max_vars=128)
    aw = extract_abi_widths(ir, max_vars=128)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    dim = _trunk_dim(block)
    device = torch.device("cpu")
    dtype = torch.float32
    col_vs = int(block.abi["viability_score"])

    optimizer = torch.optim.Adam(block.parameters(), lr=0.5)

    for epoch in range(200):
        optimizer.zero_grad(set_to_none=True)
        h = _batch_inputs_to_tensor(
            data, block.abi, dim, device=device, dtype=dtype, abi_widths=aw
        )
        block.train()
        out = block(h)
        viability = out[:, col_vs]
        loss = -torch.mean(viability)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            with torch.no_grad():
                mv = float(viability.mean().item())
            print(f"Epoch {epoch + 1}  Mean Viability Score {mv:.4f}")

    model = AxiomModel(block)
    test_cell = data[0]
    trace = model.explain(test_cell)

    print("\n========== PHARMA AUTOPSY (first cell) ==========")
    print(f"  target_polarity   : {trace.get('target_polarity', test_cell['target_polarity'])}")
    print(f"  drug_polarity     : {trace.get('drug_polarity', '—')}")
    print(f"  carbon_angle      : {trace.get('carbon_angle', '—')}")
    print(f"  physics_penalty   : {trace.get('physics_penalty', '—')}")
    print(f"  molecular_weight  : {trace.get('molecular_weight', '—')}")
    print(f"  weight_penalty    : {trace.get('weight_penalty', '—')}")
    print(f"  binding_affinity  : {trace.get('binding_affinity', '—')}")
    print(f"  viability_score   : {trace.get('viability_score', '—')}")
    print("================================================\n")

    source = AX_PATH.read_text(encoding="utf-8")
    model.export_report(test_cell, str(REPORT_PATH.resolve()), source_code=source)
    print(f"Open {REPORT_PATH} to view the biochemical execution trace!")


if __name__ == "__main__":
    main()
