"""
Phase 36: train ``portfolio.ax`` on synthetic finance CSV, then compare neuro-symbolic vs pure symbolic MSE.

Run from repo root (editable install): ``python examples/train_portfolio.py``
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.datasets import load_finance_mock
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.dataloader import AxiomDataset, load_csv_to_dicts


def _trunk_dim_from_abi(abi: dict, aw: dict) -> int:
    return max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=16)


@torch.no_grad()
def dataset_mse(block: InterpretedBlock, loader: DataLoader, pos_col: int) -> float:
    block.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        out = block(x)
        pred = out[:, pos_col].unsqueeze(1)
        tot += F.mse_loss(pred, y, reduction="sum").item()
        n += y.numel()
    return tot / max(n, 1)


def main() -> None:
    csv_path = load_finance_mock(2000, seed=42)
    try:
        rows = load_csv_to_dicts(csv_path)
        reset_parser()
        ax_path = Path(__file__).resolve().parent / "portfolio.ax"
        ir = ast_to_ir(parse_ax_file(ax_path))
        abi = extract_global_abi(ir, max_vars=64)
        aw = extract_abi_widths(ir, max_vars=64)
        dim = _trunk_dim_from_abi(abi, aw)
        pos_col = abi["position"]
        block = InterpretedBlock(ir, abi, abi_widths=aw)
        ds = AxiomDataset(rows, abi, trunk_dim=dim, target_key="target_position", abi_widths=aw)
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        eval_loader = DataLoader(ds, batch_size=256, shuffle=False)

        opt = torch.optim.Adam(block.parameters(), lr=1e-2)
        block.train()
        for epoch in range(50):
            ep_loss = 0.0
            for x, y in loader:
                opt.zero_grad(set_to_none=True)
                out = block(x)
                pred = out[:, pos_col].unsqueeze(1)
                loss = F.mse_loss(pred, y)
                loss.backward()
                opt.step()
                ep_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"epoch {epoch + 1} train_loss {ep_loss / len(loader):.6f}")

        trained_mse = dataset_mse(block, eval_loader, pos_col)
        print(f"Trained neuro-symbolic MSE (full data): {trained_mse:.6f}")

        backup = block.neural_registry
        block.neural_registry = nn.ModuleDict()
        symbolic_mse = dataset_mse(block, eval_loader, pos_col)
        block.neural_registry = backup
        print(f"Pure symbolic baseline MSE (neural disabled): {symbolic_mse:.6f}")

        improve = symbolic_mse - trained_mse
        pct = 100.0 * improve / symbolic_mse if symbolic_mse > 1e-12 else 0.0
        print(
            f"Summary: neural adapter reduced MSE by {improve:.6f} "
            f"({pct:.1f}% vs symbolic-only clamp(1 - base_risk, [0,1]))."
        )
    finally:
        try:
            os.unlink(csv_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
