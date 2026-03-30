"""Phase 36: synthetic finance CSV, portfolio.ax, neuro-symbolic vs symbolic MSE."""

import math
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


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_load_finance_mock_csv_columns_and_target_formula():
    path = load_finance_mock(20, seed=0)
    try:
        rows = load_csv_to_dicts(path)
        assert len(rows) == 20
        for r in rows:
            assert set(r.keys()) >= {
                "volatility",
                "drawdown",
                "momentum",
                "volume",
                "target_position",
            }
            assert 0.1 <= r["volatility"] <= 1.0
            assert 0.0 <= r["drawdown"] <= 0.5
            assert -1.0 <= r["momentum"] <= 1.0
            assert 0.5 <= r["volume"] <= 2.0
            v, d, m, vol = r["volatility"], r["drawdown"], r["momentum"], r["volume"]
            base = 1.0
            if v > 0.5:
                base -= 0.5
            if d > 0.2:
                base -= 0.3
            want = max(0.0, min(1.0, base + 0.2 * math.sin(m * vol)))
            assert abs(r["target_position"] - want) < 1e-9
    finally:
        os.unlink(path)


def test_portfolio_ax_parses_with_neural_and_max_min():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "portfolio.ax"))
    flat = str(ir)
    assert "OP_NEURAL" in flat
    assert "OP_MATH_BINARY" in flat
    abi = extract_global_abi(ir, max_vars=64)
    aw = extract_abi_widths(ir, max_vars=64)
    for name in ("volatility", "drawdown", "momentum", "volume", "position"):
        assert name in abi
    assert aw.get("features", 1) == 3


@torch.no_grad()
def _mse(block: InterpretedBlock, loader: DataLoader, pos_col: int) -> float:
    block.eval()
    tot, n = 0.0, 0
    for x, y in loader:
        out = block(x)
        pred = out[:, pos_col].unsqueeze(1)
        tot += F.mse_loss(pred, y, reduction="sum").item()
        n += y.numel()
    return tot / max(n, 1)


def test_neuro_symbolic_mse_below_symbolic_after_short_training():
    path = load_finance_mock(512, seed=7)
    try:
        rows = load_csv_to_dicts(path)
        reset_parser()
        ir = ast_to_ir(parse_ax_file(_root() / "examples" / "portfolio.ax"))
        abi = extract_global_abi(ir, max_vars=64)
        aw = extract_abi_widths(ir, max_vars=64)
        dim = max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=16)
        pos_col = abi["position"]
        block = InterpretedBlock(ir, abi, abi_widths=aw)
        ds = AxiomDataset(
            rows, abi, trunk_dim=dim, target_key="target_position", abi_widths=aw
        )
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        eval_loader = DataLoader(ds, batch_size=128, shuffle=False)

        opt = torch.optim.Adam(block.parameters(), lr=5e-2)
        block.train()
        for _ in range(12):
            for x, y in loader:
                opt.zero_grad(set_to_none=True)
                out = block(x)
                loss = F.mse_loss(out[:, pos_col].unsqueeze(1), y)
                loss.backward()
                opt.step()

        trained = _mse(block, eval_loader, pos_col)
        backup = block.neural_registry
        block.neural_registry = nn.ModuleDict()
        symbolic = _mse(block, eval_loader, pos_col)
        block.neural_registry = backup
        assert trained < symbolic
    finally:
        os.unlink(path)
