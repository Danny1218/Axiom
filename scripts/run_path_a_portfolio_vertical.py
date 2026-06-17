"""
Path A vertical: finance portfolio neuro-symbolic vs symbolic baseline (readme Path A).

Run from repo root: ``python scripts/run_path_a_portfolio_vertical.py``
Writes ``artifacts/path_a_portfolio/report.json`` and ``artifacts/path_a_portfolio/model.axb``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from axiom.compiler.deserializer import load_bundle
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.compiler.serializer import bundle_weights_path, save_bundle
from axiom.datasets import load_finance_mock
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.dataloader import AxiomDataset, load_csv_to_dicts
from axiom.engine.inference import _inputs_to_tensor


def _trunk_dim(abi: dict, aw: dict) -> int:
    return max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=16)


@torch.no_grad()
def _dataset_mse(block: InterpretedBlock, loader: DataLoader, pos_col: int) -> float:
    block.eval()
    total, n = 0.0, 0
    for x, y in loader:
        out = block(x)
        pred = out[:, pos_col].unsqueeze(1)
        total += F.mse_loss(pred, y, reduction="sum").item()
        n += y.numel()
    return total / max(n, 1)


def main() -> int:
    out_dir = ROOT / "artifacts" / "path_a_portfolio"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = load_finance_mock(2000, seed=42)
    try:
        rows = load_csv_to_dicts(csv_path)
        reset_parser()
        ax_path = ROOT / "examples" / "portfolio.ax"
        ir = ast_to_ir(parse_ax_file(ax_path))
        abi = extract_global_abi(ir, max_vars=64)
        aw = extract_abi_widths(ir, max_vars=64)
        dim = _trunk_dim(abi, aw)
        pos_col = abi["position"]
        block = InterpretedBlock(ir, abi, abi_widths=aw)
        ds = AxiomDataset(rows, abi, trunk_dim=dim, target_key="target_position", abi_widths=aw)
        train_loader = DataLoader(ds, batch_size=64, shuffle=True)
        eval_loader = DataLoader(ds, batch_size=256, shuffle=False)

        opt = torch.optim.Adam(block.parameters(), lr=1e-2)
        block.train()
        for _ in range(30):
            for x, y in train_loader:
                opt.zero_grad(set_to_none=True)
                out = block(x)
                pred = out[:, pos_col].unsqueeze(1)
                loss = F.mse_loss(pred, y)
                loss.backward()
                opt.step()

        trained_mse = _dataset_mse(block, eval_loader, pos_col)
        backup = block.neural_registry
        block.neural_registry = nn.ModuleDict()
        symbolic_mse = _dataset_mse(block, eval_loader, pos_col)
        block.neural_registry = backup
        improve = symbolic_mse - trained_mse
        pct = 100.0 * improve / symbolic_mse if symbolic_mse > 1e-12 else 0.0

        bundle_path = out_dir / "model.axb"
        save_bundle(block, bundle_path)
        reloaded = load_bundle(bundle_path)
        sample = {
            "volatility": 0.6,
            "drawdown": 0.1,
            "momentum": -0.8,
            "volume": 1.5,
        }
        h = _inputs_to_tensor(sample, abi, dim, device=torch.device("cpu"), dtype=torch.float32, abi_widths=aw)
        block.eval()
        reloaded.eval()
        with torch.no_grad():
            parity_ok = torch.allclose(block(h), reloaded(h), atol=1e-5, rtol=1e-4)

        report = {
            "kind": "axiom.path_a_portfolio",
            "domain": "finance_risk",
            "trained_mse": trained_mse,
            "symbolic_baseline_mse": symbolic_mse,
            "mse_improvement": improve,
            "mse_improvement_pct": pct,
            "bundle_path": str(bundle_path.relative_to(ROOT)),
            "weights_sidecar": str(bundle_weights_path(bundle_path).relative_to(ROOT)),
            "bundle_parity_ok": bool(parity_ok),
            "beats_baseline": trained_mse < symbolic_mse,
        }
        report_path = out_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"Path A portfolio vertical")
        print(f"  trained MSE:          {trained_mse:.6f}")
        print(f"  symbolic baseline:    {symbolic_mse:.6f}")
        print(f"  improvement:          {improve:.6f} ({pct:.1f}%)")
        print(f"  beats baseline:       {report['beats_baseline']}")
        print(f"  bundle parity:        {report['bundle_parity_ok']}")
        print(f"  wrote {report_path}")
        print(f"  wrote {bundle_path}")
        return 0 if report["beats_baseline"] and report["bundle_parity_ok"] else 1
    finally:
        try:
            os.unlink(csv_path)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
