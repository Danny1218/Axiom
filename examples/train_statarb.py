"""
Institutional-style stat arb: cross-sectional ``batch_mean`` + custom grouped training loop.

Run from repo root: ``python examples/train_statarb.py``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from axiom import AxiomModel
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.inference import _inputs_to_tensor


def _trunk_dim(block: InterpretedBlock) -> int:
    abi, aw = block.abi, getattr(block, "abi_widths", {}) or {}
    return max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=16)


def make_mock_panel() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict[str, float | int]] = []
    for day in range(10):
        for ticker in range(50):
            momentum = float(rng.normal(0.0, 1.0))
            volatility = float(rng.uniform(0.01, 0.05))
            future_return = float(0.1 * momentum + rng.normal(0.0, 0.02))
            rows.append(
                {
                    "day": day,
                    "ticker": ticker,
                    "momentum": momentum,
                    "volatility": volatility,
                    "future_return": future_return,
                }
            )
    return pd.DataFrame(rows)


def _rows_to_float_dicts(df: pd.DataFrame) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for rec in df.to_dict(orient="records"):
        row: dict[str, float] = {}
        for k, v in rec.items():
            row[str(k)] = float(v.item()) if hasattr(v, "item") else float(v)
        out.append(row)
    return out


def main() -> None:
    reset_parser()
    ax_path = Path(__file__).resolve().parent / "statarb.ax"
    ir = ast_to_ir(parse_ax_file(ax_path))
    abi = extract_global_abi(ir, max_vars=64)
    aw = extract_abi_widths(ir, max_vars=64)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    dim = _trunk_dim(block)
    tw_col = abi["target_weight"]
    opt = torch.optim.Adam(block.parameters(), lr=0.01)

    df = make_mock_panel()

    for epoch in range(20):
        ep_loss = 0.0
        n_batches = 0
        for _, day_df in df.groupby("day", sort=True):
            rows = _rows_to_float_dicts(day_df)
            parts = [
                _inputs_to_tensor(
                    r,
                    block.abi,
                    dim,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    abi_widths=aw,
                )
                for r in rows
            ]
            h = torch.cat(parts, dim=0)
            opt.zero_grad(set_to_none=True)
            out = block(h)
            weights = out[:, tw_col]
            if weights.dim() > 1:
                weights = weights.squeeze(-1)
            rets = torch.tensor(
                day_df["future_return"].values,
                dtype=torch.float32,
                device=weights.device,
            )
            loss = -(weights * rets).sum()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            n_batches += 1
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1} avg_batch_loss {-ep_loss / max(n_batches, 1):.6f}")

    block.eval()
    model = AxiomModel(block)
    day0 = df[df["day"] == 0].reset_index(drop=True)
    results = model.predict(day0)
    mna = sum(float(r["market_neutral_alpha"]) for r in results)
    tw_sum = sum(float(r["target_weight"]) for r in results)
    print("\n--- Proof (day 0 cross-section) ---")
    print(f"Sum market_neutral_alpha (should be ~0): {mna:.8f}")
    print(f"Sum target_weight (vol-scaled; not forced to 0): {tw_sum:.6f}")
    j = int(day0["volatility"].values.argmax())
    print(f"Highest-volatility row target_weight: {results[j]['target_weight']:.6f}")


if __name__ == "__main__":
    main()
