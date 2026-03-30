"""
Live SPY features, train ``spy_alpha.ax`` (neural alpha + volatility circuit breaker), backtest via ``axiom.load``.

Requires: ``pip install yfinance pandas``

Run from repo root: ``python examples/train_spy.py``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import axiom
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.compiler.serializer import save_bundle
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.dataloader import AxiomDataset

_EXAMPLES = Path(__file__).resolve().parent
AX_PATH = _EXAMPLES / "spy_alpha.ax"
BUNDLE_PATH = _EXAMPLES / "spy_trained.axb"
TEST_TAIL = 500


def add_spy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum / volatility / next-day target; expects yfinance OHLCV column names."""
    out = df.copy()
    out["momentum_1d"] = out["Close"].pct_change(1)
    out["momentum_5d"] = out["Close"].pct_change(5)
    out["volatility"] = (out["High"] - out["Low"]) / out["Open"]
    out["target_return"] = out["Close"].pct_change(1).shift(-1)
    return out.dropna()


def chronological_split(
    df: pd.DataFrame, test_rows: int = TEST_TAIL
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n <= test_rows:
        raise ValueError(f"need more than {test_rows} rows after dropna; got {n}")
    return df.iloc[:-test_rows].copy(), df.iloc[-test_rows:].copy()


def position_from_prediction(p: float) -> int:
    if p > 0.0:
        return 1
    if p < 0.0:
        return -1
    return 0


def cumulative_return_from_series(returns: pd.Series) -> float:
    return float((1.0 + returns).prod() - 1.0)


def backtest_metrics(
    test_df: pd.DataFrame, predictions: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Align ``model.predict`` outputs with ``test_df`` rows; return cumulative strategy vs buy-hold."""
    if len(predictions) != len(test_df):
        raise ValueError("predictions length must match test_df rows")
    df_test = test_df.reset_index(drop=True).copy()
    df_test["model_prediction"] = [float(r["prediction"]) for r in predictions]
    df_test["position"] = df_test["model_prediction"].map(position_from_prediction)
    df_test["strategy_return"] = df_test["position"] * df_test["target_return"]
    strat = cumulative_return_from_series(df_test["strategy_return"])
    bh = cumulative_return_from_series(df_test["target_return"])
    return {"cumulative_strategy": strat, "cumulative_buy_hold": bh}


def fetch_spy_frame(period: str = "6y") -> pd.DataFrame:
    import yfinance as yf

    return yf.Ticker("SPY").history(period=period)


def main() -> None:
    reset_parser()
    ir = ast_to_ir(parse_ax_file(AX_PATH))
    abi = extract_global_abi(ir, max_vars=64)
    aw = extract_abi_widths(ir, max_vars=64)
    dim = max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)
    pred_col = abi["prediction"]
    block = InterpretedBlock(ir, abi, abi_widths=aw)

    raw = fetch_spy_frame("6y")
    df = add_spy_features(raw)
    train_df, test_df = chronological_split(df, TEST_TAIL)

    train_rows: List[Dict[str, float]] = train_df.to_dict(orient="records")
    for r in train_rows:
        for k, v in list(r.items()):
            if hasattr(v, "item"):
                r[k] = float(v.item())
            else:
                r[k] = float(v)

    ds = AxiomDataset(train_rows, abi, trunk_dim=dim, target_key="target_return", abi_widths=aw)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    opt = torch.optim.Adam(block.parameters(), lr=0.01)
    block.train()
    for epoch in range(50):
        ep = 0.0
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            out = block(x)
            pred = out[:, pred_col].unsqueeze(1)
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
            ep += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1} train_loss {ep / max(len(loader), 1):.6f}")

    block.eval()
    save_bundle(block, BUNDLE_PATH)
    print(f"Saved {BUNDLE_PATH}")

    print("\n--- RUNNING OUT-OF-SAMPLE BACKTEST ---\n")
    model = axiom.load(BUNDLE_PATH)
    results = model.predict(test_df)
    metrics = backtest_metrics(test_df, results)
    print(f"Cumulative strategy return (OOS): {metrics['cumulative_strategy']:.4f}")
    print(f"Cumulative buy-and-hold return (OOS): {metrics['cumulative_buy_hold']:.4f}")


if __name__ == "__main__":
    main()
