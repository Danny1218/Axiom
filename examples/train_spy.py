"""
Live SPY features, train ``spy_alpha.ax`` (neural alpha + volatility circuit breaker), backtest via ``axiom.load``.

Requires: ``pip install yfinance pandas``

Run from repo root: ``python examples/train_spy.py``
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import axiom
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi, extract_neural_node_specs
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.compiler.serializer import save_bundle
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.dataloader import AxiomDataset

_EXAMPLES = Path(__file__).resolve().parent
AX_PATH = _EXAMPLES / "spy_alpha.ax"
BUNDLE_PATH = _EXAMPLES / "spy_trained.axb"
TEST_TAIL = 500


def make_spy_alpha_custom_brain() -> nn.Module:
    return nn.Sequential(
        nn.Linear(6, 32),
        nn.GELU(),
        nn.Dropout(0.2),
        nn.Linear(32, 16),
        nn.GELU(),
        nn.Linear(16, 1),
    )


def add_spy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum, volatility, SMA divergence, 20d vol, next-day target; expects yfinance OHLCV names."""
    out = df.copy()
    out["momentum_1d"] = out["Close"].pct_change(1)
    out["momentum_5d"] = out["Close"].pct_change(5)
    out["volatility"] = (out["High"] - out["Low"]) / out["Open"]
    out["sma_10"] = out["Close"].rolling(window=10).mean() / out["Close"] - 1.0
    out["sma_50"] = out["Close"].rolling(window=50).mean() / out["Close"] - 1.0
    out["volatility_20d"] = out["Close"].pct_change().rolling(window=20).std()
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


def annualized_sharpe(returns: pd.Series, periods_per_year: float = 252.0) -> float:
    m = returns.mean()
    s = returns.std()
    if s == 0 or pd.isna(s) or len(returns) < 2:
        return float("nan")
    return float((m / s) * math.sqrt(periods_per_year))


def max_drawdown_from_returns(returns: pd.Series) -> float:
    """Largest peak-to-trough drop on the equity curve (1+r).cumprod(); negative fraction."""
    equity = (1.0 + returns).cumprod()
    peak = equity.cummax()
    dd = (equity / peak) - 1.0
    return float(dd.min())


def backtest_metrics(
    test_df: pd.DataFrame, predictions: List[Dict[str, Any]]
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Align ``model.predict`` outputs with ``test_df`` rows; return metrics and augmented OOS frame."""
    if len(predictions) != len(test_df):
        raise ValueError("predictions length must match test_df rows")
    df_test = test_df.copy()
    df_test["model_prediction"] = [float(r["prediction"]) for r in predictions]
    df_test["position"] = df_test["model_prediction"].map(position_from_prediction)
    df_test["strategy_return"] = df_test["position"] * df_test["target_return"]
    strat = cumulative_return_from_series(df_test["strategy_return"])
    bh = cumulative_return_from_series(df_test["target_return"])
    sh_s = annualized_sharpe(df_test["strategy_return"])
    sh_bh = annualized_sharpe(df_test["target_return"])
    mdd_s = max_drawdown_from_returns(df_test["strategy_return"])
    mdd_bh = max_drawdown_from_returns(df_test["target_return"])
    return (
        {
            "cumulative_strategy": strat,
            "cumulative_buy_hold": bh,
            "sharpe_strategy": sh_s,
            "sharpe_buy_hold": sh_bh,
            "max_drawdown_strategy": mdd_s,
            "max_drawdown_buy_hold": mdd_bh,
        },
        df_test,
    )


def row_series_to_plain_dict(row: "pd.Series") -> Dict[str, Any]:
    """Scalar floats for ``AxiomModel.explain`` (one trading row)."""
    out: Dict[str, Any] = {}
    for k in row.index:
        v = row[k]
        if hasattr(v, "item"):
            out[str(k)] = float(v.item())
        elif isinstance(v, (float, int)):
            out[str(k)] = float(v)
        else:
            out[str(k)] = float(v)
    return out


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

    probe = InterpretedBlock(ir, abi, abi_widths=aw)
    neural_keys = list(probe.neural_registry.keys())
    print("Neural node id(s):", neural_keys)
    del probe
    nid = neural_keys[0]
    spec = extract_neural_node_specs(ir, aw)
    w0, _arch0 = spec.get(nid, (0, "mlp"))
    if w0 != 6:
        raise RuntimeError(f"expected neural input width 6, got {w0}")
    custom_brain = make_spy_alpha_custom_brain()
    block = InterpretedBlock(
        ir, abi, abi_widths=aw, custom_neural_registry={nid: custom_brain}
    )

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
    model = axiom.load(BUNDLE_PATH, custom_neural_registry={nid: make_spy_alpha_custom_brain()})
    results = model.predict(test_df)
    metrics, df_oos = backtest_metrics(test_df, results)
    print(f"Cumulative strategy return (OOS): {metrics['cumulative_strategy']:.4f}")
    print(f"Cumulative buy-and-hold return (OOS): {metrics['cumulative_buy_hold']:.4f}")
    print(f"Sharpe ratio (strategy, ann.): {metrics['sharpe_strategy']:.4f}")
    print(f"Sharpe ratio (buy-and-hold, ann.): {metrics['sharpe_buy_hold']:.4f}")
    print(f"Max drawdown (strategy): {metrics['max_drawdown_strategy']*100:.2f}%")
    print(f"Max drawdown (buy-and-hold): {metrics['max_drawdown_buy_hold']*100:.2f}%")

    worst_idx = df_oos["strategy_return"].idxmin()
    worst_row = df_oos.loc[worst_idx]
    worst_dict = row_series_to_plain_dict(worst_row)
    trace = model.explain(worst_dict)
    print("\n--- THE AUTOPSY (WORST TRADE) ---\n")
    dname = worst_row.name
    if hasattr(dname, "strftime"):
        dstr = dname.strftime("%Y-%m-%d")
    else:
        dstr = str(dname)
    print(f"Date: {dstr}")
    print(f"Market return that day (target_return): {float(worst_row['target_return'])*100:.2f}%")
    print(f"Strategy return that day: {float(worst_row['strategy_return'])*100:.2f}%")
    print("AI internal trace:")
    print(json.dumps(trace, indent=2))

    source = AX_PATH.read_text(encoding="utf-8")
    report_path = _EXAMPLES / "worst_trade_report.html"
    model.export_report(worst_dict, str(report_path.resolve()), source_code=source)
    print(
        f"\nGlass Box Report generated at {report_path}. Open it in your browser!"
    )


if __name__ == "__main__":
    main()
