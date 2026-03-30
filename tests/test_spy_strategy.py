"""SPY neuro-symbolic example: ``spy_alpha.ax``, ``train_spy`` helpers, bundle + ``axiom.load`` backtest."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import axiom
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.compiler.serializer import save_bundle
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.dataloader import AxiomDataset
from axiom.engine.inference import _inputs_to_tensor


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _train_spy_module():
    pytest.importorskip("pandas")
    p = _root() / "examples" / "train_spy.py"
    spec = importlib.util.spec_from_file_location("train_spy_mod", p)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_spy_alpha_ax_exists_and_parses():
    ax_path = _root() / "examples" / "spy_alpha.ax"
    assert ax_path.is_file()
    reset_parser()
    ir = ast_to_ir(parse_ax_file(ax_path))
    assert any(s[0] == "OP_CONDITIONAL" for s in ir)
    assert any("OP_NEURAL" in str(s) for s in ir)
    abi = extract_global_abi(ir, max_vars=64)
    assert abi["prediction"] >= 0 and "volatility" in abi


def test_volatility_circuit_breaker_forces_zero_prediction():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "spy_alpha.ax"))
    abi = extract_global_abi(ir, max_vars=64)
    aw = extract_abi_widths(ir, max_vars=64)
    dim = max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    block.eval()
    row = {"momentum_1d": 0.01, "momentum_5d": 0.02, "volatility": 0.05}
    h = _inputs_to_tensor(
        row, abi, dim, device=torch.device("cpu"), dtype=torch.float32, abi_widths=aw
    )
    with torch.no_grad():
        out = block(h)
    from axiom.engine.inference import _abi_outputs_from_trunk_row

    dec = _abi_outputs_from_trunk_row(out[0], abi, aw)
    assert dec["prediction"] == pytest.approx(0.0)


def test_add_spy_features_and_chronological_split():
    pd = pytest.importorskip("pandas")
    ts = _train_spy_module()
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0] * 20,
            "High": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0] * 20,
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0] * 20,
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5] * 20,
        }
    )
    df = ts.add_spy_features(raw)
    assert "momentum_1d" in df.columns and "target_return" in df.columns
    tr, te = ts.chronological_split(df, test_rows=10)
    assert len(te) == 10 and len(tr) == len(df) - 10


def test_backtest_metrics_matches_manual():
    pd = pytest.importorskip("pandas")
    ts = _train_spy_module()
    test_df = pd.DataFrame(
        {
            "target_return": [0.01, -0.02, 0.03],
        }
    )
    preds = [
        {"prediction": 1.0},
        {"prediction": -1.0},
        {"prediction": 0.0},
    ]
    m = ts.backtest_metrics(test_df, preds)
    # pos 1, -1, 0 -> strategy returns = target * [1,-1,0]
    strat = (1 + 0.01) * (1 + 0.02) * (1 + 0.0) - 1.0
    bh = (1 + 0.01) * (1 - 0.02) * (1 + 0.03) - 1.0
    assert m["cumulative_strategy"] == pytest.approx(strat)
    assert m["cumulative_buy_hold"] == pytest.approx(bh)


def test_mini_train_save_load_predict_backtest(tmp_path):
    pd = pytest.importorskip("pandas")
    ts = _train_spy_module()
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "spy_alpha.ax"))
    abi = extract_global_abi(ir, max_vars=64)
    aw = extract_abi_widths(ir, max_vars=64)
    dim = max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)
    pred_col = abi["prediction"]
    block = InterpretedBlock(ir, abi, abi_widths=aw)

    raw = pd.DataFrame(
        {
            "Open": [100.0 + i * 0.1 for i in range(120)],
            "High": [101.0 + i * 0.1 for i in range(120)],
            "Low": [99.0 + i * 0.1 for i in range(120)],
            "Close": [100.2 + i * 0.11 for i in range(120)],
        }
    )
    df = ts.add_spy_features(raw)
    train_df, test_df = ts.chronological_split(df, test_rows=25)
    train_rows = train_df.to_dict(orient="records")
    for r in train_rows:
        for k, v in list(r.items()):
            r[k] = float(v.item()) if hasattr(v, "item") else float(v)

    ds = AxiomDataset(
        train_rows, abi, trunk_dim=dim, target_key="target_return", abi_widths=aw
    )
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    opt = torch.optim.Adam(block.parameters(), lr=0.01)
    block.train()
    for _ in range(3):
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            out = block(x)
            loss = F.mse_loss(out[:, pred_col].unsqueeze(1), y)
            loss.backward()
            opt.step()

    axb = tmp_path / "mini.axb"
    block.eval()
    save_bundle(block, axb)
    model = axiom.load(axb)
    results = model.predict(test_df)
    assert len(results) == len(test_df)
    m = ts.backtest_metrics(test_df, results)
    assert "cumulative_strategy" in m and "cumulative_buy_hold" in m
    assert all(abs(m[k]) < 10.0 for k in m)


@pytest.mark.integration
def test_yfinance_spy_fetch_smoke():
    pytest.importorskip("yfinance")
    ts = _train_spy_module()
    df = ts.fetch_spy_frame("5d")
    assert len(df) >= 1 and "Close" in df.columns
