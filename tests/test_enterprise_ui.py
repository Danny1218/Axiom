"""Phase 50: enterprise_ui Streamlit app — helpers, source contract, gateway wiring."""

import importlib.util
import random
import sys
from pathlib import Path
from unittest.mock import MagicMock

from axiom.api import AxiomModel
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.inference import _batch_inputs_to_tensor


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_enterprise_ui_module():
    """Load enterprise_ui with Streamlit mocked so CI never opens a server."""
    ex = str(_root() / "examples")
    if ex not in sys.path:
        sys.path.insert(0, ex)
    path = _root() / "examples" / "enterprise_ui.py"
    mock_st = MagicMock()
    mock_st.set_page_config = MagicMock()

    def _cache_resource(*_a, **_kw):
        def _wrap(f):
            return f

        return _wrap

    mock_st.cache_resource = _cache_resource
    sys.modules["streamlit"] = mock_st
    spec = importlib.util.spec_from_file_location("enterprise_ui", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, mock_st


def test_telemetry_row_scales():
    mod, _ = _load_enterprise_ui_module()
    sig = {"has_pii_data": 1.0, "mentions_competitor": 0.0, "text_toxicity": 0.15}
    tr = {"intent_risk": 0.5, "is_approved": 0.0}
    row = mod.telemetry_row(sig, tr)
    assert row["pii"] == 1.0
    assert row["competitor"] == 0.0
    assert abs(row["toxicity"] - 0.5) < 1e-6
    assert 0.0 <= row["intent_risk_bar"] <= 1.0
    assert row["approved"] == 0.0


def test_enterprise_ui_source_contract():
    text = (_root() / "examples" / "enterprise_ui.py").read_text(encoding="utf-8")
    assert "st.chat_input" in text
    assert "st.cache_resource" in text
    assert "st.download_button" in text
    assert "build_trained_policy" in text
    assert "scan_text" in text
    assert "live_audit.html" in text
    assert "export_report" in text


def test_blocked_path_writes_live_audit(tmp_path: Path):
    """Mirror UI blocked branch: export_report creates HTML."""
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "enterprise_policy.ax"))
    abi = extract_global_abi(ir, max_vars=128)
    aw = extract_abi_widths(ir, max_vars=128)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    model = AxiomModel(b)
    audit = tmp_path / "live_audit.html"
    src = (_root() / "examples" / "enterprise_policy.ax").read_text(encoding="utf-8")
    signals = {"has_pii_data": 1.0, "mentions_competitor": 0.0, "text_toxicity": 0.1}
    model.export_report(signals, str(audit.resolve()), source_code=src)
    assert audit.is_file()
    body = audit.read_text(encoding="utf-8").lower()
    assert "html" in body


def test_chat_with_onyx_verbose_false_no_prints(capsys):
    import importlib.util

    gw_path = _root() / "examples" / "onyx_gateway.py"
    spec = importlib.util.spec_from_file_location("onyx_gateway_t", gw_path)
    assert spec and spec.loader
    gw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gw)
    model, _, src = gw.build_trained_policy(epochs=5, lr=0.1, seed=3)
    gw.chat_with_onyx(model, src, "safe text", verbose=False, text_rng=random.Random(1))
    out = capsys.readouterr().out
    assert "[signals]" not in out
    assert "[trace]" not in out
