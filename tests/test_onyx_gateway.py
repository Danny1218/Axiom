"""Phase 49: enterprise_policy.ax + onyx_gateway (signals, symbolic gates, optional Onyx POST)."""

import importlib.util
import random
from pathlib import Path

import torch

from axiom.api import AxiomModel
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi, extract_neural_node_specs
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.inference import _batch_inputs_to_tensor


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_onyx_gateway():
    path = _root() / "examples" / "onyx_gateway.py"
    spec = importlib.util.spec_from_file_location("onyx_gateway", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_enterprise_policy_ax_structure():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "enterprise_policy.ax"))
    aw = extract_abi_widths(ir, max_vars=128)
    spec = extract_neural_node_specs(ir, aw)
    assert len(spec) == 1
    assert list(spec.values())[0][1] == "liquid"
    abi = extract_global_abi(ir, max_vars=128)
    assert abi.get("is_approved") is not None
    for k in ("has_pii_data", "mentions_competitor", "text_toxicity"):
        assert k in abi


def test_scan_text_ssn_and_competitors():
    gw = _load_onyx_gateway()
    rng = random.Random(0)
    s1 = gw.scan_text("My SSN is 123-45-6789 please help", rng=rng)
    assert s1["has_pii_data"] == 1.0
    s2 = gw.scan_text("nothing sensitive here", rng=rng)
    assert s2["has_pii_data"] == 0.0
    assert gw.scan_text("We use OpenAI", rng=rng)["mentions_competitor"] == 1.0
    assert gw.scan_text("OPENAI in caps", rng=rng)["mentions_competitor"] == 1.0
    assert gw.scan_text("Anthropic models", rng=rng)["mentions_competitor"] == 1.0
    assert 0.0 <= gw.scan_text("x", rng=rng)["text_toxicity"] <= 0.3


def test_symbolic_pii_and_competitor_force_deny():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "enterprise_policy.ax"))
    abi = extract_global_abi(ir, max_vars=128)
    aw = extract_abi_widths(ir, max_vars=128)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    dim = max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)
    m = AxiomModel(b)

    t_pii = m.explain({"has_pii_data": 1.0, "mentions_competitor": 0.0, "text_toxicity": 0.1})
    assert float(t_pii["is_approved"]) < 0.5

    t_comp = m.explain({"has_pii_data": 0.0, "mentions_competitor": 1.0, "text_toxicity": 0.1})
    assert float(t_comp["is_approved"]) < 0.5


def test_training_makes_clean_rows_more_approved_than_toxic_batch():
    gw = _load_onyx_gateway()
    model, block, _src = gw.build_trained_policy(epochs=40, lr=0.1, seed=7)
    abi, aw = block.abi, getattr(block, "abi_widths", {}) or {}
    dim = gw._trunk_dim(block)
    col = int(abi["is_approved"])
    device, dtype = torch.device("cpu"), torch.float32
    clean = [{"has_pii_data": 0.0, "mentions_competitor": 0.0, "text_toxicity": 0.1} for _ in range(8)]
    toxic = [{"has_pii_data": 0.0, "mentions_competitor": 0.0, "text_toxicity": 0.95} for _ in range(8)]
    with torch.no_grad():
        hc = _batch_inputs_to_tensor(clean, abi, dim, device=device, dtype=dtype, abi_widths=aw)
        ht = _batch_inputs_to_tensor(toxic, abi, dim, device=device, dtype=dtype, abi_widths=aw)
        mean_clean = float(block(hc)[:, col].mean().item())
        mean_toxic = float(block(ht)[:, col].mean().item())
    assert mean_clean > mean_toxic


def test_chat_blocked_exports_audit(tmp_path: Path):
    gw = _load_onyx_gateway()
    model, _, source = gw.build_trained_policy(epochs=15, lr=0.1, seed=1)
    audit = tmp_path / "blocked.html"
    out = gw.chat_with_onyx(
        model,
        source,
        "123-45-6789",
        audit_path=audit,
        text_rng=random.Random(0),
    )
    assert out is None
    assert audit.is_file()
    assert "html" in audit.read_text(encoding="utf-8").lower()


def test_chat_approved_custom_post():
    gw = _load_onyx_gateway()
    model, _, source = gw.build_trained_policy(epochs=15, lr=0.1, seed=2)

    def _post(_url: str, msg: str):
        class R:
            text = f"echo:{msg[:20]}"

        return R()

    out = gw.chat_with_onyx(
        model,
        source,
        "sort a list in Python",
        post_fn=_post,
        text_rng=random.Random(99),
    )
    assert out is not None
    assert "echo:" in out
