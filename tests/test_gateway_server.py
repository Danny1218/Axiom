"""Axiom gateway server (policy gate + optional downstream forward)."""

import importlib.util
import random
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from axiom.api import AxiomModel
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.engine.block_executor import InterpretedBlock
from axiom.gateway.core import (
    build_block_audit,
    default_scan_text,
    forward_to_downstream,
    is_approved,
    policy_explain,
    resolve_signals,
)


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_onyx_gateway():
    path = _root() / "examples" / "onyx_gateway.py"
    spec = importlib.util.spec_from_file_location("onyx_gateway_gw", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_scan_matches_ssn_and_competitors():
    rng = random.Random(0)
    assert default_scan_text("My SSN is 123-45-6789", rng=rng)["has_pii_data"] == 1.0
    assert default_scan_text("nothing", rng=rng)["has_pii_data"] == 0.0
    assert default_scan_text("We use OpenAI", rng=rng)["mentions_competitor"] == 1.0


def test_resolve_signals_uses_explicit_or_scan():
    assert resolve_signals("x", {"a": 1.0}) == {"a": 1.0}
    s = resolve_signals("no pii here", None, scan_fn=default_scan_text)
    assert "has_pii_data" in s


def test_policy_explain_and_approve_untrained_symbolic_deny():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "enterprise_policy.ax"))
    abi = extract_global_abi(ir, max_vars=128)
    aw = extract_abi_widths(ir, max_vars=128)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    m = AxiomModel(b)
    sig = {"has_pii_data": 1.0, "mentions_competitor": 0.0, "text_toxicity": 0.1}
    tr = policy_explain(m, sig)
    assert not is_approved(tr)


def test_forward_to_downstream_post_fn():
    seen: list[tuple[str, dict]] = []

    def _post(url: str, body: dict) -> dict:
        seen.append((url, body))
        return {"ok": True}

    out = forward_to_downstream(
        "http://example.test/chat",
        "hello",
        post_fn=_post,
    )
    assert out == {"ok": True}
    assert seen[0][0] == "http://example.test/chat"
    assert seen[0][1] == {"message": "hello"}


def test_build_block_audit_html_and_optional_path(tmp_path: Path):
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "enterprise_policy.ax"))
    abi = extract_global_abi(ir, max_vars=128)
    aw = extract_abi_widths(ir, max_vars=128)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    m = AxiomModel(b)
    src = (_root() / "examples" / "enterprise_policy.ax").read_text(encoding="utf-8")
    sig = {"has_pii_data": 0.0, "mentions_competitor": 0.0, "text_toxicity": 0.1}
    html, none_path = build_block_audit(m, sig, source_code=src, audit_path=None)
    assert "<!DOCTYPE html>" in html
    assert none_path is None
    out = tmp_path / "a.html"
    html2, p = build_block_audit(m, sig, source_code=src, audit_path=out)
    assert p == str(out.resolve())
    assert out.is_file()
    assert html2.startswith("<!DOCTYPE html>")


def test_gateway_chat_blocked_and_approved():
    from fastapi.testclient import TestClient

    from axiom.gateway.server import create_gateway_app

    gw = _load_onyx_gateway()
    model, _, src = gw.build_trained_policy(epochs=12, lr=0.1, seed=4)

    def _fwd(_url: str, body: dict) -> dict:
        return {"reply": f"ok:{body.get('message', '')[:10]}"}

    app = create_gateway_app(
        model,
        src,
        downstream_url="http://downstream.invalid/api",
        forward_post_fn=_fwd,
    )
    c = TestClient(app)
    r = c.post("/gateway/chat", json={"message": "123-45-6789"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "blocked"
    assert data.get("audit_html")
    r2 = c.post(
        "/gateway/chat",
        json={
            "message": "safe",
            "signals": {
                "has_pii_data": 0.0,
                "mentions_competitor": 0.0,
                "text_toxicity": 0.1,
            },
        },
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["status"] == "approved"
    assert "reply" in d2["downstream"]


def test_gateway_health():
    from fastapi.testclient import TestClient

    from axiom.gateway.server import create_gateway_app

    gw = _load_onyx_gateway()
    model, _, src = gw.build_trained_policy(epochs=5, lr=0.1, seed=0)
    app = create_gateway_app(model, src, downstream_url="http://x", forward_post_fn=lambda u, b: {})
    c = TestClient(app)
    assert c.get("/health").json()["mode"] == "gateway"


def test_cli_gateway_serve_help():
    from axiom.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["gateway-serve", "--help"])
    assert exc.value.code == 0
