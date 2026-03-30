"""HTTP gateway: policy bundle gate + optional downstream forward."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from axiom.api import AxiomModel, load
from axiom.gateway.core import (
    build_block_audit,
    default_scan_text,
    forward_to_downstream,
    is_approved,
    policy_explain,
    resolve_signals,
)
from axiom.serve import verify_api_key


class GatewayChatIn(BaseModel):
    message: str = Field(..., min_length=1)
    signals: Optional[Dict[str, float]] = None


class GatewayBlocked(BaseModel):
    status: Literal["blocked"] = "blocked"
    signals: Dict[str, float]
    trace: Dict[str, Any]
    reason: str = "policy_denied"
    audit_html: Optional[str] = None
    audit_path: Optional[str] = None


class GatewayApproved(BaseModel):
    status: Literal["approved"] = "approved"
    signals: Dict[str, float]
    trace: Dict[str, Any]
    downstream: Any


def create_gateway_app(
    model: AxiomModel,
    policy_source: str | None,
    *,
    downstream_url: str,
    approve_threshold: float = 0.5,
    audit_path_on_block: str | Path | None = None,
    forward_post_fn: Callable[[str, Dict[str, Any]], Any] | None = None,
    scan_fn: Callable[[str], Dict[str, float]] | None = None,
) -> FastAPI:
    """FastAPI app with ``POST /gateway/chat``.

    * **Blocked:** JSON with ``audit_html`` and optional ``audit_path`` when a file was written.
    * **Approved:** JSON with ``downstream`` payload from :func:`axiom.gateway.core.forward_to_downstream`.
    """
    app = FastAPI(title="Axiom Gateway", version="1.0")
    app.state.model = model
    app.state.policy_source = policy_source
    app.state.downstream_url = downstream_url
    app.state.approve_threshold = float(approve_threshold)
    app.state.audit_path_on_block = str(audit_path_on_block) if audit_path_on_block else None
    app.state.forward_post_fn = forward_post_fn
    app.state.scan_fn = scan_fn or default_scan_text

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "mode": "gateway"}

    @app.post("/gateway/chat")
    def gateway_chat(
        body: GatewayChatIn,
        _auth: None = Depends(verify_api_key),
    ) -> GatewayBlocked | GatewayApproved:
        m: AxiomModel = app.state.model
        sig = resolve_signals(body.message, body.signals, scan_fn=app.state.scan_fn)
        trace = policy_explain(m, sig)
        if not is_approved(trace, threshold=app.state.approve_threshold):
            html, path = build_block_audit(
                m,
                sig,
                source_code=app.state.policy_source,
                audit_path=app.state.audit_path_on_block,
            )
            return GatewayBlocked(
                signals=sig,
                trace=trace,
                audit_html=html,
                audit_path=path,
            )
        downstream = forward_to_downstream(
            app.state.downstream_url,
            body.message,
            post_fn=app.state.forward_post_fn,
        )
        return GatewayApproved(signals=sig, trace=trace, downstream=downstream)

    return app


def create_app() -> FastAPI:
    """Uvicorn factory: ``uvicorn axiom.gateway.server:create_app --factory`` (reads env)."""
    return gateway_app_from_env()


def gateway_app_from_env() -> FastAPI:
    """Build :func:`create_gateway_app` from environment (for ``uvicorn`` CLI)."""
    bundle = os.environ.get("AXIOM_GATEWAY_BUNDLE", "").strip()
    src = os.environ.get("AXIOM_GATEWAY_POLICY_SOURCE", "").strip()
    url = os.environ.get("AXIOM_GATEWAY_DOWNSTREAM_URL", "").strip()
    if not bundle or not Path(bundle).is_file():
        raise RuntimeError("Set AXIOM_GATEWAY_BUNDLE to a .axb file.")
    if not url:
        raise RuntimeError("Set AXIOM_GATEWAY_DOWNSTREAM_URL to the downstream HTTP endpoint.")
    policy_text: str | None = None
    if src:
        p = Path(src)
        if p.is_file():
            policy_text = p.read_text(encoding="utf-8")
    thr = float(os.environ.get("AXIOM_GATEWAY_APPROVE_THRESHOLD", "0.5"))
    audit = os.environ.get("AXIOM_GATEWAY_AUDIT_PATH", "").strip() or None
    model = load(bundle)
    return create_gateway_app(
        model,
        policy_text,
        downstream_url=url,
        approve_threshold=thr,
        audit_path_on_block=audit,
    )
