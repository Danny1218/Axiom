"""FastAPI HTTP server for a single loaded ``.axb`` bundle (``axiom serve``).

This module's ``create_app`` is the **bundle inference** API (``/health``, ``/predict``, …). Do not
confuse with ``axiom.gateway.server:create_app``, which is the **policy gateway** factory for
uvicorn. External copilots may treat this as a headless ``AxiomModel`` over HTTP.
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Union

import torch.nn as nn
from fastapi import Depends, FastAPI, HTTPException, Request

from axiom.api import AxiomModel, load
from axiom.engine.expert_call import ExpertHandler, ExpertRuntimeError
from axiom.engine.expert_registry import (
    ExpertRuntimeRegistry,
    expert_runtime_wiring_sufficient,
    interpreted_block_ir_contains_expert,
)
from axiom.api_models import (
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReportRequest,
    ReportResponse,
)
from axiom.engine.strict import StrictInferenceError
from axiom.security.bundle_trust import (
    bundle_trust_from_env,
    report_output_dir_from_env,
    resolve_report_output_path,
)
from axiom.security.serve_policy import verify_serve_startup
from axiom.tools.html_exporter import render_html_report


def _expected_api_key() -> Optional[str]:
    k = os.environ.get("AXIOM_API_KEY", "").strip()
    return k or None


def _api_key_matches(provided: str, expected: str) -> bool:
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def verify_api_key(request: Request) -> None:
    """Require Authorization: Bearer <key> or X-API-Key when ``AXIOM_API_KEY`` is set."""
    expected = _expected_api_key()
    if not expected:
        return
    auth = request.headers.get("Authorization", "")
    x_key = request.headers.get("X-API-Key", "")
    ok = False
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        ok = _api_key_matches(token, expected)
    if x_key:
        ok = ok or _api_key_matches(x_key, expected)
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def get_model(request: Request) -> AxiomModel:
    return request.app.state.model


def _require_op_expert_wiring(model: AxiomModel) -> None:
    """Reject predict/explain/report when IR uses ``expert()`` but nothing can satisfy it at runtime."""
    if interpreted_block_ir_contains_expert(model.block) and not expert_runtime_wiring_sufficient(
        model.block
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Bundle uses expert() but no OP_EXPERT runtime wiring is configured. "
                "Pass expert_registry=..., expert_handler=..., or expert_fallback=... to create_app(), "
                "or call model.set_expert_registry(...) / set_expert_handler / set_expert_fallback after load."
            ),
        )


def _strict_from_env() -> bool:
    v = os.environ.get("AXIOM_STRICT", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _health_disclose_path() -> bool:
    v = os.environ.get("AXIOM_HEALTH_DISCLOSE_PATH", "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _health_bundle_path(resolved_path: str) -> str:
    """Return basename by default; full resolved path only when ``AXIOM_HEALTH_DISCLOSE_PATH=1``."""
    if _health_disclose_path():
        return resolved_path
    return Path(resolved_path).name


def _raise_inference_http(exc: BaseException) -> None:
    """Map predictable user/input/runtime errors to stable HTTP status codes."""
    if isinstance(exc, StrictInferenceError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, ExpertRuntimeError):
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if isinstance(exc, (TypeError, ValueError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


def create_app(
    bundle_path: str | Path,
    *,
    custom_neural_registry: Optional[Dict[str, nn.Module]] = None,
    expert_registry: Optional[Union[ExpertRuntimeRegistry, Mapping[str, ExpertHandler]]] = None,
    expert_handler: Optional[ExpertHandler] = None,
    expert_fallback: Optional[float] = None,
    trusted: Optional[bool] = None,
    strict: Optional[bool] = None,
    report_output_dir: Optional[str | Path] = None,
    allow_insecure: bool = False,
) -> FastAPI:
    """Load ``bundle_path`` once and return a FastAPI app serving ``/health``, ``/predict``, ``/explain``, ``/report``.

    Optional ``expert_*`` / ``custom_neural_registry`` mirror :func:`axiom.api.load` for bundles that use
    ``expert()`` or custom ``neural()`` modules.
    """
    path = Path(bundle_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Bundle not found: {path}")

    model = load(
        path,
        custom_neural_registry=custom_neural_registry,
        expert_registry=expert_registry,
        trusted=trusted if trusted is not None else bundle_trust_from_env(),
    )
    if strict is None:
        strict = _strict_from_env()
    model.strict = bool(strict)
    if os.environ.get("AXIOM_REQUIRE_API_KEY", "").strip().lower() in ("1", "true", "yes", "on") and not _expected_api_key():
        raise RuntimeError("AXIOM_REQUIRE_API_KEY is set but AXIOM_API_KEY is empty")
    sandbox = (
        Path(report_output_dir).expanduser().resolve()
        if report_output_dir
        else report_output_dir_from_env()
    )
    if expert_handler is not None:
        model.set_expert_handler(expert_handler)
    if expert_fallback is not None:
        model.set_expert_fallback(expert_fallback)
    app = FastAPI(title="Axiom Bundle Server", version="1.0")
    app.state.model = model
    app.state.bundle_path = str(path)
    app.state.report_output_dir = str(sandbox) if sandbox is not None else None

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            bundle_path=_health_bundle_path(app.state.bundle_path),
        )

    @app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
    def predict(
        body: PredictRequest,
        model_: AxiomModel = Depends(get_model),
    ) -> PredictResponse:
        _require_op_expert_wiring(model_)
        try:
            out = model_.predict(body.inputs)
        except (StrictInferenceError, ExpertRuntimeError, TypeError, ValueError) as e:
            _raise_inference_http(e)
        return PredictResponse(outputs=out)

    @app.post("/explain", response_model=ExplainResponse, dependencies=[Depends(verify_api_key)])
    def explain(
        body: ExplainRequest,
        model_: AxiomModel = Depends(get_model),
    ) -> ExplainResponse:
        _require_op_expert_wiring(model_)
        try:
            trace = model_.explain(body.inputs)
        except (StrictInferenceError, ExpertRuntimeError, TypeError, ValueError) as e:
            _raise_inference_http(e)
        return ExplainResponse(trace=trace)

    @app.post("/report", response_model=ReportResponse, dependencies=[Depends(verify_api_key)])
    def report(
        body: ReportRequest,
        request: Request,
        model_: AxiomModel = Depends(get_model),
    ) -> ReportResponse:
        _require_op_expert_wiring(model_)
        try:
            html = render_html_report(model_, body.inputs, body.source_code)
        except (StrictInferenceError, ExpertRuntimeError, TypeError, ValueError) as e:
            _raise_inference_http(e)
        if body.output_path:
            sandbox_raw = getattr(request.app.state, "report_output_dir", None)
            if not sandbox_raw:
                raise HTTPException(
                    status_code=400,
                    detail="output_path requires AXIOM_REPORT_OUTPUT_DIR or report_output_dir on create_app",
                )
            try:
                out = resolve_report_output_path(body.output_path, Path(sandbox_raw))
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html, encoding="utf-8")
            return ReportResponse(html=None, output_path=str(out.resolve()))
        return ReportResponse(html=html, output_path=None)

    return app
