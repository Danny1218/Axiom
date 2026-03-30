"""FastAPI HTTP server for a single loaded ``.axb`` bundle (``axiom serve``).

This module's ``create_app`` is the **bundle inference** API (``/health``, ``/predict``, …). Do not
confuse with ``axiom.gateway.server:create_app``, which is the **policy gateway** factory for
uvicorn. External copilots may treat this as a headless ``AxiomModel`` over HTTP.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request

from axiom.api import AxiomModel, load
from axiom.api_models import (
    ExplainRequest,
    ExplainResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReportRequest,
    ReportResponse,
)
from axiom.tools.html_exporter import render_html_report


def _expected_api_key() -> Optional[str]:
    k = os.environ.get("AXIOM_API_KEY", "").strip()
    return k or None


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
        ok = token == expected
    if x_key == expected:
        ok = True
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def get_model(request: Request) -> AxiomModel:
    return request.app.state.model


def create_app(bundle_path: str | Path) -> FastAPI:
    """Load ``bundle_path`` once and return a FastAPI app serving ``/health``, ``/predict``, ``/explain``, ``/report``."""
    path = Path(bundle_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Bundle not found: {path}")

    model = load(path)
    app = FastAPI(title="Axiom Bundle Server", version="1.0")
    app.state.model = model
    app.state.bundle_path = str(path)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", bundle_path=app.state.bundle_path)

    @app.post("/predict", response_model=PredictResponse, dependencies=[Depends(verify_api_key)])
    def predict(
        body: PredictRequest,
        model_: AxiomModel = Depends(get_model),
    ) -> PredictResponse:
        out = model_.predict(body.inputs)
        return PredictResponse(outputs=out)

    @app.post("/explain", response_model=ExplainResponse, dependencies=[Depends(verify_api_key)])
    def explain(
        body: ExplainRequest,
        model_: AxiomModel = Depends(get_model),
    ) -> ExplainResponse:
        trace = model_.explain(body.inputs)
        return ExplainResponse(trace=trace)

    @app.post("/report", response_model=ReportResponse, dependencies=[Depends(verify_api_key)])
    def report(
        body: ReportRequest,
        model_: AxiomModel = Depends(get_model),
    ) -> ReportResponse:
        html = render_html_report(model_, body.inputs, body.source_code)
        if body.output_path:
            out = Path(body.output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(html, encoding="utf-8")
            return ReportResponse(html=None, output_path=str(out.resolve()))
        return ReportResponse(html=html, output_path=None)

    return app
