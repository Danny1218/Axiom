"""Pydantic models for the FastAPI bundle server (``axiom serve``)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    bundle_path: str


class PredictRequest(BaseModel):
    """Single row dict or batch of row dicts (ABI feature names)."""

    inputs: Union[Dict[str, Any], List[Dict[str, Any]]] = Field(
        ...,
        description="One feature dict or a list of dicts for batch inference.",
    )


class PredictResponse(BaseModel):
    outputs: Union[Dict[str, Any], List[Dict[str, Any]]]


class ExplainRequest(BaseModel):
    inputs: Dict[str, Any]


class ExplainResponse(BaseModel):
    trace: Dict[str, Any]


class ReportRequest(BaseModel):
    inputs: Dict[str, Any]
    source_code: Optional[str] = None
    """Optional strategy `.ax` source for the HTML report."""

    output_path: Optional[str] = None
    """If set, write HTML to this path; if omitted, return HTML in the response body."""


class ReportResponse(BaseModel):
    html: Optional[str] = None
    output_path: Optional[str] = None
    """Absolute path written when ``output_path`` was provided in the request."""
