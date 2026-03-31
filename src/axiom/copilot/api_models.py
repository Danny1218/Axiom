"""Pydantic request/response models for the copilot FastAPI server (Phase 67)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class CopilotHealthResponse(BaseModel):
    status: str = "ok"
    server: str = "axiom.copilot"


class ExampleRow(BaseModel):
    inputs: Dict[str, Any]
    expected: Dict[str, Any]


class DraftRequest(BaseModel):
    goal: str
    domain_context: Optional[str] = None


class DraftResponse(BaseModel):
    ax_source: str
    backend_name: str
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    goal: str
    domain_context: Optional[str] = None
    max_iterations: int = Field(default=8, ge=1, le=256)
    compile_only: bool = False
    examples: Optional[List[ExampleRow]] = None
    summarize_traces: bool = False
    artifact_dir: Optional[str] = None


class SearchResponse(BaseModel):
    converged: bool
    best_source: str
    best_evaluation: Dict[str, Any]
    final_evaluation: Dict[str, Any]
    iterations: List[Dict[str, Any]]


class SummarizeRequest(BaseModel):
    goal: str
    program: str
    trace: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)


class SummarizeResponse(BaseModel):
    summary: str
