"""Pydantic request/response models for the copilot FastAPI server (Phase 67)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


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


class TrainTabularPayload(BaseModel):
    """In-process training eval for search (matches :func:`~axiom.copilot.tabular_json.parse_tabular_json_dict`)."""

    target_var: str = Field(min_length=1)
    train_rows: List[ExampleRow] = Field(min_length=1)
    eval_rows: List[ExampleRow] = Field(min_length=1)
    epochs: int = Field(default=30, ge=0, le=100_000)
    learning_rate: float = Field(default=0.01, ge=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    batch_size: int = Field(default=32, ge=1, le=8192)


class SearchRequest(BaseModel):
    goal: str
    domain_context: Optional[str] = None
    max_iterations: int = Field(default=8, ge=1, le=256)
    compile_only: bool = False
    examples: Optional[List[ExampleRow]] = None
    train_tabular: Optional[TrainTabularPayload] = None
    summarize_traces: bool = False
    artifact_dir: Optional[str] = None

    @model_validator(mode="after")
    def _train_tabular_exclusive(self) -> SearchRequest:
        if self.train_tabular is not None and self.examples:
            raise ValueError("Set either train_tabular or examples, not both.")
        if self.compile_only and self.train_tabular is not None:
            raise ValueError("compile_only cannot be true when train_tabular is set.")
        return self


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
