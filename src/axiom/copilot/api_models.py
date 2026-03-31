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
    repair_valid_with_metrics: Optional[bool] = None
    """``None``: auto — enabled for ``examples`` / ``train_tabular`` modes; ``False`` / ``True`` override."""
    metric_repair_if_below: Optional[float] = None
    """Repair while the score sort key is below this; unset uses built-in default for ``neg_mse`` (see ``plan.md``)."""
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    """Optional OpenAI-style sampling temperature for expert draft/repair (Onyx backend)."""
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    """Optional nucleus sampling ``top_p`` for expert draft/repair (Onyx backend)."""

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
    convergence_reason: str = ""
    metric_repair_enabled: bool = False
    metric_repair_threshold_effective: Optional[float] = None


class CopilotRunRequest(SearchRequest):
    """Same evaluation surface as :class:`SearchRequest` plus optional final compile-only pass."""

    final_validate: bool = True
    restarts: int = Field(default=1, ge=1, le=64)
    """Run this many independent full searches; pipeline keeps the overall best (Phase 80)."""


class CopilotRunResponse(BaseModel):
    """Pipeline output: search traces + optional ``final_validation`` + honesty ``disclaimer``."""

    disclaimer: str
    converged: bool
    best_source: str
    best_evaluation: Dict[str, Any]
    final_evaluation: Dict[str, Any]
    iterations: List[Dict[str, Any]]
    convergence_reason: str = ""
    metric_repair_enabled: bool = False
    metric_repair_threshold_effective: Optional[float] = None
    final_validation: Optional[Dict[str, Any]] = None
    semantic_summaries: Optional[Dict[str, Any]] = None
    artifact_dir: Optional[str] = None
    restarts_total: int = 1
    winning_restart_index: int = 0
    per_restart_summaries: List[Dict[str, Any]] = Field(default_factory=list)


class SummarizeRequest(BaseModel):
    goal: str
    program: str
    trace: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)


class SummarizeResponse(BaseModel):
    summary: str


class BenchmarkRunRequest(BaseModel):
    """Optional task suite override (same shape as ``benchmark_tasks_from_json_dict`` file root)."""

    tasks: Optional[Dict[str, Any]] = None
    max_iterations: int = Field(default=4, ge=1, le=64)
    draft_only: bool = False
    search_only: bool = False

    @model_validator(mode="after")
    def _draft_search_exclusive(self) -> BenchmarkRunRequest:
        if self.draft_only and self.search_only:
            raise ValueError("draft_only and search_only cannot both be true.")
        return self


class BenchmarkRunResponse(BaseModel):
    """Full output of :func:`~axiom.copilot.benchmarks.benchmark_suite_to_dict`."""

    suite: Dict[str, Any]
