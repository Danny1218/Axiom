"""Typed request/response models and the semantic expert protocol.

This layer is for **external** tools (LLMs, hosted APIs) that author or comment on ``.ax``
source. It is intentionally separate from ``OP_NEURAL``, the compiler, and FastAPI.
In-program ``expert("name", features)`` (``OP_EXPERT``) is a distinct runtime hook on
``InterpretedBlock`` (``ExpertRuntimeRegistry`` / handlers — ``plan.md`` Phase 66 + 72), not this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable


@dataclass(frozen=True)
class ExpertDraftRequest:
    """Ask an expert to propose new ``.ax`` source from a natural-language goal."""

    goal: str
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExpertRepairRequest:
    """Ask an expert to fix ``.ax`` source given a structured or textual error report."""

    goal: str
    current_program: str
    error_report: str
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExpertTraceSummaryRequest:
    """Ask an expert to narrate a run trace (e.g. Glass Box / ``explain`` output)."""

    goal: str
    program: str
    trace: Mapping[str, Any]
    metrics: Mapping[str, Any]
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExpertDraftResponse:
    """Draft or repair result: ``.ax`` text plus provenance."""

    ax_source: str
    backend_name: str
    explanation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SemanticExpert(Protocol):
    """External semantic backend: draft, repair, and narrate traces (no network in core)."""

    def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
        """Return proposed ``.ax`` source for ``request.goal``."""

    def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
        """Return revised ``.ax`` given the current program and ``error_report``."""

    def summarize_trace(self, request: ExpertTraceSummaryRequest) -> str:
        """Return a human-readable summary of ``trace`` / ``metrics`` in light of ``goal``."""
