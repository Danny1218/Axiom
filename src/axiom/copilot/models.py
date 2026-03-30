"""Structured reports for copilot compile / validate / evaluate (semantic search, expert repair)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

EvaluationMode = Literal["compile_only", "predict_rows", "train_tabular"]


@dataclass(frozen=True)
class ProgramCandidate:
    """In-memory ``.ax`` source (no filesystem required)."""

    source: str
    id: Optional[str] = None


@dataclass(frozen=True)
class ProgramFailure:
    """Single structured failure (syntax, IR, block build, predict, …)."""

    stage: str
    kind: str
    message: str
    detail: Optional[str] = None


@dataclass(frozen=True)
class ProgramMetric:
    """Named scalar from a caller ``score_fn`` or harness."""

    name: str
    value: float


@dataclass
class ProgramValidationReport:
    """Compile-only outcome (parse → IR → ``InterpretedBlock``)."""

    success: bool
    source: str
    compile_stage_reached: str
    failures: List[ProgramFailure] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ProgramEvaluationReport:
    """Full evaluation: validation plus optional batched predict and metrics."""

    success: bool
    source: str
    compile_stage_reached: str
    mode: EvaluationMode
    failures: List[ProgramFailure] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    program_metrics: List[ProgramMetric] = field(default_factory=list)
    predictions_sample: Optional[List[Dict[str, Any]]] = None
    trace_snippet: Optional[Dict[str, Any]] = None
