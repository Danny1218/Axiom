"""Copilot harness: validate and evaluate in-memory ``.ax`` for semantic search / expert repair loops."""

from axiom.copilot.evaluator import evaluate_program, validate_program
from axiom.copilot.models import (
    EvaluationMode,
    ProgramCandidate,
    ProgramEvaluationReport,
    ProgramFailure,
    ProgramMetric,
    ProgramValidationReport,
)
from axiom.copilot.search import (
    CopilotIterationRecord,
    CopilotSearchConfig,
    CopilotSearchResult,
    build_draft_context,
    build_repair_context,
    build_repair_error_report,
    format_failures_for_repair,
    format_metrics_for_repair,
    run_copilot_search,
)

__all__ = [
    "CopilotIterationRecord",
    "CopilotSearchConfig",
    "CopilotSearchResult",
    "EvaluationMode",
    "ProgramCandidate",
    "ProgramEvaluationReport",
    "ProgramFailure",
    "ProgramMetric",
    "ProgramValidationReport",
    "build_draft_context",
    "build_repair_context",
    "build_repair_error_report",
    "evaluate_program",
    "format_failures_for_repair",
    "format_metrics_for_repair",
    "run_copilot_search",
    "validate_program",
]
