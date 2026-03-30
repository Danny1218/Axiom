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

__all__ = [
    "EvaluationMode",
    "ProgramCandidate",
    "ProgramEvaluationReport",
    "ProgramFailure",
    "ProgramMetric",
    "ProgramValidationReport",
    "evaluate_program",
    "validate_program",
]
