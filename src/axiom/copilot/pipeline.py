"""Opinionated NL→``.ax`` pipeline: search + optional artifact bundle + final compile check.

This is **goal to best Axiom source and reports** — not training, not ONNX export.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from axiom.copilot.artifacts import evaluation_report_to_dict, validation_report_to_dict
from axiom.copilot.evaluator import validate_program
from axiom.copilot.models import ProgramCandidate, ProgramValidationReport
from axiom.copilot.search import CopilotSearchConfig, CopilotSearchResult, run_copilot_search

PIPELINE_DISCLAIMER = (
    "Semantic copilot pipeline: produces .ax source and JSON reports only. "
    "Does not run axiom train, save .axb bundles, or export ONNX."
)


@dataclass
class CopilotPipelineConfig:
    """Inputs for :func:`run_copilot_search` plus pipeline-only outputs."""

    search: CopilotSearchConfig
    best_ax_path: Optional[Path] = None
    """Optional extra path for ``best`` source (``artifact_dir`` still writes ``best.ax`` when set)."""
    summary_json_path: Optional[Path] = None
    final_validate: bool = True
    """If True, run :func:`~axiom.copilot.evaluator.validate_program` on the champion source after search."""


@dataclass
class CopilotPipelineResult:
    search_result: CopilotSearchResult
    final_validation: Optional[ProgramValidationReport]
    artifact_dir: Optional[Path]


def copilot_pipeline_summary_dict(
    result: CopilotPipelineResult,
    *,
    artifact_dir_resolved: Optional[Path] = None,
    summarize_traces: bool = False,
) -> Dict[str, Any]:
    """JSON-serializable run summary (shared by CLI and HTTP)."""
    sr = result.search_result
    out: Dict[str, Any] = {
        "disclaimer": PIPELINE_DISCLAIMER,
        "converged": sr.converged,
        "best_source": sr.best_source,
        "best_evaluation": evaluation_report_to_dict(sr.best_evaluation),
        "final_evaluation": evaluation_report_to_dict(sr.final_report),
        "iterations": [
            {
                "index": rec.index,
                "source": rec.source,
                "evaluation": evaluation_report_to_dict(rec.evaluation),
                "producing_payload": dict(rec.producing_payload),
                "producing_expert": dict(rec.producing_expert),
                "outgoing_repair_error_report": rec.outgoing_repair_error_report,
                "semantic_trace_summary": rec.semantic_trace_summary,
            }
            for rec in sr.iterations
        ],
        "final_validation": None,
        "artifact_dir": str(artifact_dir_resolved) if artifact_dir_resolved is not None else None,
    }
    if summarize_traces:
        out["semantic_summaries"] = {
            "enabled": True,
            "per_iteration": [
                {"index": r.index, "semantic_trace_summary": r.semantic_trace_summary}
                for r in sr.iterations
            ],
        }
    if result.final_validation is not None:
        out["final_validation"] = validation_report_to_dict(result.final_validation)
    return out


def run_copilot_pipeline(cfg: CopilotPipelineConfig) -> CopilotPipelineResult:
    """Run search, persist artifacts when ``search.artifact_dir`` is set, optional extra ``best_ax_path``."""
    result = run_copilot_search(cfg.search)
    art = cfg.search.artifact_dir
    art_resolved = art.resolve() if art is not None else None

    if cfg.best_ax_path is not None:
        p = cfg.best_ax_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(result.best_source.rstrip() + "\n", encoding="utf-8")

    final_val: Optional[ProgramValidationReport] = None
    if cfg.final_validate:
        final_val = validate_program(
            ProgramCandidate(result.best_source),
            max_unroll=int(cfg.search.max_unroll),
        )

    return CopilotPipelineResult(
        search_result=result,
        final_validation=final_val,
        artifact_dir=art_resolved,
    )


__all__ = [
    "PIPELINE_DISCLAIMER",
    "CopilotPipelineConfig",
    "CopilotPipelineResult",
    "copilot_pipeline_summary_dict",
    "run_copilot_pipeline",
]
