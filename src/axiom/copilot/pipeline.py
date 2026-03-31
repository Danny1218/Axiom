"""Opinionated NL→``.ax`` pipeline: search + optional artifact bundle + final compile check.

This is **goal to best Axiom source and reports** — not training, not ONNX export.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from axiom.copilot.artifacts import evaluation_report_to_dict, validation_report_to_dict
from axiom.copilot.evaluator import validate_program
from axiom.copilot.models import ProgramCandidate, ProgramValidationReport
from axiom.copilot.search import CopilotSearchConfig, CopilotSearchResult, _is_better, run_copilot_search

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
    restarts: int = 1
    """Run this many independent searches; keep the overall best by :func:`~axiom.copilot.search._is_better`. ``1`` = unchanged."""


@dataclass
class CopilotPipelineResult:
    search_result: CopilotSearchResult
    final_validation: Optional[ProgramValidationReport]
    artifact_dir: Optional[Path]
    restarts: int = 1
    winning_restart_index: int = 0
    # One summary dict per restart (index, converged, evaluations, …).
    per_restart: List[Dict[str, Any]] = field(default_factory=list)


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
        "convergence_reason": sr.convergence_reason,
        "metric_repair": {
            "enabled": sr.metric_repair_enabled,
            "threshold_effective": sr.metric_repair_threshold_effective,
        },
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
        "restarts": {
            "total": result.restarts,
            "winning_index": result.winning_restart_index,
            "per_restart": list(result.per_restart),
        },
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
    """Run one or more independent searches; pick overall best; optional artifacts and final validation."""
    n = max(1, int(cfg.restarts))
    base_art = cfg.search.artifact_dir
    sort_key = cfg.search.score_sort_key

    best_result: Optional[CopilotSearchResult] = None
    winning_idx = 0
    per_restart: List[Dict[str, Any]] = []
    art_resolved = base_art.resolve() if base_art is not None else None

    for r in range(n):
        art_dir: Optional[Path] = None
        if base_art is not None:
            art_dir = base_art / f"restart_{r}" if n > 1 else base_art
        scfg = replace(cfg.search, artifact_dir=art_dir)
        result = run_copilot_search(scfg)
        entry: Dict[str, Any] = {
            "index": r,
            "converged": result.converged,
            "convergence_reason": result.convergence_reason,
            "iteration_count": len(result.iterations),
            "best_source": result.best_source,
            "best_evaluation": evaluation_report_to_dict(result.best_evaluation),
            "final_evaluation": evaluation_report_to_dict(result.final_report),
            "metric_repair": {
                "enabled": result.metric_repair_enabled,
                "threshold_effective": result.metric_repair_threshold_effective,
            },
        }
        if art_dir is not None:
            entry["artifact_subdir"] = str(art_dir.resolve())
        per_restart.append(entry)
        if best_result is None or _is_better(result.best_evaluation, best_result.best_evaluation, sort_key):
            best_result = result
            winning_idx = r

    assert best_result is not None

    if cfg.best_ax_path is not None:
        p = cfg.best_ax_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(best_result.best_source.rstrip() + "\n", encoding="utf-8")

    final_val: Optional[ProgramValidationReport] = None
    if cfg.final_validate:
        final_val = validate_program(
            ProgramCandidate(best_result.best_source),
            max_unroll=int(cfg.search.max_unroll),
        )

    return CopilotPipelineResult(
        search_result=best_result,
        final_validation=final_val,
        artifact_dir=art_resolved,
        restarts=n,
        winning_restart_index=winning_idx,
        per_restart=per_restart,
    )


__all__ = [
    "PIPELINE_DISCLAIMER",
    "CopilotPipelineConfig",
    "CopilotPipelineResult",
    "copilot_pipeline_summary_dict",
    "run_copilot_pipeline",
]
