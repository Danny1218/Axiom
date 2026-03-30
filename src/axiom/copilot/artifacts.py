"""Persist copilot search runs to disk (reproducible, JSON-serializable experiment traces).

Layout when :func:`persist_copilot_artifacts` is used (fixed filenames):

- ``best.ax`` — champion program source.
- ``iterations.json`` — per-iteration candidates, eval outcomes, expert metadata.
- ``search_report.json`` — run header, best/final evaluation blobs, failures/metrics summary.

Only call persistence when an output directory is explicitly provided (no writes otherwise).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from axiom.copilot.models import ProgramEvaluationReport
from axiom.copilot.search import CopilotIterationRecord, CopilotSearchConfig, CopilotSearchResult
from axiom.experts.base import ExpertDraftResponse

COPILOT_ARTIFACT_SCHEMA_VERSION = 1

BEST_AX_NAME = "best.ax"
ITERATIONS_JSON_NAME = "iterations.json"
SEARCH_REPORT_JSON_NAME = "search_report.json"


def json_safe(obj: Any) -> Any:
    """Best-effort JSON-serializable view (unknown types → str)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(x) for x in obj]
    return str(obj)


def expert_response_to_dict(resp: ExpertDraftResponse, expert_call: str) -> Dict[str, Any]:
    """Structured expert round-trip metadata (for experiment traces)."""
    return {
        "expert_call": expert_call,
        "backend_name": resp.backend_name,
        "metadata": json_safe(dict(resp.metadata)),
        "explanation": resp.explanation,
    }


def evaluation_report_to_dict(rep: ProgramEvaluationReport) -> Dict[str, Any]:
    """Stable :class:`ProgramEvaluationReport` JSON shape (shared with CLI reporting)."""
    return {
        "success": rep.success,
        "source": rep.source,
        "compile_stage_reached": rep.compile_stage_reached,
        "mode": rep.mode,
        "failures": [
            {"stage": f.stage, "kind": f.kind, "message": f.message, "detail": f.detail}
            for f in rep.failures
        ],
        "warnings": list(rep.warnings),
        "metrics": dict(rep.metrics),
        "program_metrics": [{"name": m.name, "value": m.value} for m in rep.program_metrics],
    }


def _failure_summaries(rep: ProgramEvaluationReport) -> List[Dict[str, Any]]:
    return [
        {"stage": f.stage, "kind": f.kind, "message": f.message, "detail": f.detail}
        for f in rep.failures
    ]


def iteration_entry_to_dict(rec: CopilotIterationRecord) -> Dict[str, Any]:
    ev = rec.evaluation
    return {
        "index": rec.index,
        "candidate_source": rec.source,
        "success": ev.success,
        "compile_stage_reached": ev.compile_stage_reached,
        "metrics": dict(ev.metrics),
        "program_metrics": [{"name": m.name, "value": m.value} for m in ev.program_metrics],
        "failure_summaries": _failure_summaries(ev),
        "warnings": list(ev.warnings),
        "producing_expert": dict(rec.producing_expert),
        "producing_payload": dict(rec.producing_payload),
        "outgoing_repair_error_report": rec.outgoing_repair_error_report,
        "semantic_trace_summary": rec.semantic_trace_summary,
    }


def _backend_name_from_result(result: CopilotSearchResult) -> str:
    if not result.iterations:
        return "unknown"
    name = result.iterations[0].producing_expert.get("backend_name")
    return str(name) if name else "unknown"


def build_iterations_document(config: CopilotSearchConfig, result: CopilotSearchResult) -> Dict[str, Any]:
    return {
        "schema_version": COPILOT_ARTIFACT_SCHEMA_VERSION,
        "kind": "axiom.copilot.iterations",
        "goal": config.goal,
        "domain_context": config.domain_context,
        "backend_name": _backend_name_from_result(result),
        "evaluation_mode": config.mode,
        "max_iterations_config": config.max_iterations,
        "iteration_count": len(result.iterations),
        "score_sort_key": config.score_sort_key,
        "iterations": [iteration_entry_to_dict(rec) for rec in result.iterations],
    }


def build_search_report_document(config: CopilotSearchConfig, result: CopilotSearchResult) -> Dict[str, Any]:
    per_iter = []
    for rec in result.iterations:
        ev = rec.evaluation
        per_iter.append(
            {
                "index": rec.index,
                "success": ev.success,
                "metrics": dict(ev.metrics),
                "failure_count": len(ev.failures),
                "failure_summaries": _failure_summaries(ev),
            }
        )
    be, fe = result.best_evaluation, result.final_report
    return {
        "schema_version": COPILOT_ARTIFACT_SCHEMA_VERSION,
        "kind": "axiom.copilot.search_report",
        "goal": config.goal,
        "domain_context": config.domain_context,
        "backend_name": _backend_name_from_result(result),
        "converged": result.converged,
        "iteration_count": len(result.iterations),
        "evaluation_mode": config.mode,
        "max_iterations_config": config.max_iterations,
        "score_sort_key": config.score_sort_key,
        "artifact_files": {
            "best_ax": BEST_AX_NAME,
            "iterations": ITERATIONS_JSON_NAME,
            "search_report": SEARCH_REPORT_JSON_NAME,
        },
        "best_evaluation": evaluation_report_to_dict(be),
        "final_evaluation": evaluation_report_to_dict(fe),
        "failures_metrics_summary": {
            "per_iteration": per_iter,
            "best": {
                "success": be.success,
                "metrics": dict(be.metrics),
                "failure_count": len(be.failures),
            },
            "final": {
                "success": fe.success,
                "metrics": dict(fe.metrics),
                "failure_count": len(fe.failures),
            },
        },
        "semantic_summaries": {
            "enabled": config.summarize_traces,
            "per_iteration": [
                {"index": rec.index, "semantic_trace_summary": rec.semantic_trace_summary}
                for rec in result.iterations
            ],
        },
    }


def persist_copilot_artifacts(
    config: CopilotSearchConfig,
    result: CopilotSearchResult,
    artifact_dir: Path,
) -> None:
    """Write ``best.ax``, ``iterations.json``, and ``search_report.json`` under ``artifact_dir``."""
    root = artifact_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / BEST_AX_NAME).write_text(result.best_source.rstrip() + "\n", encoding="utf-8")
    iter_doc = build_iterations_document(config, result)
    (root / ITERATIONS_JSON_NAME).write_text(
        json.dumps(iter_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report_doc = build_search_report_document(config, result)
    (root / SEARCH_REPORT_JSON_NAME).write_text(
        json.dumps(report_doc, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


__all__ = [
    "BEST_AX_NAME",
    "COPILOT_ARTIFACT_SCHEMA_VERSION",
    "ITERATIONS_JSON_NAME",
    "SEARCH_REPORT_JSON_NAME",
    "build_iterations_document",
    "build_search_report_document",
    "evaluation_report_to_dict",
    "expert_response_to_dict",
    "iteration_entry_to_dict",
    "json_safe",
    "persist_copilot_artifacts",
]
