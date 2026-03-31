"""FastAPI app for semantic copilot (draft / search / summarize) — separate from ``axiom serve`` and gateway."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from axiom.copilot.api_models import (
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    CopilotHealthResponse,
    CopilotRunRequest,
    CopilotRunResponse,
    DraftRequest,
    DraftResponse,
    SearchRequest,
    SearchResponse,
    SummarizeRequest,
    SummarizeResponse,
    TrainTabularPayload,
)
from axiom.copilot.artifacts import evaluation_report_to_dict, json_safe
from axiom.copilot.benchmarks import benchmark_suite_to_dict, benchmark_tasks_from_json_dict, run_benchmark_suite
from axiom.copilot.pipeline import CopilotPipelineConfig, copilot_pipeline_summary_dict, run_copilot_pipeline
from axiom.copilot.search import CopilotSearchConfig, CopilotSearchResult, build_draft_context, run_copilot_search
from axiom.copilot.tabular_json import parse_tabular_json_dict
from axiom.experts.base import ExpertDraftRequest, ExpertTraceSummaryRequest, SemanticExpert


def _expected_copilot_api_key() -> Optional[str]:
    k = os.environ.get("AXIOM_COPILOT_API_KEY", "").strip()
    return k or None


def _default_neg_mse_score_fn() -> Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Dict[str, float]]:
    def score_fn(preds: List[Dict[str, Any]], exp: List[Dict[str, Any]]) -> Dict[str, float]:
        total = 0.0
        n = 0
        for p, e in zip(preds, exp):
            for k, ev in e.items():
                if k not in p:
                    continue
                total += (float(p[k]) - float(ev)) ** 2
                n += 1
        mse = total / max(n, 1)
        return {"neg_mse": float(-mse)}

    return score_fn


def _copilot_payload_from_train_tabular(section: TrainTabularPayload):
    d = {
        "target_var": section.target_var.strip(),
        "train_rows": [{"inputs": dict(r.inputs), "expected": dict(r.expected)} for r in section.train_rows],
        "eval_rows": [{"inputs": dict(r.inputs), "expected": dict(r.expected)} for r in section.eval_rows],
        "epochs": section.epochs,
        "learning_rate": section.learning_rate,
        "weight_decay": section.weight_decay,
        "batch_size": section.batch_size,
    }
    return parse_tabular_json_dict(d)


def _search_config_from_request(body: SearchRequest, exp: SemanticExpert) -> CopilotSearchConfig:
    example_in: Optional[List[Dict[str, Any]]] = None
    example_exp: Optional[List[Dict[str, Any]]] = None
    if body.examples:
        example_in = [dict(x.inputs) for x in body.examples]
        example_exp = [dict(x.expected) for x in body.examples]

    tab_train: Optional[List[Dict[str, Any]]] = None
    tab_eval: Optional[List[Dict[str, Any]]] = None
    tab_target: Optional[str] = None
    tab_params = None
    tab_eval_exp: Optional[List[Dict[str, Any]]] = None
    if body.train_tabular is not None:
        pld = _copilot_payload_from_train_tabular(body.train_tabular)
        tab_train = list(pld.train_rows)
        tab_eval = list(pld.eval_rows)
        tab_target = pld.target_var
        tab_params = pld.params
        tab_eval_exp = list(pld.eval_expected_rows)

    if body.compile_only:
        mode: str = "compile_only"
        score_fn = None
        sort_key = None
    elif body.train_tabular is not None:
        mode = "train_tabular"
        score_fn = _default_neg_mse_score_fn()
        sort_key = "neg_mse"
    elif example_in is not None:
        mode = "predict_rows"
        score_fn = _default_neg_mse_score_fn()
        sort_key = "neg_mse"
    else:
        mode = "compile_only"
        score_fn = None
        sort_key = None

    summarize = bool(body.summarize_traces)
    return CopilotSearchConfig(
        expert=exp,
        goal=body.goal.strip(),
        domain_context=body.domain_context,
        example_input_rows=example_in,
        expected_rows=example_exp,
        max_iterations=int(body.max_iterations),
        mode=mode,  # type: ignore[arg-type]
        score_fn=score_fn,
        score_sort_key=sort_key,
        include_trace_snippet=summarize,
        summarize_traces=summarize,
        artifact_dir=Path(body.artifact_dir).resolve() if body.artifact_dir else None,
        tabular_train_rows=tab_train,
        tabular_eval_rows=tab_eval,
        tabular_target_var=tab_target,
        tabular_train_params=tab_params,
        tabular_eval_expected_rows=tab_eval_exp,
    )


def _serialize_search_response(result: CopilotSearchResult, cfg: CopilotSearchConfig) -> SearchResponse:
    _ = cfg
    iters = []
    for rec in result.iterations:
        ev = rec.evaluation
        iters.append(
            {
                "index": rec.index,
                "source": rec.source,
                "evaluation": evaluation_report_to_dict(ev),
                "producing_payload": dict(rec.producing_payload),
                "producing_expert": dict(rec.producing_expert),
                "outgoing_repair_error_report": rec.outgoing_repair_error_report,
                "semantic_trace_summary": rec.semantic_trace_summary,
            }
        )
    return SearchResponse(
        converged=result.converged,
        best_source=result.best_source,
        best_evaluation=evaluation_report_to_dict(result.best_evaluation),
        final_evaluation=evaluation_report_to_dict(result.final_report),
        iterations=iters,
    )


def create_app(expert: SemanticExpert):
    """Build FastAPI app with ``expert`` injected into routes (one process-wide backend)."""
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException
    except ImportError as e:
        raise ImportError(
            'Copilot server requires FastAPI. Install with: pip install -e ".[serve]"'
        ) from e

    app = FastAPI(title="Axiom Copilot Server", version="1.0")

    def get_expert() -> SemanticExpert:
        return expert

    async def verify_copilot_api_key(
        authorization: str | None = Header(None),
        x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    ) -> None:
        expected = _expected_copilot_api_key()
        if not expected:
            return
        ok = False
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
            ok = token == expected
        if x_api_key == expected:
            ok = True
        if not ok:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.get("/health", response_model=CopilotHealthResponse)
    def health() -> CopilotHealthResponse:
        return CopilotHealthResponse()

    @app.post("/draft", response_model=DraftResponse, dependencies=[Depends(verify_copilot_api_key)])
    def draft(body: DraftRequest, exp: SemanticExpert = Depends(get_expert)) -> DraftResponse:
        ctx = build_draft_context(
            domain_context=body.domain_context,
            example_input_rows=None,
            expected_rows=None,
        )
        resp = exp.draft_program(ExpertDraftRequest(goal=body.goal.strip(), context=ctx))
        return DraftResponse(
            ax_source=resp.ax_source,
            backend_name=resp.backend_name,
            explanation=resp.explanation,
            metadata=json_safe(dict(resp.metadata)),
        )

    @app.post("/search", response_model=SearchResponse, dependencies=[Depends(verify_copilot_api_key)])
    def search(body: SearchRequest, exp: SemanticExpert = Depends(get_expert)) -> SearchResponse:
        cfg = _search_config_from_request(body, exp)
        result = run_copilot_search(cfg)
        return _serialize_search_response(result, cfg)

    @app.post("/run", response_model=CopilotRunResponse, dependencies=[Depends(verify_copilot_api_key)])
    def run_pipeline(body: CopilotRunRequest, exp: SemanticExpert = Depends(get_expert)) -> CopilotRunResponse:
        cfg = _search_config_from_request(body, exp)
        pcfg = CopilotPipelineConfig(
            search=cfg,
            best_ax_path=None,
            summary_json_path=None,
            final_validate=bool(body.final_validate),
        )
        result = run_copilot_pipeline(pcfg)
        doc = copilot_pipeline_summary_dict(
            result,
            artifact_dir_resolved=result.artifact_dir,
            summarize_traces=bool(body.summarize_traces),
        )
        return CopilotRunResponse(
            disclaimer=str(doc["disclaimer"]),
            converged=bool(doc["converged"]),
            best_source=str(doc["best_source"]),
            best_evaluation=dict(doc["best_evaluation"]),
            final_evaluation=dict(doc["final_evaluation"]),
            iterations=list(doc["iterations"]),
            final_validation=dict(doc["final_validation"]) if doc.get("final_validation") is not None else None,
            semantic_summaries=dict(doc["semantic_summaries"]) if doc.get("semantic_summaries") else None,
            artifact_dir=doc.get("artifact_dir"),
        )

    @app.post("/benchmarks/run", response_model=BenchmarkRunResponse, dependencies=[Depends(verify_copilot_api_key)])
    def benchmarks_run(body: BenchmarkRunRequest, exp: SemanticExpert = Depends(get_expert)) -> BenchmarkRunResponse:
        task_list = None
        if body.tasks is not None:
            try:
                task_list = benchmark_tasks_from_json_dict(body.tasks)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        run_draft = not body.search_only
        run_search = not body.draft_only
        suite = run_benchmark_suite(
            exp,
            tasks=task_list,
            max_iterations=int(body.max_iterations),
            run_draft=run_draft,
            run_search=run_search,
        )
        return BenchmarkRunResponse(suite=benchmark_suite_to_dict(suite))

    @app.post("/summarize", response_model=SummarizeResponse, dependencies=[Depends(verify_copilot_api_key)])
    def summarize(body: SummarizeRequest, exp: SemanticExpert = Depends(get_expert)) -> SummarizeResponse:
        req = ExpertTraceSummaryRequest(
            goal=body.goal.strip(),
            program=body.program,
            trace=dict(body.trace),
            metrics={str(k): float(v) for k, v in body.metrics.items()},
            context=dict(body.context),
        )
        text = exp.summarize_trace(req)
        s = text.strip() if isinstance(text, str) else str(text).strip()
        return SummarizeResponse(summary=s)

    return app


__all__ = ["create_app", "_expected_copilot_api_key"]
