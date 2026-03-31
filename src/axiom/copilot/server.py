"""FastAPI app for semantic copilot (draft / search / summarize) — separate from ``axiom serve`` and gateway."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from axiom.copilot.api_models import (
    CopilotHealthResponse,
    DraftRequest,
    DraftResponse,
    SearchRequest,
    SearchResponse,
    SummarizeRequest,
    SummarizeResponse,
)
from axiom.copilot.artifacts import evaluation_report_to_dict, json_safe
from axiom.copilot.search import CopilotSearchConfig, CopilotSearchResult, build_draft_context, run_copilot_search
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
        example_in: Optional[List[Dict[str, Any]]] = None
        example_exp: Optional[List[Dict[str, Any]]] = None
        if body.examples:
            example_in = [dict(x.inputs) for x in body.examples]
            example_exp = [dict(x.expected) for x in body.examples]

        if body.compile_only:
            mode: str = "compile_only"
            score_fn = None
            sort_key = None
        elif example_in is not None:
            mode = "predict_rows"
            score_fn = _default_neg_mse_score_fn()
            sort_key = "neg_mse"
        else:
            mode = "compile_only"
            score_fn = None
            sort_key = None

        summarize = bool(body.summarize_traces)
        cfg = CopilotSearchConfig(
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
        )
        result = run_copilot_search(cfg)
        return _serialize_search_response(result, cfg)

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
