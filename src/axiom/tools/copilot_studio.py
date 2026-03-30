"""Streamlit Copilot Studio: draft and search ``.ax`` programs via a semantic expert (optional UI).

Install: ``pip install -e ".[inspect,copilot]"`` (Streamlit + ``requests``). Run: ``axiom copilot-studio``.

Logic below is free of Streamlit imports so tests can import without the UI stack. :func:`main` is only
used when this file is executed under ``streamlit run``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from axiom.copilot.artifacts import build_iterations_document, build_search_report_document
from axiom.copilot.search import CopilotSearchConfig, CopilotSearchResult, build_draft_context, run_copilot_search
from axiom.experts.base import ExpertDraftRequest, ExpertDraftResponse


def _default_neg_mse_score_fn():
    """Same contract as CLI copilot-search row scoring (higher ``neg_mse`` is better)."""

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


def parse_examples_rows_json(text: str) -> Tuple[List[dict], List[dict]]:
    """Parse row-eval JSON (array of ``{"inputs": {...}, "expected": {...}}``). Raises ``ValueError`` on bad input."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    if not isinstance(raw, list):
        raise ValueError("Examples must be a JSON array.")
    if not raw:
        raise ValueError("Examples array is empty.")
    inputs: List[dict] = []
    expected: List[dict] = []
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i} must be an object.")
        if "inputs" not in row or "expected" not in row:
            raise ValueError(f'Row {i} must have "inputs" and "expected".')
        ins, exp = row["inputs"], row["expected"]
        if not isinstance(ins, dict) or not isinstance(exp, dict):
            raise ValueError(f"Row {i}: inputs and expected must be objects.")
        inputs.append(dict(ins))
        expected.append(dict(exp))
    return inputs, expected


def build_studio_expert(base_url: str, model: str, api_key: Optional[str] = None):
    """Build :class:`~axiom.experts.onyx_qwen.OnyxQwenBackend`. Raises ``ImportError`` or ``ValueError``."""
    try:
        import requests  # noqa: F401
    except ImportError as e:
        raise ImportError('Install the copilot extra: pip install -e ".[copilot]"') from e
    from axiom.experts.onyx_qwen import OnyxQwenBackend

    url = (base_url or "").strip()
    m = (model or "").strip()
    if not url or not m:
        raise ValueError("Expert base URL and model are required.")
    key = (api_key or "").strip() or None
    if not key:
        key = os.environ.get("AXIOM_EXPERT_API_KEY")
    return OnyxQwenBackend(url, m, api_key=key)


def run_studio_draft(goal: str, context: Optional[str], expert: Any) -> ExpertDraftResponse:
    ctx = build_draft_context(
        domain_context=(context or "").strip() or None,
        example_input_rows=None,
        expected_rows=None,
    )
    return expert.draft_program(ExpertDraftRequest(goal=goal.strip(), context=ctx))


def run_studio_search(
    goal: str,
    context: Optional[str],
    expert: Any,
    max_iterations: int,
    *,
    compile_only: bool,
    examples_text: Optional[str] = None,
    summarize_traces: bool = False,
) -> Tuple[CopilotSearchConfig, CopilotSearchResult]:
    example_in: Optional[List[dict]] = None
    example_exp: Optional[List[dict]] = None
    if not compile_only and examples_text and examples_text.strip():
        example_in, example_exp = parse_examples_rows_json(examples_text.strip())
    if compile_only:
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
    cfg = CopilotSearchConfig(
        expert=expert,
        goal=goal.strip(),
        domain_context=(context or "").strip() or None,
        example_input_rows=example_in,
        expected_rows=example_exp,
        max_iterations=max(1, int(max_iterations)),
        mode=mode,  # type: ignore[arg-type]
        score_fn=score_fn,
        score_sort_key=sort_key,
        include_trace_snippet=bool(summarize_traces),
        summarize_traces=bool(summarize_traces),
    )
    return cfg, run_copilot_search(cfg)


def iterations_table_rows(result: CopilotSearchResult) -> List[Dict[str, Any]]:
    """Flat rows for ``st.dataframe`` (iteration summary)."""
    rows: List[Dict[str, Any]] = []
    for rec in result.iterations:
        ev = rec.evaluation
        summ = rec.semantic_trace_summary or ""
        rows.append(
            {
                "iter": rec.index,
                "success": ev.success,
                "compile_stage": ev.compile_stage_reached,
                "metrics": json.dumps(ev.metrics, sort_keys=True) if ev.metrics else "",
                "failure_count": len(ev.failures),
                "failure_kinds": ", ".join(sorted({f.kind for f in ev.failures})),
                "trace_summary": summ[:200] + ("…" if len(summ) > 200 else ""),
            }
        )
    return rows


def build_studio_download_payload(
    config: CopilotSearchConfig,
    result: CopilotSearchResult,
) -> Dict[str, Any]:
    """Single JSON object for download (iterations log + search report + best source)."""
    return {
        "best_source": result.best_source,
        "converged": result.converged,
        "iterations_document": build_iterations_document(config, result),
        "search_report": build_search_report_document(config, result),
    }


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Axiom Copilot Studio", layout="wide")
    st.title("Copilot Studio")
    st.caption("Draft or search `.ax` programs with an Onyx/Qwen-style expert. Nothing runs until you click a button.")

    for key in (
        "draft_ax",
        "draft_error",
        "draft_explanation",
        "search_cfg",
        "search_result",
        "search_error",
    ):
        if key not in st.session_state:
            st.session_state[key] = None

    with st.sidebar:
        st.subheader("Expert (OpenAI-style chat)")
        expert_url = st.text_input("Base URL", placeholder="https://host/v1/", key="expert_url")
        expert_model = st.text_input("Model id", placeholder="qwen-7b-chat", key="expert_model")
        expert_key = st.text_input("API key (optional)", type="password", key="expert_key")
        st.caption("If key is empty, `AXIOM_EXPERT_API_KEY` is used when set.")

    st.subheader("Task")
    goal = st.text_area("Goal", placeholder="Describe the .ax program you want.", height=100, key="goal")
    context = st.text_area("Context (optional)", placeholder="Domain notes, columns, constraints.", height=80, key="ctx")
    iterations = st.number_input("Max search iterations", min_value=1, max_value=64, value=8, step=1)
    summarize_traces = st.checkbox(
        "Summarize traces (calls expert after each iteration; extra latency)",
        value=False,
        key="summarize_traces",
    )
    eval_mode = st.radio("Search evaluation", ("compile_only", "predict_rows"), horizontal=True)
    examples_text = None
    if eval_mode == "predict_rows":
        examples_text = st.text_area(
            'Examples JSON (array of {"inputs":{}, "expected":{}})',
            height=120,
            key="examples_json",
        )

    col_draft, col_search = st.columns(2)
    draft_clicked = col_draft.button("Draft once", type="primary")
    search_clicked = col_search.button("Run search", type="primary")

    if draft_clicked:
        st.session_state.search_error = None
        st.session_state.search_cfg = None
        st.session_state.search_result = None
        if not goal.strip():
            st.session_state.draft_error = "Goal is required."
            st.session_state.draft_ax = None
        else:
            try:
                expert = build_studio_expert(expert_url, expert_model, expert_key or None)
                resp = run_studio_draft(goal, context or None, expert)
                st.session_state.draft_ax = resp.ax_source
                st.session_state.draft_explanation = resp.explanation
                st.session_state.draft_error = None
            except (ImportError, ValueError) as e:
                st.session_state.draft_error = str(e)
                st.session_state.draft_ax = None

    if search_clicked:
        st.session_state.draft_error = None
        if not goal.strip():
            st.session_state.search_error = "Goal is required."
            st.session_state.search_result = None
            st.session_state.search_cfg = None
        elif eval_mode == "predict_rows" and (not examples_text or not str(examples_text).strip()):
            st.session_state.search_error = "predict_rows requires non-empty examples JSON."
            st.session_state.search_result = None
            st.session_state.search_cfg = None
        else:
            try:
                expert = build_studio_expert(expert_url, expert_model, expert_key or None)
                cfg, res = run_studio_search(
                    goal,
                    context or None,
                    expert,
                    int(iterations),
                    compile_only=(eval_mode == "compile_only"),
                    examples_text=examples_text if eval_mode == "predict_rows" else None,
                    summarize_traces=summarize_traces,
                )
                st.session_state.search_cfg = cfg
                st.session_state.search_result = res
                st.session_state.search_error = None
            except (ImportError, ValueError) as e:
                st.session_state.search_error = str(e)
                st.session_state.search_result = None
                st.session_state.search_cfg = None

    if st.session_state.draft_error:
        st.error(st.session_state.draft_error)
    if st.session_state.draft_ax:
        st.subheader("Last draft")
        st.code(st.session_state.draft_ax, language="text")
        if st.session_state.draft_explanation:
            st.info(st.session_state.draft_explanation)
        st.download_button(
            "Download draft.ax",
            data=st.session_state.draft_ax.rstrip() + "\n",
            file_name="draft.ax",
            mime="text/plain",
            key="dl_draft",
        )

    if st.session_state.search_error:
        st.error(st.session_state.search_error)
    res = st.session_state.search_result
    cfg = st.session_state.search_cfg
    if res is not None and cfg is not None:
        st.subheader("Search result")
        st.write(f"**Converged:** {res.converged}")
        st.code(res.best_source.rstrip() + "\n", language="text")
        tbl = iterations_table_rows(res)
        if tbl:
            st.subheader("Iteration summary")
            st.dataframe(tbl, use_container_width=True, hide_index=True)
        with st.expander("Best evaluation (compile / predict / metrics)"):
            st.json(
                {
                    "success": res.best_evaluation.success,
                    "compile_stage_reached": res.best_evaluation.compile_stage_reached,
                    "metrics": dict(res.best_evaluation.metrics),
                    "failures": [
                        {"stage": f.stage, "kind": f.kind, "message": f.message}
                        for f in res.best_evaluation.failures
                    ],
                }
            )
        with st.expander("Final iteration evaluation"):
            st.json(
                {
                    "success": res.final_report.success,
                    "compile_stage_reached": res.final_report.compile_stage_reached,
                    "metrics": dict(res.final_report.metrics),
                    "failures": [
                        {"stage": f.stage, "kind": f.kind, "message": f.message}
                        for f in res.final_report.failures
                    ],
                }
            )
        if any(rec.semantic_trace_summary for rec in res.iterations):
            with st.expander("Semantic trace summaries (expert)"):
                for rec in res.iterations:
                    if rec.semantic_trace_summary:
                        st.markdown(f"**Iteration {rec.index}**")
                        st.write(rec.semantic_trace_summary)
        payload = build_studio_download_payload(cfg, res)
        st.download_button(
            "Download copilot_report.json",
            data=json.dumps(payload, indent=2, ensure_ascii=False),
            file_name="copilot_report.json",
            mime="application/json",
            key="dl_json",
        )
        st.download_button(
            "Download best.ax",
            data=res.best_source.rstrip() + "\n",
            file_name="best.ax",
            mime="text/plain",
            key="dl_best",
        )


if __name__ == "__main__":
    main()
