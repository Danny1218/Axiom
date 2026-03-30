"""Small internal NL→``.ax`` benchmark harness (draft vs search); no HTTP, no large data.

Tasks are tiny, repo-local definitions. Pass any :class:`~axiom.experts.base.SemanticExpert`
(stub or remote adapter). Results are JSON-serializable via :func:`benchmark_suite_to_dict`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from axiom.copilot.evaluator import evaluate_program
from axiom.copilot.models import EvaluationMode, ProgramCandidate, ProgramEvaluationReport
from axiom.copilot.search import (
    CopilotSearchConfig,
    build_draft_context,
    run_copilot_search,
)
from axiom.experts.base import (
    ExpertDraftRequest,
    ExpertDraftResponse,
    ExpertRepairRequest,
    ExpertTraceSummaryRequest,
    SemanticExpert,
)

BENCHMARK_SUITE_SCHEMA_VERSION = 1
_BENCH_CTX_FLAG = "benchmark_task_id"


def default_neg_mse_score_fn():
    """Same scoring contract as copilot CLI row eval (higher ``neg_mse`` is better)."""

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


@dataclass(frozen=True)
class BenchmarkTask:
    """One NL synthesis target plus how to score the produced program."""

    id: str
    title: str
    goal: str
    domain_context: str = ""
    evaluation_mode: EvaluationMode = "compile_only"
    example_input_rows: Tuple[Dict[str, Any], ...] = ()
    expected_rows: Tuple[Dict[str, Any], ...] = ()
    max_unroll: int = 8
    score_sort_key: Optional[str] = None
    metric_pass_min: Optional[Tuple[str, float]] = None

    def __post_init__(self) -> None:
        if self.evaluation_mode == "predict_rows":
            if not self.example_input_rows or not self.expected_rows:
                raise ValueError(f"Task {self.id!r}: predict_rows requires non-empty rows.")
            if len(self.example_input_rows) != len(self.expected_rows):
                raise ValueError(f"Task {self.id!r}: input vs expected row count mismatch.")


def _bench_extras(task: BenchmarkTask) -> Dict[str, Any]:
    return {_BENCH_CTX_FLAG: task.id, "benchmark_suite": "axiom.copilot.benchmarks"}


def _evaluate_for_task(task: BenchmarkTask, source: str) -> ProgramEvaluationReport:
    score_fn = default_neg_mse_score_fn() if task.evaluation_mode == "predict_rows" else None
    return evaluate_program(
        ProgramCandidate(source),
        mode=task.evaluation_mode,
        max_unroll=task.max_unroll,
        input_rows=list(task.example_input_rows) if task.example_input_rows else None,
        expected_rows=list(task.expected_rows) if task.expected_rows else None,
        score_fn=score_fn,
        include_trace_snippet=False,
    )


def compile_success(report: ProgramEvaluationReport) -> bool:
    """True when the harness reached a successful compile (and predict/metrics if applicable)."""
    return bool(report.success)


def metric_success(task: BenchmarkTask, report: ProgramEvaluationReport) -> bool:
    """Task-specific metric bar; for ``compile_only`` equals :func:`compile_success`."""
    if not report.success:
        return False
    if task.evaluation_mode == "compile_only":
        return True
    if task.metric_pass_min is None:
        return True
    key, lo = task.metric_pass_min
    v = report.metrics.get(key)
    if v is None:
        return False
    return float(v) >= float(lo)


@dataclass
class BenchmarkRunRecord:
    """Outcome for one task under one strategy (draft-only or search)."""

    task_id: str
    mode: Literal["draft_only", "search"]
    source: str
    evaluation: ProgramEvaluationReport
    compile_ok: bool
    metric_ok: bool
    converged: Optional[bool] = None
    iterations_run: Optional[int] = None


@dataclass
class BenchmarkTaskComparison:
    task_id: str
    title: str
    draft_only: BenchmarkRunRecord
    search: BenchmarkRunRecord


@dataclass
class BenchmarkSummary:
    """Aggregate rates over tasks (each task contributes one boolean per rate)."""

    task_count: int
    compile_success_rate: float
    metric_success_rate: float
    compile_ok_count: int
    metric_ok_count: int


@dataclass
class BenchmarkSuiteResult:
    schema_version: int = BENCHMARK_SUITE_SCHEMA_VERSION
    tasks: List[BenchmarkTaskComparison] = field(default_factory=list)
    draft_summary: Optional[BenchmarkSummary] = None
    search_summary: Optional[BenchmarkSummary] = None


def run_benchmark_draft_only(expert: SemanticExpert, task: BenchmarkTask) -> BenchmarkRunRecord:
    """Single ``draft_program`` + in-memory evaluation (no repair)."""
    ctx = build_draft_context(
        domain_context=task.domain_context or None,
        example_input_rows=task.example_input_rows,
        expected_rows=task.expected_rows,
    )
    ctx = {**ctx, **_bench_extras(task)}
    resp = expert.draft_program(ExpertDraftRequest(goal=task.goal, context=ctx))
    rep = _evaluate_for_task(task, resp.ax_source)
    co, mo = compile_success(rep), metric_success(task, rep)
    return BenchmarkRunRecord(
        task_id=task.id,
        mode="draft_only",
        source=resp.ax_source,
        evaluation=rep,
        compile_ok=co,
        metric_ok=mo,
    )


def run_benchmark_search(
    expert: SemanticExpert,
    task: BenchmarkTask,
    *,
    max_iterations: int = 4,
) -> BenchmarkRunRecord:
    """Full copilot search (draft → eval → repair loop)."""
    score_fn = default_neg_mse_score_fn() if task.evaluation_mode == "predict_rows" else None
    sk = task.score_sort_key if task.evaluation_mode == "predict_rows" else None
    cfg = CopilotSearchConfig(
        expert=expert,
        goal=task.goal,
        domain_context=task.domain_context or None,
        example_input_rows=list(task.example_input_rows) if task.example_input_rows else None,
        expected_rows=list(task.expected_rows) if task.expected_rows else None,
        max_iterations=max(1, int(max_iterations)),
        mode=task.evaluation_mode,
        max_unroll=task.max_unroll,
        score_fn=score_fn,
        score_sort_key=sk,
        include_trace_snippet=False,
        draft_context_extras=_bench_extras(task),
        repair_context_extras=_bench_extras(task),
    )
    out = run_copilot_search(cfg)
    rep = out.best_evaluation
    co, mo = compile_success(rep), metric_success(task, rep)
    return BenchmarkRunRecord(
        task_id=task.id,
        mode="search",
        source=out.best_source,
        evaluation=rep,
        compile_ok=co,
        metric_ok=mo,
        converged=out.converged,
        iterations_run=len(out.iterations),
    )


def summarize_rates(records: Sequence[BenchmarkRunRecord]) -> BenchmarkSummary:
    n = len(records)
    if n == 0:
        return BenchmarkSummary(0, 0.0, 0.0, 0, 0)
    cc = sum(1 for r in records if r.compile_ok)
    mc = sum(1 for r in records if r.metric_ok)
    return BenchmarkSummary(n, cc / n, mc / n, cc, mc)


def run_benchmark_suite(
    expert: SemanticExpert,
    *,
    tasks: Optional[Sequence[BenchmarkTask]] = None,
    max_iterations: int = 4,
) -> BenchmarkSuiteResult:
    """Run every task under draft-only and search; compare aggregate success rates."""
    seq = tuple(tasks) if tasks is not None else DEFAULT_BENCHMARK_TASKS
    comps: List[BenchmarkTaskComparison] = []
    draft_recs: List[BenchmarkRunRecord] = []
    search_recs: List[BenchmarkRunRecord] = []
    for t in seq:
        dr = run_benchmark_draft_only(expert, t)
        sr = run_benchmark_search(expert, t, max_iterations=max_iterations)
        draft_recs.append(dr)
        search_recs.append(sr)
        comps.append(BenchmarkTaskComparison(task_id=t.id, title=t.title, draft_only=dr, search=sr))
    return BenchmarkSuiteResult(
        tasks=comps,
        draft_summary=summarize_rates(draft_recs),
        search_summary=summarize_rates(search_recs),
    )


def _record_to_dict(rec: BenchmarkRunRecord) -> Dict[str, Any]:
    ev = rec.evaluation
    return {
        "task_id": rec.task_id,
        "mode": rec.mode,
        "source": rec.source,
        "compile_ok": rec.compile_ok,
        "metric_ok": rec.metric_ok,
        "converged": rec.converged,
        "iterations_run": rec.iterations_run,
        "evaluation": {
            "success": ev.success,
            "compile_stage_reached": ev.compile_stage_reached,
            "mode": ev.mode,
            "metrics": dict(ev.metrics),
        },
    }


def benchmark_suite_to_dict(result: BenchmarkSuiteResult) -> Dict[str, Any]:
    """JSON-ready view (use with ``json.dumps``)."""
    from axiom.copilot.artifacts import json_safe

    def summ(s: Optional[BenchmarkSummary]) -> Any:
        if s is None:
            return None
        return {
            "task_count": s.task_count,
            "compile_success_rate": s.compile_success_rate,
            "metric_success_rate": s.metric_success_rate,
            "compile_ok_count": s.compile_ok_count,
            "metric_ok_count": s.metric_ok_count,
        }

    out: Dict[str, Any] = {
        "schema_version": result.schema_version,
        "kind": "axiom.copilot.benchmark_suite",
        "draft_summary": summ(result.draft_summary),
        "search_summary": summ(result.search_summary),
        "tasks": [
            {
                "task_id": c.task_id,
                "title": c.title,
                "draft_only": _record_to_dict(c.draft_only),
                "search": _record_to_dict(c.search),
            }
            for c in result.tasks
        ],
    }
    return json_safe(out)


DEFAULT_BENCHMARK_TASKS: Tuple[BenchmarkTask, ...] = (
    BenchmarkTask(
        id="finance_threshold_policy",
        title="Finance-style threshold policy",
        goal=(
            "Write an .ax program with inputs volatility, drawdown, momentum, volume (float features). "
            "Output target_position in [0,1]. Use symbolic if/else to cut exposure when volatility is high "
            "(e.g. above 0.75), and combine with momentum and volume (you may use neural([...]) for a residual)."
        ),
        domain_context="Axiom: assignments, if/else, while allowed, neural([a,b]) for liquid head; outputs are floats.",
        evaluation_mode="predict_rows",
        example_input_rows=(
            {"volatility": 0.85, "drawdown": 0.2, "momentum": 0.4, "volume": 0.6},
            {"volatility": 0.3, "drawdown": 0.05, "momentum": 0.8, "volume": 0.7},
        ),
        expected_rows=(
            {"target_position": 0.245},
            {"target_position": 0.61},
        ),
        score_sort_key="neg_mse",
        metric_pass_min=("neg_mse", -0.5),
    ),
    BenchmarkTask(
        id="simple_risk_score",
        title="Simple blended risk score",
        goal=(
            "Write .ax with inputs risk_a and risk_b in [0,1]. Output risk_score as a weighted blend "
            "(heavier on risk_b), clamped to [0,1], without using neural()."
        ),
        domain_context="Use max, min, and arithmetic only.",
        evaluation_mode="predict_rows",
        example_input_rows=({"risk_a": 0.1, "risk_b": 0.9},),
        expected_rows=({"risk_score": 0.42},),
        score_sort_key="neg_mse",
        metric_pass_min=("neg_mse", -0.25),
    ),
    BenchmarkTask(
        id="looped_numeric_counter",
        title="Small while-loop counter",
        goal=(
            "Write .ax that initializes i to 0, runs a while loop while i < 3 incrementing i by 1 each time, "
            "then sets out_i to the final value of i after the loop."
        ),
        domain_context="Axiom while syntax: while (cond) { ... }",
        evaluation_mode="compile_only",
        max_unroll=8,
    ),
)

_REFERENCE_AX_BY_TASK: Dict[str, str] = {
    "finance_threshold_policy": (
        "target_position = max(0.0, min(1.0, momentum * 0.5 + (1.0 - volatility) * 0.3));\n"
    ),
    "simple_risk_score": "risk_score = max(0.0, min(1.0, risk_a * 0.6 + risk_b * 0.4));\n",
    "looped_numeric_counter": (
        "i = 0.0;\nwhile (i < 3.0) {\n  i = i + 1.0;\n}\nout_i = i;\n"
    ),
    "risk_from_json_fixture": "risk_score = max(0.0, min(1.0, risk_a * 0.6 + risk_b * 0.4));\n",
}


class BenchmarkDispatchExpert:
    """Expert that returns reference .ax keyed by ``benchmark_task_id`` in draft/repair context (tests / offline baselines)."""

    def __init__(
        self,
        sources: Optional[Dict[str, str]] = None,
        *,
        broken_draft_by_task: Optional[Dict[str, str]] = None,
    ) -> None:
        self._sources = dict(sources or _REFERENCE_AX_BY_TASK)
        self._broken = dict(broken_draft_by_task or {})

    def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
        tid = request.context.get(_BENCH_CTX_FLAG)
        if not isinstance(tid, str) or not tid:
            raise ValueError("BenchmarkDispatchExpert requires context['benchmark_task_id'].")
        if tid in self._broken:
            ax = self._broken[tid]
        else:
            ax = self._sources[tid]
        return ExpertDraftResponse(ax_source=ax, backend_name="benchmark_dispatch", metadata={"task_id": tid})

    def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
        tid = request.context.get(_BENCH_CTX_FLAG)
        if not isinstance(tid, str) or not tid:
            raise ValueError("BenchmarkDispatchExpert repair requires context['benchmark_task_id'].")
        ax = self._sources[tid]
        return ExpertDraftResponse(ax_source=ax, backend_name="benchmark_dispatch", metadata={"task_id": tid, "call": "repair"})

    def summarize_trace(self, request: ExpertTraceSummaryRequest) -> str:
        return ""


def benchmark_tasks_from_json_dict(obj: Any) -> List[BenchmarkTask]:
    """Load tasks from a JSON object (e.g. ``json.load``)."""
    if not isinstance(obj, dict):
        raise ValueError("benchmark JSON root must be an object.")
    tasks_raw = obj.get("tasks")
    if not isinstance(tasks_raw, list):
        raise ValueError("benchmark JSON must contain a 'tasks' array.")
    out: List[BenchmarkTask] = []
    for i, row in enumerate(tasks_raw):
        if not isinstance(row, dict):
            raise ValueError(f"tasks[{i}] must be an object.")
        mpm = row.get("metric_pass_min")
        mpm_t: Optional[Tuple[str, float]] = None
        if mpm is not None:
            if not isinstance(mpm, (list, tuple)) or len(mpm) != 2:
                raise ValueError(f"tasks[{i}].metric_pass_min must be [key, float].")
            mpm_t = (str(mpm[0]), float(mpm[1]))
        mode = row.get("evaluation_mode", "compile_only")
        if mode not in ("compile_only", "predict_rows"):
            raise ValueError(f"tasks[{i}].evaluation_mode invalid.")
        ex = row.get("example_input_rows") or []
        er = row.get("expected_rows") or []
        if not isinstance(ex, list) or not isinstance(er, list):
            raise ValueError(f"tasks[{i}] example_input_rows / expected_rows must be arrays.")
        out.append(
            BenchmarkTask(
                id=str(row["id"]),
                title=str(row.get("title", row["id"])),
                goal=str(row["goal"]),
                domain_context=str(row.get("domain_context", "")),
                evaluation_mode=mode,  # type: ignore[arg-type]
                example_input_rows=tuple(dict(r) for r in ex),
                expected_rows=tuple(dict(r) for r in er),
                max_unroll=int(row.get("max_unroll", 8)),
                score_sort_key=row.get("score_sort_key"),
                metric_pass_min=mpm_t,
            )
        )
    return out


def load_benchmark_tasks_json_path(path: Path) -> List[BenchmarkTask]:
    """Load tasks from a UTF-8 JSON file (repo-local fixture)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return benchmark_tasks_from_json_dict(raw)


def default_benchmark_tasks_json_path() -> Path:
    """Path to bundled ``fixtures/benchmark_tasks.json`` (minimal extra task for JSON round-trip)."""
    return Path(__file__).resolve().parent / "fixtures" / "benchmark_tasks.json"


__all__ = [
    "BENCHMARK_SUITE_SCHEMA_VERSION",
    "DEFAULT_BENCHMARK_TASKS",
    "BenchmarkDispatchExpert",
    "BenchmarkRunRecord",
    "BenchmarkSummary",
    "BenchmarkSuiteResult",
    "BenchmarkTask",
    "BenchmarkTaskComparison",
    "benchmark_suite_to_dict",
    "benchmark_tasks_from_json_dict",
    "compile_success",
    "default_benchmark_tasks_json_path",
    "default_neg_mse_score_fn",
    "load_benchmark_tasks_json_path",
    "metric_success",
    "run_benchmark_draft_only",
    "run_benchmark_search",
    "run_benchmark_suite",
    "summarize_rates",
]
