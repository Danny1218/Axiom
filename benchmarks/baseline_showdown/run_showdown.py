"""Run symbolic extrapolation benchmark and write evidence artifacts."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.baseline_showdown.contenders import build_contenders
from benchmarks.baseline_showdown.harness import (
    GLOBAL_SEED,
    TaskResult,
    build_summary,
    render_markdown,
    write_json,
)
from benchmarks.baseline_showdown.tasks import SynthTask, all_tasks

EVIDENCE_JSON = ROOT / "docs" / "evidence" / "baseline_showdown.json"
EVIDENCE_MD = ROOT / "docs" / "evidence" / "baseline_showdown.md"


def run_task(task: SynthTask, task_ids: Optional[List[str]] = None) -> TaskResult:
    split = task.generate_split()
    contenders = build_contenders()
    results = [
        c.run(
            task,
            split.train_rows,
            split.train_y,
            split.interp_rows,
            split.interp_y,
            split.extrap_rows,
            split.extrap_y,
        )
        for c in contenders
    ]
    ml_extrap = [
        r.extrap_rmse
        for r in results
        if r.name in ("mlp", "gbr", "linear") and r.extrap_rmse is not None and math.isfinite(r.extrap_rmse)
    ]
    best_ml = min(ml_extrap) if ml_extrap else None
    ax = next((r for r in results if r.name == "axiom"), None)
    margin: Optional[float] = None
    if ax and ax.extrap_rmse is not None and best_ml and best_ml > 0 and math.isfinite(ax.extrap_rmse):
        margin = best_ml / max(ax.extrap_rmse, 1e-15)
    return TaskResult(
        task_id=task.task_id,
        family=task.family,
        in_family=task.in_family,
        ground_truth=task.ground_truth,
        contenders=results,
        best_neural_extrap_rmse=best_ml,
        axiom_extrap_margin_x=margin,
    )


def run_showdown(*, task_ids: Optional[List[str]] = None) -> dict:
    tasks = all_tasks()
    if task_ids:
        tasks = [t for t in tasks if t.task_id in task_ids]
    t0 = time.perf_counter()
    task_results = [run_task(t, task_ids) for t in tasks]
    wall_ms = (time.perf_counter() - t0) * 1000.0
    summary = build_summary(task_results, wall_ms=wall_ms)
    write_json(EVIDENCE_JSON, summary)
    EVIDENCE_MD.write_text(render_markdown(summary), encoding="utf-8")
    print(summary["narrative"])
    print(f"Wrote {EVIDENCE_JSON} and {EVIDENCE_MD}")
    return summary


def main() -> int:
    run_showdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
