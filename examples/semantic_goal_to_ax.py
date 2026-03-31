"""
Library sketch: natural-language goal → champion ``.ax`` via the copilot pipeline.

Production entrypoints are **`axiom copilot-run`** (CLI) and **`POST /run`** on
**`axiom copilot-serve`**. This script shows :func:`axiom.copilot.pipeline.run_copilot_pipeline`
with an expert you inject (e.g. Onyx/Qwen HTTP via :func:`axiom.copilot.backend.build_copilot_expert`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root on path when run as `python examples/semantic_goal_to_ax.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from axiom.copilot.pipeline import (  # noqa: E402
    CopilotPipelineConfig,
    copilot_pipeline_summary_dict,
    run_copilot_pipeline,
)
from axiom.copilot.search import CopilotSearchConfig  # noqa: E402


def main() -> None:
    url = os.environ.get("AXIOM_EXPERT_URL", "").strip()
    model = os.environ.get("AXIOM_EXPERT_MODEL", "").strip()
    if not url or not model:
        print(
            "Set AXIOM_EXPERT_URL and AXIOM_EXPERT_MODEL (and optional AXIOM_EXPERT_API_KEY), "
            "or use: axiom copilot-run --backend onyx-qwen --goal ... --expert-url ... --expert-model ...",
            file=sys.stderr,
        )
        raise SystemExit(1)
    from axiom.copilot.backend import build_copilot_expert

    expert = build_copilot_expert(
        "onyx-qwen",
        expert_url=url,
        expert_model=model,
        expert_api_key=os.environ.get("AXIOM_EXPERT_API_KEY"),
    )
    goal = os.environ.get("AXIOM_COPILOT_GOAL", "Return y = neural([x]); as a minimal policy.").strip()
    out_dir = Path(os.environ.get("AXIOM_COPILOT_OUT", "copilot_e2e_demo"))
    cfg = CopilotSearchConfig(
        expert=expert,
        goal=goal,
        max_iterations=int(os.environ.get("AXIOM_COPILOT_ITERATIONS", "4")),
        mode="compile_only",
        artifact_dir=out_dir,
    )
    result = run_copilot_pipeline(CopilotPipelineConfig(search=cfg, final_validate=True))
    summary = copilot_pipeline_summary_dict(result, artifact_dir_resolved=result.artifact_dir)
    print(summary["disclaimer"], file=sys.stderr)
    print(summary["best_source"], end="")


if __name__ == "__main__":
    main()
