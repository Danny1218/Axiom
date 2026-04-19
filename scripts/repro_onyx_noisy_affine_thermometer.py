from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from axiom.copilot.backend import build_copilot_expert  # noqa: E402
from axiom.copilot.benchmarks import load_benchmark_tasks_json_path  # noqa: E402
from axiom.copilot.search import CopilotSearchConfig, _build_copilot_draft_request  # noqa: E402
from axiom.experts.onyx_qwen import (  # noqa: E402
    OnyxQwenHTTPError,
    REQUEST_CAPTURE_DIR_CONTEXT_KEY,
    REQUEST_CAPTURE_DIR_ENV_VAR,
)


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Set {name}.")
    return value


def _load_task():
    task_json = Path(
        os.environ.get(
            "AXIOM_ONYX_REPRO_TASK_JSON",
            str(_ROOT / "benchmarks" / "copilot_symbolic_robustness_ambiguity_stress_tasks.json"),
        )
    )
    tasks = load_benchmark_tasks_json_path(task_json)
    for task in tasks:
        if task.id == "noisy_affine_thermometer":
            return task
    raise SystemExit(f"Task 'noisy_affine_thermometer' not found in {task_json}.")


def main() -> None:
    url = _require_env("AXIOM_EXPERT_URL")
    model = _require_env("AXIOM_EXPERT_MODEL")
    key = os.environ.get("AXIOM_EXPERT_API_KEY")
    max_tokens = int(os.environ.get("AXIOM_ONYX_REPRO_MAX_TOKENS", "64"))
    capture_dir = Path(
        os.environ.get(
            REQUEST_CAPTURE_DIR_ENV_VAR,
            os.environ.get("AXIOM_ONYX_REPRO_CAPTURE_DIR", str(_ROOT / "debug_onyx_request_capture")),
        )
    )
    task = _load_task()
    expert = build_copilot_expert("onyx-qwen", expert_url=url, expert_model=model, expert_api_key=key)
    cfg = CopilotSearchConfig(
        expert=expert,
        goal=task.goal,
        domain_context=task.domain_context or None,
        example_input_rows=list(task.example_input_rows) if task.example_input_rows else None,
        expected_rows=list(task.expected_rows) if task.expected_rows else None,
        mode=task.evaluation_mode,
        max_unroll=task.max_unroll,
        draft_context_extras={
            "benchmark_task_id": task.id,
            "benchmark_suite": "axiom.copilot.benchmarks",
            REQUEST_CAPTURE_DIR_CONTEXT_KEY: str(capture_dir),
        },
        completion_overrides={"temperature": 0, "max_tokens": max_tokens},
    )
    draft_req = _build_copilot_draft_request(cfg)
    try:
        resp = expert.draft_program(draft_req)
    except OnyxQwenHTTPError as e:
        capture_path = str(e.metadata.get("request_capture_path") or "")
        if capture_path:
            print(capture_path)
        raise SystemExit(str(e)) from e
    capture_path = str(resp.metadata.get("request_capture_path") or "")
    if not capture_path:
        raise SystemExit("Expected request capture artifact path in metadata.")
    print(capture_path)


if __name__ == "__main__":
    main()
