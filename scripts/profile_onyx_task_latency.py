from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from axiom.copilot.backend import build_copilot_expert  # noqa: E402
from axiom.copilot.benchmarks import load_benchmark_tasks_json_path  # noqa: E402
from axiom.copilot.search import CopilotSearchConfig, _build_copilot_draft_request  # noqa: E402
from axiom.experts.onyx_qwen import REQUEST_CAPTURE_DIR_CONTEXT_KEY, REQUEST_CAPTURE_DIR_ENV_VAR  # noqa: E402


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise SystemExit(f"Set {name}.")
    return value


def _format_elapsed(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "n/a"


def _summary_elapsed(values: list[float]) -> tuple[str, str]:
    if not values:
        return "n/a", "n/a"
    return f"{statistics.fmean(values):.6f}", f"{statistics.median(values):.6f}"


def _load_task(task_id: str, task_json: Path):
    tasks = load_benchmark_tasks_json_path(task_json)
    for task in tasks:
        if task.id == task_id:
            return task
    raise SystemExit(f"Task {task_id!r} not found in {task_json}.")


def _build_draft_request(*, expert: Any, task_id: str, task_json: Path, capture_dir: Path | None, max_tokens: int):
    task = _load_task(task_id, task_json)
    extras: dict[str, Any] = {
        "benchmark_task_id": task.id,
        "benchmark_suite": "axiom.copilot.benchmarks",
    }
    if capture_dir is not None:
        extras[REQUEST_CAPTURE_DIR_CONTEXT_KEY] = str(capture_dir)
    cfg = CopilotSearchConfig(
        expert=expert,
        goal=task.goal,
        domain_context=task.domain_context or None,
        example_input_rows=list(task.example_input_rows) if task.example_input_rows else None,
        expected_rows=list(task.expected_rows) if task.expected_rows else None,
        mode=task.evaluation_mode,
        max_unroll=task.max_unroll,
        draft_context_extras=extras,
        completion_overrides={"temperature": 0, "max_tokens": max_tokens},
    )
    return task, _build_copilot_draft_request(cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile repeated live Onyx draft latency for one benchmark task.")
    parser.add_argument("--task-id", required=True, help="Benchmark task id to profile.")
    parser.add_argument(
        "--task-json",
        default=str(_ROOT / "benchmarks" / "copilot_symbolic_robustness_ambiguity_stress_tasks.json"),
        help="Path to the benchmark task JSON file.",
    )
    parser.add_argument("--timeout", type=float, required=True, help="Expert timeout in seconds.")
    parser.add_argument("--max-tokens", type=int, required=True, help="max_tokens override for the live draft call.")
    parser.add_argument("--repeats", type=int, required=True, help="Number of repeated draft attempts to run.")
    args = parser.parse_args(argv)

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1.")

    url = _require_env("AXIOM_EXPERT_URL")
    model = _require_env("AXIOM_EXPERT_MODEL")
    api_key = os.environ.get("AXIOM_EXPERT_API_KEY")
    capture_dir_text = os.environ.get(REQUEST_CAPTURE_DIR_ENV_VAR, "").strip()
    capture_dir = Path(capture_dir_text) if capture_dir_text else None
    task_json = Path(args.task_json)
    expert = build_copilot_expert(
        "onyx-qwen",
        expert_url=url,
        expert_model=model,
        expert_api_key=api_key,
        timeout=args.timeout,
    )
    task, draft_req = _build_draft_request(
        expert=expert,
        task_id=args.task_id,
        task_json=task_json,
        capture_dir=capture_dir,
        max_tokens=args.max_tokens,
    )

    elapsed_values: list[float] = []
    success_count = 0
    timeout_count = 0

    for idx in range(1, int(args.repeats) + 1):
        started = time.perf_counter()
        status = "failure"
        failure_kind = "n/a"
        metadata: dict[str, Any] = {}
        try:
            resp = expert.draft_program(draft_req)
            metadata = dict(resp.metadata or {})
            status = "success"
            failure_kind = "n/a"
            success_count += 1
        except Exception as exc:  # pragma: no cover - live-only branch behavior varies by backend
            metadata = dict(getattr(exc, "metadata", {}) or {})
            failure_kind = str(metadata.get("failure_kind") or type(exc).__name__)
            if failure_kind == "timeout":
                timeout_count += 1
        elapsed = metadata.get("elapsed_seconds")
        if elapsed is None:
            elapsed = time.perf_counter() - started
        try:
            elapsed_values.append(float(elapsed))
        except (TypeError, ValueError):
            pass
        print(
            "ATTEMPT {0}: task_id={1} status={2} failure_kind={3} elapsed_seconds={4} payload_sha256={5} request_capture_path={6}".format(
                idx,
                task.id,
                status,
                failure_kind,
                _format_elapsed(elapsed),
                str(metadata.get("payload_sha256") or "n/a"),
                str(metadata.get("request_capture_path") or "n/a"),
            )
        )

    mean_elapsed, median_elapsed = _summary_elapsed(elapsed_values)
    print(
        "SUMMARY: repeats={0} success_count={1} timeout_count={2} mean_elapsed={3} median_elapsed={4}".format(
            int(args.repeats),
            success_count,
            timeout_count,
            mean_elapsed,
            median_elapsed,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
