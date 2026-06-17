from __future__ import annotations

import argparse
import hashlib
import json
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

from axiom.copilot.artifacts import evaluation_report_to_dict  # noqa: E402
from axiom.copilot.backend import build_copilot_expert  # noqa: E402
from axiom.copilot.benchmarks import _evaluate_for_task, compile_success, load_benchmark_tasks_json_path, metric_success  # noqa: E402
from axiom.copilot.search import CopilotSearchConfig, _build_copilot_draft_request  # noqa: E402
from axiom.experts.onyx_qwen import REQUEST_CAPTURE_DIR_CONTEXT_KEY, REQUEST_CAPTURE_DIR_ENV_VAR  # noqa: E402


def _resolve_setting(
    explicit_value: str | None,
    *,
    env_name: str,
    setting_name: str,
    required: bool,
) -> str | None:
    if explicit_value is not None and str(explicit_value).strip():
        return str(explicit_value).strip()
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return env_value
    if required:
        raise SystemExit(
            f"Missing required setting: {setting_name}. Provide --{setting_name.replace('_', '-')} or set {env_name}."
        )
    return None


def _resolve_expert_api_key(explicit_cli: str, key_file: str) -> str | None:
    """API key only. Precedence: --expert-api-key, --expert-api-key-file, AXIOM_EXPERT_API_KEY."""
    if str(explicit_cli).strip():
        return str(explicit_cli).strip()
    path = str(key_file).strip() if key_file else ""
    if path:
        p = Path(path)
        if not p.is_file():
            raise SystemExit(f"expert-api-key-file not found: {p}")
        raw = p.read_text(encoding="utf-8").strip()
        if raw:
            return raw
    env_value = os.environ.get("AXIOM_EXPERT_API_KEY", "").strip()
    if env_value:
        return env_value
    return None


def _api_key_fingerprint(key: str) -> str:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"sha256:{digest[:12]}"


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


def _summary_elapsed_number(values: list[float], kind: str) -> float | None:
    if not values:
        return None
    if kind == "mean":
        return round(float(statistics.fmean(values)), 6)
    return round(float(statistics.median(values)), 6)


def _status_code_key(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _increment_count(bucket: dict[str, int], key: str) -> None:
    bucket[key] = bucket.get(key, 0) + 1


def _attempt_diagnostics(
    *,
    status: str,
    exc: BaseException | None,
    metadata: dict[str, Any],
) -> tuple[str | None, int | None, str | None, str | None]:
    """exception_class, status_code, request_id, response_id (JSON-serializable)."""
    if status == "success":
        sc: int | None = None
        if metadata.get("status_code") is not None:
            try:
                sc = int(metadata["status_code"])
            except (TypeError, ValueError):
                sc = None
        return (
            None,
            sc,
            str(metadata["request_id"]) if metadata.get("request_id") is not None else None,
            str(metadata["response_id"]) if metadata.get("response_id") is not None else None,
        )
    assert exc is not None
    exc_class = type(exc).__name__
    sc_val: int | None = None
    if metadata.get("status_code") is not None:
        try:
            sc_val = int(metadata["status_code"])
        except (TypeError, ValueError):
            sc_val = None
    if sc_val is None and isinstance(exc, Exception) and getattr(exc, "status_code", None) is not None:
        try:
            sc_val = int(getattr(exc, "status_code"))
        except (TypeError, ValueError):
            sc_val = None
    rid = metadata.get("request_id")
    rsp = metadata.get("response_id")
    return (
        exc_class,
        sc_val,
        str(rid) if rid is not None else None,
        str(rsp) if rsp is not None else None,
    )


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


def _grade_response(task: Any, source: str) -> dict[str, Any]:
    try:
        report = _evaluate_for_task(task, source)
    except Exception as exc:  # pragma: no cover - defensive grading path
        return {
            "parse_ok": None,
            "compile_ok": None,
            "metric_ok": None,
            "grading_error": str(exc),
        }
    return {
        "parse_ok": not any(f.stage == "parse" for f in report.failures),
        "compile_ok": compile_success(report),
        "metric_ok": metric_success(task, report),
        "evaluation": evaluation_report_to_dict(report),
    }


def _warmup_failure_kind(exc: BaseException | None, metadata: dict[str, Any]) -> str:
    if exc is None:
        return "n/a"
    fk = metadata.get("failure_kind")
    if fk:
        return str(fk)
    return type(exc).__name__


def _run_warmup_drafts(
    expert: Any,
    draft_req: Any,
    warmup_n: int,
) -> list[dict[str, Any]]:
    """Run throwaway draft calls; record every outcome (never swallow failures silently)."""
    results: list[dict[str, Any]] = []
    if warmup_n <= 0:
        return results
    print(f"WARMUP: runs={warmup_n}")
    for wi in range(1, warmup_n + 1):
        exc: BaseException | None = None
        metadata: dict[str, Any] = {}
        status = "success"
        try:
            expert.draft_program(draft_req)
        except Exception as caught:  # pragma: no cover - live-only branch behavior varies by backend
            exc = caught
            status = "failure"
            metadata = dict(getattr(exc, "metadata", {}) or {})
        exc_class, status_code_val, request_id_val, response_id_val = _attempt_diagnostics(
            status=status,
            exc=exc,
            metadata=metadata,
        )
        failure_kind = _warmup_failure_kind(exc, metadata)
        rec: dict[str, Any] = {
            "index": wi,
            "status": status,
            "failure_kind": failure_kind,
            "exception_class": exc_class,
            "status_code": status_code_val,
            "request_id": request_id_val,
            "response_id": response_id_val,
        }
        results.append(rec)
        print(
            "WARMUP {0}: status={1} failure_kind={2} status_code={3} exception_class={4}".format(
                wi,
                status,
                failure_kind,
                _status_code_key(status_code_val),
                exc_class or "n/a",
            )
        )
    return results


def _write_json_out(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_live_config(args: argparse.Namespace) -> tuple[str, str, str | None, Path | None]:
    expert_url = _resolve_setting(
        args.expert_url,
        env_name="AXIOM_EXPERT_URL",
        setting_name="expert_url",
        required=True,
    )
    expert_model = _resolve_setting(
        args.expert_model,
        env_name="AXIOM_EXPERT_MODEL",
        setting_name="expert_model",
        required=True,
    )
    expert_api_key = _resolve_expert_api_key(
        str(args.expert_api_key),
        str(getattr(args, "expert_api_key_file", "") or ""),
    )
    capture_dir_text = _resolve_setting(
        args.request_capture_dir,
        env_name=REQUEST_CAPTURE_DIR_ENV_VAR,
        setting_name="request_capture_dir",
        required=False,
    )
    capture_dir = Path(capture_dir_text) if capture_dir_text else None
    assert expert_url is not None
    assert expert_model is not None
    return expert_url, expert_model, expert_api_key, capture_dir


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
    parser.add_argument("--json-out", default="", help="Optional path to write structured JSON results.")
    parser.add_argument("--expert-url", default="", help="Live expert URL. Overrides AXIOM_EXPERT_URL when provided.")
    parser.add_argument(
        "--expert-model",
        default="",
        help="Live expert model. Overrides AXIOM_EXPERT_MODEL when provided.",
    )
    parser.add_argument(
        "--expert-api-key",
        default="",
        help="Optional API key. Precedence over --expert-api-key-file and AXIOM_EXPERT_API_KEY.",
    )
    parser.add_argument(
        "--expert-api-key-file",
        default="",
        help="File containing the raw API key (used only when --expert-api-key is empty). Precedence: --expert-api-key, then this file, then AXIOM_EXPERT_API_KEY.",
    )
    parser.add_argument(
        "--request-capture-dir",
        default="",
        help=f"Optional request capture directory. Overrides {REQUEST_CAPTURE_DIR_ENV_VAR} when provided.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Optional draft calls before measured attempts (not recorded in JSON).",
    )
    args = parser.parse_args(argv)

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1.")
    if int(args.warmup_runs) < 0:
        raise SystemExit("--warmup-runs must be >= 0.")

    url, model, api_key, capture_dir = _resolve_live_config(args)
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

    warmup_n = int(args.warmup_runs)
    warmup_results = _run_warmup_drafts(expert, draft_req, warmup_n)

    elapsed_values: list[float] = []
    success_count = 0
    timeout_count = 0
    attempts: list[dict[str, Any]] = []

    for idx in range(1, int(args.repeats) + 1):
        exc: BaseException | None = None
        started = time.perf_counter()
        status = "failure"
        failure_kind = "n/a"
        metadata: dict[str, Any] = {}
        grading: dict[str, Any] = {"parse_ok": None, "compile_ok": None, "metric_ok": None}
        try:
            resp = expert.draft_program(draft_req)
            metadata = dict(resp.metadata or {})
            status = "success"
            failure_kind = "n/a"
            success_count += 1
            grading = _grade_response(task, resp.ax_source)
        except Exception as caught:  # pragma: no cover - live-only branch behavior varies by backend
            exc = caught
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
        exc_class, status_code_val, request_id_val, response_id_val = _attempt_diagnostics(
            status=status,
            exc=exc,
            metadata=metadata,
        )
        attempt = {
            "index": idx,
            "status": status,
            "failure_kind": failure_kind,
            "exception_class": exc_class,
            "status_code": status_code_val,
            "request_id": request_id_val,
            "response_id": response_id_val,
            "elapsed_seconds": round(float(elapsed), 6) if elapsed is not None else None,
            "payload_sha256": str(metadata.get("payload_sha256") or ""),
            "request_capture_path": str(metadata.get("request_capture_path") or ""),
            "parse_ok": grading.get("parse_ok"),
            "compile_ok": grading.get("compile_ok"),
            "metric_ok": grading.get("metric_ok"),
        }
        if "grading_error" in grading:
            attempt["grading_error"] = grading["grading_error"]
        if "evaluation" in grading:
            attempt["evaluation"] = grading["evaluation"]
        attempts.append(attempt)
        print(
            "ATTEMPT {0}: task_id={1} status={2} failure_kind={3} elapsed_seconds={4} payload_sha256={5} request_capture_path={6} parse_ok={7} compile_ok={8} metric_ok={9}".format(
                idx,
                task.id,
                status,
                failure_kind,
                _format_elapsed(elapsed),
                str(metadata.get("payload_sha256") or "n/a"),
                str(metadata.get("request_capture_path") or "n/a"),
                str(grading.get("parse_ok")),
                str(grading.get("compile_ok")),
                str(grading.get("metric_ok")),
            )
        )

    mean_elapsed, median_elapsed = _summary_elapsed(elapsed_values)
    failure_kind_counts: dict[str, int] = {}
    status_code_counts: dict[str, int] = {}
    for att in attempts:
        _increment_count(failure_kind_counts, str(att.get("failure_kind") or "n/a"))
        _increment_count(status_code_counts, _status_code_key(att.get("status_code")))
    summary = {
        "repeats": int(args.repeats),
        "success_count": success_count,
        "timeout_count": timeout_count,
        "mean_elapsed": _summary_elapsed_number(elapsed_values, "mean"),
        "median_elapsed": _summary_elapsed_number(elapsed_values, "median"),
        "failure_kind_counts": failure_kind_counts,
        "status_code_counts": status_code_counts,
    }
    print(
        "SUMMARY: repeats={0} success_count={1} timeout_count={2} mean_elapsed={3} median_elapsed={4}".format(
            summary["repeats"],
            summary["success_count"],
            summary["timeout_count"],
            mean_elapsed,
            median_elapsed,
        )
    )
    print(
        "AGGREGATES: failure_kind_counts={0} status_code_counts={1}".format(
            json.dumps(failure_kind_counts, sort_keys=True),
            json.dumps(status_code_counts, sort_keys=True),
        )
    )
    if args.json_out:
        _write_json_out(
            Path(args.json_out),
            {
                "config": {
                    "task_id": task.id,
                    "task_json": str(task_json),
                    "timeout": float(args.timeout),
                    "max_tokens": int(args.max_tokens),
                    "repeats": int(args.repeats),
                    "warmup_runs": warmup_n,
                },
                "warmup": warmup_results,
                "attempts": attempts,
                "summary": summary,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
