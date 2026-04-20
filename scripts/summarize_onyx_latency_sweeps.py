from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _load_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Failed to read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"Expected top-level object in {path}.")
    return data


def _require_mapping(data: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"Expected object field {key!r} in {path}.")
    return value


def _require_list(data: dict[str, Any], key: str, path: Path) -> list[dict[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list):
        raise SystemExit(f"Expected list field {key!r} in {path}.")
    out: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise SystemExit(f"Expected object items in {key!r} for {path}.")
        out.append(item)
    return out


def _require_fields(obj: dict[str, Any], keys: list[str], path: Path, label: str) -> None:
    missing = [key for key in keys if key not in obj]
    if missing:
        raise SystemExit(f"Missing {label} field(s) {missing!r} in {path}.")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _status_code_key(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _counts_from_attempts(attempts: list[dict[str, Any]], kind: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in attempts:
        if kind == "failure_kind":
            key = str(item.get("failure_kind") or "n/a")
        else:
            key = _status_code_key(item.get("status_code"))
        out[key] = out.get(key, 0) + 1
    return out


def _merge_counts(dst: dict[str, int], src: Mapping[str, int]) -> None:
    for key, val in src.items():
        dst[str(key)] = dst.get(str(key), 0) + int(val)


def _row_from_file(path: Path) -> dict[str, Any]:
    doc = _load_json(path)
    config = _require_mapping(doc, "config", path)
    attempts = _require_list(doc, "attempts", path)
    summary = _require_mapping(doc, "summary", path)
    _require_fields(config, ["task_id", "task_json", "timeout", "max_tokens", "repeats"], path, "config")
    _require_fields(
        summary,
        ["repeats", "success_count", "timeout_count", "mean_elapsed", "median_elapsed"],
        path,
        "summary",
    )
    for attempt in attempts:
        _require_fields(
            attempt,
            ["index", "status", "failure_kind", "elapsed_seconds", "payload_sha256", "request_capture_path"],
            path,
            "attempt",
        )
    fk_src = summary.get("failure_kind_counts")
    sc_src = summary.get("status_code_counts")
    failure_kind_counts: dict[str, int]
    status_code_counts: dict[str, int]
    if isinstance(fk_src, dict):
        failure_kind_counts = {str(k): int(v) for k, v in fk_src.items()}
    else:
        failure_kind_counts = _counts_from_attempts(attempts, "failure_kind")
    if isinstance(sc_src, dict):
        status_code_counts = {str(k): int(v) for k, v in sc_src.items()}
    else:
        status_code_counts = _counts_from_attempts(attempts, "status_code")
    return {
        "path": path,
        "timeout": int(config["timeout"]),
        "max_tokens": int(config["max_tokens"]),
        "repeats": int(summary["repeats"]),
        "success_count": int(summary["success_count"]),
        "timeout_count": int(summary["timeout_count"]),
        "compile_ok_count": sum(1 for item in attempts if item.get("compile_ok") is True),
        "metric_ok_count": sum(1 for item in attempts if item.get("metric_ok") is True),
        "mean_elapsed": _as_float(summary.get("mean_elapsed")),
        "median_elapsed": _as_float(summary.get("median_elapsed")),
        "failure_kind_counts": failure_kind_counts,
        "status_code_counts": status_code_counts,
    }


def _format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _fastest_key(row: dict[str, Any]) -> tuple[float, int, int]:
    mean_elapsed = row["mean_elapsed"]
    return (float("inf") if mean_elapsed is None else float(mean_elapsed), int(row["timeout"]), int(row["max_tokens"]))


def _ratio_key(row: dict[str, Any], numerator_key: str) -> tuple[float, int, int]:
    repeats = max(int(row["repeats"]), 1)
    ratio = float(row[numerator_key]) / float(repeats)
    return (-ratio, int(row["timeout"]), int(row["max_tokens"]))


def _pick_best(rows: list[dict[str, Any]], predicate, key_fn):
    candidates = [row for row in rows if predicate(row)]
    if not candidates:
        return None
    return min(candidates, key=key_fn)


def _print_best(label: str, row: dict[str, Any] | None, value_key: str | None = None) -> None:
    if row is None:
        print(f"BEST {label}: n/a")
        return
    extra = ""
    if value_key is not None:
        extra = f" value={row[value_key]:.6f}" if isinstance(row[value_key], float) else f" value={row[value_key]}"
    print(
        "BEST {0}: timeout={1} max_tokens={2} repeats={3} success_count={4} timeout_count={5} compile_ok_count={6} metric_ok_count={7}{8}".format(
            label,
            row["timeout"],
            row["max_tokens"],
            row["repeats"],
            row["success_count"],
            row["timeout_count"],
            row["compile_ok_count"],
            row["metric_ok_count"],
            extra,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize Onyx latency sweep JSON artifacts.")
    parser.add_argument("artifact_dir", help="Directory containing JSON latency sweep artifacts.")
    args = parser.parse_args(argv)

    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_dir():
        raise SystemExit(f"Not a directory: {artifact_dir}")
    paths = sorted(p for p in artifact_dir.glob("*.json") if p.is_file())
    if not paths:
        raise SystemExit(f"No JSON artifacts found in {artifact_dir}")

    rows = sorted((_row_from_file(path) for path in paths), key=lambda row: (row["timeout"], row["max_tokens"], row["path"].name))
    for row in rows:
        print(
            "ROW timeout={0} max_tokens={1} repeats={2} success_count={3} timeout_count={4} compile_ok_count={5} metric_ok_count={6} mean_elapsed={7} median_elapsed={8} failure_kind_counts={9} status_code_counts={10} file={11}".format(
                row["timeout"],
                row["max_tokens"],
                row["repeats"],
                row["success_count"],
                row["timeout_count"],
                row["compile_ok_count"],
                row["metric_ok_count"],
                _format_number(row["mean_elapsed"]),
                _format_number(row["median_elapsed"]),
                json.dumps(row["failure_kind_counts"], sort_keys=True),
                json.dumps(row["status_code_counts"], sort_keys=True),
                row["path"].name,
            )
        )

    fastest_compile = _pick_best(rows, lambda row: row["compile_ok_count"] > 0, _fastest_key)
    fastest_metric = _pick_best(rows, lambda row: row["metric_ok_count"] > 0, _fastest_key)
    best_success_ratio = _pick_best(rows, lambda row: True, lambda row: _ratio_key(row, "success_count"))
    best_metric_ratio = _pick_best(rows, lambda row: True, lambda row: _ratio_key(row, "metric_ok_count"))

    _print_best("fastest_compile_ok", fastest_compile, "mean_elapsed")
    _print_best("fastest_metric_ok", fastest_metric, "mean_elapsed")
    _print_best("highest_success_ratio", best_success_ratio)
    _print_best("highest_metric_ratio", best_metric_ratio)

    all_attempts: list[dict[str, Any]] = []
    for path in paths:
        doc = _load_json(path)
        all_attempts.extend(_require_list(doc, "attempts", path))
    total_attempts = len(all_attempts)
    agg_fk: dict[str, int] = {}
    agg_sc: dict[str, int] = {}
    for row in rows:
        _merge_counts(agg_fk, row["failure_kind_counts"])
        _merge_counts(agg_sc, row["status_code_counts"])
    print(
        "AGGREGATE total_attempts={0} status_code_counts={1} failure_kind_counts={2}".format(
            total_attempts,
            json.dumps(agg_sc, sort_keys=True),
            json.dumps(agg_fk, sort_keys=True),
        )
    )
    if (
        total_attempts > 0
        and all(_as_int_or_none(item.get("status_code")) == 401 for item in all_attempts)
    ):
        print("AUTH BLOCKER: all attempts unauthorized (401)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
