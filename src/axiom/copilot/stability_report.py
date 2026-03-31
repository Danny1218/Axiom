"""Aggregate metrics across multiple copilot artifact directories or pipeline summary JSON files."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from axiom.copilot.artifacts import BEST_AX_NAME, SEARCH_REPORT_JSON_NAME

DEFAULT_NEAR_NEG_MSE = -1e-9


@dataclass(frozen=True)
class RunSnapshot:
    """One logical run: a search artifact dir, a pipeline summary file, or a multi-restart artifact root."""

    label: str
    kind: str
    success: bool
    sort_key: Optional[str]
    primary_metric: Optional[float]
    metric_display: str
    convergence_reason: Optional[str]
    exact_hit: bool
    near_hit: bool
    winning_restart_index: Optional[int]
    best_ax_path: Optional[str]
    evaluation_mode: Optional[str]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _primary_from_best_eval(be: dict[str, Any], sort_key: Optional[str]) -> Optional[float]:
    adj = be.get("adjusted_sort_score")
    if adj is not None:
        return _f(adj)
    m = be.get("metrics") or {}
    if sort_key:
        return _f(m.get(sort_key))
    return None


def _dict_eval_better(cand: dict[str, Any], best: Optional[dict[str, Any]], sort_key: Optional[str]) -> bool:
    if best is None:
        return True
    c_ok, b_ok = bool(cand.get("success")), bool(best.get("success"))
    if c_ok and not b_ok:
        return True
    if not c_ok and b_ok:
        return False
    if not c_ok and not b_ok:
        return False
    cp, bp = _primary_from_best_eval(cand, sort_key), _primary_from_best_eval(best, sort_key)
    if cp is not None and bp is not None:
        return cp > bp
    if cp is not None:
        return True
    if bp is not None:
        return False
    return False


def _infer_sort_key_from_report(report: dict[str, Any]) -> Optional[str]:
    sk = report.get("score_sort_key")
    if isinstance(sk, str) and sk:
        return sk
    m = (report.get("best_evaluation") or {}).get("metrics") or {}
    if "neg_mse" in m:
        return "neg_mse"
    return None


def _infer_sort_key_from_eval(be: dict[str, Any]) -> Optional[str]:
    m = be.get("metrics") or {}
    return "neg_mse" if "neg_mse" in m else None


def _hits(
    be: dict[str, Any], *, mode: Optional[str], sort_key: Optional[str], near_floor: float
) -> Tuple[bool, bool]:
    success = bool(be.get("success"))
    metrics = be.get("metrics") or {}
    neg_mse = _f(metrics.get("neg_mse"))
    if mode == "compile_only" or sort_key is None:
        return success, success
    if sort_key == "neg_mse" and neg_mse is not None:
        exact = success and abs(neg_mse) <= 1e-15
        near = success and neg_mse >= near_floor
        return exact, near
    return success, success


def _metric_line(be: dict[str, Any], sort_key: Optional[str], primary: Optional[float]) -> str:
    m = be.get("metrics") or {}
    neg = _f(m.get("neg_mse"))
    if sort_key == "neg_mse" and neg is not None:
        return f"neg_mse={neg:.6g}"
    if primary is not None:
        return f"{sort_key or 'primary'}={primary:.6g}"
    return "n/a"


def _snapshot_from_search_report(
    report_root: Path,
    report: dict[str, Any],
    *,
    label: str,
    kind: str,
    near_floor: float,
    winning_restart_index: Optional[int] = None,
) -> RunSnapshot:
    be = report.get("best_evaluation") or {}
    mode = report.get("evaluation_mode") or be.get("mode")
    sort_key = _infer_sort_key_from_report(report)
    primary = _primary_from_best_eval(be, sort_key)
    exact, near = _hits(be, mode=mode, sort_key=sort_key, near_floor=near_floor)
    ax_name = (report.get("artifact_files") or {}).get("best_ax") or BEST_AX_NAME
    ax_path = report_root / ax_name
    best_ax = str(ax_path.resolve()) if ax_path.is_file() else None
    return RunSnapshot(
        label=label,
        kind=kind,
        success=bool(be.get("success")),
        sort_key=sort_key,
        primary_metric=primary,
        metric_display=_metric_line(be, sort_key, primary),
        convergence_reason=report.get("convergence_reason"),
        exact_hit=exact,
        near_hit=near,
        winning_restart_index=winning_restart_index,
        best_ax_path=best_ax,
        evaluation_mode=mode if isinstance(mode, str) else None,
    )


def _has_restart_subdirs(p: Path) -> bool:
    try:
        return any(x.is_dir() and x.name.startswith("restart_") for x in p.iterdir())
    except OSError:
        return False


def _try_search_dir(path: Path, near_floor: float) -> Optional[RunSnapshot]:
    sr = path / SEARCH_REPORT_JSON_NAME
    if not sr.is_file():
        return None
    report = _load_json(sr)
    if report.get("kind") != "axiom.copilot.search_report":
        return None
    return _snapshot_from_search_report(
        path, report, label=str(path.resolve()), kind="search_dir", near_floor=near_floor
    )


def _try_pipeline_restart_dir(path: Path, near_floor: float) -> Optional[RunSnapshot]:
    if not path.is_dir():
        return None
    subs = sorted((p for p in path.iterdir() if p.is_dir() and p.name.startswith("restart_")), key=lambda x: x.name)
    if not subs:
        return None
    parsed: list[tuple[int, Path, dict[str, Any]]] = []
    for p in subs:
        sr = p / SEARCH_REPORT_JSON_NAME
        if not sr.is_file():
            continue
        rep = _load_json(sr)
        if rep.get("kind") != "axiom.copilot.search_report":
            continue
        suffix = p.name[len("restart_") :]
        try:
            idx = int(suffix)
        except ValueError:
            idx = len(parsed)
        parsed.append((idx, p, rep))
    if not parsed:
        return None
    sort_key: Optional[str] = None
    for _, _, rep in parsed:
        sort_key = _infer_sort_key_from_report(rep)
        if sort_key:
            break
    win_idx, win_path, win_rep = parsed[0]
    best_be = win_rep["best_evaluation"]
    for idx, rp, rep in parsed[1:]:
        cand_be = rep["best_evaluation"]
        if _dict_eval_better(cand_be, best_be, sort_key):
            win_idx, win_path, win_rep = idx, rp, rep
            best_be = cand_be
    return _snapshot_from_search_report(
        win_path,
        win_rep,
        label=str(path.resolve()),
        kind="pipeline_dir",
        near_floor=near_floor,
        winning_restart_index=win_idx,
    )


def _snapshot_from_pipeline_doc(path: Path, doc: dict[str, Any], near_floor: float) -> Optional[RunSnapshot]:
    r = doc.get("restarts")
    if not isinstance(r, dict) or not isinstance(r.get("per_restart"), list):
        return None
    be = doc.get("best_evaluation") or {}
    mode = be.get("mode")
    sort_key = _infer_sort_key_from_eval(be)
    primary = _primary_from_best_eval(be, sort_key)
    exact, near = _hits(be, mode=mode if isinstance(mode, str) else None, sort_key=sort_key, near_floor=near_floor)
    win_i: Optional[int] = None
    w = r.get("winning_index")
    if w is not None:
        try:
            win_i = int(w)
        except (TypeError, ValueError):
            win_i = None
    best_ax: Optional[str] = None
    pr = r["per_restart"]
    if win_i is not None and 0 <= win_i < len(pr):
        sub = pr[win_i].get("artifact_subdir")
        if sub:
            axp = Path(sub) / BEST_AX_NAME
            best_ax = str(axp.resolve()) if axp.is_file() else str(axp)
    return RunSnapshot(
        label=str(path.resolve()),
        kind="pipeline_json",
        success=bool(be.get("success")),
        sort_key=sort_key,
        primary_metric=primary,
        metric_display=_metric_line(be, sort_key, primary),
        convergence_reason=doc.get("convergence_reason") if isinstance(doc.get("convergence_reason"), str) else None,
        exact_hit=exact,
        near_hit=near,
        winning_restart_index=win_i,
        best_ax_path=best_ax,
        evaluation_mode=mode if isinstance(mode, str) else None,
    )


def _try_pipeline_json(path: Path, near_floor: float) -> Optional[RunSnapshot]:
    if not path.is_file() or path.suffix.lower() != ".json":
        return None
    try:
        doc = _load_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return _snapshot_from_pipeline_doc(path, doc, near_floor)


def _is_restart_dir_name(name: str) -> bool:
    return bool(name.startswith("restart_") and len(name) > 8 and name[8:].isdigit())


def _norm_path_str(p: Optional[str | Path]) -> Optional[str]:
    if p is None:
        return None
    try:
        return str(Path(p).expanduser().resolve())
    except OSError:
        return str(Path(p))


def _artifact_dirs_from_pipeline_doc(doc: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    ad = _norm_path_str(doc.get("artifact_dir"))
    if ad:
        out.add(ad)
    r = doc.get("restarts")
    if isinstance(r, dict):
        for pr in r.get("per_restart") or []:
            if not isinstance(pr, dict):
                continue
            sd = _norm_path_str(pr.get("artifact_subdir"))
            if sd:
                out.add(sd)
                try:
                    p = Path(sd)
                    if _is_restart_dir_name(p.name):
                        out.add(str(p.parent.resolve()))
                except OSError:
                    pass
    return out


SEARCH_PATTERNS_DESC = (
    "**/search_report.json (kind axiom.copilot.search_report, excluding paths under restart_*/); "
    "**/*.json with restarts.per_restart (pipeline summary); "
    "directories with restart_*/ children (multi-restart artifact bundle)"
)


def _dedupe_paths(paths: Sequence[Path]) -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []
    for p in paths:
        k = str(p.resolve())
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def resolve_scan_roots(paths: Sequence[Path], parent: Optional[Path]) -> List[Path]:
    """Scan roots for recursive discovery (Phase 83b): explicit paths plus optional ``--parent`` as one root."""
    out: List[Path] = []
    for p in paths:
        out.append(p.expanduser().resolve())
    if parent is not None:
        out.append(parent.expanduser().resolve())
    return _dedupe_paths(out)


def resolve_input_paths(paths: Sequence[Path], parent: Optional[Path]) -> List[Path]:
    """Alias for :func:`resolve_scan_roots`."""
    return resolve_scan_roots(paths, parent)


def ingest_path(path: Path, near_floor: float) -> Optional[RunSnapshot]:
    if path.is_file():
        return _try_pipeline_json(path, near_floor)
    if not path.is_dir():
        return None
    if _has_restart_subdirs(path):
        got = _try_pipeline_restart_dir(path, near_floor)
        if got is not None:
            return got
    return _try_search_dir(path, near_floor)


def _discover_from_roots(roots: Sequence[Path], near_floor: float) -> tuple[list[RunSnapshot], dict[str, Any]]:
    """Recursively find pipeline summaries and search artifacts; dedupe JSON vs artifact dirs."""
    meta: dict[str, Any] = {
        "scan_roots": [str(r) for r in roots],
        "search_report_paths_total": 0,
        "search_report_paths_skipped_under_restart": 0,
        "json_files_probed_pipeline": 0,
        "pipeline_json_matched": 0,
        "pipeline_json_skipped_duplicate_artifact_dir": 0,
        "pipeline_root_dir_ingested": 0,
        "pipeline_root_dir_skipped_covered": 0,
        "search_dir_ingested": 0,
        "search_dir_skipped_covered": 0,
        "patterns": SEARCH_PATTERNS_DESC,
    }

    all_search_reports: list[Path] = []
    json_probe_list: list[Path] = []
    for root in sorted(roots, key=str):
        if root.is_file():
            if root.suffix.lower() == ".json":
                json_probe_list.append(root)
            continue
        if not root.is_dir():
            continue
        try:
            sr_all = sorted(root.rglob(SEARCH_REPORT_JSON_NAME), key=str)
        except OSError:
            sr_all = []
        meta["search_report_paths_total"] += len(sr_all)
        all_search_reports.extend(sr_all)
        try:
            for jp in sorted(root.rglob("*.json"), key=str):
                if jp.name == SEARCH_REPORT_JSON_NAME:
                    continue
                json_probe_list.append(jp)
        except OSError:
            pass

    json_probe_list = _dedupe_paths(json_probe_list)
    meta["json_files_probed_pipeline"] = len(json_probe_list)

    pipeline_snapshots: list[RunSnapshot] = []
    covered_dirs: set[str] = set()
    seen_pipeline_artifact_root: set[str] = set()

    for jp in sorted(json_probe_list, key=str):
        try:
            doc = _load_json(jp)
        except (OSError, json.JSONDecodeError):
            continue
        snap = _snapshot_from_pipeline_doc(jp, doc, near_floor)
        if snap is None:
            continue
        ad = _norm_path_str(doc.get("artifact_dir"))
        if ad and ad in seen_pipeline_artifact_root:
            meta["pipeline_json_skipped_duplicate_artifact_dir"] += 1
            continue
        pipeline_snapshots.append(snap)
        meta["pipeline_json_matched"] += 1
        covered_dirs |= _artifact_dirs_from_pipeline_doc(doc)
        if ad:
            seen_pipeline_artifact_root.add(ad)

    pipeline_root_dirs: set[Path] = set()
    search_root_dirs: set[Path] = set()
    for sp in sorted(all_search_reports, key=str):
        d = sp.parent
        if _is_restart_dir_name(d.name):
            meta["search_report_paths_skipped_under_restart"] += 1
            pipeline_root_dirs.add(d.parent.resolve())
            continue
        if _has_restart_subdirs(d):
            pipeline_root_dirs.add(d.resolve())
        else:
            search_root_dirs.add(d.resolve())

    for pd in sorted(pipeline_root_dirs, key=str):
        ns = str(pd.resolve())
        if ns in covered_dirs:
            meta["pipeline_root_dir_skipped_covered"] += 1
            continue
        snap = _try_pipeline_restart_dir(pd, near_floor)
        if snap is not None:
            pipeline_snapshots.append(snap)
            meta["pipeline_root_dir_ingested"] += 1

    search_snaps: list[RunSnapshot] = []
    for sd in sorted(search_root_dirs, key=str):
        if sd in pipeline_root_dirs:
            continue
        ns = str(sd.resolve())
        if ns in covered_dirs:
            meta["search_dir_skipped_covered"] += 1
            continue
        snap = _try_search_dir(sd, near_floor)
        if snap is not None:
            search_snaps.append(snap)
            meta["search_dir_ingested"] += 1

    runs = sorted(pipeline_snapshots + search_snaps, key=lambda r: r.label)
    meta["run_snapshots_total_before_display"] = len(runs)
    return runs, meta


def format_discovery_failure_message(meta: dict[str, Any]) -> str:
    roots = meta.get("scan_roots") or []
    return (
        "No copilot runs matched after discovery. "
        f"Scan root(s) ({len(roots)}): {roots!r}. "
        f"Found {meta.get('search_report_paths_total', 0)} {SEARCH_REPORT_JSON_NAME} path(s) "
        f"({meta.get('search_report_paths_skipped_under_restart', 0)} under restart_*/ skipped as nested). "
        f"Probed {meta.get('json_files_probed_pipeline', 0)} other JSON file(s) for pipeline summary shape; "
        f"{meta.get('pipeline_json_matched', 0)} matched. "
        f"Patterns: {meta.get('patterns', SEARCH_PATTERNS_DESC)}"
    )


def collect_run_snapshots(
    paths: Sequence[Path], parent: Optional[Path], near_floor: float
) -> tuple[list[RunSnapshot], dict[str, Any]]:
    roots = resolve_scan_roots(paths, parent)
    if not roots:
        return [], {
            "scan_roots": [],
            "search_report_paths_total": 0,
            "search_report_paths_skipped_under_restart": 0,
            "json_files_probed_pipeline": 0,
            "pipeline_json_matched": 0,
            "patterns": SEARCH_PATTERNS_DESC,
        }
    return _discover_from_roots(roots, near_floor)


def _pick_best_run(runs: Sequence[RunSnapshot]) -> Optional[RunSnapshot]:
    if not runs:
        return None
    ok = [r for r in runs if r.success]
    pool = ok if ok else list(runs)

    def key(r: RunSnapshot) -> Tuple[bool, int, float]:
        pm = r.primary_metric
        v = pm if pm is not None else float("-inf")
        return (r.success, 1 if pm is not None else 0, v)

    return max(pool, key=key)


def build_aggregate(runs: Sequence[RunSnapshot]) -> dict[str, Any]:
    conv = Counter()
    for r in runs:
        cr = r.convergence_reason or "(none)"
        conv[cr] += 1
    win_c = Counter()
    for r in runs:
        if r.winning_restart_index is not None:
            win_c[str(r.winning_restart_index)] += 1
    best = _pick_best_run(runs)
    best_blob: Optional[dict[str, Any]] = None
    if best is not None:
        best_blob = {
            "label": best.label,
            "metric_display": best.metric_display,
            "primary_metric": best.primary_metric,
            "best_ax_path": best.best_ax_path,
            "sort_key": best.sort_key,
        }
    return {
        "run_count": len(runs),
        "exact_hit_count": sum(1 for r in runs if r.exact_hit),
        "near_hit_count": sum(1 for r in runs if r.near_hit),
        "convergence_reason_counts": dict(sorted(conv.items())),
        "winning_restart_counts": dict(sorted(win_c.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0])),
        "best": best_blob,
    }


def format_stability_text(runs: Sequence[RunSnapshot], agg: dict[str, Any]) -> str:
    n = agg["run_count"]
    lines = [f"Copilot stability — {n} run(s)"]
    lines.append(f"  exact_hit: {agg['exact_hit_count']} / {n}")
    lines.append(f"  near_hit:  {agg['near_hit_count']} / {n}")
    crc = agg["convergence_reason_counts"]
    if crc:
        parts = [f"{k}={v}" for k, v in crc.items()]
        lines.append("  convergence: " + " ".join(parts))
    wrc = agg.get("winning_restart_counts") or {}
    if wrc:
        parts = [f"{k}={v}" for k, v in sorted(wrc.items(), key=lambda kv: int(kv[0]) if kv[0].lstrip("-").isdigit() else 0)]
        lines.append("  winning_restart: " + " ".join(parts))
    b = agg.get("best")
    if b:
        ax = b.get("best_ax_path") or "(unknown path)"
        lines.append(f"Best: {b.get('metric_display', 'n/a')} @ {ax}")
        lines.append(f"  run: {b.get('label')}")
    return "\n".join(lines)


def stability_report_to_dict(
    runs: Sequence[RunSnapshot],
    agg: dict[str, Any],
    *,
    discovery: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "runs": [asdict(r) for r in runs],
        "aggregate": {**agg, "summary_text": format_stability_text(runs, agg)},
    }
    if discovery is not None:
        out["discovery"] = discovery
    return out


def collect_stability_report(
    paths: Sequence[Path], *, parent: Optional[Path] = None, near_floor: float = DEFAULT_NEAR_NEG_MSE
) -> Tuple[List[RunSnapshot], dict[str, Any], str, dict[str, Any]]:
    runs, meta = collect_run_snapshots(paths, parent, near_floor)
    agg = build_aggregate(runs)
    text = format_stability_text(runs, agg)
    return runs, agg, text, meta


__all__ = [
    "DEFAULT_NEAR_NEG_MSE",
    "SEARCH_PATTERNS_DESC",
    "RunSnapshot",
    "build_aggregate",
    "collect_run_snapshots",
    "collect_stability_report",
    "format_discovery_failure_message",
    "format_stability_text",
    "ingest_path",
    "resolve_input_paths",
    "resolve_scan_roots",
    "stability_report_to_dict",
]
