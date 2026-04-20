from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from profile_onyx_task_latency import (  # noqa: E402 - loads src/ onto sys.path before axiom imports
    REQUEST_CAPTURE_DIR_ENV_VAR,
    _api_key_fingerprint,
    _build_draft_request,
    _resolve_expert_api_key,
    _resolve_live_config,
    _resolve_setting,
)

from axiom.copilot.backend import build_copilot_expert  # noqa: E402
from axiom.experts.onyx_qwen import (  # noqa: E402
    OnyxQwenHTTPError,
    OnyxQwenTimeoutError,
    OnyxQwenTransportError,
)

_ROOT = Path(__file__).resolve().parents[1]


def _safe_url_display(url: str) -> str:
    parts = urlsplit(url)
    host = parts.hostname or ""
    port = f":{parts.port}" if parts.port is not None else ""
    netloc = f"{host}{port}" if host else ""
    return urlunsplit((parts.scheme, netloc, parts.path, "", ""))


def _quoted(text: str) -> str:
    return f'"{text}"'


def _resolved_or_placeholder(value: str | None, placeholder: str) -> str:
    return value if value else placeholder


def _print_probe_field(label: str, value: object | None) -> None:
    if value is None or value == "":
        print(f"{label}: n/a")
    else:
        print(f"{label}: {value}")


def _run_auth_probe(args: argparse.Namespace) -> int:
    """One draft call via the same stack as profile_onyx_task_latency. Exit 0 only on success."""
    ns = argparse.Namespace(
        expert_url=args.expert_url,
        expert_model=args.expert_model,
        expert_api_key=args.expert_api_key,
        expert_api_key_file=args.expert_api_key_file,
        request_capture_dir=args.request_capture_dir,
    )
    url, model, api_key, capture_dir = _resolve_live_config(ns)
    expert = build_copilot_expert(
        "onyx-qwen",
        expert_url=url,
        expert_model=model,
        expert_api_key=api_key,
        timeout=45.0,
    )
    task_json = _ROOT / "benchmarks" / "copilot_symbolic_robustness_ambiguity_stress_tasks.json"
    _task, draft_req = _build_draft_request(
        expert=expert,
        task_id="noisy_affine_thermometer",
        task_json=task_json,
        capture_dir=capture_dir,
        max_tokens=16,
    )
    try:
        resp = expert.draft_program(draft_req)
        meta = dict(resp.metadata or {})
        print("probe result: success")
        sc = meta.get("status_code")
        if sc is not None:
            try:
                print(f"status_code: {int(sc)}")
            except (TypeError, ValueError):
                print("status_code: n/a")
        else:
            print("status_code: n/a")
        _print_probe_field("request_id", meta.get("request_id"))
        _print_probe_field("request_capture_path", meta.get("request_capture_path"))
        return 0
    except Exception as exc:
        meta = dict(getattr(exc, "metadata", {}) or {})
        sc_val = meta.get("status_code")
        if sc_val is None and isinstance(exc, OnyxQwenHTTPError):
            sc_val = getattr(exc, "status_code", None)
        if isinstance(exc, OnyxQwenTimeoutError):
            label = "timeout"
        elif isinstance(exc, OnyxQwenTransportError):
            label = "transport"
        elif isinstance(exc, OnyxQwenHTTPError):
            try:
                sc_int = int(sc_val) if sc_val is not None else None
            except (TypeError, ValueError):
                sc_int = None
            if sc_int == 401:
                label = "unauthorized"
            elif sc_int == 403:
                label = "forbidden"
            else:
                label = "http_error"
        else:
            label = "http_error"
        print(f"probe result: {label}")
        if sc_val is not None:
            try:
                print(f"status_code: {int(sc_val)}")
            except (TypeError, ValueError):
                print("status_code: n/a")
        else:
            print("status_code: n/a")
        _print_probe_field("request_id", meta.get("request_id"))
        _print_probe_field("request_capture_path", meta.get("request_capture_path"))
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check whether Onyx live latency tooling is configured.")
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
        "--probe",
        action="store_true",
        help="After resolving config, run one minimal live draft (same path as profile_onyx_task_latency). Exit 0 only on success.",
    )
    args = parser.parse_args(argv)

    expert_url = _resolve_setting(
        args.expert_url,
        env_name="AXIOM_EXPERT_URL",
        setting_name="expert_url",
        required=False,
    )
    expert_model = _resolve_setting(
        args.expert_model,
        env_name="AXIOM_EXPERT_MODEL",
        setting_name="expert_model",
        required=False,
    )
    expert_api_key = _resolve_expert_api_key(str(args.expert_api_key), str(args.expert_api_key_file))
    request_capture_dir = _resolve_setting(
        args.request_capture_dir,
        env_name=REQUEST_CAPTURE_DIR_ENV_VAR,
        setting_name="request_capture_dir",
        required=False,
    )

    missing: list[str] = []
    if not expert_url:
        missing.append("expert_url")
    if not expert_model:
        missing.append("expert_model")
    ready = not missing

    print(f"live execution is possible: {'yes' if ready else 'no'}")
    print(f"expert_url: {'present' if expert_url else 'missing'}")
    print(f"expert_model: {'present' if expert_model else 'missing'}")
    print(f"expert_api_key: {'present' if expert_api_key else 'missing (optional)'}")
    if expert_api_key:
        print(f"expert_api_key_fingerprint: {_api_key_fingerprint(expert_api_key)}")
    if missing:
        for setting in missing:
            print(f"missing required setting: {setting}")

    print(f"resolved expert_url: {_resolved_or_placeholder(_safe_url_display(expert_url) if expert_url else None, 'n/a')}")
    print(f"resolved expert_model: {_resolved_or_placeholder(expert_model, 'n/a')}")
    print(f"request_capture_dir: {_resolved_or_placeholder(request_capture_dir, 'n/a')}")

    sweep_parts = [
        "powershell -ExecutionPolicy Bypass -File .\\scripts\\sweep_robustness_task_latency.ps1",
        "  -TaskId noisy_affine_thermometer",
        f"  -ExpertUrl {_quoted(_resolved_or_placeholder(expert_url, '<set --expert-url or AXIOM_EXPERT_URL>'))}",
        f"  -ExpertModel {_quoted(_resolved_or_placeholder(expert_model, '<set --expert-model or AXIOM_EXPERT_MODEL>'))}",
    ]
    if expert_api_key:
        sweep_parts.append('  -ExpertApiKey "<redacted>"')
    sweep_parts.append("  -Repeats 3")
    if request_capture_dir:
        sweep_parts.append(f"  -RequestCaptureDir {_quoted(request_capture_dir)}")
    sweep_parts.append('  -OutDir ".\\debug_onyx_latency_sweeps"')
    summarize_cmd = 'python .\\scripts\\summarize_onyx_latency_sweeps.py ".\\debug_onyx_latency_sweeps"'

    print("")
    print("next command: sweep wrapper")
    print("next script: scripts/sweep_robustness_task_latency.ps1")
    print(" `\n".join(sweep_parts))
    print("")
    print("next command: summarizer")
    print("next script: scripts/summarize_onyx_latency_sweeps.py")
    print(summarize_cmd)

    if args.probe:
        print("")
        if not expert_url or not expert_model:
            print("probe result: skipped (missing expert_url or expert_model)")
            return 1
        return _run_auth_probe(args)

    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
