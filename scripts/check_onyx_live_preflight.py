from __future__ import annotations

import argparse
from urllib.parse import urlsplit, urlunsplit

from profile_onyx_task_latency import REQUEST_CAPTURE_DIR_ENV_VAR, _resolve_setting


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
        help="Optional live expert API key. Overrides AXIOM_EXPERT_API_KEY when provided.",
    )
    parser.add_argument(
        "--request-capture-dir",
        default="",
        help=f"Optional request capture directory. Overrides {REQUEST_CAPTURE_DIR_ENV_VAR} when provided.",
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
    expert_api_key = _resolve_setting(
        args.expert_api_key,
        env_name="AXIOM_EXPERT_API_KEY",
        setting_name="expert_api_key",
        required=False,
    )
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
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
