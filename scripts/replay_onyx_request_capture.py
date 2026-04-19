from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]

REPLAY_URL_ENV_VAR = "AXIOM_ONYX_REPLAY_URL"
REPLAY_API_KEY_ENV_VAR = "AXIOM_ONYX_REPLAY_API_KEY"


def _load_capture(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _headers_from_env() -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(REPLAY_API_KEY_ENV_VAR, "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _request_id_from_artifact(artifact: dict, payload_sha: str) -> str:
    request_id = str(artifact.get("request_id") or "").strip()
    if request_id:
        return request_id
    sha_prefix = payload_sha[:12] if payload_sha else "unknown"
    return f"replay-{sha_prefix}"


def replay_capture(path: Path) -> int:
    if requests is None:
        raise SystemExit('requests is required: pip install -e ".[copilot]"')
    artifact = _load_capture(path)
    payload = artifact["payload"]
    chat_url = os.environ.get(REPLAY_URL_ENV_VAR, "").strip() or str(artifact["chat_url"])
    payload_sha = str(artifact.get("payload_sha256") or "")
    request_id = _request_id_from_artifact(artifact, payload_sha)
    headers = _headers_from_env()
    if payload_sha:
        headers["X-Payload-SHA256"] = payload_sha
    headers["X-Request-ID"] = request_id
    response = requests.post(chat_url, json=payload, headers=headers, timeout=120.0)
    print(f"request_id: {request_id}")
    print(f"payload_sha256: {payload_sha}")
    print(f"status_code: {response.status_code}")
    if response.status_code >= 400:
        snippet = response.text if isinstance(response.text, str) else ""
        print(f"failure_snippet: {snippet[:2000]}")
        return 1
    try:
        body = response.json()
        print(json.dumps(body, indent=2, sort_keys=True))
    except ValueError:
        print(response.text if isinstance(response.text, str) else "")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a captured Onyx chat/completions payload.")
    parser.add_argument("capture_json", help="Path to a Phase 106 Onyx request-capture artifact JSON file.")
    args = parser.parse_args(argv)
    return replay_capture(Path(args.capture_json))


if __name__ == "__main__":
    raise SystemExit(main())
