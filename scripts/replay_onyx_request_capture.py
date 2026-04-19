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


def replay_capture(path: Path) -> int:
    if requests is None:
        raise SystemExit('requests is required: pip install -e ".[copilot]"')
    artifact = _load_capture(path)
    payload = artifact["payload"]
    chat_url = os.environ.get(REPLAY_URL_ENV_VAR, "").strip() or str(artifact["chat_url"])
    payload_sha = str(artifact.get("payload_sha256") or "")
    response = requests.post(chat_url, json=payload, headers=_headers_from_env(), timeout=120.0)
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
