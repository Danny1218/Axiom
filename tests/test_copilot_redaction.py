"""Request capture redaction defaults."""

from __future__ import annotations

import json

import pytest

from axiom.copilot.redaction import capture_mode_from_env, redact_mapping


def test_redact_mapping_truncates_long_prompts():
    data = {"system_prompt": "x" * 800, "user_prompt": "ok", "model": "m"}
    out = redact_mapping(data, redact_prompts=True)
    assert out["model"] == "m"
    assert isinstance(out["system_prompt"], dict)
    assert out["system_prompt"]["redacted"] is True


def test_capture_mode_defaults_redacted(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AXIOM_ONYX_CAPTURE_MODE", raising=False)
    assert capture_mode_from_env() == "redacted"


def test_write_request_capture_redacts_by_default(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AXIOM_ONYX_CAPTURE_MODE", raising=False)
    from axiom.experts.onyx_qwen import _write_request_capture

    path = _write_request_capture(
        tmp_path,
        request_kind="draft",
        benchmark_task_id="t1",
        chat_url="http://127.0.0.1:8000/v1/chat/completions",
        model="m",
        system_prompt="secret " * 200,
        user_prompt="user",
        prompt_char_count=100,
        system_prompt_char_count=80,
        user_prompt_char_count=20,
        completion_overrides_applied={},
        compact_benchmark_prompt_used=False,
        payload={"messages": [{"role": "user", "content": "hello"}]},
        payload_sha256="abc",
    )
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["capture_mode"] == "redacted"
    assert isinstance(doc["system_prompt"], dict)
