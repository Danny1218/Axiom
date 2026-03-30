"""``axiom.experts.onyx_qwen`` — HTTP expert with injected ``post`` (no live server)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

pytest.importorskip("requests")
import requests

from axiom.experts.base import ExpertDraftRequest, ExpertRepairRequest, ExpertTraceSummaryRequest, SemanticExpert
from axiom.experts.onyx_qwen import (
    OnyxQwenBackend,
    OnyxQwenHTTPError,
    OnyxQwenParseError,
    OnyxQwenTimeoutError,
    OnyxQwenTransportError,
    split_ax_and_prose,
    user_prompt_draft,
)


def _ok_response(content: str, *, status: int = 200, text: str = "") -> Mock:
    r = Mock()
    r.status_code = status
    r.text = text
    r.json.return_value = {"choices": [{"message": {"content": content}}]}
    return r


def test_backend_satisfies_semantic_expert_protocol():
    b = OnyxQwenBackend("http://h", "m", _post=lambda *a, **k: _ok_response("x"))
    assert isinstance(b, SemanticExpert)


def test_split_ax_prefers_fence():
    ax, expl = split_ax_and_prose("Intro\n```ax\na = 1.0;\n```\nOutro")
    assert ax == "a = 1.0;"
    assert expl is not None
    assert "Intro" in expl and "Outro" in expl


def test_split_ax_fallback_plain_text():
    ax, expl = split_ax_and_prose("  x = 2.0;\n")
    assert ax == "x = 2.0;"
    assert expl is None


def test_user_prompt_draft_is_deterministic():
    a = user_prompt_draft("g", {"b": 1, "a": 2})
    b = user_prompt_draft("g", {"a": 2, "b": 1})
    assert a == b
    assert '"a":2' in a and '"b":1' in a


def test_successful_draft():
    calls: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return _ok_response("```ax\ny = 1.0;\n```")

    b = OnyxQwenBackend("http://qwen:9999", "qwen-turbo", _post=fake_post)
    out = b.draft_program(ExpertDraftRequest("minimal constant", context={"k": "v"}))
    assert out.ax_source == "y = 1.0;"
    assert out.backend_name == "onyx_qwen"
    assert calls[0]["url"] == "http://qwen:9999/v1/chat/completions"
    assert calls[0]["json"]["model"] == "qwen-turbo"
    assert "minimal constant" in calls[0]["json"]["messages"][1]["content"]


def test_successful_repair():
    def fake_post(url, json=None, headers=None, timeout=None):
        return _ok_response("```ax\nfixed = 1.0;\n```")

    b = OnyxQwenBackend("http://h/v1/", "m", _post=fake_post)
    out = b.repair_program(
        ExpertRepairRequest(
            goal="fix",
            current_program="bad = ;",
            error_report="parse error",
            context={},
        )
    )
    assert "fixed = 1.0" in out.ax_source


def test_summarize_trace_returns_str():
    def fake_post(url, json=None, headers=None, timeout=None):
        return _ok_response("Summary: trace shows x=1.")

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    s = b.summarize_trace(
        ExpertTraceSummaryRequest(
            goal="g",
            program="x = 1.0;",
            trace={"x": 1.0},
            metrics={},
            context={},
        )
    )
    assert s == "Summary: trace shows x=1."


def test_auth_header_bearer_when_api_key():
    seen: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen["headers"] = dict(headers or {})
        return _ok_response("ok")

    b = OnyxQwenBackend("http://h", "m", api_key="secret-token", _post=fake_post)
    b.draft_program(ExpertDraftRequest("x"))
    assert seen["headers"]["Authorization"] == "Bearer secret-token"
    assert seen["headers"]["Content-Type"] == "application/json"


def test_no_auth_header_without_key():
    seen: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        seen["headers"] = dict(headers or {})
        return _ok_response("x")

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    b.draft_program(ExpertDraftRequest("x"))
    assert "Authorization" not in seen["headers"]


def test_malformed_json_raises_parse_error():
    r = Mock()
    r.status_code = 200
    r.text = "not json"
    r.json.side_effect = ValueError("nope")

    def fake_post(url, json=None, headers=None, timeout=None):
        return r

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    with pytest.raises(OnyxQwenParseError, match="valid JSON"):
        b.draft_program(ExpertDraftRequest("g"))


def test_missing_choices_raises_parse_error():
    def fake_post(url, json=None, headers=None, timeout=None):
        m = Mock()
        m.status_code = 200
        m.text = ""
        m.json.return_value = {"choices": []}
        return m

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    with pytest.raises(OnyxQwenParseError, match="missing choices"):
        b.draft_program(ExpertDraftRequest("g"))


def test_http_error_surfaces_clean_exception():
    def fake_post(url, json=None, headers=None, timeout=None):
        m = Mock()
        m.status_code = 503
        m.text = "upstream unavailable"
        return m

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    with pytest.raises(OnyxQwenHTTPError) as ei:
        b.draft_program(ExpertDraftRequest("g"))
    assert ei.value.status_code == 503
    assert "upstream" in ei.value.body_snippet


def test_timeout_surfaces_timeout_error():
    def fake_post(url, json=None, headers=None, timeout=None):
        raise requests.exceptions.ReadTimeout("read timed out")

    b = OnyxQwenBackend("http://h", "m", timeout=5.0, _post=fake_post)
    with pytest.raises(OnyxQwenTimeoutError, match="timed out"):
        b.draft_program(ExpertDraftRequest("g"))


def test_transport_error_wrapped():
    def fake_post(url, json=None, headers=None, timeout=None):
        raise requests.exceptions.ConnectionError("refused")

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    with pytest.raises(OnyxQwenTransportError, match="refused"):
        b.draft_program(ExpertDraftRequest("g"))


def test_plain_response_used_as_ax_when_no_fence():
    def fake_post(url, json=None, headers=None, timeout=None):
        return _ok_response("alpha = 3.0;")

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    out = b.draft_program(ExpertDraftRequest("g"))
    assert out.ax_source == "alpha = 3.0;"


def test_custom_chat_path_url():
    got: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        got["url"] = url
        return _ok_response("```ax\nz=0;\n```")

    OnyxQwenBackend("http://host/prefix/", "m", chat_path="openai/v1/chat", _post=fake_post).draft_program(
        ExpertDraftRequest("g")
    )
    assert got["url"] == "http://host/prefix/openai/v1/chat"
