"""``axiom.experts.onyx_qwen`` — HTTP expert with injected ``post`` (no live server)."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

pytest.importorskip("requests")
import requests

from axiom.experts.base import ExpertDraftRequest, ExpertRepairRequest, ExpertTraceSummaryRequest, SemanticExpert
from axiom.experts.onyx_qwen import (
    COMPLETION_OVERRIDES_CONTEXT_KEY,
    EXAMPLES_SEMANTICS_BLOCK,
    EXACT_SYMBOLIC_MATH_BLOCK,
    REPAIR_NEURAL_TO_SYMBOLIC_BLOCK,
    REPAIR_UNROLL_COLLAPSE_BLOCK,
    OnyxQwenBackend,
    ax_source_metadata_flags,
    normalize_onyx_chat_completion_payload,
    OnyxQwenHTTPError,
    OnyxQwenParseError,
    OnyxQwenTimeoutError,
    OnyxQwenTransportError,
    SYSTEM_DRAFT,
    SYSTEM_REPAIR,
    split_ax_and_prose,
    user_prompt_draft,
    user_prompt_repair,
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
    r = split_ax_and_prose("Intro\n```ax\na = 1.0;\n```\nOutro")
    assert r.ax_source == "a = 1.0;"
    assert r.prose is not None
    assert "Intro" in r.prose and "Outro" in r.prose
    assert r.extraction.get("extraction_mode") == "fenced_ax"


def test_split_ax_fallback_plain_text():
    r = split_ax_and_prose("  x = 2.0;\n")
    assert r.ax_source == "x = 2.0;"
    assert r.prose is None


def test_split_ax_prefers_second_ax_fence_over_macaulay2():
    """When Macaulay2 appears in another fence, use the explicit ``ax`` block."""
    raw = """See Macaulay2:
```m2
R = QQ[x,y];
ideal(x^2-y)
```
Real `.ax`:
```ax
out = max(0.0, min(1.0, x));
```
"""
    r = split_ax_and_prose(raw)
    assert "max(0.0" in r.ax_source
    assert "QQ" not in r.ax_source
    assert r.extraction.get("extraction_mode") == "fenced_ax"


def test_split_ax_skips_m2_fence_uses_heuristic_line():
    """Only m2 fence: pick code-like line outside the fence."""
    raw = """Explanation paragraph that is not code and has many words in a row without semicolons.
```m2
R = QQ[x,y];
```
Use this line only:
y = x * 2.0;
"""
    r = split_ax_and_prose(raw)
    assert r.ax_source.strip() == "y = x * 2.0;"
    assert r.extraction.get("extraction_mode") == "heuristic_lines"
    assert r.prose is not None
    assert "Explanation" in r.prose


def test_split_ax_prose_plus_code_prefers_code_block():
    raw = """Here is the policy you asked for.
The output variable should be bounded.

score = neural([a, b]);
result = max(0.0, min(1.0, score));
"""
    r = split_ax_and_prose(raw)
    assert "neural(" in r.ax_source
    assert "Here is the policy" in (r.prose or "")
    assert r.extraction.get("extraction_mode") == "heuristic_lines"


def test_split_ax_heuristic_prefers_semicolon_and_keywords():
    raw = """Blah blah prose without any semicolons and lots of filler text here.
z = 1.0;
if (z > 0.5) { out = z; } else { out = 0.0; }
"""
    r = split_ax_and_prose(raw)
    assert "if (" in r.ax_source
    assert r.ax_source.strip().endswith("}")
    assert "Blah blah" in (r.prose or "")


def test_system_draft_rejects_macaulay2_and_cas():
    assert "Macaulay2" in SYSTEM_DRAFT
    assert "theorem prover" in SYSTEM_DRAFT.lower() or "not a theorem" in SYSTEM_DRAFT.lower()
    assert "not Axiom CAS" in SYSTEM_DRAFT or "computer algebra" in SYSTEM_DRAFT.lower()


def test_system_repair_requires_ax_only():
    assert "ONLY" in SYSTEM_REPAIR or "only" in SYSTEM_REPAIR.lower()
    assert ":=" in SYSTEM_REPAIR or "colon" in SYSTEM_REPAIR.lower()
    assert "print" in SYSTEM_REPAIR.lower()


def test_draft_and_repair_prompts_forbid_backend_only_failure_patterns():
    assert "input.a" in SYSTEM_DRAFT and "input.a" in SYSTEM_REPAIR
    assert "<=" in SYSTEM_DRAFT and "<=" in SYSTEM_REPAIR
    assert "else if" in SYSTEM_DRAFT and "else if" in SYSTEM_REPAIR
    assert "0.0 0" in SYSTEM_DRAFT and "1.0 0" in SYSTEM_REPAIR


def test_prompt_contains_backend_only_fewshot_rewrites():
    p = user_prompt_draft("g", {"example_input_rows": [{"a": 1.0}], "expected_outputs": [{"score": 1.0}]})
    assert "score = max(min(input.a, input.b), input.c);" in p
    assert "score = max(min(a, b), c);" in p
    assert "copy this structure exactly" in p
    assert "else if (x < 1.0)" in p
    assert "else { if (x < 1.0)" in p
    assert "0.9999<x<1" in p
    assert "y == x;" in p
    assert "y = x;" in p
    assert "if (x != 0.0 && x < 2.0)" in p
    assert "if (x <= 0) { y = 0.0; }" in p
    assert "if (x < 0.0) { y = 0.0; } else { y = x; }" in p
    assert p.rstrip().endswith(
        "Return only valid .ax source. Every assignment statement must end with a semicolon."
    )


def test_user_prompt_draft_is_deterministic():
    a = user_prompt_draft("g", {"b": 1, "a": 2})
    b = user_prompt_draft("g", {"a": 2, "b": 1})
    assert a == b
    assert '"a":2' in a and '"b":1' in a
    assert "Syntax summary" in a and "neural(features" in a


def test_user_prompt_repair_is_deterministic():
    a = user_prompt_repair("g", "x = 1.0;", "err", {"b": 1, "a": 2})
    b = user_prompt_repair("g", "x = 1.0;", "err", {"a": 2, "b": 1})
    assert a == b
    assert "Syntax summary" in a and "Few-shot repair" in a


def test_user_prompt_repair_includes_examples_semantics_when_expected_outputs():
    ctx = {"expected_outputs": [{"y": 2.0}], "example_input_rows": [{"x": 1.0}]}
    p = user_prompt_repair("goal", "bad", "err", ctx)
    assert "Example-driven semantics" in p
    assert EXAMPLES_SEMANTICS_BLOCK.splitlines()[0] in p
    assert "y = x * 2.0" in p and "x = 5.0" in p
    assert "SINGLE general" in p
    assert "x_0" in p and "output(" in p


def test_user_prompt_draft_includes_examples_semantics_when_rows_present():
    ctx = {"example_input_rows": [{"x": 1.0}], "expected_outputs": []}
    p = user_prompt_draft("goal", ctx)
    assert "Example-driven semantics" in p
    assert "Do NOT emit row-indexed variables" in p
    assert "Prefer direct symbolic arithmetic" in p


def test_user_prompt_draft_includes_exact_symbolic_block_when_flag_and_examples():
    ctx = {
        "example_input_rows": [{"x": 1.0}],
        "expected_outputs": [{"y": 2.0}],
        "exact_symbolic_examples_task": True,
    }
    p = user_prompt_draft("risk_score = max(0, min(1, x));", ctx)
    assert EXACT_SYMBOLIC_MATH_BLOCK.splitlines()[0] in p
    assert "Do NOT" in p and "neural" in p.lower()


def test_user_prompt_repair_includes_neural_to_symbolic_when_neural_and_examples():
    ctx = {"expected_outputs": [{"y": 1.0}], "example_input_rows": [{}], "exact_symbolic_examples_task": True}
    p = user_prompt_repair("goal", 'y = neural([1.0], "liquid");\n', "err", ctx)
    assert REPAIR_NEURAL_TO_SYMBOLIC_BLOCK.splitlines()[0] in p
    assert "0.7 * risk_a" in p or "risk_score = max(min" in p


def test_ax_source_metadata_flags_neural_and_suspicious_numeric():
    src = 'risk_score = neural([0.7*risk_a, 03*risk_b], "liquid");\n'
    m = ax_source_metadata_flags(src)
    assert m.get("uses_neural") is True
    assert m.get("suspicious_numeric_literal_warning") is True


def test_split_ax_metadata_suspicious_numeric_literal():
    r = split_ax_and_prose("```ax\ny = 03 * x;\n```")
    assert r.extraction.get("suspicious_numeric_literal_warning") is True


def test_user_prompt_no_examples_block_when_empty_lists():
    p = user_prompt_repair("g", "x=1;", "e", {"expected_outputs": [], "example_input_rows": []})
    assert "Example-driven semantics" not in p
    assert "Repair focus" not in p


def test_user_prompt_repair_includes_unroll_collapse_when_indexed_names():
    p = user_prompt_repair("g", "y_0 = x_0 * 2.0;", "e", {})
    assert REPAIR_UNROLL_COLLAPSE_BLOCK.splitlines()[0] in p
    assert "Collapse" in p


def test_user_prompt_repair_includes_unroll_collapse_when_output_call():
    p = user_prompt_repair("g", "y = 1.0;\noutput(y);", "e", {})
    assert "Repair focus" in p
    assert "output(...)" in p or "output(`" in p


def test_user_prompt_repair_unroll_hint_is_deterministic():
    a = user_prompt_repair("g", "y_0 = x_0;", "e", {"b": 1, "a": 2})
    b = user_prompt_repair("g", "y_0 = x_0;", "e", {"a": 2, "b": 1})
    assert a == b


def test_split_ax_metadata_indexed_and_output_warnings():
    r = split_ax_and_prose("```ax\ny_0 = x_0 * 2.0;\noutput(z);\n```")
    assert r.extraction.get("indexed_variable_warning") is True
    assert r.extraction.get("output_call_warning") is True
    assert "output(z)" in r.ax_source


def test_split_ax_strips_leading_ax_line_inside_fence():
    r = split_ax_and_prose("```ax\nax\ny = x * 2.0;\n```")
    assert r.ax_source.strip() == "y = x * 2.0;"
    assert r.extraction.get("stripped_language_tag") == "ax"
    assert r.extraction.get("code_line_count") == 1


def test_split_ax_bare_javascript_line_then_code():
    r = split_ax_and_prose("javascript\ny = x * 2.0;\n")
    assert r.ax_source.strip() == "y = x * 2.0;"
    assert r.extraction.get("stripped_language_tag") == "javascript"
    assert r.extraction.get("extraction_mode") == "heuristic_lines"


def test_draft_metadata_code_line_count_after_strip():
    def fake_post(url, json=None, headers=None, timeout=None):
        return _ok_response("```ax\nax\na = 1.0;\nb = 2.0;\n```")

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    out = b.draft_program(ExpertDraftRequest("g"))
    assert out.ax_source.strip() == "a = 1.0;\nb = 2.0;"
    assert out.metadata.get("stripped_language_tag") == "ax"
    assert out.metadata.get("code_line_count") == 2


def test_forbidden_tokens_in_metadata_from_draft():
    def fake_post(url, json=None, headers=None, timeout=None):
        return _ok_response("```ax\nbad := 1;\nprint(x);\n```")

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    out = b.draft_program(ExpertDraftRequest("g"))
    assert "forbidden_tokens_detected" in out.metadata
    assert "assign_colon_eq" in out.metadata["forbidden_tokens_detected"]
    assert "print_call" in out.metadata["forbidden_tokens_detected"]


def test_split_ax_normalizes_colon_eq_and_trailing_dot_float():
    r = split_ax_and_prose("```ax\ny := x + 2.;\n```")
    assert r.ax_source == "y = x + 2.0;"
    assert r.extraction.get("normalized_colon_eq") is True
    assert r.extraction.get("normalized_trailing_dot_float") is True
    # Detected from raw response before normalization.
    assert "assign_colon_eq" in (r.extraction.get("forbidden_tokens_detected") or [])


def test_split_ax_normalizes_statement_eq_eq_assignment_only_for_statement_lines():
    r = split_ax_and_prose("```ax\nif (x == 0) {\n    y == x + 2.;\n}\n```")
    assert r.ax_source == "if (x == 0) {\n    y = x + 2.0;\n}"
    assert r.extraction.get("normalized_statement_eq_eq_assignment") is True
    assert r.extraction.get("normalized_trailing_dot_float") is True


def test_split_ax_normalization_does_not_introduce_dotted_access():
    r = split_ax_and_prose("```ax\nscore := max(min(a, b), c);\n```")
    assert r.ax_source == "score = max(min(a, b), c);"
    assert "input." not in r.ax_source


def test_draft_metadata_includes_normalization_flags():
    def fake_post(url, json=None, headers=None, timeout=None):
        return _ok_response("```ax\nz := 2.;\n```")

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    out = b.draft_program(ExpertDraftRequest("normalize"))
    assert out.ax_source == "z = 2.0;"
    assert out.metadata.get("normalized_colon_eq") is True
    assert out.metadata.get("normalized_trailing_dot_float") is True


def test_draft_metadata_includes_statement_eq_eq_assignment_normalization_flag():
    def fake_post(url, json=None, headers=None, timeout=None):
        return _ok_response("```ax\ny == x;\nif (x == 0) { y = 0.0; }\n```")

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    out = b.draft_program(ExpertDraftRequest("normalize"))
    assert out.ax_source == "y = x;\nif (x == 0) { y = 0.0; }"
    assert out.metadata.get("normalized_statement_eq_eq_assignment") is True


def test_successful_draft():
    calls: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return _ok_response("```ax\ny = 1.0;\n```")

    b = OnyxQwenBackend("http://qwen:9999", "qwen-turbo", _post=fake_post)
    out = b.draft_program(ExpertDraftRequest("minimal constant", context={"k": "v"}))
    assert out.ax_source == "y = 1.0;"
    assert out.backend_name == "onyx_qwen"
    assert out.metadata.get("extraction_mode") == "fenced_ax"
    assert calls[0]["url"] == "http://qwen:9999/v1/chat/completions"
    assert calls[0]["json"]["model"] == "qwen-turbo"
    assert "minimal constant" in calls[0]["json"]["messages"][1]["content"]
    assert SYSTEM_DRAFT in calls[0]["json"]["messages"][0]["content"]


def test_repair_completion_overrides_merged_and_stripped_from_user_json():
    calls: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return _ok_response("```ax\nz = 1.0;\n```")

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    from axiom.experts.onyx_qwen import COMPLETION_OVERRIDES_CONTEXT_KEY

    b.repair_program(
        ExpertRepairRequest(
            "g",
            current_program="a=1;",
            error_report="e",
            context={COMPLETION_OVERRIDES_CONTEXT_KEY: {"temperature": 0.1, "top_p": 0.9}, "k": 1},
        )
    )
    assert calls[0].get("temperature") == 0.1
    assert calls[0].get("top_p") == 0.9
    user = calls[0]["messages"][1]["content"]
    assert COMPLETION_OVERRIDES_CONTEXT_KEY not in user


def test_draft_completion_overrides_merged_and_stripped_from_user_json():
    calls: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return _ok_response("```ax\ny = 1.0;\n```")

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    b.draft_program(
        ExpertDraftRequest(
            "goal-text",
            context={COMPLETION_OVERRIDES_CONTEXT_KEY: {"temperature": 0}, "extra": 1},
        )
    )
    assert "temperature" not in calls[0]
    assert calls[0].get("do_sample") is False
    user = calls[0]["messages"][1]["content"]
    assert COMPLETION_OVERRIDES_CONTEXT_KEY not in user
    assert "extra" in user


def test_normalize_onyx_chat_completion_payload_zero_temperature_greedy():
    p = {"model": "m", "messages": [], "temperature": 0.0}
    normalize_onyx_chat_completion_payload(p)
    assert "temperature" not in p
    assert p.get("do_sample") is False


def test_normalize_onyx_chat_completion_payload_negative_temperature_greedy():
    p = {"model": "m", "messages": [], "temperature": -0.5, "top_p": 0.9}
    normalize_onyx_chat_completion_payload(p)
    assert "temperature" not in p and "top_p" not in p
    assert p.get("do_sample") is False


def test_normalize_onyx_chat_completion_payload_positive_unchanged():
    p = {"model": "m", "messages": [], "temperature": 0.7, "top_p": 0.95}
    normalize_onyx_chat_completion_payload(p)
    assert p["temperature"] == 0.7 and p["top_p"] == 0.95
    assert "do_sample" not in p


def test_normalize_onyx_chat_completion_payload_no_temperature_no_op():
    p = {"model": "m", "messages": [], "max_tokens": 10}
    normalize_onyx_chat_completion_payload(p)
    assert p == {"model": "m", "messages": [], "max_tokens": 10}


def test_greedy_completion_overrides_drop_top_p_from_http_payload():
    calls: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return _ok_response("```ax\ny = 1.0;\n```")

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    b.draft_program(
        ExpertDraftRequest(
            "g",
            context={
                COMPLETION_OVERRIDES_CONTEXT_KEY: {"temperature": 0.0, "top_p": 0.88},
            },
        )
    )
    assert "temperature" not in calls[0] and "top_p" not in calls[0]
    assert calls[0].get("do_sample") is False


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
    assert out.metadata.get("extraction_mode") == "heuristic_lines"


def test_split_ax_plain_prose_no_code_is_plain_fallback():
    r = split_ax_and_prose("This is only natural language with no semicolons or assignments.")
    assert r.ax_source == "This is only natural language with no semicolons or assignments."
    assert r.extraction.get("extraction_mode") == "plain_fallback"


def test_custom_chat_path_url():
    got: dict = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        got["url"] = url
        return _ok_response("```ax\nz=0;\n```")

    OnyxQwenBackend("http://host/prefix/", "m", chat_path="openai/v1/chat", _post=fake_post).draft_program(
        ExpertDraftRequest("g")
    )
    assert got["url"] == "http://host/prefix/openai/v1/chat"


def test_build_onyx_qwen_expert_accepts_timeout():
    from axiom.copilot.backend import build_onyx_qwen_expert

    b = build_onyx_qwen_expert(url="http://h", model="m", timeout=41.25)
    assert b._timeout == 41.25
