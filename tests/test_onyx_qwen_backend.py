"""``axiom.experts.onyx_qwen`` — HTTP expert with injected ``post`` (no live server)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("requests")
import requests

from axiom.experts.base import ExpertDraftRequest, ExpertRepairRequest, ExpertTraceSummaryRequest, SemanticExpert
from axiom.experts.onyx_qwen import (
    COMPLETION_OVERRIDES_CONTEXT_KEY,
    DRAFT_FEWSHOT,
    EXAMPLES_SEMANTICS_BLOCK,
    EXACT_SYMBOLIC_MATH_BLOCK,
    ROBUSTNESS_AMBIGUITY_FALLBACK_BLOCK,
    ROBUSTNESS_AMBIGUITY_FALLBACK_EXAMPLES_BLOCK,
    ROBUSTNESS_AMBIGUITY_REPAIR_CLEANUP_BLOCK,
    REQUEST_CAPTURE_DIR_CONTEXT_KEY,
    REQUEST_CAPTURE_DIR_ENV_VAR,
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


@pytest.fixture(autouse=True)
def _onyx_capture_full_for_contract_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing capture contract tests expect full prompts; redaction is tested separately."""
    monkeypatch.setenv("AXIOM_ONYX_CAPTURE_MODE", "full")


def _ok_response(
    content: str,
    *,
    status: int = 200,
    text: str = "",
    headers: dict[str, str] | None = None,
    body: dict | None = None,
) -> Mock:
    r = Mock()
    r.status_code = status
    r.text = text
    r.headers = headers or {}
    r.json.return_value = body or {"choices": [{"message": {"content": content}}]}
    return r


def _read_capture(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _assert_capture_timing_fields(
    data: dict,
    *,
    timeout_seconds: float,
    max_tokens: int | None,
    has_response_received_at: bool,
) -> None:
    assert isinstance(data.get("request_started_at"), str)
    assert data.get("request_started_at", "").endswith("Z")
    assert data.get("timeout_seconds") == timeout_seconds
    assert isinstance(data.get("elapsed_seconds"), float)
    assert data.get("elapsed_seconds") >= 0.0
    if max_tokens is None:
        assert "max_tokens" not in data
    else:
        assert data.get("max_tokens") == max_tokens
    if has_response_received_at:
        assert isinstance(data.get("response_received_at"), str)
        assert data.get("response_received_at", "").endswith("Z")
    else:
        assert "response_received_at" not in data


def _load_script_module(name: str, relative_path: str):
    root = Path(__file__).resolve().parents[1]
    script_path = root / relative_path
    spec = importlib.util.spec_from_file_location(name, script_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
    assert "Axiom CAS" in SYSTEM_DRAFT or "computer algebra" in SYSTEM_DRAFT.lower()
    assert "not generic pseudocode" in SYSTEM_DRAFT.lower()


def test_system_draft_includes_small_always_on_syntax_core():
    assert "Always-on syntax core" in SYSTEM_DRAFT
    assert "y = x * 2.0;" in SYSTEM_DRAFT
    assert "if (x > 0.0) { y = x; } else { y = 0.0; }" in SYSTEM_DRAFT
    assert "if (x < 0.0) {" in SYSTEM_DRAFT
    assert "score = max(0.0, min(a + b, 1.0));" in SYSTEM_DRAFT
    assert "Return only `.ax` source, no prose." in SYSTEM_DRAFT


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
    assert "If a piecewise program is needed, always use nested `else { if (...) { ... } else { ... } }`." in p
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


def test_system_draft_does_not_include_global_canonical_symbolic_family_anchor_block():
    anchor = (
        "If the goal matches one of these families, emit that canonical form directly and do not add "
        "validation/range-guard branches unless the goal explicitly asks for them."
    )
    assert DRAFT_FEWSHOT not in SYSTEM_DRAFT
    assert anchor not in SYSTEM_DRAFT


def test_user_prompt_draft_includes_canonical_symbolic_family_anchor_block_when_exact_symbolic():
    p = user_prompt_draft("goal", {"exact_symbolic_examples_task": True})
    anchor = (
        "If the goal matches one of these families, emit that canonical form directly and do not add "
        "validation/range-guard branches unless the goal explicitly asks for them."
    )
    assert DRAFT_FEWSHOT in p
    assert anchor in DRAFT_FEWSHOT
    assert "y = x * 2.0;" in DRAFT_FEWSHOT
    assert "risk_score = max(0.0, min(1.0, 0.7 * risk_a + 0.3 * risk_b));" in DRAFT_FEWSHOT
    assert "score = 0.5 * a + 0.3 * b + 0.2 * c;" in DRAFT_FEWSHOT
    assert "score = max(0.0, min(a + b, 1.0));" in DRAFT_FEWSHOT
    assert "y = a * b + a + 1.0;" in DRAFT_FEWSHOT
    assert "score = max(min(a, b), c);" in DRAFT_FEWSHOT
    assert "if (x < 0.0) {" in DRAFT_FEWSHOT
    assert "if (a > b) {" in DRAFT_FEWSHOT


def test_user_prompt_draft_includes_canonical_symbolic_family_anchor_block_for_known_family_context():
    p = user_prompt_draft("goal", {"benchmark_task_id": "minmax_blend"})
    assert DRAFT_FEWSHOT in p
    assert "score = max(0.0, min(a + b, 1.0));" in p


def test_user_prompt_draft_uses_compact_benchmark_mode_only_for_benchmark_context():
    ctx_plain = {"example_input_rows": [{"x": 1.0}], "expected_outputs": [{"y": 2.0}]}
    ctx_bench = {
        "benchmark_task_id": "noisy_affine_thermometer",
        "example_input_rows": [{"x": 1.0}],
        "expected_outputs": [{"y": 2.0}],
    }
    plain = user_prompt_draft("goal", ctx_plain)
    compact = user_prompt_draft("goal", ctx_bench)
    assert "Context (JSON, sorted keys):" in plain
    assert "Context (JSON, sorted keys):" not in compact
    assert "Benchmark task id: noisy_affine_thermometer" in compact
    assert len(compact) < len(plain)


def test_user_prompt_draft_includes_canonical_symbolic_family_anchor_block_for_known_family_goal():
    p = user_prompt_draft("Write .ax so score = max(a, b).", {})
    assert DRAFT_FEWSHOT in p
    assert "if (a > b) {" in p


def test_user_prompt_draft_omits_canonical_symbolic_family_anchor_block_for_unrelated_goal():
    p = user_prompt_draft("Write a while loop counter that increments i until it reaches n.", {})
    assert DRAFT_FEWSHOT not in p
    assert "score = 0.5 * a + 0.3 * b + 0.2 * c;" not in p
    assert "if (a > b) {" not in p


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


def test_user_prompt_repair_compact_benchmark_mode_keeps_hard_constraints():
    ctx = {
        "benchmark_task_id": "soft_cap_prefer_signal",
        "example_input_rows": [{"primary": 0.1, "backup": 0.0, "cap": 1.0}],
        "expected_outputs": [{"decision": 0.2}],
    }
    p = user_prompt_repair("goal", "bad", "err", ctx)
    assert "Benchmark task id: soft_cap_prefer_signal" in p
    assert "Benchmark compact syntax guardrails" in p
    assert "Never emit `:=`" in p
    assert "Never emit `:=`, `print`, `else if`" in p
    assert "Current program:" in p
    assert "Return the corrected full `.ax` program only" in p
    assert "Context (JSON, sorted keys):" not in p


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
    assert "For pure algebraic mappings, never introduce `if` / `else` / `while`." in p


def test_user_prompt_repair_includes_neural_to_symbolic_when_neural_and_examples():
    ctx = {"expected_outputs": [{"y": 1.0}], "example_input_rows": [{}], "exact_symbolic_examples_task": True}
    p = user_prompt_repair("goal", 'y = neural([1.0], "liquid");\n', "err", ctx)
    assert REPAIR_NEURAL_TO_SYMBOLIC_BLOCK.splitlines()[0] in p
    assert "0.7 * risk_a" in p or "risk_score = max(min" in p


def test_user_prompt_draft_includes_robustness_fallback_blocks_for_fourth_suite_task_id():
    ctx = {
        "benchmark_task_id": "noisy_affine_thermometer",
        "domain_context": (
            "Slight label noise is intentional. This should fall back to the expert backend "
            "instead of an exact affine fast path."
        ),
        "example_input_rows": [{"thermometer_reading": 0.0}],
        "expected_outputs": [{"adjusted": -0.19}],
    }
    p = user_prompt_draft("Take thermometer_reading and nudge it into adjusted.", ctx)
    assert ROBUSTNESS_AMBIGUITY_FALLBACK_BLOCK.splitlines()[0] in p
    assert ROBUSTNESS_AMBIGUITY_FALLBACK_EXAMPLES_BLOCK.splitlines()[0] in p
    assert "Never use `neural(...)`." in p
    assert "Never use `clip(...)`." in p
    assert "adjusted = 1.25 * thermometer_reading - 0.2;" in p
    assert "response = exposure * hedge - 0.5 * hedge + 0.25;" in p
    assert "decision = min(max(primary, backup + 0.2), cap);" in p
    assert "if (offset < -1.0) {" in p


def test_user_prompt_draft_detects_robustness_fallback_from_semantics_text():
    p = user_prompt_draft(
        "This near-miss mapping has slight row noise and should fall back safely.",
        {"domain_context": "Adversarial wording plus underdetermined examples."},
    )
    assert ROBUSTNESS_AMBIGUITY_FALLBACK_BLOCK.splitlines()[0] in p
    assert ROBUSTNESS_AMBIGUITY_FALLBACK_EXAMPLES_BLOCK.splitlines()[0] in p


def test_user_prompt_draft_fallback_mode_skips_exact_symbolic_anchors():
    ctx = {
        "fallback_expected": True,
        "exact_symbolic_examples_task": True,
        "domain_context": "Noisy underdetermined near-miss task that should fall back.",
        "example_input_rows": [{"x": 1.0}],
        "expected_outputs": [{"y": 2.0}],
    }
    p = user_prompt_draft("Use x to produce y.", ctx)
    assert ROBUSTNESS_AMBIGUITY_FALLBACK_BLOCK.splitlines()[0] in p
    assert EXAMPLES_SEMANTICS_BLOCK.splitlines()[0] in p
    assert EXACT_SYMBOLIC_MATH_BLOCK not in p
    assert DRAFT_FEWSHOT not in p


def test_user_prompt_repair_includes_robustness_cleanup_block_for_fourth_suite_fallback():
    ctx = {
        "benchmark_task_id": "soft_cap_prefer_signal",
        "fallback_expected": True,
        "domain_context": "Sparse near-miss relative to min(max(a,b), c); should fall back.",
    }
    current = (
        "if (primary > backup) { }\n"
        "else if (backup > primary) { decision = clip(primary, 0.0, cap); }\n"
        "// note\n"
        "score *= 2.0;\n"
    )
    p = user_prompt_repair("Prefer primary unless backup gets a 0.2 head start.", current, "err", ctx)
    assert ROBUSTNESS_AMBIGUITY_FALLBACK_BLOCK.splitlines()[0] in p
    assert ROBUSTNESS_AMBIGUITY_FALLBACK_EXAMPLES_BLOCK.splitlines()[0] in p
    assert ROBUSTNESS_AMBIGUITY_REPAIR_CLEANUP_BLOCK.splitlines()[0] in p
    assert "`clip(expr, low, high)` -> `max(low, min(expr, high))`" in p
    assert "`max(a, b, c)` -> `max(max(a, b), c)`" in p
    assert "`x *= y;`, `x += y;`, `x -= y;` -> explicit assignments such as `x = x * y;`" in p


def test_user_prompt_repair_includes_fallback_syntax_only_mode():
    ctx = {
        "benchmark_task_id": "soft_cap_prefer_signal",
        "fallback_expected": True,
        "domain_context": "Noisy underdetermined near-miss task that should fall back.",
    }
    p = user_prompt_repair("Prefer primary unless backup gets a 0.2 head start.", "bad", "parse err", ctx)
    assert "Fallback-only syntax repair mode" in p
    assert "Preserve the intended math and variable names." in p
    assert "Rewrite only into valid canonical `.ax`; do not add new behavior." in p
    assert "Never add `neural(...)`." in p
    assert "Never add comments or prose." in p
    assert "Never use 3-arg `max` or `min`" in p


def test_split_ax_cleanup_rewrites_comments_trailing_prose_shorthand_and_three_arg_extrema():
    raw = """```ax
score += bonus; // keep the same math
delta -= drag;
mass *= scale;
ratio /= total;
capped = max(a, b, c);
floored = min(low, mid, high);
This trailing prose should disappear.
```"""
    r = split_ax_and_prose(raw)
    assert r.ax_source == (
        "score = score + bonus;\n"
        "delta = delta - drag;\n"
        "mass = mass * scale;\n"
        "ratio = ratio / total;\n"
        "capped = max(max(a, b), c);\n"
        "floored = min(min(low, mid), high);"
    )
    assert r.extraction.get("stripped_line_comments") is True
    assert r.extraction.get("stripped_trailing_prose") is True
    assert r.extraction.get("normalized_shorthand_assignment") is True
    assert r.extraction.get("normalized_three_arg_max") is True
    assert r.extraction.get("normalized_three_arg_min") is True


def test_ax_source_metadata_flags_unsupported_fallback_surface_patterns():
    src = (
        "if (x > 0.0) { y = x; } else if (x < 0.0) { y = -x; }\n"
        "score = good if cond else bad;\n"
        "bounded = clip(score, 0.0, 1.0);\n"
        "ok = left && right;\n"
        "fallback = alt || base;\n"
    )
    m = ax_source_metadata_flags(src)
    assert m.get("unsupported_branch_surface_warning") is True
    assert m.get("inline_if_expression_warning") is True
    assert m.get("clip_call_warning") is True
    assert m.get("logical_operator_warning") is True


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


def test_draft_metadata_includes_request_diagnostics_for_benchmark_context():
    def fake_post(url, json=None, headers=None, timeout=None):
        return _ok_response("```ax\ny = 1.0;\n```")

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    out = b.draft_program(
        ExpertDraftRequest(
            "goal",
            context={
                "benchmark_task_id": "noisy_affine_thermometer",
                "example_input_rows": [{"x": 1.0}],
                "expected_outputs": [{"y": 2.0}],
                COMPLETION_OVERRIDES_CONTEXT_KEY: {"max_tokens": 64},
            },
        )
    )
    assert out.metadata.get("benchmark_task_id") == "noisy_affine_thermometer"
    assert out.metadata.get("compact_benchmark_prompt_used") is True
    assert out.metadata.get("completion_overrides_applied") == {"max_tokens": 64}
    assert out.metadata.get("prompt_char_count") == (
        out.metadata.get("system_prompt_char_count") + out.metadata.get("user_prompt_char_count")
    )
    assert out.metadata.get("http_failure_detail") is None


def test_request_capture_writes_success_artifact_via_env(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(dict(json or {}))
        return _ok_response(
            "```ax\ny = 1.0;\n```",
            headers={"x-request-id": "req-123", "content-type": "application/json"},
            body={"id": "chatcmpl-123", "choices": [{"message": {"content": "```ax\ny = 1.0;\n```"}}]},
        )

    monkeypatch.setenv(REQUEST_CAPTURE_DIR_ENV_VAR, str(tmp_path))
    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    out = b.draft_program(
        ExpertDraftRequest(
            "goal",
            context={
                "benchmark_task_id": "noisy_affine_thermometer",
                COMPLETION_OVERRIDES_CONTEXT_KEY: {"max_tokens": 64},
            },
        )
    )
    capture = _read_capture(str(out.metadata.get("request_capture_path")))
    assert capture["benchmark_task_id"] == "noisy_affine_thermometer"
    assert capture["chat_url"] == "http://h/v1/chat/completions"
    assert capture["payload"] == calls[0]
    assert capture["payload_sha256"] == out.metadata.get("payload_sha256")
    assert capture["system_prompt"] == calls[0]["messages"][0]["content"]
    assert capture["user_prompt"] == calls[0]["messages"][1]["content"]
    assert capture["prompt_char_count"] == (
        capture["system_prompt_char_count"] + capture["user_prompt_char_count"]
    )
    assert capture["request_id"] == "req-123"
    assert capture["response_id"] == "chatcmpl-123"
    assert capture["response_headers"]["x-request-id"] == "req-123"
    assert out.metadata.get("request_id") == "req-123"
    assert out.metadata.get("response_id") == "chatcmpl-123"
    _assert_capture_timing_fields(capture, timeout_seconds=120.0, max_tokens=64, has_response_received_at=True)
    _assert_capture_timing_fields(dict(out.metadata), timeout_seconds=120.0, max_tokens=64, has_response_received_at=True)
    assert "failure_kind" not in capture


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


def test_request_capture_writes_http500_artifact(tmp_path):
    def fake_post(url, json=None, headers=None, timeout=None):
        m = Mock()
        m.status_code = 500
        m.text = '{"detail":"CUDA error: out of memory"}'
        m.headers = {"x-request-id": "req-500", "content-type": "application/json"}
        return m

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    with pytest.raises(OnyxQwenHTTPError) as ei:
        b.draft_program(
            ExpertDraftRequest(
                "goal",
                context={
                    "benchmark_task_id": "noisy_affine_thermometer",
                    REQUEST_CAPTURE_DIR_CONTEXT_KEY: str(tmp_path),
                    COMPLETION_OVERRIDES_CONTEXT_KEY: {"max_tokens": 32},
                },
            )
        )
    capture = _read_capture(str(ei.value.metadata.get("request_capture_path")))
    assert capture["status_code"] == 500
    assert "CUDA error: out of memory" in capture["http_failure_detail"]
    assert capture["payload_sha256"] == ei.value.metadata.get("payload_sha256")
    assert capture["request_id"] == "req-500"
    assert capture["response_headers"]["x-request-id"] == "req-500"
    assert ei.value.metadata.get("request_id") == "req-500"
    _assert_capture_timing_fields(capture, timeout_seconds=120.0, max_tokens=32, has_response_received_at=True)
    _assert_capture_timing_fields(dict(ei.value.metadata), timeout_seconds=120.0, max_tokens=32, has_response_received_at=True)
    assert "failure_kind" not in capture


def test_request_capture_writes_timeout_artifact_during_repair(tmp_path):
    def fake_post(url, json=None, headers=None, timeout=None):
        raise requests.exceptions.ReadTimeout("read timed out")

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    with pytest.raises(OnyxQwenTimeoutError) as ei:
        b.repair_program(
            ExpertRepairRequest(
                "goal",
                current_program="y = 0.0;",
                error_report="metric mismatch",
                context={
                    "benchmark_task_id": "noisy_affine_thermometer",
                    REQUEST_CAPTURE_DIR_CONTEXT_KEY: str(tmp_path),
                    COMPLETION_OVERRIDES_CONTEXT_KEY: {"max_tokens": 64},
                },
            )
        )
    capture = _read_capture(str(ei.value.metadata.get("request_capture_path")))
    assert capture["request_kind"] == "repair"
    assert capture["failure_kind"] == "timeout"
    assert capture["exception_class"] == "ReadTimeout"
    assert "timed out" in capture["exception_message"]
    assert capture["payload_sha256"] == ei.value.metadata.get("payload_sha256")
    _assert_capture_timing_fields(capture, timeout_seconds=120.0, max_tokens=64, has_response_received_at=False)
    _assert_capture_timing_fields(dict(ei.value.metadata), timeout_seconds=120.0, max_tokens=64, has_response_received_at=False)


def test_request_capture_writes_transport_artifact_during_draft_without_secret_leak(tmp_path):
    def fake_post(url, json=None, headers=None, timeout=None):
        raise requests.exceptions.ConnectionError("refused")

    b = OnyxQwenBackend("http://h", "m", api_key="sk-secret-value", _post=fake_post)
    with pytest.raises(OnyxQwenTransportError) as ei:
        b.draft_program(
            ExpertDraftRequest(
                "goal",
                context={
                    "benchmark_task_id": "noisy_affine_thermometer",
                    REQUEST_CAPTURE_DIR_CONTEXT_KEY: str(tmp_path),
                    COMPLETION_OVERRIDES_CONTEXT_KEY: {"max_tokens": 64},
                },
            )
        )
    capture_path = str(ei.value.metadata.get("request_capture_path"))
    capture = _read_capture(capture_path)
    assert capture["request_kind"] == "draft"
    assert capture["failure_kind"] == "transport"
    assert capture["exception_class"] == "ConnectionError"
    assert "refused" in capture["exception_message"]
    text = Path(capture_path).read_text(encoding="utf-8")
    assert "sk-secret-value" not in text
    assert "Authorization" not in text


def test_http_error_carries_request_diagnostics_metadata():
    def fake_post(url, json=None, headers=None, timeout=None):
        m = Mock()
        m.status_code = 500
        m.text = '{"detail":"CUDA error: out of memory"}'
        return m

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    with pytest.raises(OnyxQwenHTTPError) as ei:
        b.draft_program(
            ExpertDraftRequest(
                "goal",
                context={
                    "benchmark_task_id": "noisy_affine_thermometer",
                    COMPLETION_OVERRIDES_CONTEXT_KEY: {"max_tokens": 32, "temperature": 0},
                },
            )
        )
    meta = ei.value.metadata
    assert meta.get("benchmark_task_id") == "noisy_affine_thermometer"
    assert meta.get("compact_benchmark_prompt_used") is True
    assert meta.get("completion_overrides_applied") == {"do_sample": False, "max_tokens": 32}
    assert "CUDA error: out of memory" in (meta.get("http_failure_detail") or "")


def test_request_capture_redacts_api_key_and_authorization(tmp_path):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _ok_response("```ax\ny = 1.0;\n```")

    b = OnyxQwenBackend("http://h", "m", api_key="sk-secret-value", _post=fake_post)
    out = b.draft_program(
        ExpertDraftRequest(
            "goal",
            context={REQUEST_CAPTURE_DIR_CONTEXT_KEY: str(tmp_path)},
        )
    )
    text = Path(str(out.metadata.get("request_capture_path"))).read_text(encoding="utf-8")
    assert "sk-secret-value" not in text
    assert "Authorization" not in text


def test_request_capture_payload_reflects_normalized_overrides(tmp_path):
    def fake_post(url, json=None, headers=None, timeout=None):
        return _ok_response("```ax\ny = 1.0;\n```")

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    out = b.draft_program(
        ExpertDraftRequest(
            "goal",
            context={
                REQUEST_CAPTURE_DIR_CONTEXT_KEY: str(tmp_path),
                COMPLETION_OVERRIDES_CONTEXT_KEY: {"temperature": 0.0, "top_p": 0.88},
            },
        )
    )
    capture = _read_capture(str(out.metadata.get("request_capture_path")))
    assert capture["completion_overrides_applied"] == {"do_sample": False}
    assert capture["payload"]["do_sample"] is False
    assert "temperature" not in capture["payload"]
    assert "top_p" not in capture["payload"]


def test_completion_overrides_forward_max_tokens_to_http_payload():
    calls: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        return _ok_response("```ax\ny = 1.0;\n```")

    b = OnyxQwenBackend("http://h", "m", _post=fake_post)
    b.draft_program(
        ExpertDraftRequest(
            "goal-text",
            context={COMPLETION_OVERRIDES_CONTEXT_KEY: {"max_tokens": 96}, "extra": 1},
        )
    )
    assert calls[0].get("max_tokens") == 96
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
