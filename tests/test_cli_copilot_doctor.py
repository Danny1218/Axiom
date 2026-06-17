"""``axiom copilot-doctor`` (Phase 81) — connectivity + one draft + validate_program."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

import axiom.cli as cli_mod
from axiom.cli import _COPILOT_DOCTOR_DEFAULT_GOAL, main
from axiom.compiler.parser import reset_parser
from axiom.experts import ExpertDraftRequest, ExpertDraftResponse


@pytest.fixture(autouse=True)
def _fresh_parser():
    reset_parser()
    yield
    reset_parser()


def test_copilot_doctor_help_exits_ok():
    with pytest.raises(SystemExit) as exc:
        main(["copilot-doctor", "--help"])
    assert exc.value.code == 0


def test_copilot_doctor_default_goal_constant():
    assert "y = x * 2.0" in _COPILOT_DOCTOR_DEFAULT_GOAL


def test_copilot_doctor_connection_fail_exits_1(capsys, monkeypatch):
    from axiom.experts.onyx_qwen import OnyxQwenTransportError

    class _F:
        def draft_program(self, _request: ExpertDraftRequest):
            raise OnyxQwenTransportError("nope")

    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: _F())

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "copilot-doctor",
                "--backend",
                "onyx-qwen",
                "--expert-url",
                "http://127.0.0.1:9/v1/",
                "--expert-model",
                "m",
            ]
        )
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "connection: fail" in err and "OnyxQwenTransportError" in err


def test_copilot_doctor_happy_path(capsys, monkeypatch):
    class _Ok:
        def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
            assert request.goal
            assert "_onyx_completion_overrides" in request.context
            assert request.context["_onyx_completion_overrides"] == {"temperature": 0.0}
            return ExpertDraftResponse(
                ax_source="y = x * 2.0;\n",
                backend_name="fake",
                metadata={"raw_chars": 99, "forbidden_tokens_detected": []},
            )

    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: _Ok())
    main(
        [
            "copilot-doctor",
            "--backend",
            "onyx-qwen",
            "--goal",
            "custom",
            "--expert-url",
            "http://x/",
            "--expert-model",
            "m",
        ]
    )
    out = capsys.readouterr().out
    assert "connection: ok" in out
    assert "raw_chars: 99" in out
    assert "parse: ok" in out and "ir: ok" in out and "block: ok" in out
    assert "anti_pattern: (none)" in out
    assert "neural: no" in out


def test_copilot_doctor_live_onyx_backend_sends_greedy_payload_not_temperature_zero(capsys, monkeypatch):
    pytest.importorskip("requests")
    captured: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):
        captured.append(dict(json or {}))
        r = Mock()
        r.status_code = 200
        r.text = ""
        r.json.return_value = {"choices": [{"message": {"content": "```ax\ny = x * 2.0;\n```"}}]}
        return r

    from axiom.experts.onyx_qwen import OnyxQwenBackend

    monkeypatch.setattr(
        cli_mod,
        "_make_copilot_expert",
        lambda _a: OnyxQwenBackend("http://127.0.0.1/v1/", "m", _post=fake_post),
    )
    main(
        [
            "copilot-doctor",
            "--backend",
            "onyx-qwen",
            "--expert-url",
            "http://127.0.0.1/v1/",
            "--expert-model",
            "m",
        ]
    )
    assert captured, "expected one chat/completions POST"
    body = captured[0]
    assert "temperature" not in body
    assert body.get("do_sample") is False
    assert "connection: ok" in capsys.readouterr().out


def test_copilot_doctor_respects_explicit_temperature_and_top_p(capsys, monkeypatch):
    class _Ok:
        def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
            o = request.context.get("_onyx_completion_overrides")
            assert o == {"temperature": 0.35, "top_p": 0.88}
            return ExpertDraftResponse(
                ax_source="y = x * 2.0;\n",
                backend_name="fake",
                metadata={"raw_chars": 1},
            )

    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: _Ok())
    main(
        [
            "copilot-doctor",
            "--backend",
            "onyx-qwen",
            "--expert-url",
            "http://x/",
            "--expert-model",
            "m",
            "--temperature",
            "0.35",
            "--top-p",
            "0.88",
        ]
    )
    assert "connection: ok" in capsys.readouterr().out


def test_copilot_doctor_parse_fail_exits_1(capsys, monkeypatch):
    class _Bad:
        def draft_program(self, _request: ExpertDraftRequest) -> ExpertDraftResponse:
            return ExpertDraftResponse(
                ax_source="this is not ax ;;;",
                backend_name="fake",
                metadata={"raw_chars": 10},
            )

    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: _Bad())
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "copilot-doctor",
                "--backend",
                "onyx-qwen",
                "--expert-url",
                "http://x/",
                "--expert-model",
                "m",
            ]
        )
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "parse: fail" in out


def test_copilot_doctor_anti_pattern_and_neural(capsys, monkeypatch):
    class _Warn:
        def draft_program(self, _request: ExpertDraftRequest) -> ExpertDraftResponse:
            return ExpertDraftResponse(
                ax_source="y = neural([x]);\n",
                backend_name="fake",
                metadata={
                    "raw_chars": 20,
                    "forbidden_tokens_detected": ["assign_colon_eq", "print_call"],
                    "indexed_variable_warning": True,
                    "output_call_warning": True,
                    "suspicious_numeric_literal_warning": True,
                },
            )

    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: _Warn())
    main(
        [
            "copilot-doctor",
            "--backend",
            "onyx-qwen",
            "--expert-url",
            "http://x/",
            "--expert-model",
            "m",
        ]
    )
    out = capsys.readouterr().out
    assert "anti_pattern: assign_colon_eq, print_call" in out
    assert "indexed_variable_warning" in out
    assert "output_call_warning" in out
    assert "suspicious_numeric_literal_warning" in out
    assert "neural: yes" in out


def test_copilot_doctor_examples_json_eval_ok(capsys, monkeypatch, tmp_path: Path):
    ex = tmp_path / "ex.json"
    ex.write_text(
        '[{"inputs": {"x": 1.0}, "expected": {"y": 2.0}}, {"inputs": {"x": 0.0}, "expected": {"y": 0.0}}]',
        encoding="utf-8",
    )

    class _Ok:
        def draft_program(self, _request: ExpertDraftRequest) -> ExpertDraftResponse:
            return ExpertDraftResponse(
                ax_source="y = x * 2.0;\n",
                backend_name="fake",
                metadata={"raw_chars": 1},
            )

    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: _Ok())
    main(
        [
            "copilot-doctor",
            "--backend",
            "onyx-qwen",
            "--expert-url",
            "http://x/",
            "--expert-model",
            "m",
            "--examples-json",
            str(ex),
        ]
    )
    out = capsys.readouterr().out
    assert "evaluation: ok" in out
    assert "metrics:" in out and "neg_mse" in out
    assert "examples: exact=yes" in out and "near_threshold=yes" in out


def test_copilot_doctor_examples_json_eval_fail_exits_1(capsys, monkeypatch, tmp_path: Path):
    from axiom.copilot.evaluator import ProgramEvaluationReport, ProgramFailure
    from axiom.copilot.models import ProgramCandidate

    ex = tmp_path / "ex.json"
    ex.write_text('[{"inputs": {"x": 1.0}, "expected": {"y": 2.0}}]', encoding="utf-8")

    class _Ok:
        def draft_program(self, _request: ExpertDraftRequest) -> ExpertDraftResponse:
            return ExpertDraftResponse(
                ax_source="y = x * 2.0;\n",
                backend_name="fake",
                metadata={"raw_chars": 1},
            )

    def _fake_eval(candidate: ProgramCandidate, **kwargs: object) -> ProgramEvaluationReport:
        return ProgramEvaluationReport(
            success=False,
            source=candidate.source,
            compile_stage_reached="predict",
            mode="predict_rows",
            failures=[ProgramFailure("predict", "runtime", "forced failure", "test")],
            warnings=[],
            metrics={},
            program_metrics=[],
        )

    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: _Ok())
    monkeypatch.setattr("axiom.copilot.evaluator.evaluate_program", _fake_eval)
    with pytest.raises(SystemExit) as ei:
        main(
            [
                "copilot-doctor",
                "--backend",
                "onyx-qwen",
                "--expert-url",
                "http://x/",
                "--expert-model",
                "m",
                "--examples-json",
                str(ex),
            ]
        )
    assert ei.value.code == 1
    assert "evaluation: fail" in capsys.readouterr().out


def test_copilot_doctor_passes_expert_timeout_to_builder(monkeypatch):
    seen: dict = {}

    def fake_build(backend: str, *, expert_url, expert_model, expert_api_key=None, timeout=None):
        seen["timeout"] = timeout

        class _E:
            def draft_program(self, _request: ExpertDraftRequest) -> ExpertDraftResponse:
                return ExpertDraftResponse(
                    ax_source="y = x * 2.0;\n",
                    backend_name="x",
                    metadata={"raw_chars": 1},
                )

        return _E()

    monkeypatch.setattr("axiom.copilot.backend.build_copilot_expert", fake_build)
    main(
        [
            "copilot-doctor",
            "--backend",
            "onyx-qwen",
            "--expert-url",
            "http://x/",
            "--expert-model",
            "m",
            "--timeout",
            "77",
        ]
    )
    assert seen["timeout"] == 77.0


@pytest.mark.parametrize(
    ("source", "pattern_id"),
    [
        ("this is not ax ;;;", "syntax_garbage"),
        ("y = x * 2.0", "missing_semicolon"),
        ("y = neural([x]);\n", "neural_on_exact_task"),
        ("if (x > 0) { y = x; }\n", "incomplete_if_else"),
        ("y = x / 0.0;\n", "divide_by_zero"),
    ],
)
def test_copilot_doctor_validate_source_wrong_patterns(
    tmp_path: Path,
    capsys,
    source: str,
    pattern_id: str,
):
    ax_path = tmp_path / f"{pattern_id}.ax"
    ax_path.write_text(source, encoding="utf-8")
    ex = tmp_path / "ex.json"
    ex.write_text(
        '[{"inputs": {"x": 1.0}, "expected": {"y": 2.0}}, {"inputs": {"x": 0.0}, "expected": {"y": 0.0}}]',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "copilot-doctor",
                "--validate-source",
                str(ax_path),
                "--examples-json",
                str(ex),
                "--goal",
                "Compute y as double of x.",
            ]
        )
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "parse:" in out and "block:" in out
    if pattern_id == "syntax_garbage":
        assert "parse: fail" in out
    else:
        assert "evaluation: fail" in out or "failures:" in out or "row_mismatches:" in out


def test_copilot_doctor_validate_source_row_mismatch_cues(tmp_path: Path, capsys):
    ax_path = tmp_path / "wrong_coeff.ax"
    ax_path.write_text("y = x * 3.0;\n", encoding="utf-8")
    ex = tmp_path / "ex.json"
    ex.write_text(
        '[{"inputs": {"x": 1.0}, "expected": {"y": 2.0}}, {"inputs": {"x": 0.0}, "expected": {"y": 0.0}}]',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "copilot-doctor",
                "--validate-source",
                str(ax_path),
                "--examples-json",
                str(ex),
                "--goal",
                "Compute y as double of x.",
            ]
        )
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "evaluation: fail" in out
    assert "row_mismatches:" in out
