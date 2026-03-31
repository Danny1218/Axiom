"""``axiom copilot-doctor`` (Phase 81) — connectivity + one draft + validate_program."""

from __future__ import annotations

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
            assert request.context["_onyx_completion_overrides"] == {"temperature": 0}
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
