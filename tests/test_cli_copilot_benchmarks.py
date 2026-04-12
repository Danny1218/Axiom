"""``axiom copilot-benchmark`` (Phase 70)."""

from __future__ import annotations

import json

import pytest

import axiom.cli as cli_mod
from axiom.cli import main
from axiom.compiler.parser import reset_parser
from axiom.copilot.benchmarks import BenchmarkDispatchExpert
from axiom.experts.base import ExpertDraftRequest, ExpertDraftResponse, ExpertRepairRequest, ExpertTraceSummaryRequest
from axiom.experts.onyx_qwen import COMPLETION_OVERRIDES_CONTEXT_KEY


@pytest.fixture(autouse=True)
def _fresh_parser():
    reset_parser()
    yield
    reset_parser()


def test_copilot_benchmark_help_exits_ok():
    with pytest.raises(SystemExit) as exc:
        main(["copilot-benchmark", "--help"])
    assert exc.value.code == 0


def test_copilot_benchmark_runs_with_dispatch_expert(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: BenchmarkDispatchExpert())
    out_json = tmp_path / "bench.json"
    main(
        [
            "copilot-benchmark",
            "--backend",
            "onyx-qwen",
            "--expert-url",
            "http://127.0.0.1:9/v1/",
            "--expert-model",
            "m",
            "--max-iterations",
            "2",
            "--out",
            str(out_json),
        ]
    )
    err = capsys.readouterr().err
    assert "[copilot-benchmark]" in err
    assert "draft:" in err and "search:" in err
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["kind"] == "axiom.copilot.benchmark_suite"
    assert data["run_options"] == {"draft": True, "search": True}
    assert len(data["tasks"]) >= 7
    first = data["tasks"][0]
    assert "producing_backend_name" in first["draft_only"]
    assert "backend_kind" in first["draft_only"]
    assert "winner_origin" in first["draft_only"]
    assert "producing_backend_name" in first["search"]
    assert "backend_kind" in first["search"]
    assert "winner_origin" in first["search"]


def test_copilot_benchmark_draft_only_and_task_json(tmp_path: Path, capsys, monkeypatch):
    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: BenchmarkDispatchExpert())
    from axiom.copilot.benchmarks import default_benchmark_tasks_json_path

    task_path = default_benchmark_tasks_json_path()
    out_json = tmp_path / "b2.json"
    main(
        [
            "copilot-benchmark",
            "--backend",
            "onyx-qwen",
            "--expert-url",
            "http://x/",
            "--expert-model",
            "m",
            "--draft-only",
            "--task-json",
            str(task_path),
            "--out",
            str(out_json),
        ]
    )
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["run_options"] == {"draft": True, "search": False}
    assert data["search_summary"] is None
    assert len(data["tasks"]) >= 5


def test_copilot_benchmark_accepts_temperature_and_passes_completion_override(tmp_path: Path, monkeypatch):
    calls: list[ExpertDraftRequest] = []

    class CaptureExpert:
        def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
            calls.append(request)
            return ExpertDraftResponse(ax_source="y = x * 2.0;", backend_name="onyx_qwen")

        def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
            raise AssertionError("repair should not run in draft-only benchmark")

        def summarize_trace(self, request: ExpertTraceSummaryRequest) -> str:
            return ""

    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: CaptureExpert())
    from axiom.copilot.benchmarks import default_benchmark_tasks_json_path

    out_json = tmp_path / "bench_temp.json"
    main(
        [
            "copilot-benchmark",
            "--backend",
            "onyx-qwen",
            "--expert-url",
            "http://x/",
            "--expert-model",
            "m",
            "--draft-only",
            "--task-json",
            str(default_benchmark_tasks_json_path()),
            "--temperature",
            "0",
            "--out",
            str(out_json),
        ]
    )
    assert calls
    overrides = calls[0].context.get(COMPLETION_OVERRIDES_CONTEXT_KEY)
    assert overrides == {"temperature": 0.0}


def test_copilot_benchmark_rejects_draft_and_search_together(monkeypatch):
    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: BenchmarkDispatchExpert())
    with pytest.raises(SystemExit) as e:
        main(
            [
                "copilot-benchmark",
                "--backend",
                "onyx-qwen",
                "--expert-url",
                "http://x/",
                "--expert-model",
                "m",
                "--draft-only",
                "--search",
            ]
        )
    assert "together" in str(e.value).lower()


def test_copilot_benchmark_bad_task_json(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: BenchmarkDispatchExpert())
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        main(
            [
                "copilot-benchmark",
                "--backend",
                "onyx-qwen",
                "--expert-url",
                "http://x/",
                "--expert-model",
                "m",
                "--task-json",
                str(bad),
            ]
        )
    assert "task-json" in str(e.value).lower() or "tasks" in str(e.value).lower()
