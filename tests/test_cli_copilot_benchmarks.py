"""``axiom copilot-benchmark`` (Phase 70)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import axiom.cli as cli_mod
from axiom.cli import main
from axiom.compiler.parser import reset_parser
from axiom.copilot.benchmarks import BenchmarkDispatchExpert, default_benchmark_tasks_json_path, load_benchmark_tasks_json_path
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


def test_copilot_benchmark_help_includes_max_tokens(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["copilot-benchmark", "--help"])
    assert exc.value.code == 0
    assert "--max-tokens" in capsys.readouterr().out


def test_copilot_benchmark_help_includes_timeout(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["copilot-benchmark", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--timeout" in out
    assert "--expert-timeout" in out


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


def test_copilot_benchmark_benchmark_dispatch_backend_runs_current_symbolic_suite(tmp_path: Path, capsys):
    root = Path(__file__).resolve().parents[1]
    out_json = tmp_path / "symbolic_suite.json"
    main(
        [
            "copilot-benchmark",
            "--backend",
            "benchmark-dispatch",
            "--task-json",
            str(root / "benchmarks" / "copilot_symbolic_and_generalization_tasks.json"),
            "--out",
            str(out_json),
        ]
    )
    err = capsys.readouterr().err
    assert "[copilot-benchmark]" in err
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["kind"] == "axiom.copilot.benchmark_suite"
    assert data["run_options"] == {"draft": True, "search": True}
    assert data["draft_summary"]["task_count"] == 10
    assert data["draft_summary"]["compile_ok_count"] == 10
    assert data["draft_summary"]["metric_ok_count"] == 10
    assert data["search_summary"]["task_count"] == 10
    assert data["search_summary"]["compile_ok_count"] == 10
    assert data["search_summary"]["metric_ok_count"] == 10


def test_copilot_benchmark_benchmark_dispatch_backend_runs_next_milestone_suite(tmp_path: Path, capsys):
    root = Path(__file__).resolve().parents[1]
    task_path = root / "benchmarks" / "copilot_symbolic_next_milestone_tasks.json"
    tasks = load_benchmark_tasks_json_path(task_path)
    out_json = tmp_path / "symbolic_next_suite.json"
    main(
        [
            "copilot-benchmark",
            "--backend",
            "benchmark-dispatch",
            "--task-json",
            str(task_path),
            "--out",
            str(out_json),
        ]
    )
    err = capsys.readouterr().err
    assert "[copilot-benchmark]" in err
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["kind"] == "axiom.copilot.benchmark_suite"
    assert data["run_options"] == {"draft": True, "search": True}
    n = len(tasks)
    assert data["draft_summary"]["task_count"] == n
    assert data["draft_summary"]["compile_ok_count"] == n
    assert data["draft_summary"]["metric_ok_count"] == n
    assert data["search_summary"]["task_count"] == n
    assert data["search_summary"]["compile_ok_count"] == n
    assert data["search_summary"]["metric_ok_count"] == n


def test_copilot_benchmark_benchmark_dispatch_backend_runs_generalization_stress_suite(tmp_path: Path, capsys):
    root = Path(__file__).resolve().parents[1]
    out_json = tmp_path / "symbolic_generalization_stress_suite.json"
    main(
        [
            "copilot-benchmark",
            "--backend",
            "benchmark-dispatch",
            "--task-json",
            str(root / "benchmarks" / "copilot_symbolic_generalization_stress_tasks.json"),
            "--out",
            str(out_json),
        ]
    )
    err = capsys.readouterr().err
    assert "[copilot-benchmark]" in err
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["kind"] == "axiom.copilot.benchmark_suite"
    assert data["run_options"] == {"draft": True, "search": True}
    assert data["draft_summary"]["task_count"] == 8
    assert data["draft_summary"]["compile_ok_count"] == 8
    assert data["draft_summary"]["metric_ok_count"] == 8
    assert data["search_summary"]["task_count"] == 8
    assert data["search_summary"]["compile_ok_count"] == 8
    assert data["search_summary"]["metric_ok_count"] == 8


def test_copilot_benchmark_benchmark_dispatch_backend_runs_robustness_ambiguity_stress_suite(
    tmp_path: Path, capsys
):
    root = Path(__file__).resolve().parents[1]
    out_json = tmp_path / "symbolic_robustness_ambiguity_stress_suite.json"
    main(
        [
            "copilot-benchmark",
            "--backend",
            "benchmark-dispatch",
            "--task-json",
            str(root / "benchmarks" / "copilot_symbolic_robustness_ambiguity_stress_tasks.json"),
            "--out",
            str(out_json),
        ]
    )
    err = capsys.readouterr().err
    assert "[copilot-benchmark]" in err
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["kind"] == "axiom.copilot.benchmark_suite"
    assert data["run_options"] == {"draft": True, "search": True}
    assert data["draft_summary"]["task_count"] == 8
    assert data["draft_summary"]["compile_ok_count"] == 8
    assert data["draft_summary"]["metric_ok_count"] == 8
    assert data["search_summary"]["task_count"] == 8
    assert data["search_summary"]["compile_ok_count"] == 8
    assert data["search_summary"]["metric_ok_count"] == 8


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


def test_copilot_benchmark_default_timeout_is_unchanged_when_omitted(tmp_path: Path, monkeypatch):
    seen_timeout: list[object] = []

    def fake_make_copilot_expert(args):
        seen_timeout.append(getattr(args, "expert_timeout", "missing"))
        return BenchmarkDispatchExpert()

    monkeypatch.setattr(cli_mod, "_make_copilot_expert", fake_make_copilot_expert)
    out_json = tmp_path / "bench_timeout_default.json"
    main(
        [
            "copilot-benchmark",
            "--backend",
            "benchmark-dispatch",
            "--out",
            str(out_json),
        ]
    )
    assert seen_timeout == [None]


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


def test_copilot_benchmark_gate_exits_on_failure(tmp_path: Path, monkeypatch, capsys):
    class _BrokenExpert(BenchmarkDispatchExpert):
        def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
            return ExpertDraftResponse(ax_source="this is not valid .ax", backend_name="broken")

    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: _BrokenExpert())
    out_json = tmp_path / "gate_fail.json"
    with pytest.raises(SystemExit) as e:
        main(
            [
                "copilot-benchmark",
                "--backend",
                "benchmark-dispatch",
                "--draft-only",
                "--task-json",
                str(default_benchmark_tasks_json_path()),
                "--out",
                str(out_json),
                "--gate",
            ]
        )
    assert e.value.code == 1
    err = capsys.readouterr().err
    assert "GATE FAILED" in err
