"""``axiom copilot-draft`` / ``axiom copilot-search`` (Phase 61)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

import axiom.cli as cli_mod
from axiom.cli import main
from axiom.compiler.parser import reset_parser
from axiom.experts import ExpertDraftRequest, ExpertDraftResponse, ExpertRepairRequest


@pytest.fixture(autouse=True)
def _fresh_parser():
    reset_parser()
    yield
    reset_parser()


class _FakeExpert:
    def __init__(self) -> None:
        self.draft_source = "y = neural([1.0, 2.0]);\n"
        self.repair_queue: list[str] = []
        self.draft_calls: list[ExpertDraftRequest] = []

    def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
        assert request.goal
        self.draft_calls.append(request)
        return ExpertDraftResponse(ax_source=self.draft_source, backend_name="fake")

    def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
        assert "Goal" in request.error_report
        if not self.repair_queue:
            return ExpertDraftResponse(ax_source=self.draft_source, backend_name="fake")
        return ExpertDraftResponse(ax_source=self.repair_queue.pop(0), backend_name="fake")

    def summarize_trace(self, *args, **kwargs) -> str:
        return "ok"


def test_copilot_draft_help_exits_ok():
    with pytest.raises(SystemExit) as exc:
        main(["copilot-draft", "--help"])
    assert exc.value.code == 0


def test_copilot_search_help_exits_ok():
    with pytest.raises(SystemExit) as exc:
        main(["copilot-search", "--help"])
    assert exc.value.code == 0


def test_copilot_serve_help_exits_ok():
    with pytest.raises(SystemExit) as exc:
        main(["copilot-serve", "--help"])
    assert exc.value.code == 0


def test_copilot_draft_runs_with_stub_expert(capsys, monkeypatch):
    fake = _FakeExpert()
    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: fake)
    main(
        [
            "copilot-draft",
            "--backend",
            "onyx-qwen",
            "--goal",
            "test goal",
            "--context",
            "tabular notes",
            "--expert-url",
            "http://127.0.0.1:9/v1/",
            "--expert-model",
            "qwen-test",
        ]
    )
    out = capsys.readouterr().out
    assert "neural" in out


def test_copilot_draft_writes_out(tmp_path: Path, monkeypatch):
    fake = _FakeExpert()
    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: fake)
    out_ax = tmp_path / "d.ax"
    main(
        [
            "copilot-draft",
            "--backend",
            "onyx-qwen",
            "--goal",
            "g",
            "--expert-url",
            "http://x/",
            "--expert-model",
            "m",
            "--out",
            str(out_ax),
        ]
    )
    assert out_ax.is_file() and "neural" in out_ax.read_text(encoding="utf-8")


def test_copilot_draft_examples_json_uses_fast_path_without_calling_expert(tmp_path: Path, capsys, monkeypatch):
    fake = _FakeExpert()
    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: fake)
    examples = tmp_path / "double_x.json"
    examples.write_text(
        json.dumps(
            [
                {"inputs": {"x": 1.0}, "expected": {"y": 2.0}},
                {"inputs": {"x": 2.5}, "expected": {"y": 5.0}},
                {"inputs": {"x": -3.0}, "expected": {"y": -6.0}},
            ]
        ),
        encoding="utf-8",
    )
    main(
        [
            "copilot-draft",
            "--backend",
            "onyx-qwen",
            "--goal",
            "Compute y as double of x.",
            "--expert-url",
            "http://x/",
            "--expert-model",
            "m",
            "--examples-json",
            str(examples),
        ]
    )
    out = capsys.readouterr().out
    assert out.strip() == "y = x * 2.0;"
    assert fake.draft_calls == []


def test_copilot_search_artifact_dir_writes_bundle(tmp_path: Path, monkeypatch):
    fake = _FakeExpert()
    fake.draft_source = "y = neural([1.0, 2.0]);\n"
    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: fake)
    ad = tmp_path / "artifacts"
    main(
        [
            "copilot-search",
            "--backend",
            "onyx-qwen",
            "--goal",
            "g",
            "--expert-url",
            "http://x/",
            "--expert-model",
            "m",
            "--iterations",
            "1",
            "--artifact-dir",
            str(ad),
        ]
    )
    assert (ad / "best.ax").is_file()
    assert (ad / "iterations.json").is_file()
    assert (ad / "search_report.json").is_file()


def test_copilot_search_runs_with_stub_expert(tmp_path: Path, capsys, monkeypatch):
    fake = _FakeExpert()
    fake.draft_source = "y = ++++ ;\n"
    fake.repair_queue = ["y = neural([1.0, 2.0]);\n"]
    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: fake)
    rep = tmp_path / "r.json"
    out_ax = tmp_path / "best.ax"
    examples = tmp_path / "ex.json"
    examples.write_text(
        json.dumps([{"inputs": {}, "expected": {"y": 0.5}}]),
        encoding="utf-8",
    )
    main(
        [
            "copilot-search",
            "--backend",
            "onyx-qwen",
            "--goal",
            "predict y",
            "--expert-url",
            "http://x/",
            "--expert-model",
            "m",
            "--iterations",
            "3",
            "--examples-json",
            str(examples),
            "--no-repair-valid-with-metrics",
            "--report-out",
            str(rep),
            "--out",
            str(out_ax),
        ]
    )
    err = capsys.readouterr().err
    assert "[iter 0]" in err and "[iter 1]" in err
    data = json.loads(rep.read_text(encoding="utf-8"))
    assert data["converged"] is True
    assert data["best_evaluation"]["success"] is True
    assert len(data["iterations"]) == 2
    assert "neural" in out_ax.read_text(encoding="utf-8")
    assert data["iterations"][0].get("semantic_trace_summary") is None
    assert "semantic_summaries" not in data


def test_copilot_search_report_includes_semantic_summaries_when_flag(tmp_path: Path, monkeypatch):
    fake = _FakeExpert()
    fake.draft_source = "y = neural([1.0, 2.0]);\n"
    fake.repair_queue = []
    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: fake)
    rep = tmp_path / "rs.json"
    examples = tmp_path / "ex2.json"
    examples.write_text(json.dumps([{"inputs": {}, "expected": {"y": 0.5}}]), encoding="utf-8")
    main(
        [
            "copilot-search",
            "--backend",
            "onyx-qwen",
            "--goal",
            "predict y",
            "--expert-url",
            "http://x/",
            "--expert-model",
            "m",
            "--iterations",
            "1",
            "--examples-json",
            str(examples),
            "--report-out",
            str(rep),
            "--summarize-traces",
        ]
    )
    data = json.loads(rep.read_text(encoding="utf-8"))
    assert data["semantic_summaries"]["enabled"] is True
    assert data["iterations"][0]["semantic_trace_summary"] == "ok"
    assert data["semantic_summaries"]["per_iteration"][0]["semantic_trace_summary"] == "ok"


def test_copilot_search_compile_only_ignores_predict(tmp_path: Path, monkeypatch):
    fake = _FakeExpert()
    fake.draft_source = "y = ++++ ;\n"
    fake.repair_queue = ["y = 1.0;\n"]
    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: fake)
    examples = tmp_path / "ex.json"
    examples.write_text(json.dumps([{"inputs": {}, "expected": {"y": 1.0}}]), encoding="utf-8")
    main(
        [
            "copilot-search",
            "--backend",
            "onyx-qwen",
            "--goal",
            "g",
            "--expert-url",
            "http://x/",
            "--expert-model",
            "m",
            "--iterations",
            "2",
            "--examples-json",
            str(examples),
            "--compile-only",
        ]
    )
    assert fake.repair_queue == []


def test_copilot_missing_requests_exits(monkeypatch):
    def boom() -> None:
        raise SystemExit('pip install -e ".[copilot]"')

    monkeypatch.setattr(cli_mod, "_require_requests_for_copilot", boom)
    with pytest.raises(SystemExit) as e:
        main(
            [
                "copilot-draft",
                "--backend",
                "onyx-qwen",
                "--goal",
                "g",
                "--expert-url",
                "http://x/",
                "--expert-model",
                "m",
            ]
        )
    assert "[copilot]" in str(e.value).lower() or "copilot" in str(e.value).lower()


def test_load_examples_json_valid(tmp_path: Path):
    p = tmp_path / "e.json"
    p.write_text(
        json.dumps(
            [
                {"inputs": {"a": 1}, "expected": {"b": 2.0}},
                {"inputs": {"a": 0}, "expected": {"b": 0.0}},
            ]
        ),
        encoding="utf-8",
    )
    ins, exp = cli_mod._load_examples_json(p)
    assert ins == [{"a": 1}, {"a": 0}] and exp == [{"b": 2.0}, {"b": 0.0}]


@pytest.mark.parametrize(
    "raw,msg",
    [
        ("{}", "array"),
        ("[]", "empty"),
        ('[{"inputs":1,"expected":{}}]', "object"),
        ('[{"inputs":{},"missing_expected":{}}]', "expected"),
    ],
)
def test_load_examples_json_errors(tmp_path: Path, raw: str, msg: str):
    p = tmp_path / "bad.json"
    p.write_text(raw, encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        cli_mod._load_examples_json(p)
    assert msg in str(e.value)


def _copilot_search_argv(tmp_path: Path) -> list:
    return [
        "copilot-search",
        "--backend",
        "onyx-qwen",
        "--goal",
        "g",
        "--expert-url",
        "http://x/",
        "--expert-model",
        "m",
        "--iterations",
        "1",
    ]


def test_copilot_search_train_tabular_requires_tabular_json(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: _FakeExpert())
    with pytest.raises(SystemExit) as e:
        main(_copilot_search_argv(tmp_path) + ["--train-tabular"])
    assert "tabular-json" in str(e.value).lower()


def test_copilot_search_tabular_json_requires_flag(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: _FakeExpert())
    tj = tmp_path / "t.json"
    tj.write_text(
        json.dumps(
            {
                "target_var": "y",
                "train_rows": [{"inputs": {"x": 0.0}, "expected": {"y": 0.0}}],
                "eval_rows": [{"inputs": {"x": 1.0}, "expected": {"y": 1.0}}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as e:
        main(_copilot_search_argv(tmp_path) + ["--tabular-json", str(tj)])
    assert "train-tabular" in str(e.value).lower()


def test_copilot_search_train_tabular_incompatible_with_compile_only(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: _FakeExpert())
    tj = tmp_path / "t.json"
    tj.write_text(
        json.dumps(
            {
                "target_var": "y",
                "train_rows": [{"inputs": {"x": 0.0}, "expected": {"y": 0.0}}],
                "eval_rows": [{"inputs": {"x": 1.0}, "expected": {"y": 1.0}}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as e:
        main(
            _copilot_search_argv(tmp_path)
            + ["--train-tabular", "--tabular-json", str(tj), "--compile-only"]
        )
    assert "compile-only" in str(e.value).lower()


def test_copilot_search_train_tabular_incompatible_with_examples_json(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: _FakeExpert())
    tj = tmp_path / "t.json"
    tj.write_text(
        json.dumps(
            {
                "target_var": "y",
                "train_rows": [{"inputs": {"x": 0.0}, "expected": {"y": 0.0}}],
                "eval_rows": [{"inputs": {"x": 1.0}, "expected": {"y": 1.0}}],
            }
        ),
        encoding="utf-8",
    )
    ej = tmp_path / "e.json"
    ej.write_text('[{"inputs":{},"expected":{"y":1}}]', encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        main(
            _copilot_search_argv(tmp_path)
            + ["--train-tabular", "--tabular-json", str(tj), "--examples-json", str(ej)]
        )
    assert "examples-json" in str(e.value).lower()


def test_copilot_search_train_tabular_runs(tmp_path: Path, capsys, monkeypatch):
    fake = _FakeExpert()
    fake.draft_source = "y = neural([x]);\n"
    monkeypatch.setattr(cli_mod, "_make_copilot_expert", lambda _a: fake)
    tab = tmp_path / "tab.json"
    tab.write_text(
        json.dumps(
            {
                "target_var": "y",
                "train_rows": [{"inputs": {"x": 0.1}, "expected": {"y": 0.2}}] * 6,
                "eval_rows": [{"inputs": {"x": 1.0}, "expected": {"y": 2.0}}],
                "epochs": 50,
                "learning_rate": 0.06,
                "batch_size": 4,
            }
        ),
        encoding="utf-8",
    )
    main(
        [
            "copilot-search",
            "--backend",
            "onyx-qwen",
            "--goal",
            "fit",
            "--expert-url",
            "http://x/",
            "--expert-model",
            "m",
            "--iterations",
            "1",
            "--train-tabular",
            "--tabular-json",
            str(tab),
        ]
    )
    captured = capsys.readouterr()
    assert "train" in captured.err or "metrics=" in captured.err
    assert "neural" in captured.out


def test_load_tabular_json_ok(tmp_path: Path):
    p = tmp_path / "z.json"
    p.write_text(
        json.dumps(
            {
                "target_var": "y",
                "train_rows": [{"inputs": {"x": 0.0}, "expected": {"y": 0.0}}],
                "eval_rows": [{"inputs": {"x": 1.0}, "expected": {"y": 2.0}}],
            }
        ),
        encoding="utf-8",
    )
    pld = cli_mod._load_tabular_json(p)
    assert pld.target_var == "y"
    assert len(pld.train_rows) == 1


def test_default_predict_score_fn_higher_is_better():
    fn = cli_mod._default_predict_score_fn()
    s = fn([{"y": 0.0}, {"y": 1.0}], [{"y": 0.0}, {"y": 1.0}])
    assert s["neg_mse"] == 0.0
    s2 = fn([{"y": 0.0}], [{"y": 1.0}])
    assert s2["neg_mse"] < 0.0


def test_build_copilot_search_config_metric_repair_on_by_default_for_examples(tmp_path: Path):
    from argparse import Namespace

    ex_path = tmp_path / "ex.json"
    ex_path.write_text(
        json.dumps([{"inputs": {"x": 1.0}, "expected": {"y": 2.0}}]),
        encoding="utf-8",
    )
    ns = Namespace(
        goal="g",
        context=None,
        compile_only=False,
        train_tabular=False,
        tabular_json=None,
        examples_json=ex_path,
        iterations=3,
        artifact_dir=None,
        summarize_traces=False,
        repair_valid_with_metrics=False,
        no_repair_valid_with_metrics=False,
        metric_repair_if_below=None,
        temperature=None,
        top_p=None,
    )
    cfg = cli_mod._build_copilot_search_config(ns, object())
    assert cfg.mode == "predict_rows"
    assert cfg.repair_valid_with_metrics is True
    assert cfg.completion_overrides is None


def test_build_copilot_search_config_no_repair_valid_with_metrics(tmp_path: Path):
    from argparse import Namespace

    ex_path = tmp_path / "ex.json"
    ex_path.write_text(
        json.dumps([{"inputs": {"x": 1.0}, "expected": {"y": 2.0}}]),
        encoding="utf-8",
    )
    ns = Namespace(
        goal="g",
        context=None,
        compile_only=False,
        train_tabular=False,
        tabular_json=None,
        examples_json=ex_path,
        iterations=3,
        artifact_dir=None,
        summarize_traces=False,
        repair_valid_with_metrics=False,
        no_repair_valid_with_metrics=True,
        metric_repair_if_below=None,
        temperature=None,
        top_p=None,
    )
    cfg = cli_mod._build_copilot_search_config(ns, object())
    assert cfg.repair_valid_with_metrics is False


def test_build_copilot_search_config_conflicting_repair_flags_exits(tmp_path: Path):
    from argparse import Namespace

    ex_path = tmp_path / "ex.json"
    ex_path.write_text(
        json.dumps([{"inputs": {"x": 1.0}, "expected": {"y": 2.0}}]),
        encoding="utf-8",
    )
    ns = Namespace(
        goal="g",
        context=None,
        compile_only=False,
        train_tabular=False,
        tabular_json=None,
        examples_json=ex_path,
        iterations=3,
        artifact_dir=None,
        summarize_traces=False,
        repair_valid_with_metrics=True,
        no_repair_valid_with_metrics=True,
        metric_repair_if_below=None,
        temperature=None,
        top_p=None,
    )
    with pytest.raises(SystemExit, match="Cannot combine"):
        cli_mod._build_copilot_search_config(ns, object())


def test_build_copilot_search_config_completion_overrides(tmp_path: Path):
    from argparse import Namespace

    ex_path = tmp_path / "ex.json"
    ex_path.write_text(
        json.dumps([{"inputs": {"x": 1.0}, "expected": {"y": 2.0}}]),
        encoding="utf-8",
    )
    ns = Namespace(
        goal="g",
        context=None,
        compile_only=False,
        train_tabular=False,
        tabular_json=None,
        examples_json=ex_path,
        iterations=3,
        artifact_dir=None,
        summarize_traces=False,
        repair_valid_with_metrics=False,
        no_repair_valid_with_metrics=False,
        metric_repair_if_below=None,
        temperature=0.15,
        top_p=0.9,
    )
    cfg = cli_mod._build_copilot_search_config(ns, object())
    assert cfg.completion_overrides == {"temperature": 0.15, "top_p": 0.9}


def test_build_copilot_search_config_completion_overrides_temperature_zero(tmp_path: Path):
    from argparse import Namespace

    ex_path = tmp_path / "ex.json"
    ex_path.write_text(
        json.dumps([{"inputs": {"x": 1.0}, "expected": {"y": 2.0}}]),
        encoding="utf-8",
    )
    ns = Namespace(
        goal="g",
        context=None,
        compile_only=False,
        train_tabular=False,
        tabular_json=None,
        examples_json=ex_path,
        iterations=3,
        artifact_dir=None,
        summarize_traces=False,
        repair_valid_with_metrics=False,
        no_repair_valid_with_metrics=False,
        metric_repair_if_below=None,
        temperature=0.0,
        top_p=None,
    )
    cfg = cli_mod._build_copilot_search_config(ns, object())
    assert cfg.completion_overrides == {"temperature": 0.0}


def test_copilot_run_help_includes_restarts_flag(capsys):
    with pytest.raises(SystemExit) as e:
        main(["copilot-run", "--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "--restarts" in out and "--temperature" in out


def test_copilot_search_help_includes_temperature(capsys):
    with pytest.raises(SystemExit) as e:
        main(["copilot-search", "--help"])
    assert e.value.code == 0
    assert "--temperature" in capsys.readouterr().out


def test_serialize_evaluation_report_roundtrip_keys():
    from axiom.copilot import ProgramEvaluationReport, ProgramFailure

    rep = ProgramEvaluationReport(
        success=False,
        source="x",
        compile_stage_reached="parse",
        mode="compile_only",
        failures=[ProgramFailure("parse", "syntax", "bad", "E")],
        warnings=["w"],
        metrics={},
    )
    d = cli_mod._serialize_evaluation_report(rep)
    assert d["success"] is False and d["failures"][0]["kind"] == "syntax" and d["warnings"] == ["w"]
