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

    def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
        assert request.goal
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


def test_default_predict_score_fn_higher_is_better():
    fn = cli_mod._default_predict_score_fn()
    s = fn([{"y": 0.0}, {"y": 1.0}], [{"y": 0.0}, {"y": 1.0}])
    assert s["neg_mse"] == 0.0
    s2 = fn([{"y": 0.0}], [{"y": 1.0}])
    assert s2["neg_mse"] < 0.0


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
