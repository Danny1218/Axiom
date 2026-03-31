"""Copilot Studio logic (``axiom.tools.copilot_studio``) — UI-free unit tests."""

from __future__ import annotations

import json

import pytest

from axiom.compiler.parser import reset_parser
from axiom.experts import ExpertDraftRequest, ExpertDraftResponse, ExpertRepairRequest, ExpertTraceSummaryRequest
from axiom.tools import copilot_studio as cs


@pytest.fixture(autouse=True)
def _fresh_parser():
    reset_parser()
    yield
    reset_parser()


class _ScriptedExpert:
    def __init__(self) -> None:
        self.draft_src = "y = ++++ ;\n"
        self.repairs = ["y = neural([1.0, 2.0]);\n"]

    def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
        return ExpertDraftResponse(ax_source=self.draft_src, backend_name="studio_test")

    def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
        if not self.repairs:
            raise AssertionError("unexpected repair")
        return ExpertDraftResponse(ax_source=self.repairs.pop(0), backend_name="studio_test")

    def summarize_trace(self, *args, **kwargs) -> str:
        return ""


def test_parse_examples_rows_json_ok():
    a, b = cs.parse_examples_rows_json('[{"inputs":{},"expected":{"y":0.5}}]')
    assert a == [{}] and b == [{"y": 0.5}]


def test_parse_tabular_json_studio_ok():
    raw = json.dumps(
        {
            "target_var": "y",
            "train_rows": [{"inputs": {"x": 0.0}, "expected": {"y": 0.0}}],
            "eval_rows": [{"inputs": {"x": 1.0}, "expected": {"y": 2.0}}],
            "epochs": 5,
        }
    )
    pld = cs.parse_tabular_json_studio(raw)
    assert pld.target_var == "y"
    assert pld.params.epochs == 5
    assert pld.train_rows[0]["x"] == 0.0 and pld.train_rows[0]["y"] == 0.0


def test_run_studio_search_train_tabular():
    ex = _ScriptedExpert()
    ex.draft_src = "y = neural([x]);\n"
    ex.repairs = []
    tab = json.dumps(
        {
            "target_var": "y",
            "train_rows": [{"inputs": {"x": float(i) * 0.1}, "expected": {"y": float(i) * 0.2}} for i in range(8)],
            "eval_rows": [{"inputs": {"x": 1.0}, "expected": {"y": 2.0}}],
            "epochs": 40,
            "learning_rate": 0.07,
            "batch_size": 4,
        }
    )
    cfg, out = cs.run_studio_search(
        "g", None, ex, 1, evaluation_mode="train_tabular", tabular_text=tab
    )
    assert cfg.mode == "train_tabular"
    assert out.best_evaluation.success
    assert "eval_mse" in out.best_evaluation.metrics


def test_run_studio_search_train_tabular_requires_json():
    ex = _ScriptedExpert()
    with pytest.raises(ValueError) as e:
        cs.run_studio_search("g", None, ex, 1, evaluation_mode="train_tabular", tabular_text=None)
    assert "tabular" in str(e.value).lower()


@pytest.mark.parametrize(
    "raw,sub",
    [
        ("not json", "Invalid JSON"),
        ("{}", "array"),
        ("[]", "empty"),
        ('[{"inputs":1,"expected":{}}]', "object"),
    ],
)
def test_parse_examples_rows_json_errors(raw: str, sub: str):
    with pytest.raises(ValueError) as e:
        cs.parse_examples_rows_json(raw)
    assert sub in str(e.value)


def test_build_studio_expert_requires_fields():
    with pytest.raises(ValueError):
        cs.build_studio_expert("", "m")
    with pytest.raises(ValueError):
        cs.build_studio_expert("http://x", "")


def test_studio_completion_overrides_from_text():
    assert cs._studio_completion_overrides_from_text("0.2", None) == {"temperature": 0.2}
    assert cs._studio_completion_overrides_from_text(None, "0.95") == {"top_p": 0.95}
    assert cs._studio_completion_overrides_from_text("", "") is None


def test_run_studio_draft():
    ex = _ScriptedExpert()
    r = cs.run_studio_draft("goal text", "ctx", ex)
    assert "++++" in r.ax_source


def test_run_studio_search_completion_overrides_on_config():
    ex = _ScriptedExpert()
    cfg, _ = cs.run_studio_search(
        "g",
        None,
        ex,
        1,
        evaluation_mode="compile_only",
        completion_overrides={"temperature": 0.2, "top_p": 0.9},
    )
    assert cfg.completion_overrides == {"temperature": 0.2, "top_p": 0.9}


def test_run_studio_search_compile_only():
    ex = _ScriptedExpert()
    cfg, out = cs.run_studio_search("g", None, ex, 3, evaluation_mode="compile_only")
    assert cfg.mode == "compile_only"
    assert out.converged is True
    assert "neural" in out.best_source


def test_run_studio_search_predict_rows():
    ex = _ScriptedExpert()
    ex.draft_src = "y = neural([1.0, 2.0]);\n"
    ex.repairs = []
    js = json.dumps([{"inputs": {}, "expected": {"y": 0.5}}])
    cfg, out = cs.run_studio_search("g", None, ex, 1, evaluation_mode="predict_rows", examples_text=js)
    assert cfg.mode == "predict_rows"
    assert out.iterations[0].evaluation.success is True


def test_iterations_table_rows_shape():
    ex = _ScriptedExpert()
    _, out = cs.run_studio_search("g", None, ex, 2, evaluation_mode="compile_only")
    rows = cs.iterations_table_rows(out)
    assert len(rows) == 2
    assert "iter" in rows[0] and "failure_count" in rows[0]
    assert "trace_summary" in rows[0]


def test_run_studio_search_summarize_traces():
    class _Talkative(_ScriptedExpert):
        def summarize_trace(self, *args, **kwargs) -> str:
            return "explained"

    ex = _Talkative()
    cfg, out = cs.run_studio_search("g", None, ex, 1, evaluation_mode="compile_only", summarize_traces=True)
    assert cfg.summarize_traces is True
    assert out.iterations[0].semantic_trace_summary == "explained"
    row = cs.iterations_table_rows(out)[0]
    assert row["trace_summary"] == "explained"


def test_build_studio_download_payload_keys():
    ex = _ScriptedExpert()
    cfg, out = cs.run_studio_search("goal", "dom", ex, 2, evaluation_mode="compile_only")
    blob = cs.build_studio_download_payload(cfg, out)
    json.dumps(blob)
    assert blob["converged"] is True
    assert "best_source" in blob
    assert blob["iterations_document"]["goal"] == "goal"
    assert blob["search_report"]["domain_context"] == "dom"


def test_main_with_mock_streamlit(monkeypatch):
    import sys

    pytest.importorskip("streamlit")

    class _SS:
        """Minimal Streamlit-like session_state (dict + attribute access)."""

        def __init__(self) -> None:
            object.__setattr__(self, "_d", {})

        def __contains__(self, k: object) -> bool:
            return k in self._d

        def __getitem__(self, k: str) -> object:
            return self._d[k]

        def __setitem__(self, k: str, v: object) -> None:
            self._d[k] = v

        def get(self, k: str, default=None):
            return self._d.get(k, default)

        def __getattr__(self, k: str) -> object:
            if k == "_d":
                raise AttributeError(k)
            return self._d.get(k)

        def __setattr__(self, k: str, v: object) -> None:
            if k == "_d":
                object.__setattr__(self, k, v)
            else:
                self._d[k] = v

    ss = _SS()
    from unittest.mock import MagicMock

    fake = MagicMock()
    fake.session_state = ss
    fake.text_input.side_effect = ["http://127.0.0.1/v1/", "m", "", "", ""]
    fake.text_area.side_effect = ["my goal", "", "[]"]
    fake.number_input.return_value = 8
    fake.checkbox.return_value = False
    fake.radio.return_value = "predict_rows"
    c1, c2 = MagicMock(), MagicMock()
    c1.button.return_value = False
    c2.button.return_value = False
    fake.columns.return_value = (c1, c2)

    monkeypatch.setitem(sys.modules, "streamlit", fake)
    cs.main()
    fake.set_page_config.assert_called_once()
    fake.title.assert_called()
