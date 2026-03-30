"""Phase 44: standalone HTML Glass Box exporter."""

from pathlib import Path

from axiom.tools.html_exporter import _format_cell_value, _is_adapter_key, export_html_report


def test_is_adapter_key():
    assert _is_adapter_key("alpha_signal")
    assert _is_adapter_key("prediction")
    assert _is_adapter_key("my_neural_x")
    assert not _is_adapter_key("volatility")


def test_format_cell_value():
    assert _format_cell_value(3.5) == "3.5"
    s = _format_cell_value([1.0, 2.0])
    assert "1.0" in s and "2.0" in s


def test_export_html_report_structure(tmp_path: Path):
    class M:
        def explain(self, data):
            return {"prediction": 0.1, "alpha_signal": 0.2, "quiet": 3.0}

        def predict(self, data):
            return {"prediction": 0.1, "extra_out": 99}

    p = tmp_path / "sub" / "out.html"
    export_html_report(M(), {"feat_a": 1, "feat_b": 2}, str(p))
    html = p.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "Input features" in html
    assert "feat_a" in html and "feat_b" in html
    assert "Neural adapters" in html
    assert "adapter-highlight" in html
    assert "prediction" in html and "alpha_signal" in html
    assert "Full execution trace" in html
    assert "quiet" in html


def test_export_html_report_source_block(tmp_path: Path):
    class M:
        def explain(self, data):
            return {}

        def predict(self, data):
            return {"y": 0}

    p = tmp_path / "s.html"
    export_html_report(M(), {}, str(p), source_code='y = x;\nif (true) { a = 1; }')
    html = p.read_text(encoding="utf-8")
    assert "Strategy source" in html
    assert "y = x;" in html
    assert "<script>" not in html  # not interpreted as tag; our source has no script


def test_export_html_escapes_payload(tmp_path: Path):
    class M:
        def explain(self, data):
            return {"k": "<>&"}

        def predict(self, data):
            return {"out": 1}

    p = tmp_path / "esc.html"
    export_html_report(M(), {"x": "<script>bad</script>"}, str(p))
    html = p.read_text(encoding="utf-8")
    assert "<script>bad</script>" not in html
    assert "&lt;script&gt;" in html or "&lt;" in html
