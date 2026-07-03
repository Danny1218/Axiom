"""High-value readme / pyproject / CLI contract checks (not brittle section-by-section prose)."""

from pathlib import Path

import pytest

from axiom.cli import main
from axiom.compiler.ir import ast_to_ir
from axiom.compiler.parser import parse_ax_file, reset_parser


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_project_dependencies(pyproject_text: str) -> list[str]:
    lines = pyproject_text.splitlines()
    deps: list[str] = []
    in_deps = False
    for line in lines:
        s = line.strip()
        if s.startswith("dependencies = ["):
            in_deps = True
            continue
        if in_deps:
            if s.startswith("]"):
                break
            if s.startswith('"'):
                deps.append(s.strip('",'))
    return deps


def test_pyproject_core_dependencies_minimal():
    text = (_root() / "pyproject.toml").read_text(encoding="utf-8")
    deps = _parse_project_dependencies(text)
    joined = " ".join(deps)
    assert "torch" in joined and "lark" in joined and "networkx" in joined
    assert "pytest" not in joined
    assert "streamlit" not in joined


def test_readme_version_matches_pyproject():
    readme = (_root() / "readme.md").read_text(encoding="utf-8")
    pyproject = (_root() / "pyproject.toml").read_text(encoding="utf-8")
    assert "1.3.0" in readme
    assert 'version = "1.3.0"' in pyproject


def test_readme_documents_core_python_api():
    text = (_root() / "readme.md").read_text(encoding="utf-8")
    assert "axiom.load" in text and "model.predict" in text
    assert "copilot-doctor" in text and "lmstudio" in text


def test_examples_titanic_ax_has_conditional_ir():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "titanic.ax"))
    assert any(x[0] == "OP_CONDITIONAL" for x in ir)


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "--help"],
        ["predict", "--help"],
        ["copilot-doctor", "--help"],
        ["copilot-benchmark", "--help"],
        ["serve", "--help"],
    ],
)
def test_cli_subcommands_help_exits_ok(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0
