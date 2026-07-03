"""Phase 57: frozen layout / public API contracts before semantic-copilot work (see ``plan.md``)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import axiom


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_cli_subcommands(cli_text: str) -> set[str]:
    return set(re.findall(r"sub\.add_parser\(\s*(?:\n\s*)?\"([^\"]+)\"", cli_text))


def _optional_extra_names(pyproject_text: str) -> set[str]:
    block = pyproject_text.split("[project.optional-dependencies]", 1)[1]
    return set(re.findall(r"^([a-z]+)\s*=\s*\[", block, re.MULTILINE))


def test_plan_architecture_and_status_doc():
    plan = (_root() / "plan.md").read_text(encoding="utf-8")
    assert "## What this is" in plan
    assert "## Two execution paths" in plan
    assert "Interpreted" in plan and "Compiled" in plan
    assert "## Copilot pipeline" in plan
    assert "tolerant_inference" in plan
    assert "lmstudio" in plan
    assert "## Test & benchmark status" in plan
    assert "## Intentionally frozen" in plan
    assert "axiom.load" in plan
    assert "benchmark-dispatch" in plan
    assert len(plan.splitlines()) <= 200


def test_cli_subcommands_stable():
    src = (_root() / "src" / "axiom" / "cli.py").read_text(encoding="utf-8")
    found = _parse_cli_subcommands(src)
    expected = {
        "train",
        "inspect",
        "predict",
        "lock-bundle",
        "export-onnx",
        "gateway-serve",
        "serve",
        "copilot-draft",
        "copilot-doctor",
        "copilot-search",
        "copilot-studio",
        "copilot-serve",
        "copilot-benchmark",
        "copilot-run",
        "copilot-stability-report",
    }
    assert found == expected, f"CLI subcommands changed: {found ^ expected}"


def test_pyproject_optional_extras_stable():
    text = (_root() / "pyproject.toml").read_text(encoding="utf-8")
    names = _optional_extra_names(text)
    assert names == {"spy", "cartpole", "inspect", "gateway", "serve", "lock", "export", "copilot", "dev"}


def test_axiom_root_public_api():
    assert axiom.__all__ == ["AxiomModel", "load"]
    assert set(dir(axiom)) >= {"AxiomModel", "load"}


def test_axiom_model_surface(tmp_path: Path):
    from axiom import AxiomModel, load
    from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
    from axiom.compiler.parser import parse_ax, reset_parser
    from axiom.compiler.serializer import save_bundle
    from axiom.engine.block_executor import InterpretedBlock

    reset_parser()
    ir = ast_to_ir(parse_ax("y = neural([1.0, 2.0]);"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    m = AxiomModel(block)
    assert hasattr(m, "block") and m.block is block
    for name in ("predict", "explain", "export_report"):
        assert callable(getattr(m, name))
    p = tmp_path / "tmp_baseline.axb"
    save_bundle(block, p)
    m2 = load(p)
    assert isinstance(m2, AxiomModel)


def test_gateway_package_exports():
    import axiom.gateway as gw

    expected = {
        "build_block_audit",
        "create_app",
        "create_gateway_app",
        "default_scan_text",
        "forward_to_downstream",
        "gateway_app_from_env",
        "is_approved",
        "policy_explain",
        "resolve_signals",
    }
    assert set(gw.__all__) == expected


def test_serve_create_app_not_gateway_create_app():
    pytest.importorskip("fastapi")
    from axiom.gateway import server as gws
    from axiom import serve as srv

    assert srv.create_app is not gws.create_app
    assert "bundle" in (srv.create_app.__doc__ or "").lower()


def test_bundle_server_routes(tmp_path: Path):
    pytest.importorskip("fastapi")
    from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
    from axiom.compiler.parser import parse_ax, reset_parser
    from axiom.compiler.serializer import save_bundle
    from axiom.engine.block_executor import InterpretedBlock
    from axiom.serve import create_app

    reset_parser()
    ir = ast_to_ir(parse_ax("y = neural([1.0, 2.0]);"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    axb = tmp_path / "b.axb"
    save_bundle(block, axb)
    app = create_app(axb)
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/health" in paths
    assert "/predict" in paths
    assert "/explain" in paths
    assert "/report" in paths


def test_gateway_app_has_chat_route(tmp_path: Path):
    pytest.importorskip("fastapi")
    from axiom.api import load
    from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
    from axiom.compiler.parser import parse_ax, reset_parser
    from axiom.compiler.serializer import save_bundle
    from axiom.engine.block_executor import InterpretedBlock
    from axiom.gateway.server import create_gateway_app

    reset_parser()
    ir = ast_to_ir(parse_ax("y = neural([1.0, 2.0]);"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    axb = tmp_path / "g.axb"
    save_bundle(block, axb)
    model = load(axb)
    app = create_gateway_app(model, None, downstream_url="http://127.0.0.1:9/nope")
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/gateway/chat" in paths


def test_src_axiom_package_layout():
    base = _root() / "src" / "axiom"
    for sub in (
        "compiler",
        "engine",
        "primitives",
        "tools",
        "gateway",
        "export",
        "security",
        "experts",
        "copilot",
    ):
        assert (base / sub).is_dir(), f"missing {sub}/"
    assert (base / "cli.py").is_file()
    assert (base / "api.py").is_file()
    assert (base / "serve.py").is_file()
    assert (base / "datasets.py").is_file()
