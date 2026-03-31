"""FastAPI bundle server (``axiom serve``)."""

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.compiler.serializer import save_bundle
from axiom.engine.block_executor import InterpretedBlock
from axiom.serve import create_app


@pytest.fixture
def sample_axb(tmp_path: Path) -> Path:
    reset_parser()
    ir = ast_to_ir(parse_ax("y = neural([1.0, 2.0]);"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    p = tmp_path / "s.axb"
    save_bundle(block, p)
    return p


@pytest.fixture
def expert_axb(tmp_path: Path) -> Path:
    reset_parser()
    ir = ast_to_ir(parse_ax('e = expert("demo", [x, 1.0]);'))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    p = tmp_path / "ex.axb"
    save_bundle(block, p)
    return p


def test_predict_expert_bundle_503_without_runtime_wiring(expert_axb: Path):
    from fastapi.testclient import TestClient

    app = create_app(expert_axb)
    c = TestClient(app)
    r = c.post("/predict", json={"inputs": {"x": 1.0}})
    assert r.status_code == 503
    assert "expert()" in r.json()["detail"]


def test_predict_expert_bundle_ok_with_registry(expert_axb: Path):
    from fastapi.testclient import TestClient

    from axiom.engine.expert_registry import ExpertRuntimeRegistry

    reg = ExpertRuntimeRegistry()
    reg.register("demo", lambda _n, f: float(f[0]) + 0.5)
    app = create_app(expert_axb, expert_registry=reg)
    c = TestClient(app)
    r = c.post("/predict", json={"inputs": {"x": 2.0}})
    assert r.status_code == 200
    assert r.json()["outputs"]["e"] == pytest.approx(2.5)


def test_explain_expert_bundle_ok_with_handler(expert_axb: Path):
    from fastapi.testclient import TestClient

    app = create_app(expert_axb, expert_handler=lambda _n, f: 3.0)
    c = TestClient(app)
    r = c.post("/explain", json={"inputs": {"x": 0.0}})
    assert r.status_code == 200
    assert r.json()["trace"]["e"] == pytest.approx(3.0)


def test_health_ok(sample_axb: Path):
    from fastapi.testclient import TestClient

    app = create_app(sample_axb)
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert Path(data["bundle_path"]).resolve() == sample_axb.resolve()


def test_predict_single_row(sample_axb: Path):
    from fastapi.testclient import TestClient

    app = create_app(sample_axb)
    c = TestClient(app)
    r = c.post("/predict", json={"inputs": {}})
    assert r.status_code == 200
    out = r.json()["outputs"]
    assert "y" in out
    assert isinstance(out["y"], (int, float))


def test_predict_batch(sample_axb: Path):
    from fastapi.testclient import TestClient

    app = create_app(sample_axb)
    c = TestClient(app)
    r = c.post(
        "/predict",
        json={"inputs": [{}, {}]},
    )
    assert r.status_code == 200
    outs = r.json()["outputs"]
    assert isinstance(outs, list) and len(outs) == 2
    assert "y" in outs[0]


def test_explain_trace(sample_axb: Path):
    from fastapi.testclient import TestClient

    app = create_app(sample_axb)
    c = TestClient(app)
    r = c.post("/explain", json={"inputs": {}})
    assert r.status_code == 200
    tr = r.json()["trace"]
    assert isinstance(tr, dict)


def test_report_html_inline(sample_axb: Path):
    from fastapi.testclient import TestClient

    app = create_app(sample_axb)
    c = TestClient(app)
    r = c.post("/report", json={"inputs": {}})
    assert r.status_code == 200
    body = r.json()
    assert body["output_path"] is None
    assert body["html"] is not None
    assert "<!DOCTYPE html>" in body["html"]


def test_report_writes_file(sample_axb: Path, tmp_path: Path):
    from fastapi.testclient import TestClient

    out = tmp_path / "r.html"
    app = create_app(sample_axb)
    c = TestClient(app)
    r = c.post(
        "/report",
        json={"inputs": {}, "output_path": str(out)},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["html"] is None
    assert body["output_path"] == str(out.resolve())
    assert out.is_file()
    assert "<!DOCTYPE html>" in out.read_text(encoding="utf-8")


def test_auth_required_when_env_set(sample_axb: Path, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("AXIOM_API_KEY", "secret-token")
    app = create_app(sample_axb)
    c = TestClient(app)
    assert c.get("/health").status_code == 200
    assert c.post("/predict", json={"inputs": {}}).status_code == 401
    r = c.post(
        "/predict",
        json={"inputs": {}},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert r.status_code == 200
    r2 = c.post(
        "/predict",
        json={"inputs": {}},
        headers={"X-API-Key": "secret-token"},
    )
    assert r2.status_code == 200


def test_create_app_missing_bundle_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        create_app(tmp_path / "nope.axb")


def test_cli_serve_help_exits_ok():
    from axiom.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["serve", "--help"])
    assert exc.value.code == 0


def test_cli_serve_uses_host_port_env(sample_axb: Path, monkeypatch: pytest.MonkeyPatch):
    """Container / compose set HOST and PORT; must override CLI defaults."""
    pytest.importorskip("uvicorn")
    monkeypatch.setenv("AXIOM_BUNDLE_PATH", str(sample_axb))
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9123")
    ran: list[tuple[str, int]] = []

    def fake_run(app, host, port, log_level="info"):
        ran.append((host, int(port)))

    monkeypatch.setattr("uvicorn.run", fake_run)
    from axiom.cli import main

    main(["serve"])
    assert ran == [("0.0.0.0", 9123)]


def test_cli_serve_falls_back_to_args_when_env_unset(sample_axb: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("uvicorn")
    monkeypatch.setenv("AXIOM_BUNDLE_PATH", str(sample_axb))
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    ran: list[tuple[str, int]] = []

    def fake_run(app, host, port, log_level="info"):
        ran.append((host, int(port)))

    monkeypatch.setattr("uvicorn.run", fake_run)
    from axiom.cli import main

    main(["serve", "--host", "127.0.0.1", "--port", "8000"])
    assert ran == [("127.0.0.1", 8000)]
