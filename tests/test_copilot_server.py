"""Phase 67: FastAPI copilot server (draft / search / summarize / auth)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from axiom.copilot.server import create_app
from axiom.experts.base import (
    ExpertDraftRequest,
    ExpertDraftResponse,
    ExpertRepairRequest,
    ExpertTraceSummaryRequest,
)


class _FakeCopilotExpert:
    """No network; returns tiny valid .ax for draft/repair."""

    def __init__(self) -> None:
        self._ax = "y = 1.0;\n"
        self.draft_calls: list[ExpertDraftRequest] = []
        self.repair_calls: list[ExpertRepairRequest] = []
        self.summarize_calls: list[ExpertTraceSummaryRequest] = []

    def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
        self.draft_calls.append(request)
        return ExpertDraftResponse(
            ax_source=self._ax,
            backend_name="fake",
            explanation="ok",
            metadata={"n": 1},
        )

    def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
        self.repair_calls.append(request)
        return ExpertDraftResponse(ax_source=self._ax, backend_name="fake")

    def summarize_trace(self, request: ExpertTraceSummaryRequest) -> str:
        self.summarize_calls.append(request)
        return "trace summary line"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("AXIOM_COPILOT_API_KEY", raising=False)
    from fastapi.testclient import TestClient

    exp = _FakeCopilotExpert()
    app = create_app(exp)
    c = TestClient(app)
    return c, exp


def test_health_no_auth_required(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_draft_success(client):
    c, exp = client
    r = c.post("/draft", json={"goal": "make a constant", "domain_context": "test"})
    assert r.status_code == 200
    data = r.json()
    assert "y = 1.0" in data["ax_source"]
    assert data["backend_name"] == "fake"
    assert data["explanation"] == "ok"
    assert data["metadata"] == {"n": 1}
    assert len(exp.draft_calls) == 1
    assert exp.draft_calls[0].goal == "make a constant"


def test_draft_validation_error(client):
    c, _ = client
    r = c.post("/draft", json={})
    assert r.status_code == 422


def test_search_compile_only_converges(client):
    c, exp = client
    r = c.post(
        "/search",
        json={
            "goal": "trivial",
            "max_iterations": 3,
            "compile_only": True,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["converged"] is True
    assert "y = 1.0" in data["best_source"]
    assert data["best_evaluation"]["success"] is True
    assert len(data["iterations"]) >= 1
    assert data["iterations"][0]["evaluation"]["success"] is True


def test_search_with_examples_predict_rows(client):
    c, _ = client
    r = c.post(
        "/search",
        json={
            "goal": "trivial",
            "max_iterations": 2,
            "compile_only": False,
            "examples": [{"inputs": {"x": 0.0}, "expected": {"y": 1.0}}],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["best_evaluation"]["mode"] == "predict_rows"


def test_summarize_endpoint(client):
    c, exp = client
    r = c.post(
        "/summarize",
        json={
            "goal": "g",
            "program": "y=x;",
            "trace": {"y": 1.0},
            "metrics": {"m": 0.5},
            "context": {},
        },
    )
    assert r.status_code == 200
    assert r.json()["summary"] == "trace summary line"
    assert len(exp.summarize_calls) == 1


def test_auth_bearer_required_when_env_set(monkeypatch, client):
    c, _ = client
    monkeypatch.setenv("AXIOM_COPILOT_API_KEY", "secret-copilot")
    from fastapi.testclient import TestClient

    app = create_app(_FakeCopilotExpert())
    c2 = TestClient(app)
    assert c2.post("/draft", json={"goal": "a"}).status_code == 401
    r = c2.post(
        "/draft",
        json={"goal": "a"},
        headers={"Authorization": "Bearer secret-copilot"},
    )
    assert r.status_code == 200


def test_auth_x_api_key(monkeypatch):
    monkeypatch.setenv("AXIOM_COPILOT_API_KEY", "k2")
    from fastapi.testclient import TestClient

    app = create_app(_FakeCopilotExpert())
    c = TestClient(app)
    r = c.post("/draft", json={"goal": "b"}, headers={"X-API-Key": "k2"})
    assert r.status_code == 200


def test_malformed_json(client):
    c, _ = client
    r = c.post("/draft", content=b"not-json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_search_writes_artifact_dir(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("AXIOM_COPILOT_API_KEY", raising=False)
    from fastapi.testclient import TestClient

    app = create_app(_FakeCopilotExpert())
    c = TestClient(app)
    ad = str(tmp_path / "art")
    r = c.post(
        "/search",
        json={"goal": "t", "max_iterations": 1, "compile_only": True, "artifact_dir": ad},
    )
    assert r.status_code == 200
    assert (Path(ad) / "best.ax").is_file()
    assert (Path(ad) / "iterations.json").is_file()


def test_create_app_import_error_message():
    # FastAPI is present in test env; smoke-check module loads
    assert callable(create_app)
