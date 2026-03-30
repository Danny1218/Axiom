"""``axiom.experts``: protocol, dataclasses, registry (Phase 58)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import axiom.experts as experts_mod
import pytest

from axiom.experts import (
    DuplicateExpertRegistrationError,
    ExpertDraftRequest,
    ExpertDraftResponse,
    ExpertRepairRequest,
    ExpertTraceSummaryRequest,
    SemanticExpert,
    UnknownExpertError,
    clear_registry,
    iter_registered,
    register,
    registered_names,
    resolve,
    unregister,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_registry()
    yield
    clear_registry()


class _StubExpert:
    """In-process expert: no I/O; satisfies :class:`SemanticExpert`."""

    def __init__(self, name: str = "stub") -> None:
        self._name = name

    def draft_program(self, request: ExpertDraftRequest) -> ExpertDraftResponse:
        return ExpertDraftResponse(
            ax_source=f"// goal: {request.goal}\ny = 1.0;",
            backend_name=self._name,
            explanation="stub draft",
            metadata={"kind": "draft", "ctx_keys": tuple(request.context.keys())},
        )

    def repair_program(self, request: ExpertRepairRequest) -> ExpertDraftResponse:
        return ExpertDraftResponse(
            ax_source=request.current_program + "\n// fixed",
            backend_name=self._name,
            explanation=f"stub repair for {request.error_report[:20]}",
            metadata={"kind": "repair"},
        )

    def summarize_trace(self, request: ExpertTraceSummaryRequest) -> str:
        return f"[{self._name}] goal={request.goal!r} trace_keys={sorted(request.trace)}"


def test_stub_is_semantic_expert():
    e: SemanticExpert = _StubExpert("e1")
    assert isinstance(e, SemanticExpert)


def test_draft_request_and_response():
    req = ExpertDraftRequest("build a parity checker", context={"lang": "ax"})
    assert req.goal == "build a parity checker"
    assert req.context["lang"] == "ax"
    e = _StubExpert("acme")
    out = e.draft_program(req)
    assert isinstance(out, ExpertDraftResponse)
    assert "y = 1.0" in out.ax_source
    assert out.backend_name == "acme"
    assert out.explanation == "stub draft"
    assert out.metadata["kind"] == "draft"
    assert "lang" in out.metadata["ctx_keys"]


def test_repair_request_and_response():
    req = ExpertRepairRequest(
        goal="fix syntax",
        current_program="x = 1.0;",
        error_report="parse error line 1",
        context={},
    )
    out = _StubExpert("acme").repair_program(req)
    assert "// fixed" in out.ax_source
    assert out.backend_name == "acme"
    assert out.metadata["kind"] == "repair"


def test_summarize_trace_returns_str():
    req = ExpertTraceSummaryRequest(
        goal="explain run",
        program="y = x;",
        trace={"x": 1.0},
        metrics={"loss": 0.1},
        context={},
    )
    s = _StubExpert("acme").summarize_trace(req)
    assert isinstance(s, str)
    assert "explain run" in s
    assert "x" in s


def test_register_resolve_and_names():
    register("alpha", _StubExpert("alpha"))
    assert registered_names() == ("alpha",)
    ex = resolve("alpha")
    r = ex.draft_program(ExpertDraftRequest("g"))
    assert r.backend_name == "alpha"


def test_duplicate_registration_raises():
    register("dup", _StubExpert("dup"))
    with pytest.raises(DuplicateExpertRegistrationError) as ei:
        register("dup", _StubExpert("other"))
    assert ei.value.name == "dup"


def test_allow_replace():
    register("x", _StubExpert("first"))
    register("x", _StubExpert("second"), allow_replace=True)
    assert resolve("x").draft_program(ExpertDraftRequest("g")).backend_name == "second"


def test_unknown_expert_raises():
    with pytest.raises(UnknownExpertError) as ei:
        resolve("missing")
    assert ei.value.name == "missing"


def test_unregister():
    register("tmp", _StubExpert("tmp"))
    unregister("tmp")
    with pytest.raises(UnknownExpertError):
        resolve("tmp")


def test_iter_registered_sorted():
    register("b", _StubExpert("b"))
    register("a", _StubExpert("a"))
    pairs = list(iter_registered())
    assert [n for n, _ in pairs] == ["a", "b"]


def test_frozen_request_immutable():
    req = ExpertDraftRequest("g", context={"k": 1})
    with pytest.raises(FrozenInstanceError):
        req.goal = "no"  # type: ignore[misc]


def test_package_all_exports():
    assert set(experts_mod.__all__) == {
        "DuplicateExpertRegistrationError",
        "ExpertDraftRequest",
        "ExpertDraftResponse",
        "ExpertRepairRequest",
        "ExpertTraceSummaryRequest",
        "SemanticExpert",
        "UnknownExpertError",
        "clear_registry",
        "iter_registered",
        "register",
        "registered_names",
        "resolve",
        "unregister",
    }
