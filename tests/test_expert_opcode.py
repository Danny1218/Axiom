"""Phase 66: ``expert("backend", feat_expr)`` → ``OP_EXPERT`` (non-differentiable runtime)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import axiom
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.compiler.serializer import save_bundle
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.expert_call import ExpertRuntimeError
from axiom.engine.interpreter import eval_expr, exec_stmt
from axiom.export.onnx_export import OnnxExportError, export_interpreted_block_to_onnx


def _ir() -> list:
    reset_parser()
    return ast_to_ir(parse_ax('e = expert("demo", [x, 1.0]);'))


def test_parser_accepts_expert_call():
    reset_parser()
    t = parse_ax('y = expert("b", [a, 2.0]);')
    assert t.data == "start"


def test_expert_lowers_to_op_expert():
    ir = _ir()
    assign = ir[0]
    assert assign[0] == "OP_ASSIGN"
    rhs = assign[2]
    expert_tups = [x for x in rhs if isinstance(x, tuple) and x and x[0] == "OP_EXPERT"]
    assert len(expert_tups) == 1
    assert expert_tups[0][1] == "demo"


def test_expert_reserved_builtin_name():
    reset_parser()
    with pytest.raises(ValueError, match="reserved"):
        ast_to_ir(parse_ax("def expert(a) { return a; }"))


def test_expert_requires_string_literal_backend():
    reset_parser()
    with pytest.raises(ValueError, match="string literal"):
        ast_to_ir(parse_ax("e = expert(x, [1.0]);"))


def test_expert_requires_two_args():
    reset_parser()
    with pytest.raises(ValueError, match="2 arguments"):
        ast_to_ir(parse_ax('e = expert("a");'))


def test_interpreter_fake_expert_handler():
    ir = _ir()
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    env = {
        "x": torch.tensor([2.0, 3.0]),
        "e": torch.zeros(2),
    }
    B = 2
    calls: list[tuple[str, list[float]]] = []

    def handler(name: str, feats: list[float]) -> float:
        calls.append((name, list(feats)))
        return sum(feats)

    rhs = ir[0][2]
    out = eval_expr(
        env,
        rhs,
        B=B,
        device=torch.device("cpu"),
        dtype=torch.float32,
        expert_handler=lambda n, f: handler(n, list(f)),
    )
    assert out.shape == (2,)
    assert float(out[0].item()) == pytest.approx(3.0)
    assert float(out[1].item()) == pytest.approx(4.0)
    assert calls[0] == ("demo", [2.0, 1.0])
    assert calls[1] == ("demo", [3.0, 1.0])


def test_interpreter_expert_fallback_without_handler():
    ir = _ir()
    env = {"x": torch.tensor([9.0]), "e": torch.zeros(1)}
    rhs = ir[0][2]
    out = eval_expr(
        env,
        rhs,
        B=1,
        device=torch.device("cpu"),
        dtype=torch.float32,
        expert_fallback=-1.5,
    )
    assert float(out.item()) == pytest.approx(-1.5)


def test_interpreter_expert_raises_when_no_backend():
    ir = _ir()
    env = {"x": torch.tensor([1.0]), "e": torch.zeros(1)}
    rhs = ir[0][2]
    with pytest.raises(ExpertRuntimeError, match="no runtime backend"):
        eval_expr(
            env,
            rhs,
            B=1,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_interpreted_block_forward_expert_handler():
    ir = _ir()
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)

    def handler(name: str, feats) -> float:
        assert name == "demo"
        return float(feats[0]) * 10.0 + float(feats[1])

    b = InterpretedBlock(ir, abi, abi_widths=aw, expert_handler=handler)
    h = torch.zeros(1, 16)
    h[0, abi["x"]] = 0.5
    out = b(h)
    col_e = abi["e"]
    assert float(out[0, col_e].item()) == pytest.approx(6.0)
    assert b._last_expert_trace == [{"op": "expert", "backend": "demo"}]


def test_axiom_explain_includes_expert_calls(tmp_path: Path):
    ir = _ir()
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    b = InterpretedBlock(
        ir, abi, abi_widths=aw, expert_handler=lambda _n, f: float(f[0]) + 1.0
    )
    save_bundle(b, tmp_path / "ex.axb")
    model = axiom.load(tmp_path / "ex.axb")
    model.block.expert_handler = lambda _n, f: float(f[0]) + 1.0
    trace = model.explain({"x": 3.0})
    assert trace["expert_calls"] == [{"op": "expert", "backend": "demo"}]
    assert trace["e"] == pytest.approx(4.0)


def test_onnx_export_rejects_expert_program(tmp_path: Path):
    pytest.importorskip("onnx")
    ir = _ir()
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    b = InterpretedBlock(ir, abi, abi_widths=aw, expert_fallback=0.0)
    with pytest.raises(OnnxExportError, match="OP_EXPERT"):
        export_interpreted_block_to_onnx(b, tmp_path / "x.onnx")


def test_exec_stmt_assign_with_expert():
    ir = _ir()
    stmt = ir[0]
    env = {"x": torch.tensor([1.0]), "e": torch.zeros(1)}
    exec_stmt(
        env,
        stmt,
        B=1,
        dim=16,
        max_unroll=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        expert_handler=lambda _n, f: float(f[0]) + 0.25,
    )
    assert float(env["e"].item()) == pytest.approx(1.25)


def test_expert_coexists_with_neural():
    reset_parser()
    src = 'y = neural([1.0]); z = expert("x", [y, 2.0]); w = y + z;'
    ir = ast_to_ir(parse_ax(src))
    assert "OP_NEURAL" in str(ir) and "OP_EXPERT" in str(ir)
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    b = InterpretedBlock(
        ir, abi, abi_widths=aw, expert_handler=lambda _n, f: float(f[0]) + float(f[1])
    )
    h = torch.zeros(1, 16)
    out = b(h)
    col_y, col_z = abi["y"], abi["z"]
    yv = float(out[0, col_y].item())
    zv = float(out[0, col_z].item())
    assert zv == pytest.approx(yv + 2.0)
    assert len(b._last_expert_trace) == 1
