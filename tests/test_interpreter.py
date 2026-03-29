import torch

from engine.interpreter import (
    eval_expr,
    make_seed_map,
    run_loop_snapshots,
    run_while_loop,
    truthy,
)
from engine.topology import _absorbed_prelude_indices, _prelude_stmts_before_loop

_CPU = torch.device("cpu")
_F32 = torch.float32


def test_eval_expr_arithmetic():
    env = {"a": torch.tensor([2.0]), "b": torch.tensor([3.0])}
    ir = [("OP_LOAD", "a"), ("OP_LOAD", "b"), ("OP_MUL",)]
    assert eval_expr(env, ir, B=1, device=_CPU, dtype=_F32).item() == 6.0


def test_eval_expr_grad_through_load():
    a = torch.tensor([2.0], requires_grad=True)
    env = {"a": a, "b": torch.tensor([3.0])}
    ir = [("OP_LOAD", "a"), ("OP_LOAD", "b"), ("OP_ADD",)]
    out = eval_expr(env, ir, B=1, device=_CPU, dtype=_F32)
    out.sum().backward()
    assert a.grad is not None and a.grad.reshape(-1)[0].item() == 1.0


def test_run_while_countdown_snapshots():
    env = {"i": torch.tensor([3.0], device=_CPU, dtype=_F32)}
    cond = [("OP_LOAD", "i"), ("OP_CONST", 0.0), ("OP_CMP_GT",)]
    body = [("OP_ASSIGN", "i", [("OP_LOAD", "i"), ("OP_CONST", 1.0), ("OP_SUB",)])]
    var_order = ["i", "_pad0", "_pad1"]
    snaps = run_while_loop(
        env,
        cond,
        body,
        B=1,
        dim=3,
        max_unroll=10,
        var_order=var_order,
        device=_CPU,
        dtype=_F32,
    )
    assert len(snaps) == 3
    assert snaps[0][0, 0].item() == 2.0 and snaps[-1][0, 0].item() == 0.0


def test_run_loop_snapshots_with_prelude():
    h = torch.zeros(1, 5)
    cond = [("OP_LOAD", "i"), ("OP_CONST", 0.0), ("OP_CMP_GT",)]
    body = [("OP_ASSIGN", "i", [("OP_LOAD", "i"), ("OP_CONST", 1.0), ("OP_SUB",)])]
    prelude = [("OP_ASSIGN", "i", [("OP_CONST", 3.0)])]
    seed = make_seed_map(cond, body, 5)
    mat = run_loop_snapshots(h, cond, body, dim=5, max_unroll=8, seed_map=seed, prelude_stmts=prelude)
    assert mat.shape == (1, 3, 5)
    assert mat[0, 0, 0].item() == 2.0


def test_run_loop_snapshots_grad_through_seed():
    h = torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    cond = [("OP_LOAD", "i"), ("OP_CONST", 0.0), ("OP_CMP_GT",)]
    body = [("OP_ASSIGN", "i", [("OP_LOAD", "i"), ("OP_CONST", 1.0), ("OP_SUB",)])]
    seed = make_seed_map(cond, body, 5)
    mat = run_loop_snapshots(h, cond, body, dim=5, max_unroll=8, seed_map=seed, prelude_stmts=[])
    mat.sum().backward()
    assert h.grad is not None and h.grad[0, 0].item() != 0.0


def test_truthy():
    assert truthy(torch.tensor(1.0)) and not truthy(torch.tensor(0.0))


def test_prelude_absorption_helpers():
    ir = [
        ("OP_ASSIGN", "i", [("OP_CONST", 3.0)]),
        ("OP_LOOP", [("OP_CONST", 1)], []),
    ]
    assert _absorbed_prelude_indices(ir) == {0}
    assert len(_prelude_stmts_before_loop(ir, 1)) == 1


def test_interpreted_loop_zero_iterations_falls_back():
    from engine.loop_executor import InterpretedLiquidLoop

    cond = [("OP_LOAD", "i"), ("OP_CONST", 0.0), ("OP_CMP_GT",)]
    body: list = []
    m = InterpretedLiquidLoop(4, cond, body, [], {}, num_basis=3, max_unroll=4)
    x = torch.randn(2, 4, requires_grad=True)
    y = m(x)
    assert y.shape == (2, 4)
    y.sum().backward()
    assert x.grad is not None
