"""OP_DIV: avoid torch.where(a/b, ...) NaN trap on backward when b==0."""

import torch

from engine.interpreter import eval_expr

_CPU = torch.device("cpu")
_F32 = torch.float32


def test_op_div_by_zero_forward_zero_backward_no_nan():
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    env = {"a": a, "b": b}
    ir = [("OP_LOAD", "a"), ("OP_LOAD", "b"), ("OP_DIV",)]
    out = eval_expr(env, ir, device=_CPU, dtype=_F32)
    assert out.item() == 0.0
    out.backward()
    for t in (a.grad, b.grad):
        if t is not None:
            assert not torch.isnan(t).any(), "grad must not be NaN (torch.where / div trap)"


def test_op_div_nonzero_matches_plain_div_and_grad():
    a = torch.tensor(6.0, requires_grad=True)
    b = torch.tensor(2.0, requires_grad=True)
    env = {"a": a, "b": b}
    ir = [("OP_LOAD", "a"), ("OP_LOAD", "b"), ("OP_DIV",)]
    out = eval_expr(env, ir, device=_CPU, dtype=_F32)
    assert out.item() == 3.0
    out.backward()
    assert a.grad is not None and b.grad is not None
    assert torch.isclose(a.grad, torch.tensor(0.5)).all()
    assert torch.isclose(b.grad, torch.tensor(-1.5)).all()


def test_op_div_near_zero_denominator_forward_stable():
    a = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(1e-13, requires_grad=True)
    env = {"a": a, "b": b}
    ir = [("OP_LOAD", "a"), ("OP_LOAD", "b"), ("OP_DIV",)]
    out = eval_expr(env, ir, device=_CPU, dtype=_F32)
    assert out.item() == 0.0
    out.backward()
    for t in (a.grad, b.grad):
        if t is not None:
            assert not torch.isnan(t).any()
