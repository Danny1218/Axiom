"""Phase 9: batched (B,) env matches independent per-row runs."""

import torch

from engine.interpreter import eval_expr, exec_stmt, make_seed_map, run_loop_snapshots


_CPU = torch.device("cpu")
_F32 = torch.float32


def _assert_row_matches_per_row_ref(
    vec: torch.Tensor, h_batch: torch.Tensor, cond_ir, body_ir, **kwargs
) -> None:
    """vec: (B, T, D); compare each row to run_loop_snapshots(h_batch[b:b+1], ...)."""
    B = h_batch.shape[0]
    for b in range(B):
        ref, _ref_m = run_loop_snapshots(h_batch[b : b + 1], cond_ir, body_ir, **kwargs)
        assert ref.shape[0] == 1
        Tb = ref.shape[1]
        assert torch.allclose(vec[b, :Tb, :], ref[0, :Tb, :], atol=0, rtol=0)
        if Tb == 0 and vec.shape[1] > 0:
            ref0 = vec[b, 0].clone()
            for t in range(1, vec.shape[1]):
                assert torch.allclose(vec[b, t], ref0, atol=0, rtol=0)
        elif Tb < vec.shape[1]:
            frozen = vec[b, Tb - 1 : Tb, :].expand(vec.shape[1] - Tb, -1)
            assert torch.allclose(vec[b, Tb:, :], frozen, atol=0, rtol=0)


def test_run_loop_snapshots_b4_matches_four_independent_rows():
    torch.manual_seed(0)
    cond = [("OP_LOAD", "i"), ("OP_CONST", 0.0), ("OP_CMP_GT",)]
    body = [("OP_ASSIGN", "i", [("OP_LOAD", "i"), ("OP_CONST", 1.0), ("OP_SUB",)])]
    seed = make_seed_map(cond, body, 6)
    h = torch.zeros(4, 6)
    h[:, 0] = torch.tensor([5.0, 2.0, 1.0, 1.0])
    vec, _vm = run_loop_snapshots(h, cond, body, dim=6, max_unroll=10, seed_map=seed, prelude_stmts=[])
    assert vec.shape[0] == 4
    _assert_row_matches_per_row_ref(vec, h, cond, body, dim=6, max_unroll=10, seed_map=seed, prelude_stmts=[])


def test_run_loop_snapshots_batched_grad():
    h = torch.randn(3, 4, requires_grad=True)
    cond = [("OP_LOAD", "x"), ("OP_CONST", 0.0), ("OP_CMP_GT",)]
    body = [("OP_ASSIGN", "x", [("OP_LOAD", "x"), ("OP_CONST", 0.5), ("OP_MUL",)])]
    seed = make_seed_map(cond, body, 4)
    mat, _m = run_loop_snapshots(h, cond, body, dim=4, max_unroll=6, seed_map=seed)
    mat.sum().backward()
    assert h.grad is not None
    assert torch.count_nonzero(h.grad) > 0


def test_eval_expr_batched_independent_rows():
    B = 4
    env = {
        "a": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "b": torch.tensor([1.0, 1.0, 1.0, 1.0]),
    }
    ir = [("OP_LOAD", "a"), ("OP_LOAD", "b"), ("OP_ADD",)]
    out = eval_expr(env, ir, B=B, device=_CPU, dtype=_F32)
    assert out.shape == (B,)
    assert torch.allclose(out, torch.tensor([2.0, 3.0, 4.0, 5.0]))


def test_op_conditional_branch_blend_batch():
    B = 3
    env = {
        "c": torch.tensor([1.0, 0.0, 1.0]),
        "x": torch.zeros(B),
    }
    stmt = (
        "OP_CONDITIONAL",
        [("OP_LOAD", "c")],
        [("OP_ASSIGN", "x", [("OP_CONST", 10.0)])],
        [("OP_ASSIGN", "x", [("OP_CONST", 20.0)])],
    )
    exec_stmt(env, stmt, B=B, dim=4, max_unroll=4, device=_CPU, dtype=_F32, active_mask=None)
    assert torch.allclose(env["x"], torch.tensor([10.0, 20.0, 10.0]))


def test_while_mixed_row_iteration_counts_alignment():
    """Rows exit at different times; batched T equals max per-row T; trailing frames frozen per row."""
    h = torch.zeros(2, 4)
    h[0, 0] = 1.0
    h[1, 0] = 3.0
    cond = [("OP_LOAD", "i"), ("OP_CONST", 0.0), ("OP_CMP_GT",)]
    body = [("OP_ASSIGN", "i", [("OP_LOAD", "i"), ("OP_CONST", 1.0), ("OP_SUB",)])]
    seed = make_seed_map(cond, body, 4)
    vec, _vm = run_loop_snapshots(h, cond, body, dim=4, max_unroll=8, seed_map=seed)
    _assert_row_matches_per_row_ref(vec, h, cond, body, dim=4, max_unroll=8, seed_map=seed)
