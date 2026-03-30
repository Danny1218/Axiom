"""Phase 9: batched (B,) env matches independent per-row runs."""

import torch

from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.interpreter import eval_expr, exec_stmt, make_seed_map, run_loop_snapshots


_CPU = torch.device("cpu")
_F32 = torch.float32


def _assert_row_matches_per_row_ref(
    vec: torch.Tensor, h_batch: torch.Tensor, cond_ir, body_ir, **kwargs
) -> None:
    """vec: (B, T, D); each row must match an independent run_loop_snapshots (same fixed T)."""
    B = h_batch.shape[0]
    for b in range(B):
        ref, _ref_m = run_loop_snapshots(h_batch[b : b + 1], cond_ir, body_ir, **kwargs)
        assert ref.shape[0] == 1 and ref.shape[1] == vec.shape[1]
        assert torch.allclose(vec[b], ref[0], atol=0, rtol=0)


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
    """Rows finish the loop at different steps; batched T is fixed max_unroll; trailing frames frozen per row."""
    h = torch.zeros(2, 4)
    h[0, 0] = 1.0
    h[1, 0] = 3.0
    cond = [("OP_LOAD", "i"), ("OP_CONST", 0.0), ("OP_CMP_GT",)]
    body = [("OP_ASSIGN", "i", [("OP_LOAD", "i"), ("OP_CONST", 1.0), ("OP_SUB",)])]
    seed = make_seed_map(cond, body, 4)
    vec, _vm = run_loop_snapshots(h, cond, body, dim=4, max_unroll=8, seed_map=seed)
    _assert_row_matches_per_row_ref(vec, h, cond, body, dim=4, max_unroll=8, seed_map=seed)


def test_literal_vector_times_scalar_interpreted_block():
    reset_parser()
    ir = ast_to_ir(parse_ax("x = [1.0, 2.0]; y = x * 2.0;"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    h = torch.zeros(2, 16)
    out = block(h)
    yc = abi["y"]
    assert torch.allclose(out[:, yc : yc + 2], torch.tensor([[2.0, 4.0], [2.0, 4.0]]))


def test_vector_times_scalar_batch_gt2():
    """(B, K) * (B,) with B>2: batch scalars promote to (B, 1) (Phase 32)."""
    reset_parser()
    ir = ast_to_ir(parse_ax("x = [1.0, 2.0, 3.0]; y = x * 2.0;"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    h = torch.zeros(4, 16)
    out = block(h)
    yc = abi["y"]
    want = torch.tensor([[2.0, 4.0, 6.0]] * 4)
    assert torch.allclose(out[:, yc : yc + 3], want)


def test_math_unary_abs_exp_vector_literal_matches_torch():
    """``abs`` / ``exp`` preserve (B,K); match PyTorch element-wise."""
    reset_parser()
    ir = ast_to_ir(parse_ax("x = [-1.0, 2.0]; y = abs(x); z = exp(y);"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    h = torch.zeros(2, 16)
    out = block(h)
    xc, yc, zc = abi["x"], abi["y"], abi["z"]
    x_t = torch.tensor([[-1.0, 2.0], [-1.0, 2.0]])
    want_y = torch.abs(x_t)
    want_z = torch.exp(want_y)
    assert torch.allclose(out[:, yc : yc + 2], want_y)
    assert torch.allclose(out[:, zc : zc + 2], want_z)


def test_math_unary_all_preserve_width_and_match_torch():
    cases = [
        ("w1", "abs(x)", torch.abs),
        ("w2", "exp(x)", torch.exp),
        ("w3", "log(x)", torch.log),
        ("w4", "sqrt(x)", torch.sqrt),
        ("w5", "sin(x)", torch.sin),
        ("w6", "cos(x)", torch.cos),
    ]
    for name, rhs, fn in cases:
        reset_parser()
        ir = ast_to_ir(parse_ax(f"x = [0.25, 0.5]; {name} = {rhs};"))
        abi = extract_global_abi(ir, max_vars=32)
        aw = extract_abi_widths(ir, max_vars=32)
        block = InterpretedBlock(ir, abi, abi_widths=aw)
        h = torch.zeros(3, 32)
        out = block(h)
        x_t = torch.tensor([[0.25, 0.5]] * 3)
        col = abi[name]
        got = out[:, col : col + 2]
        assert torch.allclose(got, fn(x_t), equal_nan=True)


def test_sum_mean_dot_on_vector_literal():
    reset_parser()
    ir = ast_to_ir(
        parse_ax("x = [1.0, 2.0, 3.0]; y = sum(x); z = dot(x, [2.0, 2.0, 2.0]); w = mean(x);")
    )
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    h = torch.zeros(3, 16)
    out = block(h)
    assert torch.allclose(out[:, abi["y"]], torch.tensor([6.0, 6.0, 6.0]))
    assert torch.allclose(out[:, abi["z"]], torch.tensor([12.0, 12.0, 12.0]))
    assert torch.allclose(out[:, abi["w"]], torch.tensor([2.0, 2.0, 2.0]))


def test_loop_snapshots_preserve_vector_column_width():
    """``snapshot_env`` stacks (B,K) state when ``abi_widths`` names a width > 1."""
    reset_parser()
    ir = ast_to_ir(
        parse_ax(
            "x = [1.0, 2.0];\n"
            "while (i > 0) {\n"
            "  x = x * 2.0;\n"
            "  i = i - 1;\n"
            "}\n"
        )
    )
    loop = next(s for s in ir if s[0] == "OP_LOOP")
    cond_ir, body_ir = loop[1], loop[2]
    prelude = [ir[0]]
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    seed = make_seed_map(cond_ir, body_ir, 16)
    h = torch.zeros(2, 16)
    h[:, abi["i"]] = torch.tensor([2.0, 1.0])
    seq, m = run_loop_snapshots(
        h,
        cond_ir,
        body_ir,
        dim=16,
        max_unroll=3,
        seed_map=seed,
        prelude_stmts=prelude,
        abi_widths=aw,
    )
    assert seq.shape[0] == 2 and seq.shape[1] == 3
    assert seq.shape[2] >= 16
