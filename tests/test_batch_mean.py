"""Phase 42: ``batch_mean`` / ``OP_REDUCE_BATCH_MEAN`` (cross-sectional mean, dim=0)."""

import pytest
import torch

from axiom.compiler.ir import _infer_expr_output_width, ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.engine.block_executor import InterpretedBlock


def test_batch_mean_in_ir_and_width_preserved():
    reset_parser()
    ir = ast_to_ir(parse_ax("a = batch_mean(x);"))
    expr = list(ir[0][2])
    assert expr[-1] == ("OP_REDUCE_BATCH_MEAN",)
    w = _infer_expr_output_width(expr, {"x": 1})
    assert w == 1


def test_batch_mean_cross_section_sum_neutral():
    reset_parser()
    ir = ast_to_ir(
        parse_ax(
            "features = [momentum, volatility];\n"
            "raw_alpha = neural(features);\n"
            "market_neutral_alpha = raw_alpha - batch_mean(raw_alpha);\n"
        )
    )
    abi = extract_global_abi(ir, max_vars=64)
    aw = extract_abi_widths(ir, max_vars=64)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    B = 50
    D = max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)
    h = torch.zeros(B, D)
    gen = torch.Generator()
    gen.manual_seed(0)
    h[:, abi["momentum"]] = torch.randn(B, generator=gen)
    h[:, abi["volatility"]] = torch.rand(B, generator=gen) * 0.04 + 0.01
    with torch.no_grad():
        out = b(h)
    col = abi["market_neutral_alpha"]
    mna = out[:, col]
    assert float(mna.sum().item()) == pytest.approx(0.0, abs=1e-4)


def test_batch_mean_assign_broadcasts_constant_across_batch():
    reset_parser()
    ir = ast_to_ir(parse_ax("a = batch_mean(x);"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    h = torch.zeros(8, 16)
    h[:, abi["x"]] = torch.arange(8, dtype=torch.float32)
    with torch.no_grad():
        out = b(h)
    want = float(torch.arange(8, dtype=torch.float32).mean().item())
    assert torch.allclose(out[:, abi["a"]], torch.full((8,), want))
