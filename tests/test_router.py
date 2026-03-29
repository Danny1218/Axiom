import pytest
import torch

from engine.router import SinkhornRouter, sinkhorn_balance


def test_sinkhorn_balance_row_col_sums():
    torch.manual_seed(0)
    B, E = 5, 3
    K = torch.rand(B, E, dtype=torch.double) + 0.1
    row_target = torch.ones(B, dtype=torch.double)
    col_target = torch.full((E,), float(B) / E, dtype=torch.double)
    P = sinkhorn_balance(K, row_target=row_target, col_target=col_target, num_iters=32)
    assert torch.allclose(P.sum(dim=1), row_target, atol=1e-5, rtol=1e-5)
    assert torch.allclose(P.sum(dim=0), col_target, atol=1e-5, rtol=1e-5)


def test_sinkhorn_router_full_mask_balanced_columns():
    torch.manual_seed(1)
    r = SinkhornRouter(6, 4, num_iters=48, epsilon=0.2)
    x = torch.randn(7, 6)
    w = r(x)
    assert w.shape == (7, 4)
    assert torch.allclose(w.sum(dim=1), torch.ones(7), atol=2e-2, rtol=2e-2)
    expect_col = torch.full((4,), 7 / 4.0)
    assert torch.allclose(w.sum(dim=0), expect_col, atol=2e-2, rtol=2e-2)


def test_sinkhorn_router_masked_subset():
    torch.manual_seed(2)
    r = SinkhornRouter(4, 5, num_iters=20, epsilon=0.15)
    x = torch.randn(8, 4)
    mask = torch.tensor([1, 0, 1, 0, 0], dtype=torch.bool)
    w = r(x, expert_mask=mask)
    assert w.shape == (8, 5)
    assert (w[:, [1, 3, 4]] == 0).all()
    assert torch.allclose(w.sum(dim=1), torch.ones(8), atol=1e-4, rtol=1e-4)
    active = w[:, [0, 2]].sum(dim=0)
    assert torch.allclose(active, torch.tensor([4.0, 4.0]), atol=2e-3, rtol=2e-3)


def test_router_grad_flow():
    torch.manual_seed(3)
    r = SinkhornRouter(5, 3, num_iters=6, epsilon=0.3)
    x = torch.randn(4, 5, requires_grad=True)
    w = r(x)
    (w.sum() * x.mean()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert any(p.grad is not None for p in r.parameters())


def test_router_empty_mask_returns_zeros():
    r = SinkhornRouter(3, 2)
    x = torch.randn(2, 3)
    mask = torch.zeros(2, dtype=torch.bool)
    w = r(x, expert_mask=mask)
    assert w.shape == (2, 2) and (w == 0).all()
