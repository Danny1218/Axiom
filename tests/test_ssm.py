import pytest
import torch

from engine.ssm import LiquidKANNode, _hat_basis
from primitives.liquid_tensor import LiquidStateTensor


def test_hat_basis_simple():
    u = torch.tensor([[0.5]])
    b = _hat_basis(u, 5)
    assert b.shape == (1, 5) and (b.sum(dim=-1) > 0).all()


def test_liquid_kan_forward_shape_and_grad():
    torch.manual_seed(0)
    m = LiquidKANNode(6, num_basis=6, max_unroll=4)
    x = torch.randn(3, 6, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None and m.coeffs.grad is not None


def test_liquid_kan_forward_sequence():
    torch.manual_seed(1)
    d = 5
    node = LiquidKANNode(d, num_basis=5, max_unroll=3)
    s0 = LiquidStateTensor(d)
    s1 = LiquidStateTensor(d)
    s2 = LiquidStateTensor(d)
    with torch.no_grad():
        s0.data.normal_(0, 0.1)
        s1.data.normal_(0, 0.1)
        s2.data.normal_(0, 0.1)
    out = node.forward_sequence([s0, s1, s2])
    assert out.shape == (d,)
    out.sum().backward()
    assert s0.data.grad is not None


def test_liquid_kan_forward_sequence_tensors_batched():
    torch.manual_seed(2)
    d = 4
    node = LiquidKANNode(d, num_basis=4, max_unroll=2)
    seq = torch.randn(3, 2, d, requires_grad=True)
    h0 = torch.randn(3, d, requires_grad=True)
    out = node.forward_sequence_tensors(seq, h_init=h0)
    assert out.shape == (3, d)
    out.sum().backward()
    assert seq.grad is not None and h0.grad is not None


def test_forward_sequence_tensors_mask_all_true_matches_no_mask():
    torch.manual_seed(3)
    d = 3
    node = LiquidKANNode(d, num_basis=3, max_unroll=4)
    B, T = 2, 3
    seq = torch.randn(B, T, d)
    h0 = torch.randn(B, d)
    m = torch.ones(B, T, dtype=torch.bool)
    o0 = node.forward_sequence_tensors(seq, h_init=h0)
    o1 = node.forward_sequence_tensors(seq, h_init=h0, mask=m)
    assert torch.allclose(o0, o1)


def test_forward_sequence_tensors_mask_bad_shape_raises():
    node = LiquidKANNode(2, num_basis=2, max_unroll=2)
    seq = torch.zeros(2, 3, 2)
    with pytest.raises(ValueError, match="mask shape"):
        node.forward_sequence_tensors(seq, mask=torch.zeros(2, 2, dtype=torch.bool))


def test_forward_sequence_tensors_mask_freezes_phantom_steps():
    """Row 0 mask False at t>=1: output matches integrating only t=0 then holding."""
    torch.manual_seed(4)
    d = 2
    node = LiquidKANNode(d, num_basis=3, max_unroll=5)
    seq = torch.randn(1, 4, d)
    h0 = torch.randn(1, d)
    mask = torch.tensor([[True, False, False, False]])
    out_masked = node.forward_sequence_tensors(seq, h_init=h0, mask=mask)
    seq1 = seq[:, :1, :]
    out_one = node.forward_sequence_tensors(seq1, h_init=h0)
    assert torch.allclose(out_masked, out_one, atol=1e-6, rtol=1e-5)
