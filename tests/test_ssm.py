import pytest
import torch

from engine.ssm import LiquidKANNode, _rbf_basis
from primitives.liquid_tensor import LiquidStateTensor


def test_rbf_basis_simple():
    fused = torch.zeros(1, 5)
    b = _rbf_basis(fused, 5)
    assert b.shape == (1, 5) and torch.isfinite(b).all() and (b > 0).all()


def test_rbf_basis_num_basis_one():
    fused = torch.randn(2, 3)
    b = _rbf_basis(fused, 1)
    assert b.shape == (2, 1) and (b == 1).all()


def test_liquid_kan_forward_shape_and_grad():
    torch.manual_seed(0)
    m = LiquidKANNode(6, num_basis=6, max_unroll=4)
    x = torch.randn(3, 6, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None and m.coeffs.grad is not None
    assert m.fuse_proj.weight.grad is not None and m.w_gate.weight.grad is not None


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


def test_sequence_input_changes_output_vs_dummy_forward():
    """x_t is fused into KAN: same h_init with different seq last step should differ from zero-input recurrence."""
    torch.manual_seed(5)
    d = 4
    node = LiquidKANNode(d, num_basis=4, max_unroll=3)
    B, T = 2, 3
    h0 = torch.randn(B, d)
    seq_a = torch.zeros(B, T, d)
    seq_b = seq_a.clone()
    seq_b[:, -1, :] = torch.randn(B, d) * 3.0
    out_a = node.forward_sequence_tensors(seq_a, h_init=h0)
    out_b = node.forward_sequence_tensors(seq_b, h_init=h0)
    assert not torch.allclose(out_a, out_b)


def test_kan_update_respects_x_t_gradient():
    torch.manual_seed(6)
    d = 3
    node = LiquidKANNode(d, num_basis=3, max_unroll=2)
    h = torch.randn(2, d, requires_grad=True)
    x = torch.randn(2, d, requires_grad=True)
    h0 = torch.randn(2, d, requires_grad=True)
    tn = torch.zeros(2, 1)
    prop = node._kan_update(h, x, h0, tn)
    prop.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
