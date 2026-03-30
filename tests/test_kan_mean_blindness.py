"""Phase 22: RBF splines must not collapse (B, D) latents via mean-pooling before basis evaluation."""

import torch

from axiom.engine.ssm import LiquidKANNode, _rbf_basis


def _mean_pooled_rbf_phi(fused_norm: torch.Tensor, num_basis: int) -> torch.Tensor:
    """Reference rank-1 path: same mean for [1,-1] and [-1,1] → identical scalar coordinate."""
    u = torch.sigmoid(fused_norm.mean(dim=-1, keepdim=True))
    x = u.squeeze(-1).clamp(0.0, 1.0)
    if num_basis < 2:
        return torch.ones(x.shape[0], 1, device=fused_norm.device, dtype=fused_norm.dtype)
    centers = torch.linspace(0.0, 1.0, num_basis, device=fused_norm.device, dtype=fused_norm.dtype).view(
        1, -1
    )
    width = 1.0 / max(num_basis - 1, 1)
    diff = (x.unsqueeze(-1) - centers) / width
    return torch.exp(-(diff**2))


def test_rbf_basis_vectorized_not_mean_blind():
    """Opposite vectors with zero mean must produce different (B, D, K) basis activations."""
    x1 = torch.tensor([[1.0, -1.0]])
    x2 = torch.tensor([[-1.0, 1.0]])
    assert torch.allclose(x1.mean(), x2.mean())
    k = 5
    b1 = _rbf_basis(x1, k)
    b2 = _rbf_basis(x2, k)
    assert b1.shape == (1, 2, k) and b2.shape == (1, 2, k)
    assert not torch.allclose(b1, b2), "per-channel RBF should distinguish permuted latents"


def test_mean_pooled_reference_collapses_opposites():
    """Sanity: old scalar pooling makes [1,-1] and [-1,1] identical (documents the bug)."""
    x1 = torch.tensor([[1.0, -1.0]])
    x2 = torch.tensor([[-1.0, 1.0]])
    k = 5
    p1 = _mean_pooled_rbf_phi(x1, k)
    p2 = _mean_pooled_rbf_phi(x2, k)
    assert p1.shape == (1, k) and torch.allclose(p1, p2)


def test_liquid_kan_with_identity_fuse_separates_permuted_h():
    """fuse_proj copies h from cat([h, 0]); permuted h with same mean → different forward if not blind."""
    torch.manual_seed(0)
    d = 2
    kan = LiquidKANNode(d, num_basis=4, max_unroll=2)
    with torch.no_grad():
        kan.fuse_proj.weight.zero_()
        kan.fuse_proj.weight[0, 0] = 1.0
        kan.fuse_proj.weight[1, 1] = 1.0
        kan.w_gate.weight.zero_()
        kan.w_gate.bias.fill_(8.0)
    h1 = torch.tensor([[1.0, -1.0]])
    h2 = torch.tensor([[-1.0, 1.0]])
    y1 = kan(h1)
    y2 = kan(h2)
    assert y1.shape == (1, 2) and y2.shape == (1, 2)
    assert not torch.allclose(y1, y2)
