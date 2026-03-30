"""Phase 16: supervise only the ABI target column — avoid latent space collapse from full-trunk MSE."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def test_column_sliced_mse_zero_when_only_that_column_matches():
    out = torch.zeros(2, 4)
    out[:, 1] = torch.tensor([5.0, 10.0])
    out[:, 2] = 99.0
    y = torch.tensor([[5.0], [10.0]])
    crit = nn.MSELoss()
    col_loss = crit(out[:, 1:2], y.view(-1, 1))
    assert col_loss.item() == 0.0


def test_full_trunk_mse_penalizes_free_latent_columns():
    """Broadcasting scalar targets across all columns forces every channel toward the same value."""
    out = torch.zeros(2, 4)
    out[:, 1] = torch.tensor([5.0, 10.0])
    out[:, 2] = 99.0
    y = torch.tensor([[5.0], [10.0]])
    crit = nn.MSELoss()
    y_broadcast = y.expand(-1, 4)
    full_loss = crit(out, y_broadcast)
    assert full_loss.item() > 100.0


def test_f_mse_loss_shadow_slice_same_as_main():
    """Shadow localized loss uses the same (B,1) slice as the main objective."""
    loc = torch.zeros(3, 5)
    loc[:, 2] = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([[1.0], [2.0], [3.0]])
    sl = F.mse_loss(loc[:, 2:3], y.view(-1, 1))
    assert sl.shape == ()
    assert sl.item() == 0.0
