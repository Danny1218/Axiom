import torch

from engine.dataloader import LiquidSequenceLoader, sequential_to_features


def test_sequential_to_features_shape():
    s = torch.linspace(0, 1, 10)
    f = sequential_to_features(s, 5)
    assert f.shape == (10, 5)


def test_liquid_loader_noise_and_target_match():
    torch.manual_seed(0)
    seq = torch.randn(40)
    L = LiquidSequenceLoader(seq, feature_dim=6, batch_size=11, baseline_var=0.1, shuffle=False)
    batches = list(L)
    assert len(batches) == 4
    x, y = batches[0]
    assert x.shape == (11, 6) and y.shape == (11, 6)
    assert not torch.allclose(x, y)


def test_liquid_loader_len():
    L = LiquidSequenceLoader(torch.arange(25.0), 3, batch_size=10, shuffle=False)
    assert len(L) == 3
