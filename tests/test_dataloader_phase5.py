import torch

from engine.dataloader import AxiomDataset, LiquidSequenceLoader, sequential_to_features


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


def test_axiom_dataset_abi_maps_columns_and_missing_zero():
    abi = {"a": 1, "b": 0, "x": 2}
    ds = AxiomDataset(
        [{"a": 3.0, "b": 4.0, "target": 7.0}],
        abi,
        trunk_dim=5,
        target_key="target",
    )
    assert len(ds) == 1
    x, y = ds[0]
    assert x.shape == (5,) and y.shape == (1,)
    assert x[1].item() == 3.0 and x[0].item() == 4.0
    assert x[2].item() == 0.0
    assert y.item() == 7.0


def test_axiom_dataset_skips_out_of_range_abi_columns():
    ds = AxiomDataset(
        [{"k": 1.0, "target": 0.0}],
        {"k": 99},
        trunk_dim=4,
        target_key="target",
    )
    x, _ = ds[0]
    assert x.abs().sum().item() == 0.0


def test_axiom_dataset_broadcast_target_shape():
    ds = AxiomDataset(
        [{"a": 1.0, "target": 5.0}],
        {"a": 0},
        trunk_dim=4,
        target_key="target",
        broadcast_target=True,
    )
    _, y = ds[0]
    assert y.shape == (4,) and (y == 5.0).all()
