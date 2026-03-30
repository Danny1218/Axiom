"""Phase 17: target column must not appear in model inputs (AxiomDataset blinding)."""

from axiom.engine.dataloader import AxiomDataset


def test_axiom_dataset_blinds_target_abi_column():
    abi = {"a": 0, "price": 1}
    data = [{"a": 5.0, "price": 100.0}]
    ds = AxiomDataset(data, abi, trunk_dim=4, target_key="price")
    x, y = ds[0]
    assert x[1].item() == 0.0
    assert y.item() == 100.0
    assert x[0].item() == 5.0


def test_target_key_not_in_abi_does_not_blind_column():
    abi = {"a": 0, "b": 1}
    ds = AxiomDataset(
        [{"a": 1.0, "b": 2.0, "label": 3.0}],
        abi,
        trunk_dim=4,
        target_key="label",
    )
    x, y = ds[0]
    assert x[0].item() == 1.0 and x[1].item() == 2.0
    assert y.item() == 3.0
