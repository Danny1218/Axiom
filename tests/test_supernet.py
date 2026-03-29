import pytest
import torch

from engine.supernet import LatentSupernet, TTLoRAAdapter


def test_trunk_frozen():
    m = LatentSupernet(8, ("e0", "e1"))
    assert all(not p.requires_grad for p in m.trunk.parameters())


def test_adapters_trainable():
    m = LatentSupernet(8, ("e0", "e1"))
    assert any(p.requires_grad for p in m.adapters.parameters())


def test_mask_zero_skips_adapter():
    torch.manual_seed(0)
    m = LatentSupernet(4, ("a", "b"), rank=2)
    x = torch.randn(2, 4)
    base = m.trunk(x).clone()
    m.set_adapter_mask("a", 0.0)
    m.set_adapter_mask("b", 0.0)
    y = m(x)
    assert torch.allclose(y, base)


def test_mask_one_applies_adapter():
    torch.manual_seed(1)
    m = LatentSupernet(4, ("a",), rank=2)
    x = torch.randn(3, 4)
    m.set_adapter_mask("a", 0.0)
    y0 = m(x)
    m.set_adapter_mask("a", 1.0)
    y1 = m(x)
    assert not torch.allclose(y0, y1)


def test_tt_adapter_shape():
    a = TTLoRAAdapter(6, rank=3)
    x = torch.randn(4, 6)
    assert a(x).shape == x.shape


def test_set_masks_dict():
    m = LatentSupernet(3, ("p", "q"))
    m.set_masks({"p": 1.0, "q": 0.0})
    assert m.adapter_mask[0] == 1.0 and m.adapter_mask[1] == 0.0


def test_custom_trunk_respected():
    trunk = torch.nn.Linear(5, 5, bias=False)
    m = LatentSupernet(5, ("z",), trunk=trunk)
    assert m.trunk is trunk
