import torch
import torch.nn as nn

from engine.meta_compiler import MetaCompiler
from engine.router import SinkhornRouter
from engine.supernet import LatentSupernet


def test_meta_unmasks_on_signal():
    sn = LatentSupernet(4, ("a", "b", "c"), rank=2)
    sn.set_masks({"a": 1.0, "b": 1.0})
    r = SinkhornRouter(4, 2, mutation_entropy_norm_threshold=0.5)
    nn.init.zeros_(r.proj.weight)
    nn.init.zeros_(r.proj.bias)
    x = torch.randn(6, 4)
    r(x)
    assert r.last_mutation_signal.triggered
    mc = MetaCompiler(sn)
    names = mc.react_to_router_signals([r], max_unmasks=1)
    assert names == ["c"]
    assert sn.adapter_mask[2] >= 0.5
    assert sn.is_shadow[2].item() is True


def test_meta_respects_max_unmasks():
    sn = LatentSupernet(3, ("x", "y", "z"), rank=2)
    sn.set_masks({"x": 1.0})
    r1 = SinkhornRouter(3, 2, mutation_entropy_norm_threshold=0.3)
    r2 = SinkhornRouter(3, 2, mutation_entropy_norm_threshold=0.3)
    for r in (r1, r2):
        nn.init.zeros_(r.proj.weight)
        nn.init.zeros_(r.proj.bias)
        r(torch.randn(4, 3))
    mc = MetaCompiler(sn)
    out = mc.react_to_router_signals([r1, r2], max_unmasks=1)
    assert len(out) == 1
