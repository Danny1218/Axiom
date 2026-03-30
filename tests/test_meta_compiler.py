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
    _, ent = r(x)
    assert float(ent.item()) >= 0.5
    mc = MetaCompiler(sn)
    names = mc.react_to_signals(
        {"cond_0": ent}, sn, max_unmasks=1, block_thresholds={"cond_0": 0.5}
    )
    assert names == ["c"]
    assert sn.adapter_mask[2] >= 0.5
    assert sn.is_shadow[2].item() is True


def test_meta_respects_max_unmasks():
    sn = LatentSupernet(3, ("x", "y", "z"), rank=2)
    sn.set_masks({"x": 1.0})
    r1 = SinkhornRouter(3, 2, mutation_entropy_norm_threshold=0.3)
    r2 = SinkhornRouter(3, 2, mutation_entropy_norm_threshold=0.3)
    sigs = {}
    for key, r in (("cond_0", r1), ("cond_1", r2)):
        nn.init.zeros_(r.proj.weight)
        nn.init.zeros_(r.proj.bias)
        _, ent = r(torch.randn(4, 3))
        sigs[key] = ent
    mc = MetaCompiler(sn)
    thr = {"cond_0": 0.3, "cond_1": 0.3}
    out = mc.react_to_signals(sigs, sn, max_unmasks=1, block_thresholds=thr)
    assert len(out) == 1
