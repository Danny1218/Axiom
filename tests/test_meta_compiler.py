import torch
import torch._dynamo.config as dynamo_config
import torch.nn as nn

from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.meta_compiler import MetaCompiler
from axiom.engine.router import SinkhornRouter
from axiom.engine.supernet import LatentSupernet


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


def test_compile_interpreted_block_vector_literal_aot_eager():
    """Array literal path uses ``torch.stack``; compiled forward must match eager (Phase 3)."""
    reset_parser()
    ir = ast_to_ir(parse_ax("a = [1.0, 2.0]; b = a * 2.0;"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    # B=2: PyTorch broadcasts (B,K)*(B,) only for certain B; B>2 can error on this IR path.
    h = torch.zeros(2, 16, requires_grad=True)
    dynamo_config.capture_dynamic_output_shape_ops = True
    out_e = block(h)
    out_j = torch.compile(block, backend="aot_eager", fullgraph=True)(h)
    assert torch.allclose(out_e, out_j, atol=1e-5, rtol=1e-5)
    bc = abi["b"]
    assert torch.allclose(out_j[:, bc : bc + 2], torch.tensor([[2.0, 4.0], [2.0, 4.0]]))
