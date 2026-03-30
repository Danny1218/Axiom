"""Phase 14: global feature ABI — trunk columns match IR first-seen order, not alphabetical keys."""

import torch

from compiler.flow import wire_execution_graph
from compiler.ir import ast_to_ir, extract_global_abi
from compiler.parser import parse_ax, reset_parser
from engine.inference import AxiomRunner, _abi_rows_to_tensor, _batch_inputs_to_tensor
from engine.loop_executor import InterpretedLiquidLoop
from engine.supernet import LatentSupernet


def test_extract_global_abi_loop_z_before_a():
    reset_parser()
    ir = ast_to_ir(parse_ax("while (z > 0) { a = a + 1; }"))
    abi = extract_global_abi(ir, max_vars=8)
    assert abi["z"] == 0 and abi["a"] == 1


def test_predict_tensor_aligns_with_loop_seed_map_not_alphabetical_order():
    reset_parser()
    ir = ast_to_ir(parse_ax("while (z > 0) { a = a + 1; }"))
    dim = 6
    sn = LatentSupernet(dim, ("latent_0", "latent_1"), rank=2)
    g = wire_execution_graph(ir, sn, [], loop_max_unroll=4, loop_num_basis=4)
    assert g.abi["z"] == 0 and g.abi["a"] == 1

    loop = next(m for m in g.node_modules.values() if isinstance(m, InterpretedLiquidLoop))
    assert loop.seed_map["z"] == 0 and loop.seed_map["a"] == 1

    dev = torch.device("cpu")
    t = _batch_inputs_to_tensor(
        [{"a": 1.0, "z": 5.0}],
        g.abi,
        dim,
        device=dev,
        dtype=torch.float32,
    )
    assert t[0, 0] == 5.0 and t[0, 1] == 1.0

    runner = AxiomRunner(g)
    out = runner.predict({"a": 1.0, "z": 5.0})
    assert out.shape == (1, dim)
    assert torch.isfinite(out).all()


def test_abi_ignores_unknown_input_keys():
    dev = torch.device("cpu")
    abi = {"z": 0, "a": 1}
    t = _abi_rows_to_tensor([{"z": 2.0, "noise": 99.0, "a": 3.0}], abi, 4, device=dev, dtype=torch.float32)
    assert t[0, 0] == 2.0 and t[0, 1] == 3.0


def test_abi_defaults_missing_names_to_zero():
    dev = torch.device("cpu")
    abi = {"z": 0, "a": 1}
    t = _abi_rows_to_tensor([{"z": 7.0}], abi, 4, device=dev, dtype=torch.float32)
    assert t[0, 0] == 7.0 and t[0, 1] == 0.0
