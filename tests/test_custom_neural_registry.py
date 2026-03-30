"""Phase 40 Part B: optional ``custom_neural_registry`` on ``InterpretedBlock``."""

import torch
import torch.nn as nn

from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi, extract_neural_node_specs
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.engine.block_executor import InterpretedBlock


def test_custom_registry_replaces_default_mlp():
    reset_parser()
    ir = ast_to_ir(parse_ax("y = neural([1.0]);"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    spec = extract_neural_node_specs(ir, aw)
    nid = next(iter(spec.keys()))
    tiny = nn.Sequential(nn.Linear(1, 1), nn.Identity())
    b = InterpretedBlock(ir, abi, abi_widths=aw, custom_neural_registry={nid: tiny})
    assert b.neural_registry[nid] is tiny


def test_forward_with_custom_registry_runs():
    reset_parser()
    ir = ast_to_ir(parse_ax("y = neural([1.0]);"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    spec = extract_neural_node_specs(ir, aw)
    nid = next(iter(spec.keys()))
    custom = nn.Sequential(nn.Linear(1, 1))
    b = InterpretedBlock(ir, abi, abi_widths=aw, custom_neural_registry={nid: custom})
    h = torch.zeros(2, 16)
    with torch.no_grad():
        out = b(h)
    assert out.shape == (2, 16)
