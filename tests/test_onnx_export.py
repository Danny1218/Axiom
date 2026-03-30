"""ONNX export for InterpretedBlock ``.axb`` bundles (optional ``[export]`` extra)."""

from pathlib import Path

import pytest
import torch

from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.compiler.serializer import save_bundle
from axiom.engine.block_executor import InterpretedBlock
from axiom.export.onnx_export import OnnxExportError, export_bundle_to_onnx, export_interpreted_block_to_onnx

pytest.importorskip("onnx")


def _tiny_block() -> InterpretedBlock:
    reset_parser()
    ir = ast_to_ir(parse_ax("y = neural([1.0, 2.0]);"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    return InterpretedBlock(ir, abi, abi_widths=aw)


def test_export_bundle_writes_onnx_and_matches_torch(tmp_path: Path):
    block = _tiny_block()
    axb = tmp_path / "m.axb"
    onnx_path = tmp_path / "m.onnx"
    save_bundle(block, axb)
    export_bundle_to_onnx(axb, onnx_path)
    assert onnx_path.is_file()
    assert onnx_path.stat().st_size > 0

    block.eval()
    dim = max(
        (block.abi[n] + max(1, int(block.abi_widths.get(n, 1))) for n in block.abi),
        default=16,
    )
    h = torch.randn(3, dim, dtype=torch.float32)
    with torch.no_grad():
        want = block(h)

    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    got = torch.tensor(sess.run(None, {name: h.numpy()})[0])
    assert got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


def test_export_interpreted_block_direct(tmp_path: Path):
    block = _tiny_block()
    p = tmp_path / "direct.onnx"
    export_interpreted_block_to_onnx(block, p)
    assert p.is_file()


def test_empty_abi_rejected():
    block = InterpretedBlock([], {}, abi_widths={})
    with pytest.raises(OnnxExportError, match="empty ABI"):
        export_interpreted_block_to_onnx(block, "nope.onnx")


def test_cli_export_onnx_runs(tmp_path: Path):
    from axiom.cli import main

    block = _tiny_block()
    axb = tmp_path / "c.axb"
    out = tmp_path / "c.onnx"
    save_bundle(block, axb)
    main(["export-onnx", "--bundle", str(axb), "--output", str(out)])
    assert out.is_file()
