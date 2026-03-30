"""Phase 38: root ``axiom.load`` / ``AxiomModel.predict`` (dict, batch, optional DataFrame)."""

from pathlib import Path

import pytest
import torch

import axiom
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.compiler.serializer import save_bundle
from axiom.engine.block_executor import InterpretedBlock


def _simple_double_block() -> InterpretedBlock:
    reset_parser()
    ir = ast_to_ir(parse_ax("y = x * 2.0;"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    return InterpretedBlock(ir, abi, abi_widths=aw)


def test_axiom_root_exports():
    assert hasattr(axiom, "load")
    assert hasattr(axiom, "AxiomModel")
    assert "load" in axiom.__all__
    assert "AxiomModel" in axiom.__all__


def test_predict_single_dict(tmp_path: Path):
    block = _simple_double_block()
    axb = tmp_path / "m.axb"
    save_bundle(block, axb)
    model = axiom.load(axb)
    res = model.predict({"x": 2.0})
    assert res["y"] == pytest.approx(4.0)


def test_predict_batch_list(tmp_path: Path):
    block = _simple_double_block()
    axb = tmp_path / "m.axb"
    save_bundle(block, axb)
    model = axiom.load(axb)
    res_batch = model.predict([{"x": 2.0}, {"x": 3.0}])
    assert res_batch[0]["y"] == pytest.approx(4.0)
    assert res_batch[1]["y"] == pytest.approx(6.0)


def test_predict_empty_list_errors(tmp_path: Path):
    block = _simple_double_block()
    axb = tmp_path / "m.axb"
    save_bundle(block, axb)
    model = axiom.load(axb)
    with pytest.raises(ValueError, match="non-empty"):
        model.predict([])


def test_predict_matches_direct_block_forward(tmp_path: Path):
    block = _simple_double_block()
    axb = tmp_path / "m.axb"
    save_bundle(block, axb)
    model = axiom.load(axb)
    dim = 16
    h = torch.zeros(1, dim)
    h[0, block.abi["x"]] = 1.5
    with torch.no_grad():
        out = block(h)
    want = float(out[0, block.abi["y"]].item())
    assert model.predict({"x": 1.5})["y"] == pytest.approx(want)


def test_predict_dataframe_when_pandas_installed(tmp_path: Path):
    pd = pytest.importorskip("pandas")
    block = _simple_double_block()
    axb = tmp_path / "m.axb"
    save_bundle(block, axb)
    model = axiom.load(axb)
    df = pd.DataFrame([{"x": 2.0}, {"x": 3.0}])
    res = model.predict(df)
    assert res[1]["y"] == pytest.approx(6.0)


def test_predict_rejects_non_dict_row_in_batch(tmp_path: Path):
    block = _simple_double_block()
    axb = tmp_path / "m.axb"
    save_bundle(block, axb)
    model = axiom.load(axb)
    with pytest.raises(TypeError, match="dict"):
        model.predict([{"x": 1.0}, "bad"])


def test_predict_rejects_invalid_type(tmp_path: Path):
    block = _simple_double_block()
    axb = tmp_path / "m.axb"
    save_bundle(block, axb)
    model = axiom.load(axb)
    with pytest.raises(TypeError):
        model.predict("not-a-dict")
