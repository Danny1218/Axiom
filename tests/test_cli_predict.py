"""``axiom predict`` for InterpretedBlock .axb bundles."""

import json
from pathlib import Path

import torch

from axiom.cli import main
from axiom.compiler.deserializer import load_bundle
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax, reset_parser
from axiom.compiler.serializer import save_bundle
from axiom.engine.block_executor import InterpretedBlock


def test_cli_predict_help_exits_ok():
    import pytest

    with pytest.raises(SystemExit) as exc:
        main(["predict", "--help"])
    assert exc.value.code == 0


def test_cli_predict_runs_on_saved_bundle(tmp_path: Path, capsys):
    reset_parser()
    ir = ast_to_ir(parse_ax("y = neural([1.0, 2.0]);"))
    abi = extract_global_abi(ir, max_vars=16)
    aw = extract_abi_widths(ir, max_vars=16)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    axb = tmp_path / "p.axb"
    save_bundle(block, axb)

    main(
        [
            "predict",
            "--bundle",
            str(axb),
            "--input",
            "{}",
        ]
    )
    out = capsys.readouterr().out
    data = json.loads(out)
    assert "y" in data
    b2 = load_bundle(axb)
    h = torch.zeros(1, 16)
    with torch.no_grad():
        want = float(b2(h)[0, abi["y"]].item())
    assert abs(float(data["y"]) - want) < 1e-5
