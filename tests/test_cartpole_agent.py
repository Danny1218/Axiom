"""Phase 45: neuro-symbolic CartPole brain (IR, safety rail, optional Gymnasium smoke)."""

from pathlib import Path

import pytest
import torch

from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.inference import _inputs_to_tensor


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    g = 0.0
    out: list[float] = []
    for r in reversed(rewards):
        g = float(r) + gamma * g
        out.append(g)
    return out[::-1]


def test_discounted_returns_matches_gamma():
    assert _discounted_returns([1.0, 1.0], 0.9) == pytest.approx(
        [1.0 + 0.9 * 1.0, 1.0], rel=1e-5
    )


def test_cartpole_ax_compiles_and_abi():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "cartpole.ax"))
    flat = str(ir)
    assert "OP_NEURAL" in flat
    assert "OP_CONDITIONAL" in flat or "OP_MATH_UNARY" in flat
    abi = extract_global_abi(ir, max_vars=64)
    aw = extract_abi_widths(ir, max_vars=64)
    assert "prob_right" in abi
    assert "pole_angle" in abi
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    dim = max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)
    assert dim >= 4


def test_safety_rail_pushes_prob_right_when_pole_tilts_right():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "cartpole.ax"))
    abi = extract_global_abi(ir, max_vars=64)
    aw = extract_abi_widths(ir, max_vars=64)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    dim = max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)
    col = int(b.abi["prob_right"])
    neutral = {
        "cart_pos": 0.0,
        "cart_vel": 0.0,
        "pole_angle": 0.0,
        "pole_vel": 0.0,
    }
    danger_r = {**neutral, "pole_angle": 0.2}
    h0 = _inputs_to_tensor(
        neutral, b.abi, dim, device=torch.device("cpu"), dtype=torch.float32, abi_widths=aw
    )
    h1 = _inputs_to_tensor(
        danger_r, b.abi, dim, device=torch.device("cpu"), dtype=torch.float32, abi_widths=aw
    )
    with torch.no_grad():
        p0 = b(h0)[0, col].item()
        p1 = b(h1)[0, col].item()
    assert p1 > p0
    assert p1 > 0.9


def test_safety_rail_pushes_prob_left_when_pole_tilts_left():
    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "cartpole.ax"))
    abi = extract_global_abi(ir, max_vars=64)
    aw = extract_abi_widths(ir, max_vars=64)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    dim = max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)
    col = int(b.abi["prob_right"])
    danger_l = {
        "cart_pos": 0.0,
        "cart_vel": 0.0,
        "pole_angle": -0.2,
        "pole_vel": 0.0,
    }
    h = _inputs_to_tensor(
        danger_l, b.abi, dim, device=torch.device("cpu"), dtype=torch.float32, abi_widths=aw
    )
    with torch.no_grad():
        p = b(h)[0, col].item()
    assert p < 0.1


def test_gymnasium_cartpole_one_episode_runs_with_block():
    pytest.importorskip("gymnasium")
    import gymnasium as gym

    reset_parser()
    ir = ast_to_ir(parse_ax_file(_root() / "examples" / "cartpole.ax"))
    abi = extract_global_abi(ir, max_vars=64)
    aw = extract_abi_widths(ir, max_vars=64)
    b = InterpretedBlock(ir, abi, abi_widths=aw)
    dim = max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)
    col = int(b.abi["prob_right"])
    env = gym.make("CartPole-v1")
    state, _ = env.reset()
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 50:
        data = {k: float(state[i]) for i, k in enumerate(["cart_pos", "cart_vel", "pole_angle", "pole_vel"])}
        h = _inputs_to_tensor(
            data, b.abi, dim, device=torch.device("cpu"), dtype=torch.float32, abi_widths=aw
        )
        out = b(h)
        prob = out[0, col].clamp(1e-6, 1.0 - 1e-6)
        assert 0.0 < float(prob) < 1.0
        action = int(torch.bernoulli(prob).item())
        state, _, terminated, truncated, _ = env.step(action)
        steps += 1
    env.close()
    assert steps >= 1
