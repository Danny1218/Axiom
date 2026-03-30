"""
Neuro-symbolic CartPole-v1 with REINFORCE (Phase 45).

Requires: pip install gymnasium

Run from repo root: python examples/train_cartpole.py
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import torch
from torch.distributions import Bernoulli

import axiom
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.compiler.serializer import save_bundle
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.inference import _inputs_to_tensor

_EXAMPLES = Path(__file__).resolve().parent
AX_PATH = _EXAMPLES / "cartpole.ax"
BUNDLE_PATH = _EXAMPLES / "cartpole_trained.axb"


def _trunk_dim(block: InterpretedBlock) -> int:
    abi, aw = block.abi, getattr(block, "abi_widths", {}) or {}
    return max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    g = 0.0
    out: list[float] = []
    for r in reversed(rewards):
        g = float(r) + gamma * g
        out.append(g)
    return out[::-1]


def train() -> bool:
    reset_parser()
    ir = ast_to_ir(parse_ax_file(AX_PATH))
    abi = extract_global_abi(ir, max_vars=64)
    aw = extract_abi_widths(ir, max_vars=64)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    dim = _trunk_dim(block)
    device = torch.device("cpu")
    dtype = torch.float32

    optimizer = torch.optim.Adam(block.parameters(), lr=0.01)
    env = gym.make("CartPole-v1")
    gamma = 0.99

    for episode in range(1000):
        state, _ = env.reset()
        saved_log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            data = {
                "cart_pos": float(state[0]),
                "cart_vel": float(state[1]),
                "pole_angle": float(state[2]),
                "pole_vel": float(state[3]),
            }
            h = _inputs_to_tensor(
                data,
                block.abi,
                dim,
                device=device,
                dtype=dtype,
                abi_widths=aw,
            )
            block.train()
            out = block(h)
            col = int(block.abi["prob_right"])
            prob = out[0, col].clamp(1e-6, 1.0 - 1e-6)
            m = Bernoulli(probs=prob)
            action = m.sample()
            saved_log_probs.append(m.log_prob(action))
            state, reward, terminated, truncated, _ = env.step(int(action.item()))
            rewards.append(float(reward))

        dr = torch.tensor(_discounted_returns(rewards, gamma), device=device, dtype=dtype)
        if dr.numel() > 1:
            dr = (dr - dr.mean()) / (dr.std().clamp_min(1e-8))
        elif dr.numel() == 1:
            dr = torch.ones_like(dr)

        optimizer.zero_grad(set_to_none=True)
        terms = [-lp * g for lp, g in zip(saved_log_probs, dr)]
        if not terms:
            continue
        loss = torch.stack(terms).sum()
        loss.backward()
        optimizer.step()

        total_r = sum(rewards)
        print(f"Episode {episode + 1} Total Reward {total_r:.0f}")
        if total_r >= 500:
            print("Solved!")
            save_bundle(block, str(BUNDLE_PATH))
            env.close()
            return True

    env.close()
    return False


def watch() -> None:
    print("\n--- WATCHING THE NEURO-SYMBOLIC AGENT ---\n")
    if not BUNDLE_PATH.is_file():
        print(f"No bundle at {BUNDLE_PATH}; train until solved first.")
        return
    model = axiom.load(BUNDLE_PATH)
    block = model.block
    dim = _trunk_dim(block)
    aw = dict(getattr(block, "abi_widths", {}) or {})

    try:
        eval_env = gym.make("CartPole-v1", render_mode="human")
    except Exception as e:
        print(f"Could not open rendered env (display / pygame): {e}")
        return

    state, _ = eval_env.reset()
    terminated = False
    truncated = False
    try:
        while not (terminated or truncated):
            data = {
                "cart_pos": float(state[0]),
                "cart_vel": float(state[1]),
                "pole_angle": float(state[2]),
                "pole_vel": float(state[3]),
            }
            result = model.predict(data)
            pr = float(result["prob_right"])
            action = 1 if pr > 0.5 else 0
            state, _, terminated, truncated, _ = eval_env.step(action)
    finally:
        eval_env.close()


def main() -> None:
    if train():
        watch()
    else:
        print("Did not reach 500 reward in 1000 episodes.")


if __name__ == "__main__":
    main()
