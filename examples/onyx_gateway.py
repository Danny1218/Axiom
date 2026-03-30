"""
Neuro-symbolic gateway: text -> risk signals -> Axiom policy -> optional Onyx LLM.

Requires: pip install requests  (or pip install -e ".[gateway]")

Run from repo root: python examples/onyx_gateway.py
"""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from axiom.api import AxiomModel
from axiom.compiler.ir import ast_to_ir, extract_abi_widths, extract_global_abi
from axiom.compiler.parser import parse_ax_file, reset_parser
from axiom.engine.block_executor import InterpretedBlock
from axiom.engine.inference import _batch_inputs_to_tensor

_EXAMPLES = Path(__file__).resolve().parent
AX_PATH = _EXAMPLES / "enterprise_policy.ax"
AUDIT_PATH = _EXAMPLES / "blocked_audit.html"
ONYX_URL = "http://localhost:8000/api/chat"

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def _trunk_dim(block: InterpretedBlock) -> int:
    abi, aw = block.abi, getattr(block, "abi_widths", {}) or {}
    return max((abi[n] + max(1, int(aw.get(n, 1))) for n in abi), default=8)


def scan_text(prompt: str, *, rng: random.Random | None = None) -> dict[str, float]:
    """Regex + demo toxicity draw (0..0.3) -> policy feature dict."""
    r = rng or random.Random()
    has_pii = 1.0 if _SSN_RE.search(prompt) else 0.0
    low = prompt.lower()
    comp = 1.0 if ("openai" in low or "anthropic" in low) else 0.0
    return {
        "has_pii_data": has_pii,
        "mentions_competitor": comp,
        "text_toxicity": r.uniform(0.0, 0.3),
    }


def build_trained_policy(
    *,
    epochs: int = 50,
    lr: float = 0.1,
    seed: int = 42,
) -> tuple[AxiomModel, InterpretedBlock, str]:
    """Parse policy, quick Adam fit so low-toxicity rows target approve, high-toxicity rows deny."""
    random.seed(seed)
    torch.manual_seed(seed)

    rows: list[dict[str, float]] = []
    targets: list[float] = []
    intent_targets: list[float] = []
    for _ in range(40):
        rows.append(
            {"has_pii_data": 0.0, "mentions_competitor": 0.0, "text_toxicity": random.uniform(0.0, 0.25)}
        )
        targets.append(1.0)
        intent_targets.append(-0.3)
    for _ in range(40):
        rows.append(
            {"has_pii_data": 0.0, "mentions_competitor": 0.0, "text_toxicity": random.uniform(0.85, 1.0)}
        )
        targets.append(0.0)
        intent_targets.append(0.95)
    for _ in range(10):
        rows.append({"has_pii_data": 1.0, "mentions_competitor": 0.0, "text_toxicity": 0.1})
        targets.append(0.0)
        intent_targets.append(0.0)
    for _ in range(10):
        rows.append({"has_pii_data": 0.0, "mentions_competitor": 1.0, "text_toxicity": 0.1})
        targets.append(0.0)
        intent_targets.append(0.0)

    reset_parser()
    ir = ast_to_ir(parse_ax_file(AX_PATH))
    abi = extract_global_abi(ir, max_vars=128)
    aw = extract_abi_widths(ir, max_vars=128)
    block = InterpretedBlock(ir, abi, abi_widths=aw)
    dim = _trunk_dim(block)
    col = int(block.abi["is_approved"])
    col_ir = int(block.abi["intent_risk"])
    device, dtype = torch.device("cpu"), torch.float32
    y = torch.tensor(targets, device=device, dtype=dtype)
    y_ir = torch.tensor(intent_targets, device=device, dtype=dtype)
    opt = torch.optim.Adam(block.parameters(), lr=lr)

    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        h = _batch_inputs_to_tensor(rows, block.abi, dim, device=device, dtype=dtype, abi_widths=aw)
        block.train()
        out = block(h)
        loss = F.mse_loss(out[:, col], y) + F.mse_loss(out[:, col_ir], y_ir)
        loss.backward()
        opt.step()

    source = AX_PATH.read_text(encoding="utf-8")
    return AxiomModel(block), block, source


def chat_with_onyx(
    model: AxiomModel,
    source_code: str,
    user_prompt: str,
    *,
    audit_path: Path | str = AUDIT_PATH,
    onyx_url: str = ONYX_URL,
    post_fn: Callable[[str, str], Any] | None = None,
    text_rng: random.Random | None = None,
    verbose: bool = True,
) -> str | None:
    """
    Policy gate then POST to Onyx. ``post_fn(url, message)`` overrides HTTP for tests.
    Returns assistant text if approved, else None after audit export.
    """
    signals = scan_text(user_prompt, rng=text_rng)
    if verbose:
        print(f"  [signals] {signals}")
    trace = model.explain(signals)
    approved = float(trace["is_approved"])
    if verbose:
        print(f"  [trace] intent_risk={trace.get('intent_risk')} is_approved={approved}")

    if approved < 0.5:
        if verbose:
            print(
                "\n*** AXIOM SECURITY OVERRIDE: Request blocked before reaching LLM. ***\n"
                f"    Audit written to {audit_path}\n"
            )
        model.export_report(signals, str(Path(audit_path).resolve()), source_code=source_code)
        return None

    if verbose:
        print("\n*** AXIOM APPROVED: Routing to Onyx LLM... ***\n")

    if post_fn is not None:
        resp = post_fn(onyx_url, user_prompt)
        if hasattr(resp, "text"):
            return str(resp.text)
        return str(resp)

    try:
        import requests
    except ImportError:
        return "[Onyx Mock]: Hello, I am your enterprise AI."

    try:
        r = requests.post(onyx_url, json={"message": user_prompt}, timeout=8)
        r.raise_for_status()
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if isinstance(data, dict) and "reply" in data:
            return str(data["reply"])
        return r.text or "[Onyx]: (empty body)"
    except Exception:
        return "[Onyx Mock]: Hello, I am your enterprise AI."


def main() -> None:
    print("Compiling and training enterprise policy (liquid + symbolic gates)...\n")
    model, _block, source = build_trained_policy(epochs=50, lr=0.1)

    demos = [
        "Write a python script to sort a list.",
        "My SSN is 123-45-6789, what is my credit score?",
        "Why is OpenAI better than you?",
    ]
    for i, prompt in enumerate(demos, 1):
        print(f"--- Demo {i} ---\n  Prompt: {prompt!r}")
        reply = chat_with_onyx(model, source, prompt)
        if reply is not None:
            print(f"  [LLM reply] {reply}")
        print()


if __name__ == "__main__":
    main()
