from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch

Stmt = Tuple
ExprIR = List[Tuple]


def _batch_zeros(B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros(B, device=device, dtype=dtype)


def _batch_ones(B: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.ones(B, device=device, dtype=dtype)


def _all_active(B: int, device: torch.device) -> torch.Tensor:
    return torch.ones(B, dtype=torch.bool, device=device)


def collect_load_names_from_stmts(stmts: List[Stmt]) -> List[str]:
    found: Set[str] = set()

    def walk_expr(ir: ExprIR) -> None:
        for tup in ir:
            if isinstance(tup, tuple) and tup[0] == "OP_LOAD" and len(tup) > 1:
                found.add(str(tup[1]))

    def walk_stmt(stmt: Stmt) -> None:
        op = stmt[0]
        if op == "OP_ASSIGN":
            found.add(str(stmt[1]))
            walk_expr(stmt[2])
        elif op == "OP_EXPR_STMT":
            walk_expr(stmt[1])
        elif op == "OP_CONDITIONAL":
            walk_expr(stmt[1])
            for s in stmt[2]:
                walk_stmt(s)
            for s in stmt[3]:
                walk_stmt(s)
        elif op == "OP_LOOP":
            walk_expr(stmt[1])
            for s in stmt[2]:
                walk_stmt(s)

    for st in stmts:
        walk_stmt(st)
    return sorted(found)


def collect_load_names(cond_ir: ExprIR, body_ir: List[Stmt]) -> List[str]:
    found: Set[str] = set()

    def walk_expr(ir: ExprIR) -> None:
        for tup in ir:
            if not isinstance(tup, tuple):
                continue
            if tup[0] == "OP_LOAD" and len(tup) > 1:
                found.add(str(tup[1]))

    def walk_stmt(stmt: Stmt) -> None:
        op = stmt[0]
        if op == "OP_ASSIGN":
            found.add(str(stmt[1]))
            walk_expr(stmt[2])
        elif op == "OP_EXPR_STMT":
            walk_expr(stmt[1])
        elif op == "OP_CONDITIONAL":
            walk_expr(stmt[1])
            for s in stmt[2]:
                walk_stmt(s)
            for s in stmt[3]:
                walk_stmt(s)
        elif op == "OP_LOOP":
            walk_expr(stmt[1])
            for s in stmt[2]:
                walk_stmt(s)

    walk_expr(cond_ir)
    for st in body_ir:
        walk_stmt(st)
    return sorted(found)


def make_seed_map(cond_ir: ExprIR, body_ir: List[Stmt], dim: int) -> Dict[str, int]:
    """Loop-shaped ABI slice (name -> column) in first-seen order for cond then body; caps at ``dim`` names."""
    from compiler.ir import extract_global_abi

    return extract_global_abi([("OP_LOOP", list(cond_ir), list(body_ir))], max_vars=dim)


def build_var_order(
    cond_ir: ExprIR,
    body_ir: List[Stmt],
    dim: int,
    *,
    seed_names: Optional[Set[str]] = None,
    env_keys: Optional[Set[str]] = None,
) -> List[str]:
    loads = set(collect_load_names(cond_ir, body_ir))
    if seed_names:
        loads |= seed_names
    if env_keys:
        loads |= env_keys
    order = sorted(loads)
    i = 0
    while len(order) < dim:
        order.append(f"_pad{i}")
        i += 1
    return order[:dim]


def eval_expr(
    env: Dict[str, torch.Tensor],
    ir: ExprIR,
    *,
    B: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    stack: List[torch.Tensor] = []
    z = _batch_zeros(B, device, dtype)
    o = _batch_ones(B, device, dtype)
    for tup in ir:
        op = tup[0]
        if op == "OP_CONST":
            stack.append(torch.full((B,), float(tup[1]), device=device, dtype=dtype, requires_grad=False))
        elif op == "OP_LOAD":
            stack.append(env.get(str(tup[1]), z))
        elif op == "OP_NEG":
            stack.append(-stack.pop())
        elif op == "OP_ADD":
            b, a = stack.pop(), stack.pop()
            stack.append(a + b)
        elif op == "OP_SUB":
            b, a = stack.pop(), stack.pop()
            stack.append(a - b)
        elif op == "OP_MUL":
            b, a = stack.pop(), stack.pop()
            stack.append(a * b)
        elif op == "OP_DIV":
            b, a = stack.pop(), stack.pop()
            mask = b.abs() > 1e-12
            safe_b = torch.where(mask, b, o)
            safe_div = a / safe_b
            stack.append(torch.where(mask, safe_div, z))
        elif op == "OP_CMP_GT":
            b, a = stack.pop(), stack.pop()
            stack.append(torch.where(a > b, o, z))
        elif op == "OP_CMP_LT":
            b, a = stack.pop(), stack.pop()
            stack.append(torch.where(a < b, o, z))
        elif op == "OP_CMP_EQ":
            b, a = stack.pop(), stack.pop()
            stack.append(torch.where(a == b, o, z))
        elif op == "OP_CMP_NE":
            b, a = stack.pop(), stack.pop()
            stack.append(torch.where(a != b, o, z))
        else:
            raise ValueError(f"unknown expr op {op}")
    if len(stack) != 1:
        raise ValueError(f"expr stack size {len(stack)}")
    return stack[0]


def truthy(v: torch.Tensor) -> bool:
    """Scalar check (e.g. B=1 tests); batched code uses masks, not this."""
    return v.detach().reshape(-1)[0].item() != 0.0


def snapshot_env(
    env: Dict[str, torch.Tensor],
    var_order: List[str],
    *,
    B: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    z = _batch_zeros(B, device, dtype)
    cols = [env.get(k, z) for k in var_order]
    return torch.stack(cols, dim=1)


def run_while_loop(
    env: Dict[str, torch.Tensor],
    cond_ir: ExprIR,
    body_ir: List[Stmt],
    *,
    B: int,
    dim: int,
    max_unroll: int,
    var_order: List[str],
    device: torch.device,
    dtype: torch.dtype,
    parent_active: Optional[torch.Tensor] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Runs exactly ``max_unroll`` iterations (no early ``break``). When the condition is false,
    the body is a no-op for that row (``torch.where`` in ``exec_stmt``) and masks record False;
    snapshots still advance so length is always ``max_unroll``.
    """
    if parent_active is None:
        scope = _all_active(B, device)
    else:
        scope = parent_active.clone()
    snaps: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    for _ in range(max_unroll):
        cond_val = eval_expr(env, cond_ir, B=B, device=device, dtype=dtype)
        entering = scope & (cond_val != 0)
        for st in body_ir:
            exec_stmt(
                env,
                st,
                B=B,
                dim=dim,
                max_unroll=max_unroll,
                device=device,
                dtype=dtype,
                active_mask=entering,
            )
        snaps.append(snapshot_env(env, var_order, B=B, device=device, dtype=dtype))
        masks.append(entering.clone())
    return snaps, masks


def exec_stmt(
    env: Dict[str, torch.Tensor],
    stmt: Stmt,
    *,
    B: int,
    dim: int,
    max_unroll: int,
    device: torch.device,
    dtype: torch.dtype,
    active_mask: Optional[torch.Tensor] = None,
) -> None:
    if active_mask is None:
        active_mask = _all_active(B, device)
    op = stmt[0]
    if op == "OP_ASSIGN":
        nv = eval_expr(env, stmt[2], B=B, device=device, dtype=dtype)
        k = str(stmt[1])
        env[k] = torch.where(active_mask, nv, env[k])
    elif op == "OP_EXPR_STMT":
        eval_expr(env, stmt[1], B=B, device=device, dtype=dtype)
    elif op == "OP_CONDITIONAL":
        cond_vec = eval_expr(env, stmt[1], B=B, device=device, dtype=dtype)
        base = {k: v.clone() for k, v in env.items()}
        then_env = {k: v.clone() for k, v in env.items()}
        for s in stmt[2]:
            exec_stmt(then_env, s, B=B, dim=dim, max_unroll=max_unroll, device=device, dtype=dtype, active_mask=active_mask)
        else_env = {k: v.clone() for k, v in env.items()}
        for s in stmt[3]:
            exec_stmt(else_env, s, B=B, dim=dim, max_unroll=max_unroll, device=device, dtype=dtype, active_mask=active_mask)
        sel = cond_vec != 0
        for k in env.keys():
            picked = torch.where(sel, then_env[k], else_env[k])
            env[k] = torch.where(active_mask, picked, base[k])
    elif op == "OP_LOOP":
        inner_order = build_var_order(stmt[1], stmt[2], dim, env_keys=set(env.keys()))
        _, _ = run_while_loop(
            env,
            stmt[1],
            stmt[2],
            B=B,
            dim=dim,
            max_unroll=max_unroll,
            var_order=inner_order,
            device=device,
            dtype=dtype,
            parent_active=active_mask,
        )
    else:
        raise ValueError(f"unknown stmt {op}")


def run_loop_snapshots(
    h_batch: torch.Tensor,
    cond_ir: ExprIR,
    body_ir: List[Stmt],
    *,
    dim: int,
    max_unroll: int,
    seed_map: Dict[str, int],  # global ABI: variable name -> trunk column index
    prelude_stmts: Optional[List[Stmt]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(seq, seq_mask)`` with **fixed** ``T = max_unroll`` (or ``T=0`` if ``max_unroll==0``).

    ``seq`` is ``(B, T, D)``, ``seq_mask`` is ``(B, T)`` bool — True = that row executed the body that
    step. After the loop condition goes false, later steps are no-ops (env frozen) and masks are False;
    this matches Phase 9 SIMT semantics without a Python ``break`` (TorchDynamo–friendly).
    """
    if h_batch.dim() == 1:
        h_batch = h_batch.unsqueeze(0)
    if h_batch.dim() != 2:
        raise ValueError("h_batch must be (B, D) or (D,)")
    prelude_stmts = prelude_stmts or []
    B, D_in = h_batch.shape
    dev = device if device is not None else h_batch.device
    dt = dtype if dtype is not None else h_batch.dtype
    seed_vals = set(seed_map.keys())
    extra = set(collect_load_names_from_stmts(prelude_stmts))
    var_order = build_var_order(cond_ir, body_ir, dim, seed_names=seed_vals | extra)
    env: Dict[str, torch.Tensor] = {k: _batch_zeros(B, dev, dt) for k in var_order}
    for name, idx in seed_map.items():
        if idx < D_in:
            env[name] = h_batch[:, idx]
    for st in prelude_stmts:
        exec_stmt(env, st, B=B, dim=dim, max_unroll=max_unroll, device=dev, dtype=dt, active_mask=None)
    if max_unroll == 0:
        z = torch.zeros(B, 0, dim, device=dev, dtype=dt)
        m = torch.empty(B, 0, dtype=torch.bool, device=dev)
        return z, m
    snaps, masks = run_while_loop(
        env,
        cond_ir,
        body_ir,
        B=B,
        dim=dim,
        max_unroll=max_unroll,
        var_order=var_order,
        device=dev,
        dtype=dt,
        parent_active=None,
    )
    mat = torch.stack(snaps, dim=1)
    seq_mask = torch.stack(masks, dim=1)
    if mat.shape[2] < dim:
        pad = torch.zeros(B, mat.shape[1], dim - mat.shape[2], device=mat.device, dtype=mat.dtype)
        mat = torch.cat([mat, pad], dim=2)
    elif mat.shape[2] > dim:
        mat = mat[:, :, :dim]
    return mat, seq_mask
