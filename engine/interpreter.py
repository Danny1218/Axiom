from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch

Stmt = Tuple
ExprIR = List[Tuple]


def _scalar_zero(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.zeros((), device=device, dtype=dtype)


def _scalar_one(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.ones((), device=device, dtype=dtype)


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


def make_seed_map(cond_ir: ExprIR, body_ir: List[Stmt], dim: int) -> Dict[int, str]:
    names = collect_load_names(cond_ir, body_ir)
    return {i: names[i] for i in range(min(len(names), dim))}


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
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    stack: List[torch.Tensor] = []
    for tup in ir:
        op = tup[0]
        if op == "OP_CONST":
            stack.append(torch.tensor(float(tup[1]), device=device, dtype=dtype, requires_grad=False))
        elif op == "OP_LOAD":
            stack.append(env.get(str(tup[1]), _scalar_zero(device, dtype)))
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
            # Avoid computing a/b at b==0 (forward Inf; backward 0*Inf -> NaN in torch.where).
            mask = b.abs() > 1e-12
            safe_b = torch.where(mask, b, _scalar_one(device, dtype))
            safe_div = a / safe_b
            stack.append(torch.where(mask, safe_div, _scalar_zero(device, dtype)))
        elif op == "OP_CMP_GT":
            b, a = stack.pop(), stack.pop()
            stack.append(torch.where(a > b, _scalar_one(device, dtype), _scalar_zero(device, dtype)))
        elif op == "OP_CMP_LT":
            b, a = stack.pop(), stack.pop()
            stack.append(torch.where(a < b, _scalar_one(device, dtype), _scalar_zero(device, dtype)))
        elif op == "OP_CMP_EQ":
            b, a = stack.pop(), stack.pop()
            stack.append(torch.where(a == b, _scalar_one(device, dtype), _scalar_zero(device, dtype)))
        elif op == "OP_CMP_NE":
            b, a = stack.pop(), stack.pop()
            stack.append(torch.where(a != b, _scalar_one(device, dtype), _scalar_zero(device, dtype)))
        else:
            raise ValueError(f"unknown expr op {op}")
    if len(stack) != 1:
        raise ValueError(f"expr stack size {len(stack)}")
    return stack[0]


def truthy(v: torch.Tensor) -> bool:
    return v.detach().item() != 0.0


def snapshot_env(
    env: Dict[str, torch.Tensor],
    var_order: List[str],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.stack([env.get(k, _scalar_zero(device, dtype)) for k in var_order])


def run_while_loop(
    env: Dict[str, torch.Tensor],
    cond_ir: ExprIR,
    body_ir: List[Stmt],
    *,
    dim: int,
    max_unroll: int,
    var_order: List[str],
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    snaps: List[torch.Tensor] = []
    for _ in range(max_unroll):
        if not truthy(eval_expr(env, cond_ir, device=device, dtype=dtype)):
            break
        for st in body_ir:
            exec_stmt(env, st, dim=dim, max_unroll=max_unroll, device=device, dtype=dtype)
        snaps.append(snapshot_env(env, var_order, device=device, dtype=dtype))
    return snaps


def exec_stmt(
    env: Dict[str, torch.Tensor],
    stmt: Stmt,
    *,
    dim: int,
    max_unroll: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    op = stmt[0]
    if op == "OP_ASSIGN":
        env[str(stmt[1])] = eval_expr(env, stmt[2], device=device, dtype=dtype)
    elif op == "OP_EXPR_STMT":
        eval_expr(env, stmt[1], device=device, dtype=dtype)
    elif op == "OP_CONDITIONAL":
        block = stmt[2] if truthy(eval_expr(env, stmt[1], device=device, dtype=dtype)) else stmt[3]
        for s in block:
            exec_stmt(env, s, dim=dim, max_unroll=max_unroll, device=device, dtype=dtype)
    elif op == "OP_LOOP":
        inner_order = build_var_order(stmt[1], stmt[2], dim, env_keys=set(env.keys()))
        run_while_loop(
            env,
            stmt[1],
            stmt[2],
            dim=dim,
            max_unroll=max_unroll,
            var_order=inner_order,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"unknown stmt {op}")


def run_loop_snapshots(
    h_row: torch.Tensor,
    cond_ir: ExprIR,
    body_ir: List[Stmt],
    *,
    dim: int,
    max_unroll: int,
    seed_map: Dict[int, str],
    prelude_stmts: Optional[List[Stmt]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """(T, D) tensor; T=0 if the loop body never runs. Differentiable w.r.t. `h_row` (and env tensors)."""
    prelude_stmts = prelude_stmts or []
    dev = device if device is not None else h_row.device
    dt = dtype if dtype is not None else h_row.dtype
    seed_vals = set(seed_map.values())
    extra = set(collect_load_names_from_stmts(prelude_stmts))
    var_order = build_var_order(cond_ir, body_ir, dim, seed_names=seed_vals | extra)
    env: Dict[str, torch.Tensor] = {k: _scalar_zero(dev, dt) for k in var_order}
    flat = h_row.reshape(-1)
    for idx, name in seed_map.items():
        if idx < flat.numel():
            env[name] = flat[idx].reshape(())
    for st in prelude_stmts:
        exec_stmt(env, st, dim=dim, max_unroll=max_unroll, device=dev, dtype=dt)
    snaps = run_while_loop(
        env, cond_ir, body_ir, dim=dim, max_unroll=max_unroll, var_order=var_order, device=dev, dtype=dt
    )
    if not snaps:
        return torch.zeros(0, dim, device=dev, dtype=dt)
    mat = torch.stack(snaps, dim=0)
    if mat.shape[1] < dim:
        pad = torch.zeros(mat.size(0), dim - mat.shape[1], device=mat.device, dtype=mat.dtype)
        mat = torch.cat([mat, pad], dim=1)
    elif mat.shape[1] > dim:
        mat = mat[:, :dim]
    return mat
