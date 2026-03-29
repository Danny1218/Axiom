from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import torch

Stmt = Tuple
ExprIR = List[Tuple]


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


def eval_expr(env: Dict[str, float], ir: ExprIR) -> float:
    stack: List[float] = []
    for tup in ir:
        op = tup[0]
        if op == "OP_CONST":
            stack.append(float(tup[1]))
        elif op == "OP_LOAD":
            stack.append(float(env.get(str(tup[1]), 0.0)))
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
            stack.append(a / b if b != 0 else 0.0)
        elif op == "OP_CMP_GT":
            b, a = stack.pop(), stack.pop()
            stack.append(1.0 if a > b else 0.0)
        elif op == "OP_CMP_LT":
            b, a = stack.pop(), stack.pop()
            stack.append(1.0 if a < b else 0.0)
        elif op == "OP_CMP_EQ":
            b, a = stack.pop(), stack.pop()
            stack.append(1.0 if a == b else 0.0)
        elif op == "OP_CMP_NE":
            b, a = stack.pop(), stack.pop()
            stack.append(1.0 if a != b else 0.0)
        else:
            raise ValueError(f"unknown expr op {op}")
    if len(stack) != 1:
        raise ValueError(f"expr stack size {len(stack)}")
    return stack[0]


def truthy(v: float) -> bool:
    return v != 0.0


def snapshot_env(env: Dict[str, float], var_order: List[str]) -> List[float]:
    return [float(env.get(k, 0.0)) for k in var_order]


def run_while_loop(
    env: Dict[str, float],
    cond_ir: ExprIR,
    body_ir: List[Stmt],
    *,
    dim: int,
    max_unroll: int,
    var_order: List[str],
) -> List[List[float]]:
    snaps: List[List[float]] = []
    for _ in range(max_unroll):
        if not truthy(eval_expr(env, cond_ir)):
            break
        for st in body_ir:
            exec_stmt(env, st, dim=dim, max_unroll=max_unroll)
        snaps.append(snapshot_env(env, var_order))
    return snaps


def exec_stmt(env: Dict[str, float], stmt: Stmt, *, dim: int, max_unroll: int) -> None:
    op = stmt[0]
    if op == "OP_ASSIGN":
        env[str(stmt[1])] = eval_expr(env, stmt[2])
    elif op == "OP_EXPR_STMT":
        eval_expr(env, stmt[1])
    elif op == "OP_CONDITIONAL":
        block = stmt[2] if truthy(eval_expr(env, stmt[1])) else stmt[3]
        for s in block:
            exec_stmt(env, s, dim=dim, max_unroll=max_unroll)
    elif op == "OP_LOOP":
        inner_order = build_var_order(stmt[1], stmt[2], dim, env_keys=set(env.keys()))
        run_while_loop(env, stmt[1], stmt[2], dim=dim, max_unroll=max_unroll, var_order=inner_order)
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
) -> torch.Tensor:
    """(T, D) float tensor; T=0 if the loop body never runs."""
    prelude_stmts = prelude_stmts or []
    seed_vals = set(seed_map.values())
    extra = set(collect_load_names_from_stmts(prelude_stmts))
    var_order = build_var_order(cond_ir, body_ir, dim, seed_names=seed_vals | extra)
    env: Dict[str, float] = {k: 0.0 for k in var_order}
    flat = h_row.reshape(-1)
    for idx, name in seed_map.items():
        if idx < flat.numel():
            env[name] = float(flat[idx].item())
    for st in prelude_stmts:
        exec_stmt(env, st, dim=dim, max_unroll=max_unroll)
    snaps = run_while_loop(env, cond_ir, body_ir, dim=dim, max_unroll=max_unroll, var_order=var_order)
    if not snaps:
        return torch.zeros(0, dim, device=h_row.device, dtype=h_row.dtype)
    mat = torch.tensor(snaps, device=h_row.device, dtype=h_row.dtype)
    if mat.shape[1] < dim:
        pad = torch.zeros(mat.size(0), dim - mat.size(1), device=mat.device, dtype=mat.dtype)
        mat = torch.cat([mat, pad], dim=1)
    elif mat.shape[1] > dim:
        mat = mat[:, :dim]
    return mat
