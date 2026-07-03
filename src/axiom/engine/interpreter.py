from __future__ import annotations

# ``exec_stmt`` mutates ``env`` in place; ``InterpretedBlock.forward(..., return_env=True)`` exposes it (Phase 41).

from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from axiom.engine.expert_call import ExpertHandler, ExpertRuntimeError
from axiom.engine.expert_registry import ExpertRuntimeRegistry
from axiom.engine.strict import StrictInferenceError, mark_defined

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
            if not isinstance(tup, tuple) or not tup:
                continue
            if tup[0] == "OP_LOAD" and len(tup) > 1:
                found.add(str(tup[1]))
            elif tup[0] == "OP_NEURAL" and len(tup) >= 3:
                walk_expr(list(tup[2]))
            elif tup[0] == "OP_EXPERT" and len(tup) >= 3:
                walk_expr(list(tup[2]))
            elif tup[0] == "OP_CALL":
                for a in tup[2]:
                    walk_expr(list(a))

    def walk_stmt(stmt: Stmt) -> None:
        op = stmt[0]
        if op == "OP_ASSIGN":
            found.add(str(stmt[1]))
            walk_expr(stmt[2])
        elif op == "OP_BLEND_ASSIGN":
            found.add(str(stmt[1]))
            walk_expr(list(stmt[2]))
            walk_expr(list(stmt[3]))
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


def collect_assigned_names_from_stmts(stmts: List[Stmt]) -> Set[str]:
    found: Set[str] = set()

    def walk_stmt(stmt: Stmt) -> None:
        op = stmt[0]
        if op == "OP_ASSIGN":
            found.add(str(stmt[1]))
            walk_expr(stmt[2])
        elif op == "OP_BLEND_ASSIGN":
            found.add(str(stmt[1]))
            walk_expr(list(stmt[2]))
            walk_expr(list(stmt[3]))
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

    def walk_expr(ir: ExprIR) -> None:
        for tup in ir:
            if not isinstance(tup, tuple) or not tup:
                continue
            if tup[0] == "OP_NEURAL" and len(tup) >= 3:
                walk_expr(list(tup[2]))
            elif tup[0] == "OP_EXPERT" and len(tup) >= 3:
                walk_expr(list(tup[2]))
            elif tup[0] == "OP_CALL":
                for a in tup[2]:
                    walk_expr(list(a))

    for st in stmts:
        walk_stmt(st)
    return found


def collect_load_names(cond_ir: ExprIR, body_ir: List[Stmt]) -> List[str]:
    found: Set[str] = set()

    def walk_expr(ir: ExprIR) -> None:
        for tup in ir:
            if not isinstance(tup, tuple) or not tup:
                continue
            if tup[0] == "OP_LOAD" and len(tup) > 1:
                found.add(str(tup[1]))
            elif tup[0] == "OP_NEURAL" and len(tup) >= 3:
                walk_expr(list(tup[2]))
            elif tup[0] == "OP_EXPERT" and len(tup) >= 3:
                walk_expr(list(tup[2]))
            elif tup[0] == "OP_CALL":
                for a in tup[2]:
                    walk_expr(list(a))

    def walk_stmt(stmt: Stmt) -> None:
        op = stmt[0]
        if op == "OP_ASSIGN":
            found.add(str(stmt[1]))
            walk_expr(stmt[2])
        elif op == "OP_BLEND_ASSIGN":
            found.add(str(stmt[1]))
            walk_expr(list(stmt[2]))
            walk_expr(list(stmt[3]))
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
    from axiom.compiler.ir import extract_global_abi

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
    base = sorted(loads)
    if len(base) >= dim:
        return base[:dim]
    need = dim - len(base)
    return base + [f"_pad{i}" for i in range(need)]


def _broadcast_mask(mask_1d: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """``mask_1d`` (B,) → shape broadcastable with ``t`` (B,) or (B,K)."""
    if t.dim() == 1:
        return mask_1d
    return mask_1d.view(mask_1d.shape[0], *([1] * (t.dim() - 1)))


_TORCH_MATH_UNARY = {
    "abs": torch.abs,
    "exp": torch.exp,
    "log": torch.log,
    "sqrt": torch.sqrt,
    "sin": torch.sin,
    "cos": torch.cos,
}


def _promote_batch_binop(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """If one side is ``(B, K)`` and the other ``(B,)`` (batch-aligned 1D), promote ``(B,)`` → ``(B, 1)``."""
    if a.dim() == 0 or b.dim() == 0:
        return a, b
    if a.shape[0] != b.shape[0]:
        return a, b
    if a.dim() == 2 and b.dim() == 1:
        return a, b.unsqueeze(-1)
    if b.dim() == 2 and a.dim() == 1:
        return a.unsqueeze(-1), b
    return a, b


def eval_expr(
    env: Dict[str, torch.Tensor],
    ir: ExprIR,
    *,
    B: int,
    device: torch.device,
    dtype: torch.dtype,
    neural_registry: Optional[Union[Dict[str, nn.Module], nn.ModuleDict]] = None,
    expert_handler: Optional[ExpertHandler] = None,
    expert_fallback: Optional[float] = None,
    expert_registry: Optional[ExpertRuntimeRegistry] = None,
    expert_audit: Optional[List[Dict[str, Any]]] = None,
    strict: bool = False,
    env_defined: Optional[Set[str]] = None,
) -> torch.Tensor:
    stack: List[torch.Tensor] = []
    z = _batch_zeros(B, device, dtype)
    o = _batch_ones(B, device, dtype)
    for tup in ir:
        op = tup[0]
        if op == "OP_CONST":
            stack.append(torch.full((B,), float(tup[1]), device=device, dtype=dtype, requires_grad=False))
        elif op == "OP_LOAD":
            name = str(tup[1])
            if strict and env_defined is not None and name not in env_defined:
                raise StrictInferenceError(f"load of unset variable {name!r}")
            stack.append(env.get(name, z))
        elif op == "OP_NEG":
            stack.append(-stack.pop())
        elif op == "OP_VEC_PACK":
            n = int(tup[1])
            if n < 1:
                raise ValueError("OP_VEC_PACK requires n >= 1")
            parts = [stack.pop() for _ in range(n)]
            parts.reverse()
            stack.append(torch.stack(parts, dim=1))
        elif op == "OP_INDEX":
            idx_t = stack.pop()
            arr = stack.pop()
            if arr.dim() == 1:
                arr2 = arr.unsqueeze(1)
                k = 1
            else:
                arr2 = arr
                k = int(arr2.shape[1])
            idx_raw = idx_t.to(dtype=torch.int64)
            if strict:
                bad = (idx_raw < 0) | (idx_raw >= k)
                if bool(bad.any()):
                    raise StrictInferenceError(f"index out of range (width={k})")
            idx = idx_raw.clamp(0, max(k - 1, 0))
            gathered = torch.gather(arr2, 1, idx.unsqueeze(1)).squeeze(1)
            stack.append(gathered)
        elif op == "OP_ADD":
            b, a = stack.pop(), stack.pop()
            a, b = _promote_batch_binop(a, b)
            stack.append(a + b)
        elif op == "OP_SUB":
            b, a = stack.pop(), stack.pop()
            a, b = _promote_batch_binop(a, b)
            stack.append(a - b)
        elif op == "OP_MUL":
            b, a = stack.pop(), stack.pop()
            a, b = _promote_batch_binop(a, b)
            stack.append(a * b)
        elif op == "OP_DIV":
            b, a = stack.pop(), stack.pop()
            a, b = _promote_batch_binop(a, b)
            mask = b.abs() > 1e-12
            if strict and not bool(mask.all()):
                raise StrictInferenceError("division by zero")
            safe_b = torch.where(mask, b, o)
            safe_div = a / safe_b
            stack.append(torch.where(mask, safe_div, z))
        elif op == "OP_REDUCE_SUM":
            v = stack.pop()
            if v.dim() <= 1:
                stack.append(v)
            else:
                stack.append(torch.sum(v, dim=-1))
        elif op == "OP_REDUCE_MEAN":
            v = stack.pop()
            if v.dim() <= 1:
                stack.append(v)
            else:
                stack.append(torch.mean(v, dim=-1))
        elif op == "OP_REDUCE_BATCH_MEAN":
            v = stack.pop()
            if v.dim() == 0:
                stack.append(v)
            else:
                stack.append(torch.mean(v, dim=0, keepdim=True))
        elif op == "OP_DOT":
            b, a = stack.pop(), stack.pop()
            if a.dim() == 1 and b.dim() == 1:
                a, b = a.unsqueeze(-1), b.unsqueeze(-1)
            a, b = _promote_batch_binop(a, b)
            stack.append(torch.sum(a * b, dim=-1))
        elif op == "OP_MATH_UNARY":
            fn = _TORCH_MATH_UNARY.get(str(tup[1]))
            if fn is None:
                raise ValueError(f"unknown OP_MATH_UNARY {tup[1]!r}")
            stack.append(fn(stack.pop()))
        elif op == "OP_MATH_BINARY":
            b, a = stack.pop(), stack.pop()
            a, b = _promote_batch_binop(a, b)
            fn = str(tup[1])
            if fn == "max":
                stack.append(torch.maximum(a, b))
            elif fn == "min":
                stack.append(torch.minimum(a, b))
            else:
                raise ValueError(f"unknown OP_MATH_BINARY {fn!r}")
        elif op == "OP_NEURAL" and len(tup) >= 3:
            # tup[3] is architecture tag for the executor; stack eval is unchanged.
            feats = eval_expr(
                env,
                list(tup[2]),
                B=B,
                device=device,
                dtype=dtype,
                neural_registry=neural_registry,
                expert_handler=expert_handler,
                expert_fallback=expert_fallback,
                expert_registry=expert_registry,
                expert_audit=expert_audit,
                strict=strict,
                env_defined=env_defined,
            )
            feats2 = feats.unsqueeze(-1) if feats.dim() == 1 else feats
            reg = neural_registry
            nid = str(tup[1])
            mod = reg[nid] if reg is not None and nid in reg else None
            if mod is not None:
                raw = mod(feats2)
            else:
                raw = torch.zeros(B, 1, device=device, dtype=dtype)
            if raw.dim() == 1:
                out = raw.unsqueeze(-1)
            else:
                out = raw
            if out.shape[-1] != 1:
                raise ValueError(f"neural {nid!r} must output width 1, got shape {tuple(out.shape)}")
            stack.append(out.squeeze(-1))
        elif op == "OP_EXPERT" and len(tup) >= 3:
            name = str(tup[1])
            feats = eval_expr(
                env,
                list(tup[2]),
                B=B,
                device=device,
                dtype=dtype,
                neural_registry=neural_registry,
                expert_handler=expert_handler,
                expert_fallback=expert_fallback,
                expert_registry=expert_registry,
                expert_audit=expert_audit,
                strict=strict,
                env_defined=env_defined,
            )
            fd = feats.detach()
            if fd.dim() == 1:
                fd = fd.unsqueeze(-1)
            if fd.dim() != 2 or int(fd.shape[0]) != B:
                raise ValueError(
                    f"expert({name!r}) features must be (B,) or (B,K), got {tuple(feats.shape)}"
                )
            rows = fd.cpu().tolist()
            out_vals: List[float] = []
            for bi in range(B):
                row = [float(x) for x in rows[bi]]
                fn: Optional[ExpertHandler] = None
                if expert_registry is not None:
                    fn = expert_registry.resolve(name)
                if fn is None:
                    fn = expert_handler
                if fn is not None:
                    try:
                        v = float(fn(name, row))
                    except Exception as e:
                        raise ExpertRuntimeError(
                            f"expert backend {name!r} handler raised: {e}"
                        ) from e
                elif expert_fallback is not None:
                    v = float(expert_fallback)
                else:
                    raise ExpertRuntimeError(
                        f"expert({name!r}) has no runtime backend: register a handler with "
                        f"ExpertRuntimeRegistry.register({name!r}, fn), set "
                        f"InterpretedBlock(..., expert_handler=callable), expert_fallback=float, "
                        f"or AxiomModel.set_expert_registry(...)"
                    )
                out_vals.append(v)
            if expert_audit is not None:
                expert_audit.append({"op": "expert", "backend": name})
            stack.append(
                torch.tensor(out_vals, device=device, dtype=dtype, requires_grad=False)
            )
        elif op == "OP_CMP_GT":
            b, a = stack.pop(), stack.pop()
            a, b = _promote_batch_binop(a, b)
            stack.append(torch.where(a > b, o, z))
        elif op == "OP_CMP_LT":
            b, a = stack.pop(), stack.pop()
            a, b = _promote_batch_binop(a, b)
            stack.append(torch.where(a < b, o, z))
        elif op == "OP_CMP_EQ":
            b, a = stack.pop(), stack.pop()
            a, b = _promote_batch_binop(a, b)
            stack.append(torch.where(a == b, o, z))
        elif op == "OP_CMP_NE":
            b, a = stack.pop(), stack.pop()
            a, b = _promote_batch_binop(a, b)
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
    var_widths: Optional[Dict[str, int]] = None,
) -> torch.Tensor:
    z1 = _batch_zeros(B, device, dtype)
    vw = var_widths or {}
    cols: List[torch.Tensor] = []
    for k in var_order:
        w = max(1, int(vw.get(k, 1)))
        t = env.get(k)
        if t is None:
            cols.append(torch.zeros(B, w, device=device, dtype=dtype))
        elif w == 1:
            if t.dim() != 1 or int(t.shape[0]) != B:
                raise ValueError(
                    f"snapshot_env: variable {k!r} expected (B,) with B={B}, got {tuple(t.shape)}"
                )
            cols.append(t.unsqueeze(1))
        else:
            if t.dim() != 2 or int(t.shape[0]) != B or int(t.shape[1]) != w:
                raise ValueError(
                    f"snapshot_env: variable {k!r} expected (B, {w}), got {tuple(t.shape)}"
                )
            cols.append(t)
    return torch.cat(cols, dim=1) if cols else torch.zeros(B, 0, device=device, dtype=dtype)


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
    var_widths: Optional[Dict[str, int]] = None,
    neural_registry: Optional[Union[Dict[str, nn.Module], nn.ModuleDict]] = None,
    expert_handler: Optional[ExpertHandler] = None,
    expert_fallback: Optional[float] = None,
    expert_registry: Optional[ExpertRuntimeRegistry] = None,
    expert_audit: Optional[List[Dict[str, Any]]] = None,
    strict: bool = False,
    env_defined: Optional[Set[str]] = None,
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
        cond_val = eval_expr(
            env,
            cond_ir,
            B=B,
            device=device,
            dtype=dtype,
            neural_registry=neural_registry,
            expert_handler=expert_handler,
            expert_fallback=expert_fallback,
            expert_registry=expert_registry,
            expert_audit=expert_audit,
            strict=strict,
            env_defined=env_defined,
        )
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
                abi_widths=var_widths,
                neural_registry=neural_registry,
                expert_handler=expert_handler,
                expert_fallback=expert_fallback,
                expert_registry=expert_registry,
                expert_audit=expert_audit,
                strict=strict,
                env_defined=env_defined,
            )
        snaps.append(
            snapshot_env(
                env, var_order, B=B, device=device, dtype=dtype, var_widths=var_widths
            )
        )
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
    abi_widths: Optional[Dict[str, int]] = None,
    neural_registry: Optional[Union[Dict[str, nn.Module], nn.ModuleDict]] = None,
    expert_handler: Optional[ExpertHandler] = None,
    expert_fallback: Optional[float] = None,
    expert_registry: Optional[ExpertRuntimeRegistry] = None,
    expert_audit: Optional[List[Dict[str, Any]]] = None,
    strict: bool = False,
    env_defined: Optional[Set[str]] = None,
) -> None:
    if active_mask is None:
        active_mask = _all_active(B, device)
    aw = abi_widths or {}
    op = stmt[0]
    if op == "OP_ASSIGN":
        nv = eval_expr(
            env,
            stmt[2],
            B=B,
            device=device,
            dtype=dtype,
            neural_registry=neural_registry,
            expert_handler=expert_handler,
            expert_fallback=expert_fallback,
            expert_registry=expert_registry,
            expert_audit=expert_audit,
            strict=strict,
            env_defined=env_defined,
        )
        k = str(stmt[1])
        old = env[k]
        if (
            old.dim() == 1
            and nv.dim() == 2
            and nv.shape[0] == old.shape[0]
            and nv.shape[1] == 1
        ):
            nv = nv.squeeze(-1)
        m = _broadcast_mask(active_mask, nv)
        env[k] = torch.where(m, nv, old)
        if env_defined is not None:
            mark_defined(env_defined, k)
    elif op == "OP_BLEND_ASSIGN":
        k = str(stmt[1])
        path_a = eval_expr(
            env,
            list(stmt[2]),
            B=B,
            device=device,
            dtype=dtype,
            neural_registry=neural_registry,
            expert_handler=expert_handler,
            expert_fallback=expert_fallback,
            expert_registry=expert_registry,
            expert_audit=expert_audit,
            strict=strict,
            env_defined=env_defined,
        ).to(dtype=dtype)
        nv = eval_expr(
            env,
            list(stmt[3]),
            B=B,
            device=device,
            dtype=dtype,
            neural_registry=neural_registry,
            expert_handler=expert_handler,
            expert_fallback=expert_fallback,
            expert_registry=expert_registry,
            expert_audit=expert_audit,
            strict=strict,
            env_defined=env_defined,
        )
        old = env[k]
        path_a, nv2 = _promote_batch_binop(path_a, nv)
        nv = nv2
        _, old2 = _promote_batch_binop(path_a, old)
        old = old2
        parent_m = _broadcast_mask(active_mask, nv)
        aa = path_a * parent_m
        env[k] = aa * nv + (1.0 - aa) * old
        if env_defined is not None:
            mark_defined(env_defined, k)
    elif op == "OP_EXPR_STMT":
        eval_expr(
            env,
            stmt[1],
            B=B,
            device=device,
            dtype=dtype,
            neural_registry=neural_registry,
            expert_handler=expert_handler,
            expert_fallback=expert_fallback,
            expert_registry=expert_registry,
            expert_audit=expert_audit,
            strict=strict,
            env_defined=env_defined,
        )
    elif op == "OP_CONDITIONAL":
        cond_vec = eval_expr(
            env,
            stmt[1],
            B=B,
            device=device,
            dtype=dtype,
            neural_registry=neural_registry,
            expert_handler=expert_handler,
            expert_fallback=expert_fallback,
            expert_registry=expert_registry,
            expert_audit=expert_audit,
            strict=strict,
            env_defined=env_defined,
        )
        base = {k: v.clone() for k, v in env.items()}
        parent_defined = env_defined
        then_defined = set(parent_defined) if parent_defined is not None else None
        else_defined = set(parent_defined) if parent_defined is not None else None
        then_env = {k: v.clone() for k, v in env.items()}
        for s in stmt[2]:
            exec_stmt(
                then_env,
                s,
                B=B,
                dim=dim,
                max_unroll=max_unroll,
                device=device,
                dtype=dtype,
                active_mask=active_mask,
                abi_widths=aw,
                neural_registry=neural_registry,
                expert_handler=expert_handler,
                expert_fallback=expert_fallback,
                expert_registry=expert_registry,
                expert_audit=expert_audit,
                strict=strict,
                env_defined=then_defined,
            )
        else_env = {k: v.clone() for k, v in env.items()}
        for s in stmt[3]:
            exec_stmt(
                else_env,
                s,
                B=B,
                dim=dim,
                max_unroll=max_unroll,
                device=device,
                dtype=dtype,
                active_mask=active_mask,
                abi_widths=aw,
                neural_registry=neural_registry,
                expert_handler=expert_handler,
                expert_fallback=expert_fallback,
                expert_registry=expert_registry,
                expert_audit=expert_audit,
                strict=strict,
                env_defined=else_defined,
            )
        sel = cond_vec != 0
        for k in env.keys():
            te, ee = then_env[k], else_env[k]
            if te.dim() == 1:
                picked = torch.where(sel, te, ee)
            else:
                sm = sel.view(sel.shape[0], *([1] * (te.dim() - 1)))
                picked = torch.where(sm, te, ee)
            m = _broadcast_mask(active_mask, picked)
            env[k] = torch.where(m, picked, base[k])
        if parent_defined is not None and then_defined is not None and else_defined is not None:
            for name in then_defined & else_defined:
                mark_defined(parent_defined, name)
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
            var_widths=aw,
            neural_registry=neural_registry,
            expert_handler=expert_handler,
            expert_fallback=expert_fallback,
            expert_registry=expert_registry,
            expert_audit=expert_audit,
            strict=strict,
            env_defined=env_defined,
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
    trunk_dim: Optional[int] = None,
    abi_widths: Optional[Dict[str, int]] = None,
    neural_registry: Optional[Union[Dict[str, nn.Module], nn.ModuleDict]] = None,
    expert_handler: Optional[ExpertHandler] = None,
    expert_fallback: Optional[float] = None,
    expert_registry: Optional[ExpertRuntimeRegistry] = None,
    expert_audit: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns ``(seq, seq_mask)`` with **fixed** ``T = max_unroll`` (or ``T=0`` if ``max_unroll==0``).

    ``seq`` is ``(B, T, D)``, ``seq_mask`` is ``(B, T)`` bool — True = that row executed the body that
    step. After the loop condition goes false, later steps are no-ops (env frozen) and masks are False;
    this matches Phase 9 SIMT semantics without a Python ``break`` (TorchDynamo–friendly).

    If ``trunk_dim`` is set and the stacked feature width is smaller (e.g. ``dim`` only sized to script
    variables), the sequence is zero-padded on the last axis to ``trunk_dim`` so ``LiquidKANNode`` matches
    the trunk width (latent channel padding / avoids ABI vs capacity mismatch).
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
    aw = abi_widths or {}
    for name, idx in seed_map.items():
        w = max(1, int(aw.get(name, 1)))
        if idx + w <= D_in:
            if w == 1:
                env[name] = h_batch[:, idx]
            else:
                env[name] = h_batch[:, idx : idx + w].clone()
    for st in prelude_stmts:
        exec_stmt(
            env,
            st,
            B=B,
            dim=dim,
            max_unroll=max_unroll,
            device=dev,
            dtype=dt,
            active_mask=None,
            abi_widths=aw,
            neural_registry=neural_registry,
            expert_handler=expert_handler,
            expert_fallback=expert_fallback,
            expert_registry=expert_registry,
            expert_audit=expert_audit,
        )
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
        var_widths=aw,
        neural_registry=neural_registry,
        expert_handler=expert_handler,
        expert_fallback=expert_fallback,
        expert_registry=expert_registry,
        expert_audit=expert_audit,
    )
    mat = torch.stack(snaps, dim=1)
    seq_mask = torch.stack(masks, dim=1)
    if mat.shape[2] < dim:
        pad = torch.zeros(B, mat.shape[1], dim - mat.shape[2], device=mat.device, dtype=mat.dtype)
        mat = torch.cat([mat, pad], dim=2)
    elif mat.shape[2] > dim:
        mat = mat[:, :, :dim]
    if trunk_dim is not None and mat.shape[2] < trunk_dim:
        mat = F.pad(mat, (0, trunk_dim - mat.shape[2]))
    return mat, seq_mask
