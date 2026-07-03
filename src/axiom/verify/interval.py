"""Static interval analysis over InterpretedBlock IR for safety certificates."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple, Union

from axiom.engine.block_executor import InterpretedBlock

ExprIR = List[Tuple[Any, ...]]
Stmt = Tuple[Any, ...]
Bounds = Tuple[float, float]
Env = Dict[str, Bounds]

INF = float("inf")
NINF = float("-inf")


def _import_version() -> str:
    from importlib.metadata import version

    return version("axiom-engine")


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    @staticmethod
    def unknown() -> Interval:
        return Interval(NINF, INF)

    @staticmethod
    def point(v: float) -> Interval:
        return Interval(v, v)

    @staticmethod
    def from_bounds(b: Bounds) -> Interval:
        return Interval(float(b[0]), float(b[1]))

    def union(self, other: Interval) -> Interval:
        return Interval(min(self.lo, other.lo), max(self.hi, other.hi))

    def as_tuple(self) -> Bounds:
        return (self.lo, self.hi)


def _neg(a: Interval) -> Interval:
    return Interval(-a.hi, -a.lo)


def _add(a: Interval, b: Interval) -> Interval:
    return Interval(a.lo + b.lo, a.hi + b.hi)


def _sub(a: Interval, b: Interval) -> Interval:
    return Interval(a.lo - b.hi, a.hi - b.lo)


def _mul(a: Interval, b: Interval) -> Interval:
    corners = (a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi)
    return Interval(min(corners), max(corners))


def _div(a: Interval, b: Interval) -> Interval:
    if b.lo <= 0.0 <= b.hi:
        return Interval.unknown()
    corners = (a.lo / b.lo, a.lo / b.hi, a.hi / b.lo, a.hi / b.hi)
    return Interval(min(corners), max(corners))


def _min_bin(a: Interval, b: Interval) -> Interval:
    return Interval(min(a.lo, b.lo), min(a.hi, b.hi))


def _max_bin(a: Interval, b: Interval) -> Interval:
    return Interval(max(a.lo, b.lo), max(a.hi, b.hi))


def _cmp_interval() -> Interval:
    return Interval(0.0, 1.0)


def _cmp_truth(op: str, a: Interval, b: Interval) -> Optional[bool]:
    if op == "OP_CMP_GT":
        if a.lo > b.hi:
            return True
        if a.hi <= b.lo:
            return False
    elif op == "OP_CMP_LT":
        if a.hi < b.lo:
            return True
        if a.lo >= b.hi:
            return False
    elif op == "OP_CMP_EQ":
        if a.lo == a.hi == b.lo == b.hi:
            return True
        if a.hi < b.lo or b.hi < a.lo:
            return False
    elif op == "OP_CMP_NE":
        if a.hi < b.lo or b.hi < a.lo:
            return True
        if a.lo == a.hi == b.lo == b.hi:
            return False
    return None


def eval_cond_truth(
    env: Env,
    ir: ExprIR,
    *,
    node_bounds: Mapping[str, Bounds],
    unsupported: List[str],
) -> Optional[bool]:
    """Return True/False when condition is provably always so; None if unknown."""
    stack: List[Interval] = []
    for tup in ir:
        op = tup[0]
        if op == "OP_CONST":
            stack.append(Interval.point(float(tup[1])))
        elif op == "OP_LOAD":
            stack.append(_load_bounds(env, str(tup[1])))
        elif op in ("OP_CMP_GT", "OP_CMP_LT", "OP_CMP_EQ", "OP_CMP_NE"):
            b, a = stack.pop(), stack.pop()
            truth = _cmp_truth(op, a, b)
            if truth is True:
                stack.append(Interval.point(1.0))
            elif truth is False:
                stack.append(Interval.point(0.0))
            else:
                stack.append(_cmp_interval())
        else:
            return None
    if len(stack) != 1:
        return None
    iv = stack[0]
    if iv.lo == iv.hi == 1.0:
        return True
    if iv.lo == iv.hi == 0.0:
        return False
    return None


@dataclass
class Certificate:
    axiom_version: str
    bundle_source_hash: str
    input_region: Dict[str, Bounds]
    assumptions: Dict[str, Bounds]
    proven_output_bounds: Dict[str, Bounds]
    timestamp: str
    unsupported_ops: List[str] = field(default_factory=list)
    status: str = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axiom_version": self.axiom_version,
            "bundle_source_hash": self.bundle_source_hash,
            "input_region": self.input_region,
            "assumptions": self.assumptions,
            "proven_output_bounds": self.proven_output_bounds,
            "timestamp": self.timestamp,
            "unsupported_ops": self.unsupported_ops,
            "status": self.status,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"


def _hash_source(path: Optional[Path], block: InterpretedBlock) -> str:
    if path is not None and path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    payload = json.dumps(block.ir_stmts, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_bounds(env: Env, name: str) -> Interval:
    if name in env:
        return Interval.from_bounds(env[name])
    return Interval.unknown()


def eval_expr_interval(
    env: Env,
    ir: ExprIR,
    *,
    node_bounds: Mapping[str, Bounds],
    unsupported: List[str],
) -> Interval:
    stack: List[Interval] = []
    for tup in ir:
        op = tup[0]
        if op == "OP_CONST":
            stack.append(Interval.point(float(tup[1])))
        elif op == "OP_LOAD":
            stack.append(_load_bounds(env, str(tup[1])))
        elif op == "OP_NEG":
            stack.append(_neg(stack.pop()))
        elif op == "OP_ADD":
            b, a = stack.pop(), stack.pop()
            stack.append(_add(a, b))
        elif op == "OP_SUB":
            b, a = stack.pop(), stack.pop()
            stack.append(_sub(a, b))
        elif op == "OP_MUL":
            b, a = stack.pop(), stack.pop()
            stack.append(_mul(a, b))
        elif op == "OP_DIV":
            b, a = stack.pop(), stack.pop()
            stack.append(_div(a, b))
        elif op == "OP_MATH_BINARY":
            b, a = stack.pop(), stack.pop()
            fn = str(tup[1])
            if fn == "min":
                stack.append(_min_bin(a, b))
            elif fn == "max":
                stack.append(_max_bin(a, b))
            else:
                unsupported.append(f"OP_MATH_BINARY:{fn}")
                stack.append(Interval.unknown())
        elif op in ("OP_CMP_GT", "OP_CMP_LT", "OP_CMP_EQ", "OP_CMP_NE"):
            b, a = stack.pop(), stack.pop()
            truth = _cmp_truth(op, a, b)
            if truth is True:
                stack.append(Interval.point(1.0))
            elif truth is False:
                stack.append(Interval.point(0.0))
            else:
                stack.append(_cmp_interval())
        elif op == "OP_NEURAL":
            unsupported.append(f"OP_NEURAL:{tup[1]}")
            stack.append(Interval.unknown())
        elif op == "OP_EXPERT" and len(tup) >= 3:
            name = str(tup[1])
            eval_expr_interval(env, list(tup[2]), node_bounds=node_bounds, unsupported=unsupported)
            nb = node_bounds.get(name)
            stack.append(Interval.from_bounds(nb) if nb else Interval.unknown())
        elif op == "OP_VEC_PACK":
            n = int(tup[1])
            parts = [stack.pop() for _ in range(n)]
            parts.reverse()
            stack.append(Interval(min(p.lo for p in parts), max(p.hi for p in parts)))
        elif op == "OP_INDEX":
            stack.pop()
            stack.pop()
            stack.append(Interval.unknown())
        elif op in ("OP_REDUCE_SUM", "OP_REDUCE_MEAN", "OP_REDUCE_BATCH_MEAN", "OP_DOT"):
            stack.pop()
            if op == "OP_DOT":
                stack.pop()
            stack.append(Interval.unknown())
        elif op == "OP_MATH_UNARY":
            stack.pop()
            stack.append(Interval.unknown())
        else:
            unsupported.append(str(op))
            stack.append(Interval.unknown())
    if len(stack) != 1:
        raise ValueError(f"expr stack size {len(stack)}")
    return stack[0]


def exec_stmt_interval(
    env: Env,
    stmt: Stmt,
    *,
    node_bounds: Mapping[str, Bounds],
    unsupported: List[str],
) -> Env:
    op = stmt[0]
    if op == "OP_ASSIGN":
        iv = eval_expr_interval(env, list(stmt[2]), node_bounds=node_bounds, unsupported=unsupported)
        env[str(stmt[1])] = iv.as_tuple()
    elif op == "OP_BLEND_ASSIGN":
        a_iv = eval_expr_interval(env, list(stmt[2]), node_bounds=node_bounds, unsupported=unsupported)
        b_iv = eval_expr_interval(env, list(stmt[3]), node_bounds=node_bounds, unsupported=unsupported)
        old = Interval.from_bounds(env.get(str(stmt[1]), (NINF, INF)))
        env[str(stmt[1])] = old.union(a_iv).union(b_iv).as_tuple()
    elif op == "OP_EXPR_STMT":
        eval_expr_interval(env, list(stmt[1]), node_bounds=node_bounds, unsupported=unsupported)
    elif op == "OP_CONDITIONAL":
        cond_ir = list(stmt[1])
        truth = eval_cond_truth(
            env, cond_ir, node_bounds=node_bounds, unsupported=unsupported
        )
        then_env = dict(env)
        for s in stmt[2]:
            then_env = exec_stmt_interval(
                then_env,
                tuple(s) if isinstance(s, list) else s,
                node_bounds=node_bounds,
                unsupported=unsupported,
            )
        else_env = dict(env)
        for s in stmt[3]:
            else_env = exec_stmt_interval(
                else_env,
                tuple(s) if isinstance(s, list) else s,
                node_bounds=node_bounds,
                unsupported=unsupported,
            )
        if truth is True:
            env = then_env
        elif truth is False:
            env = else_env
        else:
            keys: Set[str] = set(then_env) | set(else_env) | set(env)
            for k in keys:
                t = Interval.from_bounds(then_env.get(k, env.get(k, (NINF, INF))))
                e = Interval.from_bounds(else_env.get(k, env.get(k, (NINF, INF))))
                env[k] = t.union(e).as_tuple()
    elif op == "OP_LOOP":
        unsupported.append("OP_LOOP")
    else:
        unsupported.append(str(op))
    return env


def certify(
    block: InterpretedBlock,
    input_bounds: Mapping[str, Bounds],
    *,
    node_bounds: Optional[Mapping[str, Bounds]] = None,
    source_path: Optional[Union[str, Path]] = None,
) -> Certificate:
    nb = dict(node_bounds or {})
    env: Env = {str(k): (float(v[0]), float(v[1])) for k, v in input_bounds.items()}
    unsupported: List[str] = []
    for stmt in block.ir_stmts:
        st = tuple(stmt) if isinstance(stmt, list) else stmt
        if st[0] == "OP_LOOP":
            unsupported.append("OP_LOOP")
            break
        env = exec_stmt_interval(env, st, node_bounds=nb, unsupported=unsupported)
    path = Path(source_path) if source_path else None
    outputs = {k: env[k] for k in block.abi if k in env}
    status = "unsupported" if unsupported else "ok"
    return Certificate(
        axiom_version=_import_version(),
        bundle_source_hash=_hash_source(path, block),
        input_region={str(k): (float(v[0]), float(v[1])) for k, v in input_bounds.items()},
        assumptions={str(k): (float(v[0]), float(v[1])) for k, v in nb.items()},
        proven_output_bounds=outputs,
        timestamp=datetime.now(timezone.utc).isoformat(),
        unsupported_ops=sorted(set(unsupported)),
        status=status,
    )


__all__ = ["Certificate", "Interval", "certify"]
