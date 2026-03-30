from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from lark import Token, Tree


def _child_trees(t: Tree) -> list:
    return [c for c in t.children if isinstance(c, Tree)]

IRList = List[tuple]


_CMP = {"GT": "OP_CMP_GT", "LT": "OP_CMP_LT", "EQ": "OP_CMP_EQ", "NE": "OP_CMP_NE"}

# Built-in reducers: compile to OP_REDUCE_* / OP_DOT (not user ``OP_CALL`` / inlining). ``batch_mean`` → batch dim.
RESERVED_REDUCTION_BUILTINS = frozenset({"sum", "mean", "dot", "batch_mean"})
# Element-wise unary math: compile to ``OP_MATH_UNARY`` (same stack shape as input).
RESERVED_MATH_BUILTINS = frozenset({"abs", "exp", "log", "sqrt", "sin", "cos"})
RESERVED_MATH_BINARY = frozenset({"max", "min"})
RESERVED_NEURAL_BUILTIN = "neural"
RESERVED_EXPERT_BUILTIN = "expert"
RESERVED_BUILTIN_NAMES = (
    RESERVED_REDUCTION_BUILTINS
    | RESERVED_MATH_BUILTINS
    | RESERVED_MATH_BINARY
    | {RESERVED_NEURAL_BUILTIN, RESERVED_EXPERT_BUILTIN}
)


@dataclass(frozen=True)
class FunctionDef:
    """User function for macro inlining; body ends with ``OP_RETURN`` (MVP: single tail return)."""

    name: str
    params: Tuple[str, ...]
    body: IRList


ExprIR = List[Any]  # opcode tuples and compile-time fragments like ``StringLiteral``
Stmt = Tuple[Any, ...]


class Expression:
    """Marker base for structured expression fragments (e.g. string literals for built-ins)."""


@dataclass(frozen=True)
class StringLiteral(Expression):
    """Compile-time string (e.g. second argument to ``neural(expr, "kan")``)."""

    value: str


@dataclass(frozen=True)
class ReturnStatement:
    """Structured form of ``return expr`` (lowers to ``("OP_RETURN", expr_ir)``)."""

    value_ir: ExprIR


@dataclass(frozen=True)
class FunctionCall:
    """Structured call (IR embeds ``("OP_CALL", name, arg_irs)`` in ``ExprIR``)."""

    name: str
    args: Tuple[ExprIR, ...]


def parse_program(tree: Tree) -> Tuple[Dict[str, FunctionDef], IRList]:
    """Split top-level ``function_def`` nodes from main statements; main may contain ``OP_CALL``."""
    assert tree.data == "start"
    funcs: Dict[str, FunctionDef] = {}
    main: IRList = []
    for child in tree.children:
        if not isinstance(child, Tree):
            continue
        if child.data == "function_def":
            fd = _function_def_from_tree(child)
            if fd.name in funcs:
                raise ValueError(f"duplicate function {fd.name!r}")
            funcs[fd.name] = fd
        else:
            main.extend(_stmt(child, allow_return=False))
    return funcs, main


def ast_to_ir(tree: Tree) -> IRList:
    funcs, main = parse_program(tree)
    return expand_function_calls(main, funcs)


def expand_function_calls(ir: IRList, funcs: Dict[str, FunctionDef]) -> IRList:
    """Inline ``OP_CALL`` into mangled assignments (static graph; no call stack)."""
    if not funcs and not _ir_contains_op_call(ir):
        return list(ir)
    ctr = [0]
    out: IRList = []
    for st in ir:
        out.extend(expand_stmt(st, funcs, ctr))
    return out


def _ir_contains_op_call(ir: IRList) -> bool:
    for st in ir:
        if _stmt_contains_op_call(st):
            return True
    return False


def _stmt_contains_op_call(stmt: Stmt) -> bool:
    op = stmt[0]
    if op == "OP_ASSIGN":
        return _expr_contains_op_call(list(stmt[2]))
    if op == "OP_BLEND_ASSIGN":
        return _expr_contains_op_call(list(stmt[2])) or _expr_contains_op_call(
            list(stmt[3])
        )
    if op == "OP_EXPR_STMT":
        return _expr_contains_op_call(list(stmt[1]))
    if op == "OP_CONDITIONAL":
        if _expr_contains_op_call(list(stmt[1])):
            return True
        for s in stmt[2]:
            if _stmt_contains_op_call(tuple(s) if isinstance(s, list) else s):
                return True
        for s in stmt[3]:
            if _stmt_contains_op_call(tuple(s) if isinstance(s, list) else s):
                return True
        return False
    if op == "OP_LOOP":
        if _expr_contains_op_call(list(stmt[1])):
            return True
        for s in stmt[2]:
            if _stmt_contains_op_call(tuple(s) if isinstance(s, list) else s):
                return True
        return False
    return False


def _expr_contains_op_call(expr: ExprIR) -> bool:
    for tup in expr:
        if not isinstance(tup, tuple) or not tup:
            continue
        if tup[0] == "OP_CALL":
            return True
        if tup[0] == "OP_NEURAL" and len(tup) > 2 and _expr_contains_op_call(list(tup[2])):
            return True
        if tup[0] == "OP_EXPERT" and len(tup) > 2 and _expr_contains_op_call(list(tup[2])):
            return True
    return False


def expand_stmt(stmt: Stmt, funcs: Dict[str, FunctionDef], ctr: List[int]) -> IRList:
    op = stmt[0]
    if op == "OP_ASSIGN":
        h, rhs = expand_expr(list(stmt[2]), funcs, ctr)
        return h + [("OP_ASSIGN", str(stmt[1]), rhs)]
    if op == "OP_BLEND_ASSIGN":
        h1, air = expand_expr(list(stmt[2]), funcs, ctr)
        h2, vir = expand_expr(list(stmt[3]), funcs, ctr)
        return h1 + h2 + [("OP_BLEND_ASSIGN", str(stmt[1]), air, vir)]
    if op == "OP_EXPR_STMT":
        h, e = expand_expr(list(stmt[1]), funcs, ctr)
        return h + [("OP_EXPR_STMT", e)]
    if op == "OP_CONDITIONAL":
        hc, cond = expand_expr(list(stmt[1]), funcs, ctr)
        then_ir: IRList = []
        for s in stmt[2]:
            then_ir.extend(expand_stmt(tuple(s) if isinstance(s, list) else s, funcs, ctr))
        else_ir: IRList = []
        for s in stmt[3]:
            else_ir.extend(expand_stmt(tuple(s) if isinstance(s, list) else s, funcs, ctr))
        return hc + [("OP_CONDITIONAL", cond, then_ir, else_ir)]
    if op == "OP_LOOP":
        hc, cond = expand_expr(list(stmt[1]), funcs, ctr)
        body_ir: IRList = []
        for s in stmt[2]:
            body_ir.extend(expand_stmt(tuple(s) if isinstance(s, list) else s, funcs, ctr))
        return hc + [("OP_LOOP", cond, body_ir)]
    raise ValueError(f"expand_stmt: unknown {op}")


def expand_expr(expr: ExprIR, funcs: Dict[str, FunctionDef], ctr: List[int]) -> Tuple[IRList, ExprIR]:
    hoists: IRList = []
    out: ExprIR = []
    for tup in expr:
        if isinstance(tup, StringLiteral):
            out.append(tup)
            continue
        if isinstance(tup, tuple) and tup and tup[0] == "OP_NEURAL":
            arch = str(tup[3]) if len(tup) >= 4 else "mlp"
            h, inner = expand_expr(list(tup[2]), funcs, ctr)
            hoists.extend(h)
            out.append(("OP_NEURAL", str(tup[1]), inner, arch))
            continue
        if isinstance(tup, tuple) and tup and tup[0] == "OP_EXPERT":
            h, inner = expand_expr(list(tup[2]), funcs, ctr)
            hoists.extend(h)
            out.append(("OP_EXPERT", str(tup[1]), inner))
            continue
        if isinstance(tup, tuple) and tup and tup[0] == "OP_CALL":
            name = str(tup[1])
            if name in RESERVED_REDUCTION_BUILTINS:
                h, tail = _expand_builtin_reduction_call(tup, funcs, ctr)
                hoists.extend(h)
                out.extend(tail)
                continue
            if name in RESERVED_MATH_BINARY:
                h, tail = _expand_math_binary_call(tup, funcs, ctr)
                hoists.extend(h)
                out.extend(tail)
                continue
            if name == RESERVED_NEURAL_BUILTIN:
                h, tail = _expand_neural_call(tup, funcs, ctr)
                hoists.extend(h)
                out.extend(tail)
                continue
            if name == RESERVED_EXPERT_BUILTIN:
                h, tail = _expand_expert_call(tup, funcs, ctr)
                hoists.extend(h)
                out.extend(tail)
                continue
            if name in RESERVED_MATH_BUILTINS:
                h, tail = _expand_math_builtin_call(tup, funcs, ctr)
                hoists.extend(h)
                out.extend(tail)
                continue
            h, repl = _expand_call_op(tup, funcs, ctr)
            hoists.extend(h)
            out.extend(repl)
        else:
            out.append(tup)
    return hoists, out


def _expand_builtin_reduction_call(
    tup: Tuple[Any, ...], funcs: Dict[str, FunctionDef], ctr: List[int]
) -> Tuple[IRList, ExprIR]:
    """Lower ``OP_CALL`` to reducers when name is ``sum`` / ``mean`` / ``dot`` (not user inline)."""
    name = str(tup[1])
    arg_irs: Tuple[ExprIR, ...] = tuple(tuple(x) for x in tup[2])
    if name == "sum":
        if len(arg_irs) != 1:
            raise ValueError("sum() expects exactly 1 argument")
        h, e = expand_expr(list(arg_irs[0]), funcs, ctr)
        return h, e + [("OP_REDUCE_SUM",)]
    if name == "mean":
        if len(arg_irs) != 1:
            raise ValueError("mean() expects exactly 1 argument")
        h, e = expand_expr(list(arg_irs[0]), funcs, ctr)
        return h, e + [("OP_REDUCE_MEAN",)]
    if name == "dot":
        if len(arg_irs) != 2:
            raise ValueError("dot() expects exactly 2 arguments")
        h0, e0 = expand_expr(list(arg_irs[0]), funcs, ctr)
        h1, e1 = expand_expr(list(arg_irs[1]), funcs, ctr)
        return h0 + h1, e0 + e1 + [("OP_DOT",)]
    if name == "batch_mean":
        if len(arg_irs) != 1:
            raise ValueError("batch_mean() expects exactly 1 argument")
        h, e = expand_expr(list(arg_irs[0]), funcs, ctr)
        return h, e + [("OP_REDUCE_BATCH_MEAN",)]
    raise ValueError(f"unknown built-in {name!r}")


def _expand_math_builtin_call(
    tup: Tuple[Any, ...], funcs: Dict[str, FunctionDef], ctr: List[int]
) -> Tuple[IRList, ExprIR]:
    """Lower ``OP_CALL`` for ``abs`` / ``exp`` / … to ``("OP_MATH_UNARY", name)``."""
    name = str(tup[1])
    arg_irs: Tuple[ExprIR, ...] = tuple(tuple(x) for x in tup[2])
    if len(arg_irs) != 1:
        raise ValueError(f"{name}() expects exactly 1 argument")
    h, e = expand_expr(list(arg_irs[0]), funcs, ctr)
    return h, e + [("OP_MATH_UNARY", name)]


def _expand_math_binary_call(
    tup: Tuple[Any, ...], funcs: Dict[str, FunctionDef], ctr: List[int]
) -> Tuple[IRList, ExprIR]:
    """Lower ``max(a,b)`` / ``min(a,b)`` to postfix ``OP_MATH_BINARY`` (like ``OP_DOT``)."""
    name = str(tup[1])
    arg_irs: Tuple[ExprIR, ...] = tuple(tuple(x) for x in tup[2])
    if len(arg_irs) != 2:
        raise ValueError(f"{name}() expects exactly 2 arguments")
    h0, e0 = expand_expr(list(arg_irs[0]), funcs, ctr)
    h1, e1 = expand_expr(list(arg_irs[1]), funcs, ctr)
    return h0 + h1, e0 + e1 + [("OP_MATH_BINARY", name)]


def _expand_neural_call(
    tup: Tuple[Any, ...], funcs: Dict[str, FunctionDef], ctr: List[int]
) -> Tuple[IRList, ExprIR]:
    arg_irs: Tuple[ExprIR, ...] = tuple(tuple(x) for x in tup[2])
    if len(arg_irs) == 1:
        arch = "mlp"
        feat_ir = arg_irs[0]
    elif len(arg_irs) == 2:
        feat_ir = arg_irs[0]
        s_ir = list(arg_irs[1])
        if len(s_ir) == 1 and isinstance(s_ir[0], StringLiteral):
            arch = str(s_ir[0].value)
        elif (
            len(s_ir) == 1
            and isinstance(s_ir[0], tuple)
            and s_ir[0]
            and s_ir[0][0] == "OP_CONST_STR"
        ):
            arch = str(s_ir[0][1])
        else:
            raise ValueError("neural() second argument must be a string literal")
    else:
        raise ValueError("neural() expects 1 or 2 arguments")
    nid = f"neural_node_{uuid.uuid4().hex[:8]}"
    h, e = expand_expr(list(feat_ir), funcs, ctr)
    return h, [("OP_NEURAL", nid, e, arch)]


def _expand_expert_call(
    tup: Tuple[Any, ...], funcs: Dict[str, FunctionDef], ctr: List[int]
) -> Tuple[IRList, ExprIR]:
    """Lower ``expert("backend", feat_expr)`` → ``("OP_EXPERT", name, feat_ir)`` (scalar float out)."""
    arg_irs: Tuple[ExprIR, ...] = tuple(tuple(x) for x in tup[2])
    if len(arg_irs) != 2:
        raise ValueError("expert() expects exactly 2 arguments")
    s_ir = list(arg_irs[0])
    if len(s_ir) == 1 and isinstance(s_ir[0], StringLiteral):
        backend = str(s_ir[0].value)
    elif (
        len(s_ir) == 1
        and isinstance(s_ir[0], tuple)
        and s_ir[0]
        and s_ir[0][0] == "OP_CONST_STR"
    ):
        backend = str(s_ir[0][1])
    else:
        raise ValueError("expert() first argument must be a string literal")
    feat_ir = arg_irs[1]
    h, e = expand_expr(list(feat_ir), funcs, ctr)
    return h, [("OP_EXPERT", backend, e)]


def _load_e(name: str) -> ExprIR:
    return [("OP_LOAD", name)]


def _truth01_expr(cond_ir: ExprIR) -> ExprIR:
    return list(cond_ir) + [("OP_CONST", 0.0), ("OP_CMP_NE",)]


def _expr_mul_ir(a: ExprIR, b: ExprIR) -> ExprIR:
    return list(a) + list(b) + [("OP_MUL",)]


def _expr_add_ir(a: ExprIR, b: ExprIR) -> ExprIR:
    return list(a) + list(b) + [("OP_ADD",)]


def _one_minus_rd(rd: str) -> ExprIR:
    return [("OP_CONST", 1.0), ("OP_LOAD", rd), ("OP_SUB",)]


def _contrib_expr(pm: str, rd: str) -> ExprIR:
    return _expr_mul_ir(_load_e(pm), _one_minus_rd(rd))


def _zero_expr_for_width(w: int) -> ExprIR:
    w = max(1, int(w))
    if w == 1:
        return [("OP_CONST", 0.0)]
    ir: ExprIR = []
    for _ in range(w):
        ir.append(("OP_CONST", 0.0))
    ir.append(("OP_VEC_PACK", w))
    return ir


def _max_return_width_stmt(stmt: Stmt, known: Dict[str, int]) -> int:
    stmt = tuple(stmt) if isinstance(stmt, list) else stmt
    op = stmt[0]
    if op == "OP_RETURN":
        return _infer_expr_output_width(list(stmt[1]), known)
    if op == "OP_CONDITIONAL":
        return max(
            _max_return_width_body(stmt[2], known),
            _max_return_width_body(stmt[3], known),
        )
    if op == "OP_LOOP":
        return _max_return_width_body(stmt[2], known)
    return 1


def _max_return_width_body(body: IRList, known: Dict[str, int]) -> int:
    w = 1
    for st in body:
        w = max(w, _max_return_width_stmt(tuple(st) if isinstance(st, list) else st, known))
    return w


def _return_inside_any_loop(body: IRList) -> bool:
    for st in body:
        if _stmt_loop_contains_return(tuple(st) if isinstance(st, list) else st):
            return True
    return False


def _stmt_loop_contains_return(stmt: Stmt) -> bool:
    stmt = tuple(stmt) if isinstance(stmt, list) else stmt
    op = stmt[0]
    if op == "OP_LOOP":
        return any(_stmt_contains_return_deep(s) for s in stmt[2])
    if op == "OP_CONDITIONAL":
        return any(_stmt_loop_contains_return(s) for s in stmt[2]) or any(
            _stmt_loop_contains_return(s) for s in stmt[3]
        )
    return False


def _body_contains_any_return(body: IRList) -> bool:
    for st in body:
        if _stmt_contains_return_deep(tuple(st) if isinstance(st, list) else st):
            return True
    return False


def _stmt_contains_return_deep(stmt: Stmt) -> bool:
    stmt = tuple(stmt) if isinstance(stmt, list) else stmt
    op = stmt[0]
    if op == "OP_RETURN":
        return True
    if op == "OP_CONDITIONAL":
        return any(_stmt_contains_return_deep(s) for s in stmt[2]) or any(
            _stmt_contains_return_deep(s) for s in stmt[3]
        )
    if op == "OP_LOOP":
        return any(_stmt_contains_return_deep(s) for s in stmt[2])
    return False


def _function_needs_masked_returns(body: IRList) -> bool:
    if not body:
        return False
    last = tuple(body[-1]) if isinstance(body[-1], list) else body[-1]
    if last[0] != "OP_RETURN":
        return True
    return _returns_before_tail(body[:-1])


def _inline_fn_emit_return(
    stmt: Stmt,
    prefix: str,
    pm: str,
    rd: str,
    ra: str,
    mp: Dict[str, str],
    funcs: Dict[str, FunctionDef],
    ctr: List[int],
) -> IRList:
    ret_ir = _mangle_expr(list(stmt[1]), mp)
    hret, rv = expand_expr(ret_ir, funcs, ctr)
    ct = f"{prefix}__ct"
    out: IRList = list(hret)
    out.append(("OP_ASSIGN", ct, _contrib_expr(pm, rd)))
    out.append(
        (
            "OP_ASSIGN",
            ra,
            _expr_add_ir(_load_e(ra), _expr_mul_ir(_load_e(ct), rv)),
        )
    )
    out.append(
        (
            "OP_ASSIGN",
            rd,
            _expr_add_ir(_load_e(rd), _expr_mul_ir(_load_e(ct), _one_minus_rd(rd))),
        )
    )
    return out


def _inline_fn_emit_blend_assign(
    stmt: Stmt, pm: str, rd: str, mp: Dict[str, str], funcs: Dict[str, FunctionDef], ctr: List[int]
) -> IRList:
    k = str(stmt[1])
    rhs = _mangle_expr(list(stmt[2]), mp)
    h, vir = expand_expr(rhs, funcs, ctr)
    return h + [
        ("OP_BLEND_ASSIGN", k, _contrib_expr(pm, rd), vir),
    ]


def _inline_fn_emit_conditional(
    stmt: Stmt,
    prefix: str,
    mp: Dict[str, str],
    pm: str,
    rd: str,
    ra: str,
    save_ctr: List[int],
    funcs: Dict[str, FunctionDef],
    ctr: List[int],
) -> IRList:
    save_ctr[0] += 1
    sav = f"{prefix}__savpm_{save_ctr[0]}"
    cond_ir = _mangle_expr(list(stmt[1]), mp)
    then_body = [_mangle_stmt(s, mp) for s in stmt[2]]
    else_body = [_mangle_stmt(s, mp) for s in stmt[3]]
    then_ir: IRList = [
        ("OP_ASSIGN", sav, _load_e(pm)),
        ("OP_ASSIGN", pm, _expr_mul_ir(_load_e(sav), _truth01_expr(cond_ir))),
    ]
    then_ir.extend(
        _inline_function_body_flat(then_body, prefix, mp, pm, rd, ra, save_ctr, funcs, ctr)
    )
    then_ir.append(("OP_ASSIGN", pm, _load_e(sav)))
    else_ir: IRList = [
        ("OP_ASSIGN", sav, _load_e(pm)),
        (
            "OP_ASSIGN",
            pm,
            _expr_mul_ir(
                _load_e(sav),
                _expr_add_ir(
                    [("OP_CONST", 1.0)],
                    _expr_mul_ir([("OP_CONST", -1.0)], _truth01_expr(cond_ir)),
                ),
            ),
        ),
    ]
    else_ir.extend(
        _inline_function_body_flat(else_body, prefix, mp, pm, rd, ra, save_ctr, funcs, ctr)
    )
    else_ir.append(("OP_ASSIGN", pm, _load_e(sav)))
    return [("OP_CONDITIONAL", list(cond_ir), then_ir, else_ir)]


def _inline_function_body_flat(
    body: IRList,
    prefix: str,
    mp: Dict[str, str],
    pm: str,
    rd: str,
    ra: str,
    save_ctr: List[int],
    funcs: Dict[str, FunctionDef],
    ctr: List[int],
) -> IRList:
    acc: IRList = []
    for st in body:
        st = tuple(st) if isinstance(st, list) else st
        op = st[0]
        if op == "OP_RETURN":
            acc.extend(_inline_fn_emit_return(st, prefix, pm, rd, ra, mp, funcs, ctr))
        elif op == "OP_ASSIGN":
            acc.extend(_inline_fn_emit_blend_assign(st, pm, rd, mp, funcs, ctr))
        elif op == "OP_EXPR_STMT":
            acc.extend(expand_stmt(st, funcs, ctr))
        elif op == "OP_CONDITIONAL":
            acc.extend(
                _inline_fn_emit_conditional(st, prefix, mp, pm, rd, ra, save_ctr, funcs, ctr)
            )
        elif op == "OP_LOOP":
            if any(_stmt_contains_return_deep(s) for s in st[2]):
                raise ValueError(
                    "return inside while (user function) is not supported with early return"
                )
            acc.extend(expand_stmt(st, funcs, ctr))
        else:
            raise ValueError(f"unsupported statement in inlined function: {op}")
    return acc


def _expand_call_op(
    tup: Tuple[Any, ...], funcs: Dict[str, FunctionDef], ctr: List[int]
) -> Tuple[IRList, ExprIR]:
    name = str(tup[1])
    arg_irs: Tuple[ExprIR, ...] = tuple(tuple(x) for x in tup[2])
    if name not in funcs:
        raise ValueError(f"undefined function {name!r}")
    fd = funcs[name]
    if len(arg_irs) != len(fd.params):
        raise ValueError(f"{name}: expected {len(fd.params)} args, got {len(arg_irs)}")
    pre: IRList = []
    cid = ctr[0]
    ctr[0] += 1
    prefix = f"_inline_{name}_{cid}_"
    locals_ = _function_locals(fd.body, fd.params)
    mp = {n: prefix + n for n in locals_}
    arg_tmps: List[str] = []
    for i, air in enumerate(arg_irs):
        h_sub, eir = expand_expr(list(air), funcs, ctr)
        pre.extend(h_sub)
        an = f"{prefix}__arg{i}"
        pre.append(("OP_ASSIGN", an, eir))
        arg_tmps.append(an)
    for pname, an in zip(fd.params, arg_tmps):
        pre.append(("OP_ASSIGN", mp[pname], [("OP_LOAD", an)]))
    body = fd.body
    if _function_needs_masked_returns(body):
        pm, rd, ra = f"{prefix}__pm", f"{prefix}__rd", f"{prefix}__ra"
        pw = {mp[p]: 1 for p in fd.params}
        rw = max(1, _max_return_width_body(body, pw))
        pre.append(("OP_ASSIGN", pm, [("OP_CONST", 1.0)]))
        pre.append(("OP_ASSIGN", rd, [("OP_CONST", 0.0)]))
        pre.append(("OP_ASSIGN", ra, _zero_expr_for_width(rw)))
        mangled = [_mangle_stmt(tuple(s) if isinstance(s, list) else s, mp) for s in body]
        save_ctr = [0]
        pre.extend(
            _inline_function_body_flat(
                mangled, prefix, mp, pm, rd, ra, save_ctr, funcs, ctr
            )
        )
        return pre, [("OP_LOAD", ra)]
    assert body[-1][0] == "OP_RETURN"
    for st in body[:-1]:
        mst = _mangle_stmt(tuple(st) if isinstance(st, list) else st, mp)
        pre.extend(expand_stmt(mst, funcs, ctr))
    ret_ir = list(body[-1][1])
    ret_ir = _mangle_expr(ret_ir, mp)
    hret, re = expand_expr(ret_ir, funcs, ctr)
    pre.extend(hret)
    res = f"{prefix}ret"
    pre.append(("OP_ASSIGN", res, re))
    return pre, [("OP_LOAD", res)]


def _function_locals(body: IRList, params: Tuple[str, ...]) -> Set[str]:
    names: Set[str] = set(params)
    for st in body:
        st = tuple(st) if isinstance(st, list) else st
        if st[0] == "OP_RETURN":
            continue
        names |= _names_assigned_in_stmt(st)
    return names


def _names_assigned_in_stmt(stmt: Stmt) -> Set[str]:
    op = stmt[0]
    if op == "OP_ASSIGN":
        return {str(stmt[1])}
    if op == "OP_BLEND_ASSIGN":
        return {str(stmt[1])}
    if op == "OP_CONDITIONAL":
        s: Set[str] = set()
        for x in stmt[2]:
            s |= _names_assigned_in_stmt(tuple(x) if isinstance(x, list) else x)
        for x in stmt[3]:
            s |= _names_assigned_in_stmt(tuple(x) if isinstance(x, list) else x)
        return s
    if op == "OP_LOOP":
        s = set()
        for x in stmt[2]:
            s |= _names_assigned_in_stmt(tuple(x) if isinstance(x, list) else x)
        return s
    return set()


def _mangle_stmt(stmt: Stmt, mp: Dict[str, str]) -> Stmt:
    op = stmt[0]
    if op == "OP_ASSIGN":
        k = str(stmt[1])
        return ("OP_ASSIGN", mp.get(k, k), _mangle_expr(list(stmt[2]), mp))
    if op == "OP_BLEND_ASSIGN":
        k = str(stmt[1])
        return (
            "OP_BLEND_ASSIGN",
            mp.get(k, k),
            _mangle_expr(list(stmt[2]), mp),
            _mangle_expr(list(stmt[3]), mp),
        )
    if op == "OP_EXPR_STMT":
        return ("OP_EXPR_STMT", _mangle_expr(list(stmt[1]), mp))
    if op == "OP_CONDITIONAL":
        return (
            "OP_CONDITIONAL",
            _mangle_expr(list(stmt[1]), mp),
            [_mangle_stmt(tuple(x) if isinstance(x, list) else x, mp) for x in stmt[2]],
            [_mangle_stmt(tuple(x) if isinstance(x, list) else x, mp) for x in stmt[3]],
        )
    if op == "OP_LOOP":
        return (
            "OP_LOOP",
            _mangle_expr(list(stmt[1]), mp),
            [_mangle_stmt(tuple(x) if isinstance(x, list) else x, mp) for x in stmt[2]],
        )
    if op == "OP_RETURN":
        return ("OP_RETURN", _mangle_expr(list(stmt[1]), mp))
    raise ValueError(f"_mangle_stmt: unknown {op}")


def _mangle_expr(expr: ExprIR, mp: Dict[str, str]) -> ExprIR:
    out: ExprIR = []
    for tup in expr:
        if not isinstance(tup, tuple) or not tup:
            out.append(tup)
            continue
        op = tup[0]
        if op == "OP_LOAD" and len(tup) > 1:
            k = str(tup[1])
            out.append(("OP_LOAD", mp.get(k, k)))
        elif op == "OP_CALL":
            args = tuple(_mangle_expr(list(a), mp) for a in tup[2])
            out.append(("OP_CALL", str(tup[1]), args))
        elif op == "OP_CONST":
            out.append(tup)
        elif op == "OP_NEG":
            out.append(tup)
        elif op == "OP_VEC_PACK":
            out.append(tup)
        elif op == "OP_NEURAL" and len(tup) >= 3:
            arch = str(tup[3]) if len(tup) >= 4 else "mlp"
            out.append(("OP_NEURAL", str(tup[1]), _mangle_expr(list(tup[2]), mp), arch))
        elif op == "OP_EXPERT" and len(tup) >= 3:
            out.append(("OP_EXPERT", str(tup[1]), _mangle_expr(list(tup[2]), mp)))
        elif op in (
            "OP_ADD",
            "OP_SUB",
            "OP_MUL",
            "OP_DIV",
            "OP_INDEX",
            "OP_REDUCE_SUM",
            "OP_REDUCE_MEAN",
            "OP_REDUCE_BATCH_MEAN",
            "OP_DOT",
            "OP_MATH_UNARY",
            "OP_MATH_BINARY",
        ) or (isinstance(op, str) and op.startswith("OP_CMP_")):
            out.append(tup)
        else:
            raise ValueError(f"_mangle_expr: unknown {op}")
    return out


def _function_def_from_tree(t: Tree) -> FunctionDef:
    assert t.data == "function_def"
    ch = list(t.children)
    name = str(ch[0])
    if len(ch) == 2:
        params = ()
        inner = ch[1]
    elif len(ch) == 3:
        mid, inner = ch[1], ch[2]
        if mid is None:
            params = ()
        elif isinstance(mid, Tree) and mid.data == "param_list":
            params = tuple(str(x) for x in mid.children)
        else:
            raise ValueError("malformed function_def")
    else:
        raise ValueError("malformed function_def")
    if not isinstance(inner, Tree) or inner.data != "inner":
        raise ValueError("malformed function_def")
    if name in RESERVED_BUILTIN_NAMES:
        raise ValueError(f"cannot define function {name!r} — reserved built-in")
    body = _inner(inner, allow_return=True)
    _validate_fn_body(body)
    return FunctionDef(name=name, params=params, body=body)


def _validate_fn_body(body: IRList) -> None:
    if not _body_contains_any_return(body):
        raise ValueError("function body must contain at least one return statement")
    if _function_needs_masked_returns(body) and _return_inside_any_loop(body):
        raise ValueError("return inside while (in user function) is not supported yet")


def _returns_before_tail(stmts: IRList) -> bool:
    for st in stmts:
        if _stmt_has_return(tuple(st) if isinstance(st, list) else st):
            return True
    return False


def _stmt_has_return(stmt: Stmt) -> bool:
    op = stmt[0]
    if op == "OP_RETURN":
        return True
    if op == "OP_CONDITIONAL":
        for x in stmt[2]:
            if _stmt_has_return(tuple(x) if isinstance(x, list) else x):
                return True
        for x in stmt[3]:
            if _stmt_has_return(tuple(x) if isinstance(x, list) else x):
                return True
    if op == "OP_LOOP":
        for x in stmt[2]:
            if _stmt_has_return(tuple(x) if isinstance(x, list) else x):
                return True
    return False


def _function_name_from_postfix(t: Tree) -> str:
    if t.data == "postfix_expr" and len(t.children) == 1:
        t = t.children[0]
    if t.data != "atom":
        raise ValueError("only direct name calls are supported (e.g. add(1, 2))")
    ch = t.children
    if len(ch) != 1 or not isinstance(ch[0], Token) or ch[0].type != "NAME":
        raise ValueError("call target must be a simple identifier")
    return str(ch[0])


def ir_to_digraph(ir: IRList) -> nx.DiGraph:
    """Linear opcode sequence as a chain of nodes (lightweight view; see `compiler.flow.wire_execution_graph` for module DAG)."""
    G = nx.DiGraph()
    for i, instr in enumerate(ir):
        op = instr[0] if isinstance(instr, tuple) else instr
        G.add_node(i, op=op)
        if i:
            G.add_edge(i - 1, i)
    return G


def _stmt(t: Tree, *, allow_return: bool = False) -> IRList:
    if t.data == "return_stmt":
        if not allow_return:
            raise ValueError("return outside function")
        return [("OP_RETURN", _expr(t.children[0]))]
    if t.data == "assign_stmt":
        name = str(t.children[0])
        return [("OP_ASSIGN", name, _expr(t.children[1]))]
    if t.data == "expr_stmt":
        return [("OP_EXPR_STMT", _expr(t.children[0]))]
    if t.data == "if_stmt":
        cond = _expr(t.children[0])
        then_ir = _inner(t.children[1], allow_return=allow_return)
        if len(t.children) > 2:
            eb = t.children[2]
            assert eb.data == "else_block"
            else_ir = _inner(eb.children[0], allow_return=allow_return)
        else:
            else_ir = []
        return [("OP_CONDITIONAL", cond, then_ir, else_ir)]
    if t.data == "while_stmt":
        trees = _child_trees(t)
        cond = _expr(trees[0])
        body_ir = _inner(trees[1], allow_return=allow_return)
        return [("OP_LOOP", cond, body_ir)]
    raise ValueError(f"unknown statement {t.data}")


def _inner(t: Tree, *, allow_return: bool = False) -> IRList:
    assert t.data == "inner"
    acc: IRList = []
    for c in t.children:
        acc.extend(_stmt(c, allow_return=allow_return))
    return acc


def _expr(t: Tree) -> List[tuple]:
    if t.data == "comparison":
        return _comparison(t)
    if t.data == "sum":
        return _sum(t)
    if t.data == "product":
        return _product(t)
    if t.data == "postfix_expr":
        return _postfix_expr(t)
    if t.data == "atom":
        return _atom(t)
    raise ValueError(f"unknown expr {t.data}")


def _comparison(t: Tree) -> List[tuple]:
    kids = t.children
    if len(kids) == 1:
        return _expr(kids[0])
    left, op_tree, right = kids[0], kids[1], kids[2]
    return _cmp_operand(left) + _cmp_operand(right) + [(_CMP[str(op_tree.children[0].type)],)]


def _cmp_operand(t: Tree) -> List[tuple]:
    if t.data == "atom":
        return _atom(t)
    if t.data == "postfix_expr":
        return _postfix_expr(t)
    return _expr(t)


def _sum(t: Tree) -> List[tuple]:
    terms = t.children[0::2]
    ops = t.children[1::2]
    acc = _mul_group(terms[0])
    for i, op in enumerate(ops):
        acc += _mul_group(terms[i + 1])
        acc += [("OP_ADD",) if op.type == "ADD" else ("OP_SUB",)]
    return acc


def _mul_group(t: Tree) -> List[tuple]:
    if t.data == "product":
        return _product(t)
    if t.data == "postfix_expr":
        return _postfix_expr(t)
    if t.data == "atom":
        return _atom(t)
    raise ValueError(f"expected product|postfix_expr|atom under sum, got {t.data}")


def _product(t: Tree) -> List[tuple]:
    terms = t.children[0::2]
    ops = t.children[1::2]
    acc = _postfix_expr(terms[0])
    for i, op in enumerate(ops):
        acc += _postfix_expr(terms[i + 1])
        acc += [("OP_MUL",) if op.type == "MUL" else ("OP_DIV",)]
    return acc


def _string_literal_from_expr_tree(t: Tree) -> str:
    """Used for ``neural(expr, "arch")``: second argument must be a string literal."""
    ir = _expr(t)
    if len(ir) == 1 and isinstance(ir[0], StringLiteral):
        return ir[0].value
    raise ValueError("expected a string literal")


def _postfix_expr(t: Tree) -> List[tuple]:
    if t.data == "atom":
        return _atom(t)
    if t.data == "postfix_expr":
        ch = t.children
        if len(ch) == 1:
            c0 = ch[0]
            if not isinstance(c0, Tree):
                raise ValueError("postfix_expr expects Tree child")
            if c0.data == "atom":
                return _atom(c0)
            return _postfix_expr(c0)
        base, second = ch[0], ch[1]
        if isinstance(second, Tree) and second.data == "call_args":
            fname = _function_name_from_postfix(base)
            args = list(second.children)
            if fname == "sum":
                if len(args) != 1:
                    raise ValueError("sum() expects exactly 1 argument")
                return _expr(args[0]) + [("OP_REDUCE_SUM",)]
            if fname == "mean":
                if len(args) != 1:
                    raise ValueError("mean() expects exactly 1 argument")
                return _expr(args[0]) + [("OP_REDUCE_MEAN",)]
            if fname == "batch_mean":
                if len(args) != 1:
                    raise ValueError("batch_mean() expects exactly 1 argument")
                return _expr(args[0]) + [("OP_REDUCE_BATCH_MEAN",)]
            if fname == "dot":
                if len(args) != 2:
                    raise ValueError("dot() expects exactly 2 arguments")
                return _expr(args[0]) + _expr(args[1]) + [("OP_DOT",)]
            if fname in RESERVED_MATH_BINARY:
                if len(args) != 2:
                    raise ValueError(f"{fname}() expects exactly 2 arguments")
                return _expr(args[0]) + _expr(args[1]) + [("OP_MATH_BINARY", fname)]
            if fname == RESERVED_NEURAL_BUILTIN:
                if len(args) == 1:
                    arch = "mlp"
                    feat_t = args[0]
                elif len(args) == 2:
                    feat_t = args[0]
                    arch = _string_literal_from_expr_tree(args[1])
                else:
                    raise ValueError("neural() expects 1 or 2 arguments")
                nid = f"neural_node_{uuid.uuid4().hex[:8]}"
                return [("OP_NEURAL", nid, _expr(feat_t), arch)]
            if fname == RESERVED_EXPERT_BUILTIN:
                if len(args) != 2:
                    raise ValueError("expert() expects exactly 2 arguments")
                backend = _string_literal_from_expr_tree(args[0])
                return [("OP_EXPERT", backend, _expr(args[1]))]
            if fname in RESERVED_MATH_BUILTINS:
                if len(args) != 1:
                    raise ValueError(f"{fname}() expects exactly 1 argument")
                return _expr(args[0]) + [("OP_MATH_UNARY", fname)]
            arg_irs = tuple(_expr(c) for c in args)
            return [("OP_CALL", fname, arg_irs)]
        return _postfix_expr(base) + _expr(second) + [("OP_INDEX",)]
    raise ValueError(f"unknown postfix_expr {t.data}")


def _array_literal(t: Tree) -> List[tuple]:
    if not t.children:
        raise ValueError("empty array literal is not allowed")
    ir: List[tuple] = []
    for c in t.children:
        ir += _expr(c)
    ir.append(("OP_VEC_PACK", len(t.children)))
    return ir


def _string_tree_to_literal(t: Tree) -> List[Any]:
    assert t.data == "string"
    tok = t.children[0]
    raw = str(tok)
    if len(raw) >= 2 and raw[0] in "\"'" and raw[0] == raw[-1]:
        return [StringLiteral(raw[1:-1])]
    return [StringLiteral(raw)]


def _atom(t: Tree) -> List[tuple]:
    ch = t.children
    if len(ch) == 1:
        x = ch[0]
        if isinstance(x, Token):
            if x.type == "NUMBER":
                return [("OP_CONST", _number(x))]
            if x.type == "NAME":
                return [("OP_LOAD", str(x))]
        if isinstance(x, Tree):
            if x.data == "atom":
                return _atom(x) + [("OP_NEG",)]
            if x.data == "array_literal":
                return _array_literal(x)
            if x.data == "string":
                return _string_tree_to_literal(x)
            if x.data == "comparison":
                return _comparison(x)
            if x.data == "expr":
                return _expr(x.children[0])
            if x.data in ("sum", "product", "comparison", "postfix_expr"):
                return _expr(x)
    raise ValueError(f"bad atom {ch!r}")


def _number(tok: Token) -> int | float:
    s = str(tok)
    if "." in s or "e" in s.lower():
        return float(s)
    return int(s)


def _infer_expr_output_width(expr: ExprIR, known: Dict[str, int]) -> int:
    """Infer trailing tensor width (1 = scalar/(B,), K = (B,K)) from stack IR + current name widths."""
    stack: List[int] = []
    for tup in expr:
        if not isinstance(tup, tuple) or not tup:
            continue
        op = tup[0]
        if op == "OP_CONST":
            stack.append(1)
        elif op == "OP_LOAD":
            stack.append(max(1, int(known.get(str(tup[1]), 1))))
        elif op == "OP_NEG":
            stack.append(stack.pop())
        elif op == "OP_VEC_PACK":
            n = int(tup[1])
            for _ in range(n):
                stack.pop()
            stack.append(n)
        elif op == "OP_INDEX":
            stack.pop()
            stack.pop()
            stack.append(1)
        elif op in ("OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_MATH_BINARY"):
            b, a = stack.pop(), stack.pop()
            stack.append(max(a, b))
        elif isinstance(op, str) and op.startswith("OP_CMP_"):
            stack.pop()
            stack.pop()
            stack.append(1)
        elif op in ("OP_REDUCE_SUM", "OP_REDUCE_MEAN"):
            stack.pop()
            stack.append(1)
        elif op == "OP_REDUCE_BATCH_MEAN":
            w = stack.pop()
            stack.append(w)
        elif op == "OP_DOT":
            stack.pop()
            stack.pop()
            stack.append(1)
        elif op == "OP_MATH_UNARY":
            stack.append(stack.pop())
        elif op == "OP_NEURAL" and len(tup) >= 3:
            _infer_expr_output_width(list(tup[2]), known)
            stack.append(1)
        elif op == "OP_EXPERT" and len(tup) >= 3:
            _infer_expr_output_width(list(tup[2]), known)
            stack.append(1)
        elif op == "OP_CALL":
            raise ValueError("OP_CALL in width inference — expand calls before ABI layout")
        else:
            raise ValueError(f"unknown expr op for width inference: {op}")
    return stack[-1] if stack else 1


def _accum_widths_from_stmt(stmt: Stmt, widths: Dict[str, int]) -> None:
    if not isinstance(stmt, tuple) or not stmt:
        return
    op = stmt[0]
    if op == "OP_ASSIGN":
        nm = str(stmt[1])
        rhs_w = _infer_expr_output_width(list(stmt[2]), widths)
        widths[nm] = max(widths.get(nm, 1), rhs_w)
    elif op == "OP_BLEND_ASSIGN":
        nm = str(stmt[1])
        rhs_w = _infer_expr_output_width(list(stmt[3]), widths)
        widths[nm] = max(widths.get(nm, 1), rhs_w)
    elif op == "OP_CONDITIONAL":
        snap = dict(widths)
        wt = dict(widths)
        for s in stmt[2]:
            _accum_widths_from_stmt(tuple(s) if isinstance(s, list) else s, wt)
        we = dict(widths)
        for s in stmt[3]:
            _accum_widths_from_stmt(tuple(s) if isinstance(s, list) else s, we)
        for k in set(wt) | set(we):
            widths[k] = max(snap.get(k, 1), wt.get(k, 1), we.get(k, 1))
    elif op == "OP_LOOP":
        for _ in range(64):
            before = dict(widths)
            for s in stmt[2]:
                _accum_widths_from_stmt(tuple(s) if isinstance(s, list) else s, widths)
            if widths == before:
                break


def extract_abi_layout(
    ir: IRList, *, max_vars: Optional[int] = None
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Map each variable name to a **start** trunk column and optional width (default 1).

    Vector assignments (``OP_VEC_PACK``) reserve ``width`` consecutive columns for that name.
    """
    widths: Dict[str, int] = {}
    for instr in ir:
        _accum_widths_from_stmt(tuple(instr) if isinstance(instr, list) else instr, widths)

    order: List[str] = []
    seen: Set[str] = set()

    def add_name(name: str) -> bool:
        n = str(name)
        if n in seen:
            return True
        seen.add(n)
        order.append(n)
        return True

    def walk_expr(expr: ExprIR) -> None:
        for tup in expr:
            if not isinstance(tup, tuple) or not tup:
                continue
            if tup[0] == "OP_LOAD" and len(tup) > 1:
                add_name(str(tup[1]))
            elif tup[0] == "OP_NEURAL" and len(tup) >= 3:
                walk_expr(list(tup[2]))
            elif tup[0] == "OP_EXPERT" and len(tup) >= 3:
                walk_expr(list(tup[2]))
            elif tup[0] == "OP_CALL":
                for a in tup[2]:
                    walk_expr(list(a))

    def walk_stmt(stmt: Stmt) -> None:
        if not isinstance(stmt, tuple) or not stmt:
            return
        op = stmt[0]
        if op == "OP_ASSIGN":
            add_name(str(stmt[1]))
            walk_expr(list(stmt[2]))
        elif op == "OP_BLEND_ASSIGN":
            add_name(str(stmt[1]))
            walk_expr(list(stmt[2]))
            walk_expr(list(stmt[3]))
        elif op == "OP_EXPR_STMT":
            walk_expr(list(stmt[1]))
        elif op == "OP_CONDITIONAL":
            walk_expr(list(stmt[1]))
            for s in stmt[2]:
                walk_stmt(tuple(s) if isinstance(s, list) else s)
            for s in stmt[3]:
                walk_stmt(tuple(s) if isinstance(s, list) else s)
        elif op == "OP_LOOP":
            walk_expr(list(stmt[1]))
            for s in stmt[2]:
                walk_stmt(tuple(s) if isinstance(s, list) else s)

    for instr in ir:
        walk_stmt(tuple(instr) if isinstance(instr, list) else instr)

    starts: Dict[str, int] = {}
    col = 0
    for name in order:
        w = max(1, int(widths.get(name, 1)))
        if max_vars is not None and col + w > max_vars:
            break
        starts[name] = col
        col += w
    return starts, widths


def extract_global_abi(ir: IRList, *, max_vars: Optional[int] = None) -> Dict[str, int]:
    """Variable name → **start** column (width may be >1; see ``extract_abi_layout``)."""
    starts, _ = extract_abi_layout(ir, max_vars=max_vars)
    return starts


def extract_abi_widths(ir: IRList, *, max_vars: Optional[int] = None) -> Dict[str, int]:
    """Per-name trunk column span (1 for scalars)."""
    starts, widths = extract_abi_layout(ir, max_vars=max_vars)
    return {n: max(1, int(widths.get(n, 1))) for n in starts}


def extract_neural_node_specs(ir: IRList, known_widths: Dict[str, int]) -> Dict[str, Tuple[int, str]]:
    """``node_id`` → ``(feature_width, arch_type)`` for each ``OP_NEURAL`` (``arch_type`` defaults to ``mlp``)."""
    out: Dict[str, Tuple[int, str]] = {}

    def walk_expr(expr: ExprIR) -> None:
        for tup in expr:
            if not isinstance(tup, tuple) or not tup:
                continue
            if tup[0] == "OP_NEURAL" and len(tup) >= 3:
                walk_expr(list(tup[2]))
                nid = str(tup[1])
                arch = str(tup[3]) if len(tup) >= 4 else "mlp"
                out[nid] = (
                    max(1, int(_infer_expr_output_width(list(tup[2]), known_widths))),
                    arch,
                )
            elif tup[0] == "OP_EXPERT" and len(tup) >= 3:
                walk_expr(list(tup[2]))
            elif tup[0] == "OP_CALL":
                for a in tup[2]:
                    walk_expr(list(a))

    def walk_stmt(stmt: Stmt) -> None:
        if not isinstance(stmt, tuple) or not stmt:
            return
        op = stmt[0]
        if op == "OP_ASSIGN":
            walk_expr(list(stmt[2]))
        elif op == "OP_BLEND_ASSIGN":
            walk_expr(list(stmt[2]))
            walk_expr(list(stmt[3]))
        elif op == "OP_EXPR_STMT":
            walk_expr(list(stmt[1]))
        elif op == "OP_CONDITIONAL":
            walk_expr(list(stmt[1]))
            for s in stmt[2]:
                walk_stmt(tuple(s) if isinstance(s, list) else s)
            for s in stmt[3]:
                walk_stmt(tuple(s) if isinstance(s, list) else s)
        elif op == "OP_LOOP":
            walk_expr(list(stmt[1]))
            for s in stmt[2]:
                walk_stmt(tuple(s) if isinstance(s, list) else s)

    for instr in ir:
        walk_stmt(tuple(instr) if isinstance(instr, list) else instr)
    return out
