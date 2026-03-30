from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from lark import Token, Tree


def _child_trees(t: Tree) -> list:
    return [c for c in t.children if isinstance(c, Tree)]

IRList = List[tuple]


_CMP = {"GT": "OP_CMP_GT", "LT": "OP_CMP_LT", "EQ": "OP_CMP_EQ", "NE": "OP_CMP_NE"}


@dataclass(frozen=True)
class FunctionDef:
    """User function for macro inlining; body ends with ``OP_RETURN`` (MVP: single tail return)."""

    name: str
    params: Tuple[str, ...]
    body: IRList


ExprIR = List[Tuple[Any, ...]]
Stmt = Tuple[Any, ...]


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
    return False


def expand_stmt(stmt: Stmt, funcs: Dict[str, FunctionDef], ctr: List[int]) -> IRList:
    op = stmt[0]
    if op == "OP_ASSIGN":
        h, rhs = expand_expr(list(stmt[2]), funcs, ctr)
        return h + [("OP_ASSIGN", str(stmt[1]), rhs)]
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
        if isinstance(tup, tuple) and tup and tup[0] == "OP_CALL":
            h, repl = _expand_call_op(tup, funcs, ctr)
            hoists.extend(h)
            out.extend(repl)
        else:
            out.append(tup)
    return hoists, out


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
    for st in body[:-1]:
        names |= _names_assigned_in_stmt(tuple(st) if isinstance(st, list) else st)
    return names


def _names_assigned_in_stmt(stmt: Stmt) -> Set[str]:
    op = stmt[0]
    if op == "OP_ASSIGN":
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
        elif op in ("OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV", "OP_INDEX") or (
            isinstance(op, str) and op.startswith("OP_CMP_")
        ):
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
    body = _inner(inner, allow_return=True)
    _validate_fn_body(body)
    return FunctionDef(name=name, params=params, body=body)


def _validate_fn_body(body: IRList) -> None:
    if not body or body[-1][0] != "OP_RETURN":
        raise ValueError("function must end with a return statement")
    if _returns_before_tail(body[:-1]):
        raise ValueError("MVP: only a single return allowed, at end of function body")


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
            arg_irs = tuple(_expr(c) for c in second.children)
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
        elif op in ("OP_ADD", "OP_SUB", "OP_MUL", "OP_DIV"):
            b, a = stack.pop(), stack.pop()
            stack.append(max(a, b))
        elif isinstance(op, str) and op.startswith("OP_CMP_"):
            stack.pop()
            stack.pop()
            stack.append(1)
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

    def walk_stmt(stmt: Stmt) -> None:
        if not isinstance(stmt, tuple) or not stmt:
            return
        op = stmt[0]
        if op == "OP_ASSIGN":
            add_name(str(stmt[1]))
            walk_expr(list(stmt[2]))
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
