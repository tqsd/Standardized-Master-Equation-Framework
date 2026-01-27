from __future__ import annotations

from typing import Sequence

import numpy as np

from smef.core.ir.ops import EmbeddedKron, LocalSymbolOp, OpExpr, OpExprKind, SymbolOp
from smef.core.ir.protocols import OpMaterializeContextProto


def _prod_int(xs: Sequence[int]) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out


def _as_dense(mat: np.ndarray, D: int) -> np.ndarray:
    m = np.asarray(mat, dtype=complex)
    if m.shape != (D, D):
        raise ValueError(f"Expected matrix shape {(D, D)}, got {m.shape}")
    return m


def _kron_all(mats: Sequence[np.ndarray]) -> np.ndarray:
    out = np.asarray(mats[0], dtype=complex)
    for m in mats[1:]:
        out = np.kron(out, np.asarray(m, dtype=complex))
    return out


def materialize_op_expr(
    expr: OpExpr,
    *,
    dims: Sequence[int],
    ctx: OpMaterializeContextProto,
) -> np.ndarray:
    """
    Materialize an OpExpr into a dense (D, D) complex matrix.

    Convention:
    - full space ordering follows dims order: kron(mode0, mode1, ...)
    - EmbeddedKron places local ops on selected indices, identity elsewhere.
    """
    D = _prod_int(dims)

    if expr.kind == OpExprKind.ATOM:
        if expr.atom is None:
            raise ValueError("ATOM expr must have atom set")
        return _materialize_atom(expr.atom, dims=dims, ctx=ctx, D=D)

    if expr.kind == OpExprKind.SCALE:
        if expr.scalar is None or len(expr.args) != 1:
            raise ValueError("SCALE requires scalar and exactly one arg")
        base = materialize_op_expr(expr.args[0], dims=dims, ctx=ctx)
        return complex(expr.scalar) * base

    if expr.kind == OpExprKind.SUM:
        if not expr.args:
            raise ValueError("SUM requires at least one arg")
        acc = np.zeros((D, D), dtype=complex)
        for a in expr.args:
            acc += materialize_op_expr(a, dims=dims, ctx=ctx)
        return acc

    if expr.kind == OpExprKind.PROD:
        if not expr.args:
            raise ValueError("PROD requires at least one arg")
        out = materialize_op_expr(expr.args[0], dims=dims, ctx=ctx)
        for a in expr.args[1:]:
            out = out @ materialize_op_expr(a, dims=dims, ctx=ctx)
        return out

    raise ValueError(f"Unknown OpExprKind: {expr.kind}")


def _materialize_atom(
    atom: object,
    *,
    dims: Sequence[int],
    ctx: OpMaterializeContextProto,
    D: int,
) -> np.ndarray:
    if isinstance(atom, SymbolOp):
        mat = ctx.resolve_symbol(atom.symbol, dims)
        return _as_dense(mat, D)

    if isinstance(atom, EmbeddedKron):
        return _materialize_embedded_kron(atom, dims=dims, ctx=ctx)

    raise TypeError(f"Unsupported atom type: {type(atom)}")


def _materialize_embedded_kron(
    ek: EmbeddedKron,
    *,
    dims: Sequence[int],
    ctx: OpMaterializeContextProto,
) -> np.ndarray:
    n = len(dims)
    if any((i < 0 or i >= n) for i in ek.indices):
        raise IndexError(
            f"EmbeddedKron index out of range: {
                ek.indices} for n={n}"
        )
    if len(set(ek.indices)) != len(ek.indices):
        raise ValueError(f"EmbeddedKron indices must be unique: {ek.indices}")

    # Build per-mode local matrices (default identity).
    locals_by_idx = {int(i): op for i, op in zip(ek.indices, ek.locals)}

    mats = []
    for mode_index, dim in enumerate(dims):
        if mode_index in locals_by_idx:
            lop = locals_by_idx[mode_index]
            if not isinstance(lop, LocalSymbolOp):
                raise TypeError(f"Expected LocalSymbolOp, got {type(lop)}")
            # Convention: local symbols are resolved as full-space by ctx using dims,
            # OR: you can decide ctx will resolve locals separately. For now, we assume
            # local symbols are resolved as (dim, dim) by encoding their target dim.
            # Best practice: materializer should expose a local resolver later.
            mat = ctx.resolve_symbol(lop.symbol, (int(dim),))
            mat = np.asarray(mat, dtype=complex)
            if mat.shape != (int(dim), int(dim)):
                raise ValueError(
                    f"Local op {lop.symbol} expected shape {
                        (int(dim), int(dim))}, got {mat.shape}"
                )
            mats.append(mat)
        else:
            mats.append(np.eye(int(dim), dtype=complex))

    return _kron_all(mats)
