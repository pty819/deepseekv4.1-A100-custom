"""Entropy / conditional-entropy estimators used by the FP4 compressibility study.

Everything is reported in bits per symbol (a symbol is one FP4 nibble = one weight, unless stated).

Two numbers are always produced for a conditional model:
  * H_plugin   -- the plug-in (maximum-likelihood) conditional entropy on all samples.  This is the
                  optimistic "theoretical limit" and is biased DOWN by about (K-1)/(2 N ln2) bits per
                  context, so with many contexts it lies about how much is really there.
  * H_test     -- cross entropy of a held-out half under KT(+1/2)-smoothed counts estimated on the
                  other half.  This is what a real two-pass coder with a shipped table would spend
                  (excluding the table itself), and it does not reward overfitting.
Plus the cost of shipping the tables, reported separately in bits/weight for two amortizations.
"""
from __future__ import annotations

import numpy as np

K = 16  # FP4 nibble alphabet


def h(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def order0(sym: np.ndarray, k: int = K) -> float:
    c = np.bincount(sym.ravel(), minlength=k).astype(np.float64)
    return h(c / c.sum())


def counts(ctx: np.ndarray, sym: np.ndarray, nctx: int, k: int = K) -> np.ndarray:
    idx = ctx.astype(np.int64) * k + sym
    return np.bincount(idx, minlength=nctx * k).reshape(nctx, k).astype(np.float64)


def cond(ctx: np.ndarray, sym: np.ndarray, nctx: int, train: np.ndarray | None = None,
         k: int = K, alpha: float = 0.5) -> dict:
    """Conditional entropy of sym given ctx (ids in [0, nctx)).

    train: boolean mask selecting the estimation half; the complement is used for the held-out
    cross entropy.  Returns bits/symbol plus table-size bookkeeping.
    """
    ca = counts(ctx, sym, nctx, k)
    tot = ca.sum(1)
    used = int((tot > 0).sum())
    p = ca / np.maximum(tot, 1)[:, None]
    hp = np.array([h(row) for row in p])
    n = tot.sum()
    H_plugin = float((tot * hp).sum() / n)
    out = {"H_plugin": H_plugin, "n_ctx": nctx, "n_ctx_used": used, "n_samples": float(n)}
    # first-order bias of the plug-in estimate (Miller-Madow): +(K_used-1)/(2 N ln2) per context
    kused = (ca > 0).sum(1)
    out["H_bias_corrected"] = H_plugin + float((np.maximum(kused - 1, 0)).sum() / (2 * n * np.log(2)))
    if train is not None:
        ctr = counts(ctx[train], sym[train], nctx, k)
        cte = counts(ctx[~train], sym[~train], nctx, k)
        q = (ctr + alpha) / (ctr.sum(1, keepdims=True) + alpha * k)
        out["H_test"] = float(-(cte * np.log2(q)).sum() / cte.sum())
    # table cost: 12-bit quantized frequencies, k-1 per used context
    out["table_bits"] = float(used * (k - 1) * 12)
    return out


def ctx_of(parts: list[tuple[np.ndarray, int]]) -> tuple[np.ndarray, int]:
    """Combine (array, cardinality) pairs into a single context id array."""
    ids = np.zeros(len(parts[0][0]), dtype=np.int64)
    n = 1
    for a, c in parts:
        ids = ids * c + a.astype(np.int64)
        n *= c
    return ids, n
