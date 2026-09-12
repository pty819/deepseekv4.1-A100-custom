"""Lossy side: how many bits per weight can the FP4 experts actually be reduced to, and at what cost?

The source here is the *already quantized* checkpoint: E2M1 codes with a shared per-32 E8M0 scale.  The
original bf16 weights are gone, so every lossy scheme is a re-quantization of an already noisy signal
and its damage adds to the damage FP4 did in the first place.  This script measures both.

  L1  the discrete source: alphabet, power, and the exact rate-distortion bound R(D) (Blahut-Arimoto)
  L2  concrete schemes on that alphabet: Lloyd-Max k-level (fixed rate and entropy coded), E2M0
      (drop the mantissa bit), 2:4 structured sparsity (A100 mma.sp), magnitude pruning + bitmap,
      coarser scale blocks
  L3  what each scheme does to a REAL expert: ||dW||/||W||, the GEMM output, and the full SwiGLU expert
  L4  the FP4 noise floor itself (Gaussian -> per-32 amax -> E2M1), so the numbers can be read as
      "n times the noise the checkpoint already has"

Read-only with respect to the checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

FP4 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
ROWS = []


def row(**kw):
    ROWS.append(kw)


# ------------------------------------------------------------------ L1: source and the R(D) bound
def blahut_arimoto(p: np.ndarray, x: np.ndarray, y: np.ndarray, lambdas) -> list[tuple[float, float]]:
    """R(D) of a discrete source with squared-error distortion, as (D, R) pairs in bits."""
    d = (x[:, None] - y[None, :]) ** 2
    out = []
    for lam in lambdas:
        q = np.ones(len(y)) / len(y)
        e = np.exp(-lam * d)
        for _ in range(3000):
            num = q[None, :] * e
            Q = num / num.sum(1, keepdims=True)
            qn = p @ Q
            if np.abs(qn - q).max() < 1e-12:
                q = qn
                break
            q = qn
        num = q[None, :] * e
        Q = num / num.sum(1, keepdims=True)
        D = float((p[:, None] * Q * d).sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            r = Q * np.log2(np.where(Q > 0, Q, 1) / np.maximum(q[None, :], 1e-300))
        R = float((p[:, None] * np.where(Q > 0, r, 0)).sum())
        out.append((D, max(R, 0.0)))
    return sorted(out)


def lloyd_max(p: np.ndarray, v: np.ndarray, k: int, iters: int = 200, restarts: int = 12, seed: int = 0):
    """MSE-optimal k-level quantizer of the discrete source (p, v).  Returns (codebook, assignment, D)."""
    rng = np.random.default_rng(seed)
    best = None
    for r in range(restarts):
        if r == 0:
            c = np.quantile(np.repeat(v, (p * 10000).astype(int) + 1), np.linspace(0.5 / k, 1 - 0.5 / k, k))
        else:
            c = rng.choice(v, k, replace=False, p=p / p.sum())
        c = np.sort(c)
        for _ in range(iters):
            a = np.abs(v[:, None] - c[None, :]).argmin(1)
            cn = c.copy()
            for j in range(k):
                m = a == j
                if p[m].sum() > 0:
                    cn[j] = (p[m] * v[m]).sum() / p[m].sum()
            if np.allclose(cn, c):
                c = cn
                break
            c = cn
        a = np.abs(v[:, None] - c[None, :]).argmin(1)
        D = float((p * (v - c[a]) ** 2).sum())
        if best is None or D < best[2]:
            best = (c, a, D)
    return best


def entropy_of(p: np.ndarray, a: np.ndarray, k: int) -> float:
    q = np.bincount(a, weights=p, minlength=k)
    q = q[q > 0]
    return float(-(q * np.log2(q)).sum())


def fp4_noise_floor(n=4_000_000, block=32, seed=0):
    """Relative MSE the checkpoint's own FP4 step costs on a Gaussian source (per-32 amax, pow2 scale)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n // block, block))
    amax = np.abs(x).max(1, keepdims=True)
    s = np.exp2(np.ceil(np.log2(amax / 6.0)))
    q = x / s
    a = np.abs(q)
    r = np.where(a < 2.0, np.round(a * 2) / 2, np.where(a < 4.0, np.round(a), np.round(a / 2) * 2))
    r = np.clip(r, None, 6.0) * np.sign(q)
    xh = r * s
    return float(((x - xh) ** 2).sum() / (x ** 2).sum())



def vq_codebook(nib: np.ndarray, d: int, K: int, iters: int = 40, seed: int = 0):
    """Weighted k-means over d-tuples of FP4 codes (the tuple alphabet is finite, 16^d, so the whole
    training set is one histogram).  Returns (encode LUT 16^d -> K, codebook [K, d] in E2M1 units)."""
    t = nib.reshape(-1, d).astype(np.int64)
    idx = np.zeros(len(t), np.int64)
    for j in range(d):
        idx = idx * 16 + t[:, j]
    hist = np.bincount(idx, minlength=16 ** d).astype(np.float64)
    used = np.flatnonzero(hist)
    w = hist[used]
    pts = np.stack([FP4[(used >> (4 * (d - 1 - j))) & 15] for j in range(d)], 1)
    rng = np.random.default_rng(seed)
    c = pts[rng.choice(len(pts), K, replace=False, p=w / w.sum())]
    def assign(c):
        cn = (c * c).sum(1)
        return (cn[None, :] - 2.0 * (pts @ c.T)).argmin(1)

    for _ in range(iters):
        a = assign(c)
        num = np.zeros_like(c)
        den = np.zeros(K)
        np.add.at(num, a, pts * w[:, None])
        np.add.at(den, a, w)
        keep = den > 0
        cn = c.copy()
        cn[keep] = num[keep] / den[keep, None]
        if np.allclose(cn, c):
            c = cn
            break
        c = cn
    a = assign(c)
    D = float((w[:, None] * (pts - c[a]) ** 2).sum() / w.sum() / d)
    enc = np.zeros(16 ** d, np.int32)
    enc[used] = a
    pw = float((w[:, None] * pts ** 2).sum() / w.sum() / d)
    # grid-snapped variant: every centroid coordinate rounded onto the E2M1 grid, so a codebook entry
    # is 4 nibbles and the existing FP4 bit-placement decode still works after the LUT
    cg = FP4[np.abs(c[:, :, None] - FP4[None, None, :]).argmin(2)]
    cgn = (cg * cg).sum(1)
    ag = (cgn[None, :] - 2.0 * (pts @ cg.T)).argmin(1)
    Dg = float((w[:, None] * (pts - cg[ag]) ** 2).sum() / w.sum() / d)
    encg = np.zeros(16 ** d, np.int32)
    encg[used] = ag
    return enc, c, D, pw, encg, cg, Dg


# ------------------------------------------------------------------ L3: schemes on a real expert
def apply_codebook(code: np.ndarray, recon: np.ndarray) -> np.ndarray:
    """code [..] uint8 -> reconstructed E2M1-unit values via a 16-entry lookup."""
    return recon[code]


def sparsity_24(val: np.ndarray) -> np.ndarray:
    """Keep the 2 largest magnitudes of every 4 consecutive k (what A100 mma.sp needs)."""
    g = val.reshape(*val.shape[:-1], -1, 4)
    order = np.argsort(-np.abs(g), axis=-1)
    keep = np.zeros_like(g, dtype=bool)
    np.put_along_axis(keep, order[..., :2], True, axis=-1)
    return (g * keep).reshape(val.shape)


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum() / (a ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--sample", default="data/sample.npz")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--expert", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--out", default="results/lossy")
    a = ap.parse_args()

    d = np.load(a.sample)
    nib = d["nib"]
    cnt = np.bincount(nib.ravel(), minlength=16).astype(np.float64)
    p = cnt / cnt.sum()
    v = FP4.copy()
    power = float((p * v ** 2).sum())
    H0 = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    floor = fp4_noise_floor()
    print("=== L1. the source ===")
    print(f"  16 E2M1 levels, H = {H0:.4f} bit, E[v^2] = {power:.4f} (E2M1 units)")
    print(f"  FP4's own step already costs rel-MSE {floor:.5f} = {np.sqrt(floor)*100:.2f}% RMS "
          f"({-10*np.log10(floor):.1f} dB SQNR) on a Gaussian source  <- the noise floor everything adds to")

    # exact R(D) for this source (reconstruction on a fine grid)
    grid = np.linspace(-6, 6, 241)
    rd = blahut_arimoto(p, v, grid, np.concatenate([np.logspace(-2, 1.6, 40), [1e4]]))
    Ds = np.array([x[0] for x in rd])
    Rs = np.array([x[1] for x in rd])
    print("\n=== L2. rate-distortion: theoretical bound vs concrete schemes ===")
    print(f"  {'scheme':<40} {'bit/w':>6} {'+scale':>7} {'ratio':>6} {'rel-RMSE':>9} {'SQNR dB':>8} {'x floor':>8}")

    def show(name, bits, relmse, note="", scale_bits=8.0):
        tot = bits + scale_bits / 32
        rmse = np.sqrt(relmse)
        sqnr = -10 * np.log10(max(relmse, 1e-12))
        print(f"  {name:<40} {bits:6.3f} {tot:7.3f} {tot/4.25:6.3f} {rmse*100:8.2f}% {sqnr:8.1f} "
              f"{relmse/floor:8.2f}")
        row(scheme=name, payload_bits=bits, total_bits=tot, ratio_vs_4p25=tot / 4.25,
            rel_mse_vs_fp4=relmse, rel_rmse_pct=rmse * 100, sqnr_db=sqnr,
            times_fp4_noise=relmse / floor, note=note)

    show("stored FP4 (baseline)", 4.0, 0.0, "lossless reference")
    # the bound, read at the rates of interest
    for target in (3.5, 3.0, 2.5, 2.0, 1.5):
        i = np.searchsorted(-Rs, -target)
        i = min(max(i, 1), len(Rs) - 1)
        # interpolate D at R = target
        D = float(np.interp(target, Rs[::-1], Ds[::-1]))
        show(f"R(D) bound at {target} bit", target, D / power, "Blahut-Arimoto, any scheme at this rate")
    # Lloyd-Max
    lm = {}
    for k in (2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 16):
        c, asg, D = lloyd_max(p, v, k)
        lm[k] = (c, asg, D)
        Hk = entropy_of(p, asg, k)
        show(f"Lloyd-Max {k} levels (fixed {int(np.ceil(np.log2(k)))} bit)", float(np.ceil(np.log2(k))),
             D / power, f"codebook {np.round(c,3).tolist()}")
        if abs(Hk - np.ceil(np.log2(k))) > 0.02:
            show(f"Lloyd-Max {k} levels + entropy coding", Hk, D / power, "same distortion, coded rate")
        # fixed-length base-k packing: n symbols in ceil(n*log2 k) bits, best n <= 32
        best_n = min(((np.ceil(n * np.log2(k)) / n, n) for n in range(1, 33)))
        if best_n[0] < np.ceil(np.log2(k)) - 0.02:
            show(f"Lloyd-Max {k} levels, base-{k} packing ({best_n[1]} in "
                 f"{int(np.ceil(best_n[1]*np.log2(k)))} bit)", float(best_n[0]), D / power,
                 "fixed length, LUT decode, no entropy coder")
    # E2M0: drop the mantissa bit (sign + 2-bit exponent)
    e2m0 = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0])
    e2m0 = np.concatenate([e2m0, -e2m0])
    D = float((p * (v - e2m0) ** 2).sum())
    show("E2M0: drop the mantissa bit", 3.0, D / power, "trivial decode, 6.0 clips to 4.0")
    e2m0b = np.array([0.0, 0.5, 1.0, 1.5, 3.0, 3.0, 6.0, 6.0])   # keep the top, drop mid resolution
    e2m0b = np.concatenate([e2m0b, -e2m0b])
    D = float((p * (v - e2m0b) ** 2).sum())
    show("3 bit, top-preserving codebook", 3.0, D / power, "no clipping of 6.0")

    # magnitude pruning + bitmap (kept weights still 4 bit)
    mag = np.abs(v)
    for thr in (0.25, 0.75, 1.25, 1.75, 2.5):
        keep = mag > thr
        s = float(p[keep].sum())
        D = float((p * np.where(keep, 0.0, v ** 2)).sum())
        show(f"prune |v| <= {thr} + bitmap (keep {s*100:.1f}%)", 1.0 + 4.0 * s, D / power,
             "1 bit mask + 4 bit per surviving weight")
    # vector quantization over d-tuples (the only way to actually reach the bound at a fixed rate)
    vqs = {}
    for (dd, K) in ((2, 64), (2, 128), (4, 1024), (4, 2048), (4, 4096), (4, 8192), (4, 16384), (4, 32768)):
        enc, cb, D, pw, encg, cbg, Dg = vq_codebook(nib, dd, K)
        bits = np.log2(K) / dd
        vqs[(dd, K)] = (enc, cb)
        vqs[("grid", dd, K)] = (encg, cbg)
        show(f"VQ dim {dd}, {K} entries ({int(np.log2(K))} bit/group)", bits, D / pw,
             f"LUT decode: {K} x {dd} float table, fixed rate")
        show(f"VQ dim {dd}, {K} entries, E2M1-snapped", bits, Dg / pw,
             f"codebook entries are {dd} nibbles: {K}x{dd*4} bit LUT, then the existing FP4 decode")
    json.dump({"rd_curve": [[float(x), float(y)] for x, y in rd], "p": p.tolist(), "power": power,
               "fp4_noise_floor": floor}, open(a.out + "_rd.json", "w"))

    # ------------------------------------------------------------------ L3 on a real expert
    print("\n=== L3. the same schemes on a real expert (layer %d, expert %d) ===" % (a.layer, a.expert))
    from st import Checkpoint
    ck = Checkpoint(a.ckpt)
    W, S = {}, {}
    for m in ("w1", "w2", "w3"):
        b = ck.bytes(f"layers.{a.layer}.ffn.experts.{a.expert}.{m}.weight")
        s = ck.bytes(f"layers.{a.layer}.ffn.experts.{a.expert}.{m}.scale")
        n, kh = b.shape
        code = np.empty((n, kh * 2), np.uint8)
        code[:, 0::2] = b & 0x0F
        code[:, 1::2] = b >> 4
        W[m] = code
        S[m] = np.exp2(s.astype(np.float32) - 127.0)
    inter, dim = W["w1"].shape        # w1/w3 are [inter, dim]; w2 is [dim, inter]

    def dequant(m, recon=FP4, sparse=False, scale_block=32, vq=None):
        if vq is not None:
            enc, cb, dd = vq
            t = W[m].reshape(-1, dd).astype(np.int64)
            idx = np.zeros(len(t), np.int64)
            for j in range(dd):
                idx = idx * 16 + t[:, j]
            val = cb[enc[idx]].reshape(W[m].shape).astype(np.float32)
        else:
            val = recon[W[m]].astype(np.float32)
        if sparse:
            val = sparsity_24(val)
        sc = S[m]
        if scale_block != 32:
            g = scale_block // 32
            sc = np.repeat(sc.reshape(sc.shape[0], -1, g).max(2), g, axis=1)
        return val * np.repeat(sc, 32, axis=1)

    ref = {m: dequant(m) for m in ("w1", "w2", "w3")}
    rng = np.random.default_rng(0)
    x = rng.standard_normal((a.tokens, dim)).astype(np.float32)

    def expert_out(w):
        h1, h3 = x @ w["w1"].T, x @ w["w3"].T
        h = (h1 / (1 + np.exp(-h1))) * h3
        return h @ w["w2"].T

    y_ref = expert_out(ref)
    print(f"  {'scheme':<40} {'bit/w':>6} {'ratio':>6} {'||dW||/||W||':>12} {'GEMM dy':>9} {'expert dy':>10}")

    def real(name, bits, **kw):
        cur = {m: dequant(m, **kw) for m in ("w1", "w2", "w3")}
        dw = np.sqrt(sum(((cur[m] - ref[m]) ** 2).sum() for m in cur) / sum((ref[m] ** 2).sum() for m in ref))
        g = rel_err(x @ ref["w1"].T, x @ cur["w1"].T)
        e = rel_err(y_ref, expert_out(cur))
        tot = bits + 8 / 32
        print(f"  {name:<40} {bits:6.3f} {tot/4.25:6.3f} {dw*100:11.2f}% {g*100:8.2f}% {e*100:9.2f}%")
        row(scheme="[real expert] " + name, payload_bits=bits, total_bits=tot, ratio_vs_4p25=tot / 4.25,
            weight_rel_err_pct=dw * 100, gemm_rel_err_pct=g * 100, expert_rel_err_pct=e * 100)
        return e

    real("stored FP4 (baseline)", 4.0)
    for k in (12, 8, 6, 4):
        c, asg, _ = lm[k]
        real(f"Lloyd-Max {k} levels", float(np.ceil(np.log2(k))), recon=c[asg].astype(np.float32))
    real("E2M0: drop the mantissa bit", 3.0, recon=e2m0.astype(np.float32))
    real("3 bit, top-preserving codebook", 3.0, recon=e2m0b.astype(np.float32))
    for (dd, K) in ((2, 64), (4, 1024), (4, 2048), (4, 4096), (4, 8192), (4, 16384), (4, 32768)):
        enc, cb = vqs[(dd, K)]
        real(f"VQ dim {dd}, {K} entries", float(np.log2(K) / dd), vq=(enc, cb, dd))
        eg, cg = vqs[("grid", dd, K)]
        real(f"VQ dim {dd}, {K} entries, E2M1-snapped", float(np.log2(K) / dd), vq=(eg, cg, dd))
    real("2:4 structured sparsity (A100 mma.sp)", 3.0, sparse=True)
    real("scale block 32 -> 64 (4 bit payload)", 4.0, scale_block=64)
    real("scale block 32 -> 128 (4 bit payload)", 4.0, scale_block=128)

    os.makedirs("results", exist_ok=True)
    keys = sorted({k for r in ROWS for k in r})
    head = ["scheme", "payload_bits", "total_bits", "ratio_vs_4p25"]
    keys = head + [k for k in keys if k not in head]
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in ROWS:
            w.writerow(r)
    json.dump({"fp4_noise_floor": floor, "rows": ROWS}, open(a.out + ".json", "w"), indent=1, default=float)
    print(f"\nwrote {a.out}.csv / .json / _rd.json")


if __name__ == "__main__":
    main()
