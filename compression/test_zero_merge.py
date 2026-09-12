"""Is merging the FP4 codes 0 (+0) and 8 (-0) numerically transparent?

E2M1 code 8 dequantizes to -0.0 and code 0 to +0.0.  The claim to test is that rewriting every 8 to a 0
changes the stored bytes but not a single output bit of the expert GEMM.

Argument: the accumulator starts at +0.0 and x * (+-0.0) = +-0.0; in round-to-nearest
c + (+-0.0) = c for every c != 0, and +0.0 + (-0.0) = +0.0, so an accumulator that starts at +0.0 can
never see a difference.  This script checks it on the real kernels with a real expert.

Read-only: the checkpoint is never written; the merged copy lives in GPU/CPU memory.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsv41 import cukern                                    # noqa: E402
from dsv41.quant import dequant_fp4, fake_quant_fp8, tile_fp4, tile_fp4_scales  # noqa: E402

from st import Checkpoint                                   # noqa: E402


def merge_zeros(w: torch.Tensor) -> torch.Tensor:
    """uint8 packed FP4 -> same, with every nibble 8 (-0) rewritten to 0 (+0)."""
    lo, hi = w & 0x0F, w >> 4
    lo = torch.where(lo == 8, torch.zeros_like(lo), lo)
    hi = torch.where(hi == 8, torch.zeros_like(hi), hi)
    return lo | (hi << 4)


def pick_device(need_gb=3.0) -> torch.device:
    best, bestfree = None, 0.0
    for d in range(torch.cuda.device_count()):
        free = torch.cuda.mem_get_info(d)[0] / 2**30
        if free > bestfree:
            best, bestfree = d, free
    if bestfree < need_gb:
        raise RuntimeError(f"no GPU with {need_gb} GB free (best cuda:{best} has {bestfree:.1f})")
    print(f"using cuda:{best} ({bestfree:.1f} GB free)")
    return torch.device(f"cuda:{best}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--experts", type=int, default=6)
    ap.add_argument("--tokens", type=int, default=8)
    a = ap.parse_args()

    dev = pick_device()
    ck = Checkpoint(a.ckpt)
    torch.manual_seed(0)
    ws, ss = [], []
    for e in range(a.experts):
        ws.append(torch.from_numpy(ck.bytes(f"layers.{a.layer}.ffn.experts.{e}.w1.weight").copy()))
        ss.append(torch.from_numpy(ck.bytes(f"layers.{a.layer}.ffn.experts.{e}.w1.scale").copy()))
    w = torch.stack(ws).to(dev)
    s = torch.stack(ss).to(dev)
    E, N, Kh = w.shape
    K = Kh * 2
    wm = merge_zeros(w)
    changed = int((w != wm).sum())
    nib = torch.stack([w & 0x0F, w >> 4])
    n8 = int((nib == 8).sum())
    print(f"expert w1 [{N}, {K}] x {E}: {n8:,} of {nib.numel():,} nibbles are -0 "
          f"({n8/nib.numel()*100:.2f}%), {changed:,} bytes change")

    # 1. dequantized weights: identical except for the sign of zeros
    wb = torch.stack([dequant_fp4(w[e], s[e]) for e in range(E)])
    wbm = torch.stack([dequant_fp4(wm[e], s[e]) for e in range(E)])
    diff = wb.view(torch.int16) != wbm.view(torch.int16)
    both_zero = (wb == 0) & (wbm == 0)
    print(f"  dequantized bf16: {int(diff.sum()):,} bit patterns differ, all of them zeros: "
          f"{bool((diff <= both_zero).all())}   (value equality: {bool((wb == wbm).all())})")

    # 2. exact fp32 reference GEMM
    x = fake_quant_fp8(torch.randn(a.tokens, K, device=dev, dtype=torch.bfloat16) * 2, 32)
    r1 = torch.stack([x.float() @ wb[e].float().T for e in range(E)])
    r2 = torch.stack([x.float() @ wbm[e].float().T for e in range(E)])
    print(f"  fp32 reference GEMM bit-identical: {bool((r1.view(torch.int32) == r2.view(torch.int32)).all())}")

    # 3. the real kernels
    xp = cukern.permute_x(x)
    experts = torch.arange(E, device=dev, dtype=torch.int32)
    M = a.tokens
    grp_start = torch.arange(0, E * M + 1, M, device=dev, dtype=torch.int32)
    pair_tok = torch.arange(M, device=dev, dtype=torch.int32).repeat(E)
    P = E * M
    wt, st_ = (tile_fp4(w), tile_fp4_scales(s)) if cukern.FP4_TILED else (w, s)
    wmt = tile_fp4(wm) if cukern.FP4_TILED else wm
    y1 = cukern.fp4_gemm_tc(xp, wt, st_, experts, grp_start, pair_tok, P, M)
    y2 = cukern.fp4_gemm_tc(xp, wmt, st_, experts, grp_start, pair_tok, P, M)
    same = bool((y1.view(torch.int16) == y2.view(torch.int16)).all())
    print(f"  fp4_gemm_tc ({M} tokens x {E} experts, tiled={cukern.FP4_TILED}) bit-identical: {same}"
          f"   max|dy| = {float((y1.float()-y2.float()).abs().max()):g}")

    pair_expert = experts.repeat_interleave(M)
    ones = torch.ones(P, device=dev)
    g1 = cukern.fp4_gemv_pairs(x, wt, st_, pair_tok, pair_expert, ones, P)
    g2 = cukern.fp4_gemv_pairs(x, wmt, st_, pair_tok, pair_expert, ones, P)
    sameg = bool((g1.view(torch.int32) == g2.view(torch.int32)).all())
    print(f"  fp4_gemv_pairs bit-identical: {sameg}   max|dy| = {float((g1-g2).abs().max()):g}")

    # 4. how much would it actually save?
    W_PER_EXPERT = 3 * 2304 * 5120
    SCALES = 2 * 2304 * 160 + 5120 * 72
    p_zero = n8 / nib.numel() * 2          # both codes 0 and 8 are zero-magnitude
    free_bits = p_zero * W_PER_EXPERT      # one sign bit per zero weight is now unused
    print(f"\n  zero-magnitude weights: {p_zero*100:.2f}%  ->  {free_bits/8/1e3:.1f} kB of unused "
          f"sign bits per expert (18,800.6 kB total)")
    print(f"  scales entropy-coded need {1.0041*SCALES/8/1e3:.1f} kB; fixed 4-bit need {4*SCALES/8/1e3:.1f} kB")
    print("  => the freed bits can carry the entropy-coded scales, but NOT a fixed 4-bit scale field")


if __name__ == "__main__":
    main()
