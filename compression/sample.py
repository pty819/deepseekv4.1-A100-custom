"""Sample FP4 expert weights (packed payload + E8M0 block scales) into a compact .npz.

Checkpoint layout (DeepSeek-V4.1-Flash, quantization_config.expert_dtype = fp4):
  layers.{L}.ffn.experts.{E}.w1.weight  uint8 [inter=2304, dim/2=2560]   E2M1, two per byte along k
                          .w1.scale     uint8 [2304, dim/32=160]         E8M0, one per row per 32 k
                           w3.*         same as w1
                           w2.weight    uint8 [dim=5120, inter/2=1152]
                           w2.scale     uint8 [5120, inter/32=72]
Low nibble = even k (dsv41/quant.py unpack_fp4). A "block" is the 32 weights (16 bytes) that share one scale.

The sample keeps whole rows, so blocks of one row stay contiguous along k for the Markov / run tests.
Read-only: the checkpoint is memory-mapped and never written.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from st import Checkpoint

MATS = ("w1", "w2", "w3")


def sample(ckpt_path: str, layers: list[int], n_experts: int, n_rows: int, seed: int, out: str):
    ck = Checkpoint(ckpt_path)
    nib_l, scl_l, meta_l = [], [], []
    t0 = time.time()
    for L in layers:
        rng = np.random.default_rng(seed + 1000 * L)
        experts = np.sort(rng.choice(384, n_experts, replace=False))
        for E in experts:
            for mi, m in enumerate(MATS):
                wn = f"layers.{L}.ffn.experts.{E}.{m}.weight"
                sn = f"layers.{L}.ffn.experts.{E}.{m}.scale"
                N, Kh = ck.meta(wn)[1]
                rows = np.sort(rng.choice(N, n_rows, replace=False))
                w = ck.rows(wn, rows)              # [n_rows, Kh]
                s = ck.rows(sn, rows)              # [n_rows, Kh//16]
                nb = Kh // 16                      # blocks per row
                lo = (w & 0x0F).reshape(n_rows, nb, 16)
                hi = (w >> 4).reshape(n_rows, nb, 16)
                nib = np.empty((n_rows, nb, 32), dtype=np.uint8)
                nib[:, :, 0::2] = lo
                nib[:, :, 1::2] = hi
                nib_l.append(nib.reshape(-1, 32))
                scl_l.append(s.reshape(-1))
                nblk = n_rows * nb
                meta = np.empty((nblk, 5), dtype=np.int32)
                meta[:, 0] = L
                meta[:, 1] = E
                meta[:, 2] = mi
                meta[:, 3] = np.repeat(rows, nb)            # row id
                meta[:, 4] = np.tile(np.arange(nb), n_rows)  # block index along k
                meta_l.append(meta)
        print(f"  layer {L}: {sum(len(x) for x in nib_l):,} blocks, {time.time()-t0:.1f}s", flush=True)
    nib = np.concatenate(nib_l)
    scl = np.concatenate(scl_l)
    meta = np.concatenate(meta_l)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez(out, nib=nib, scale=scl, meta=meta, layers=np.array(layers), n_rows=n_rows, n_experts=n_experts)
    print(f"{len(nib):,} blocks ({len(nib)*32/1e6:.1f}M weights) -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--layers", default="0,5,11,17,23,29,35,39")
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="data/sample.npz")
    a = ap.parse_args()
    sample(a.ckpt, [int(x) for x in a.layers.split(",")], a.experts, a.rows, a.seed, a.out)
