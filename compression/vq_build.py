"""Train the E2M1-snapped vector quantizer and save it as a byte-pair lookup table.

Key layout fact: one VQ group is 4 consecutive k, and the packed FP4 format stores k as
(byte j low nibble = k 2j, high nibble = k 2j+1), so **a group of 4 weights is exactly 2 bytes**:

    u16 = c0 | c1<<4 | c2<<8 | c3<<12        (little endian byte pair)

Because the snapped codebook entries are themselves 4 E2M1 codes, quantizing an expert is a single
65536-entry uint16 -> uint16 lookup over the packed tensor, and the result is still a valid FP4
tensor.  That means the quality of a VQ rate can be measured with the *unmodified* kernels, and the
real encoder is the same table.

Outputs results/vq_<bits>.npz:
    lut_lo, lut_hi  uint8[65536]  packed byte pair -> quantized byte pair
    enc             int32[65536]  packed byte pair -> codebook index (what a real 12/14-bit format stores)
    codebook        uint8[K, 4]   the 4 E2M1 codes of each entry
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from lossy import FP4, vq_codebook

RATES = {3.0: 4096, 3.25: 8192, 3.5: 16384, 3.75: 32768}


def tuple_index_of_u16() -> np.ndarray:
    """u16 packed byte pair -> the (c0,c1,c2,c3) tuple index used by vq_codebook (c0 most significant)."""
    u = np.arange(65536, dtype=np.int64)
    c0, c1, c2, c3 = u & 15, (u >> 4) & 15, (u >> 8) & 15, (u >> 12) & 15
    return (((c0 * 16 + c1) * 16 + c2) * 16 + c3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="data/sample.npz")
    ap.add_argument("--bits", type=float, default=3.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    K = RATES[a.bits]
    nib = np.load(a.sample)["nib"]
    print(f"training VQ dim 4, K={K} ({a.bits} bit/weight) on {nib.size/1e6:.1f}M weights ...", flush=True)
    enc, cb, D, pw, encg, cbg, Dg = vq_codebook(nib, 4, K)
    print(f"  float codebook rel-RMSE {np.sqrt(D/pw)*100:.2f}%   E2M1-snapped {np.sqrt(Dg/pw)*100:.2f}%")

    # snapped codebook entries as E2M1 codes (note: 0.0 maps to code 0, never to the -0 code 8)
    codes = np.abs(cbg[:, :, None] - FP4[None, None, :]).argmin(2).astype(np.uint8)
    ti = tuple_index_of_u16()
    ent = encg[ti]                                   # u16 -> codebook index
    c = codes[ent]                                   # [65536, 4] quantized codes
    out_u16 = (c[:, 0].astype(np.int64) | (c[:, 1].astype(np.int64) << 4)
               | (c[:, 2].astype(np.int64) << 8) | (c[:, 3].astype(np.int64) << 12))
    lut_lo = (out_u16 & 0xFF).astype(np.uint8)
    lut_hi = (out_u16 >> 8).astype(np.uint8)
    ident = float((out_u16 == np.arange(65536)).mean())
    path = a.out or f"results/vq_{a.bits}.npz"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, lut_lo=lut_lo, lut_hi=lut_hi, enc=ent.astype(np.int32), codebook=codes,
             bits=a.bits, K=K, rel_rmse_float=float(np.sqrt(D / pw)), rel_rmse_snapped=float(np.sqrt(Dg / pw)))
    print(f"  {ident*100:.2f}% of the 65536 byte pairs are left unchanged; wrote {path}")
    json.dump({"bits": a.bits, "K": K, "rel_rmse_float": float(np.sqrt(D / pw)),
               "rel_rmse_snapped": float(np.sqrt(Dg / pw)), "identity_fraction": ident},
              open(path.replace(".npz", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
