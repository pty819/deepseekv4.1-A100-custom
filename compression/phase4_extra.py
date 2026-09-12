"""Extras: per-layer/matrix spread of the baseline entropy, the model-wide E8M0 scale alphabet, and
the +-0 alias (codes 0 and 8 both dequantize to zero -- byte-lossy but numerically identical)."""
from __future__ import annotations

import argparse
import json

import numpy as np

import entropy as ent
from st import Checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--sample", default="data/sample.npz")
    ap.add_argument("--scale-experts", type=int, default=8)
    ap.add_argument("--out", default="results/phase1_extra.json")
    a = ap.parse_args()
    out = {}

    d = np.load(a.sample)
    nib, meta, layers = d["nib"], d["meta"], d["layers"]
    sym = nib.ravel()
    lay = np.repeat(meta[:, 0], 32)
    mat = np.repeat(meta[:, 2], 32)
    print("=== per-layer / per-matrix order-0 entropy (is the sample representative?) ===")
    per = {}
    for L in layers.tolist():
        hs = [ent.order0(sym[(lay == L) & (mat == m)]) for m in range(3)]
        per[int(L)] = hs
        print(f"  layer {L:>2}: w1 {hs[0]:.4f}  w2 {hs[1]:.4f}  w3 {hs[2]:.4f}")
    out["per_layer_H0"] = per
    allh = np.array([h for v in per.values() for h in v])
    print(f"  spread over {len(allh)} (layer, matrix) groups: {allh.min():.4f} .. {allh.max():.4f}")

    print("\n=== model-wide E8M0 scale alphabet (all 40 layers) ===")
    ck = Checkpoint(a.ckpt)
    rng = np.random.default_rng(3)
    hist = np.zeros(256, np.int64)
    for L in range(40):
        for e in rng.choice(384, a.scale_experts, replace=False):
            for m in ("w1", "w2", "w3"):
                s = ck.bytes(f"layers.{L}.ffn.experts.{int(e)}.{m}.scale")
                hist += np.bincount(s.ravel(), minlength=256)
    nz = np.flatnonzero(hist)
    p = hist[nz] / hist.sum()
    H = float(-(p * np.log2(p)).sum())
    print(f"  {hist.sum():,} scale bytes from {40*a.scale_experts} experts: {len(nz)} distinct values "
          f"{nz.tolist()}  H0 = {H:.4f} bit/byte")
    print(f"  fixed-width packing needs {int(np.ceil(np.log2(len(nz))))} bit/scale; entropy coding {H:.4f}")
    out["scale_alphabet"] = {"values": nz.tolist(), "probs": p.tolist(), "H0": H,
                             "n_bytes": int(hist.sum())}

    print("\n=== +-0 alias: codes 0 and 8 both dequantize to zero ===")
    merged = np.where(sym == 8, 0, sym)
    H0 = ent.order0(sym)
    Hm = ent.order0(merged)
    pz = float((sym & 7 == 0).mean())
    print(f"  P(magnitude 0) = {pz:.4f};  H(W) = {H0:.4f} -> merged alphabet H = {Hm:.4f} bit "
          f"(saves {H0-Hm:.4f} bit/weight = {(H0-Hm)/4*100:.2f}% of the payload)")
    print("  NOT byte-lossless: the checkpoint's -0 nibbles would come back as +0.  The dequantized")
    print("  weight is the same value, and every product/accumulation is identical in round-to-nearest,")
    print("  so inference output is unchanged -- but it fails a byte-for-byte checkpoint comparison.")
    out["zero_alias"] = {"P_zero": pz, "H0": H0, "H_merged": Hm}
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
