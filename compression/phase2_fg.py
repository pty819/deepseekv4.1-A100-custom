"""Phase 1, sections F and G.

F  cross-expert / cross-matrix structure at aligned positions: is expert i's block at (row, k)
   predictable from another expert's block at the same place (nearest expert, not just the neighbour)?
G  the E8M0 block scales: how far can they be compressed, and what does that buy for a whole expert?
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

import entropy as ent
from st import Checkpoint

RESULTS = []


def unpack(b: np.ndarray) -> np.ndarray:
    """uint8 [.., Kh] packed -> uint8 [.., 2*Kh] nibbles (low nibble = even k)."""
    lo, hi = b & 0x0F, b >> 4
    out = np.empty(b.shape[:-1] + (b.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def section_f(ck: Checkpoint, layer: int, n_experts: int, n_rows: int):
    print(f"\n=== F. cross-expert structure (layer {layer}, {n_experts} experts, {n_rows} aligned rows) ===")
    rows = np.arange(n_rows)
    W = {}
    for m in ("w1", "w3"):
        W[m] = np.stack([unpack(ck.rows(f"layers.{layer}.ffn.experts.{e}.{m}.weight", rows))
                         for e in range(n_experts)])       # [E, rows, K]
    w1 = W["w1"]
    E, R, K = w1.shape
    p = np.bincount(w1.ravel(), minlength=16) / w1.size
    chance = float((p * p).sum())
    print(f"  chance nibble agreement (sum p_i^2) = {chance:.4f};  H(W) = {ent.order0(w1.ravel()):.4f}")

    flat = w1.reshape(E, -1)
    agree = np.zeros((E, E))
    for i in range(E):
        agree[i] = (flat[i][None, :] == flat).mean(1)
    off = agree[~np.eye(E, dtype=bool)]
    np.fill_diagonal(agree, -1)
    best_j = agree.argmax(1)
    print(f"  expert-pair nibble agreement: mean {off.mean():.4f}  max {off.max():.4f} "
          f"(chance {chance:.4f})  -> excess {off.max()-chance:+.4f}")
    adj = np.array([agree[i, i + 1] for i in range(E - 1)])
    print(f"  adjacent experts (i, i+1):    mean {adj.mean():.4f}")
    # residual entropy against the best-matching expert and against a fixed reference expert
    for name, idx in (("best matching expert", best_j), ("reference expert 0", np.zeros(E, int))):
        res = flat ^ flat[idx]
        keep = np.arange(E) != np.array(idx)
        H = ent.order0(res[keep].ravel())
        print(f"  XOR vs {name:<22}: H(residual) = {H:.4f} bit  ({H/4:.4f}x)")
        RESULTS.append({"section": "F", "method": f"XOR vs {name}", "bits_per_weight": H,
                        "ratio_vs_4bit": H / 4, "note": f"layer {layer} w1, aligned (row,k)"})
    # w1 vs w3 of the same expert at the same position
    res13 = w1.reshape(E, -1) ^ W["w3"].reshape(E, -1)
    ag13 = float((w1 == W["w3"]).mean())
    H13 = ent.order0(res13.ravel())
    print(f"  w1 vs w3 same expert/position: agreement {ag13:.4f} (chance {chance:.4f}), H(XOR) = {H13:.4f}")
    RESULTS.append({"section": "F", "method": "XOR w1 vs w3 (same expert, same position)",
                    "bits_per_weight": H13, "ratio_vs_4bit": H13 / 4,
                    "note": f"agreement {ag13:.4f} vs chance {chance:.4f}"})
    # aligned-block nearest neighbour: for each 32-weight block, the closest block among the other
    # experts at the SAME (row, block) position
    nb = K // 32
    blocks = w1.reshape(E, R * nb, 32)
    bits = np.unpackbits(blocks.reshape(E, -1, 32, 1), axis=3, count=4, bitorder="little").reshape(E, -1, 128)
    pm = bits.astype(np.float32) * 2 - 1
    d = np.einsum("ebk,fbk->bef", pm, pm)        # [blocks, E, E] dot products
    ham = (128.0 - d) * 0.5
    ham[:, np.arange(E), np.arange(E)] = 1e9
    nnd = ham.min(2)
    print(f"  aligned-position NN over {E-1} other experts: mean Hamming {nnd.mean():.2f} / 128 "
          f"(random pair {ham[ham<1e8].mean():.2f})")
    RESULTS.append({"section": "F", "method": f"aligned-block NN over {E-1} experts",
                    "bits_per_weight": None, "ratio_vs_4bit": None,
                    "note": f"mean Hamming {nnd.mean():.2f}/128 vs random pair {ham[ham<1e8].mean():.2f}"})
    # residual entropy of block XOR against that nearest aligned block
    argb = ham.argmin(2)
    bl = blocks.transpose(1, 0, 2)               # [blocks, E, 32]
    nn = np.take_along_axis(bl, argb[:, :, None].repeat(32, 2), axis=1)
    Hres = ent.order0((bl ^ nn).ravel())
    idxbits = np.log2(E - 1)
    tot = Hres + idxbits / 32
    print(f"  block XOR vs nearest aligned expert: H(res) = {Hres:.4f} + {idxbits/32:.4f} index "
          f"-> {tot:.4f} bit/weight ({tot/4:.4f}x)")
    RESULTS.append({"section": "F", "method": "block XOR vs nearest aligned expert",
                    "bits_per_weight": tot, "ratio_vs_4bit": tot / 4,
                    "note": f"H(res)={Hres:.4f}, + {idxbits:.1f} index bits per 32-weight block"})


def section_g(ck: Checkpoint, layers: list[int], n_experts: int):
    print(f"\n=== G. E8M0 block scales ({n_experts} experts x {len(layers)} layers) ===")
    import zstandard as zstd
    S = []
    for L in layers:
        for e in range(n_experts):
            for m in ("w1", "w2", "w3"):
                S.append(ck.bytes(f"layers.{L}.ffn.experts.{e}.{m}.scale"))
    raw = np.concatenate([s.ravel() for s in S])
    uniq = np.unique(raw)
    H0 = ent.order0(raw, k=256)
    print(f"  {len(raw):,} scale bytes, {len(uniq)} distinct {uniq.tolist()}, H0 = {H0:.4f} bit/byte")
    # 2D context: left neighbour (same row, previous block) and the row above
    left_ctx, left_sym = [], []
    for s in S:
        a = s.astype(np.int64)
        left_sym.append(a[:, 1:].ravel())
        left_ctx.append(a[:, :-1].ravel())
    ls, lc = np.concatenate(left_sym), np.concatenate(left_ctx)
    cmap = {int(v): i for i, v in enumerate(uniq)}
    lut = np.zeros(256, np.int64)
    for v, i in cmap.items():
        lut[v] = i
    H_left = ent.cond(lut[lc], lut[ls], len(uniq), k=len(uniq))["H_plugin"]
    # left + up (same column, previous row)
    sym2, c2 = [], []
    for s in S:
        a = lut[s.astype(np.int64)]
        sym2.append(a[1:, 1:].ravel())
        c2.append((a[1:, :-1] * len(uniq) + a[:-1, 1:]).ravel())
    H_lu = ent.cond(np.concatenate(c2), np.concatenate(sym2), len(uniq) ** 2, k=len(uniq))["H_plugin"]
    print(f"  H(s | left) = {H_left:.4f}   H(s | left, up) = {H_lu:.4f} bit/byte")
    blob = b"".join(s.tobytes() for s in S)
    z = {lv: len(zstd.ZstdCompressor(level=lv).compress(blob)) / len(blob) for lv in (3, 9, 19)}
    d = np.concatenate([np.diff(s.astype(np.int16), axis=1).ravel() for s in S])
    Hd = ent.order0((d + 128).astype(np.uint8), k=256)
    dz = len(zstd.ZstdCompressor(level=9).compress((d + 128).astype(np.uint8).tobytes())) / len(d)
    runs = 1.0 - float((d != 0).mean())
    print(f"  zstd on raw scales: " + "  ".join(f"lvl{k} {v:.4f}" for k, v in z.items()))
    print(f"  delta along k: H0 = {Hd:.4f} bit/byte, zstd-9 {dz:.4f}, repeat-rate {runs:.4f}")
    for name, bits in (("raw scale byte", 8.0), ("order-0 entropy", H0), ("H(s | left)", H_left),
                       ("H(s | left, up)", H_lu), ("zstd-9 on raw scales", 8 * z[9])):
        share = bits / 32          # bits per weight (one scale per 32 weights)
        total = 4.0 + share        # whole expert, payload still raw
        print(f"    {name:<22} {bits:.4f} bit/scale -> {share:.4f} bit/weight; expert total "
              f"{total:.4f} bit/weight ({total/4.25:.4f}x of 4.25)")
        RESULTS.append({"section": "G", "method": name, "bits_per_weight": share,
                        "ratio_vs_4bit": None,
                        "note": f"{bits:.4f} bit per scale byte; expert total with raw payload "
                                f"{total:.4f} bit/weight = {total/4.25:.4f}x"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--experts", type=int, default=48)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--g-layers", default="0,11,23,39")
    ap.add_argument("--g-experts", type=int, default=4)
    ap.add_argument("--out", default="results/phase1_fg")
    a = ap.parse_args()
    ck = Checkpoint(a.ckpt)
    section_f(ck, a.layer, a.experts, a.rows)
    section_g(ck, [int(x) for x in a.g_layers.split(",")], a.g_experts)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(RESULTS, open(a.out + ".json", "w"), indent=1)
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["section", "method", "bits_per_weight", "ratio_vs_4bit", "note"])
        w.writeheader()
        for r in RESULTS:
            w.writerow(r)
    print(f"\nwrote {a.out}.json / .csv")


if __name__ == "__main__":
    main()
