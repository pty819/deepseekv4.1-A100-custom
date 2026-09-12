"""Phase 1: is there any lossless headroom left in the FP4 expert payload?

Sections follow the study plan:
  A  baselines (nibble/byte/scale entropy, zstd/lz4/gzip on real expert tensors)
  B  conditional entropies (scale, position, Markov, layer/matrix/expert, running block max)
  C  bit-plane decomposition
  D  PRNG XOR residual
  E  block dictionary + XOR residual (256 .. 65536, plus "all sampled blocks" as a bound)

Nothing here writes to the checkpoint; it only reads sampled blocks (data/sample.npz) and, for the
real codecs, a few complete expert tensors.

Reported units: bits per weight (one FP4 nibble).  The stored baseline is 4.0 bits of payload plus
8 scale bits per 32 weights = 4.25 bits/weight for a whole expert.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time

import numpy as np

import entropy as ent
from st import Checkpoint

PAYLOAD_BITS = 4.0
SCALE_BITS_PER_W = 8 / 32
TOTAL_BITS = PAYLOAD_BITS + SCALE_BITS_PER_W
W_PER_EXPERT = 3 * 2304 * 5120          # w1 + w2 + w3
EXPERT_BYTES = W_PER_EXPERT // 2 + (2 * 2304 * 160 + 5120 * 72)
N_EXPERTS_MODEL = 40 * 384
W_MODEL = W_PER_EXPERT * N_EXPERTS_MODEL

RESULTS = []  # rows for the CSV/markdown table


def row(section, name, bits_per_weight=None, ratio_payload=None, note="", **kw):
    r = {"section": section, "method": name, "bits_per_weight": bits_per_weight,
         "ratio_vs_4bit": ratio_payload, "note": note}
    r.update(kw)
    RESULTS.append(r)
    return r


# --------------------------------------------------------------------------------- section A
def section_a(ck: Checkpoint, nib, scl, meta, layers, n_codec_experts: int):
    print("\n=== A. baselines ===", flush=True)
    import lz4.frame
    import zstandard as zstd

    sym = nib.ravel()
    H0 = ent.order0(sym)
    print(f"nibble order-0 entropy      H(W) = {H0:.4f} bit  ({H0/4:.4f} x 4 bit)")
    row("A", "raw FP4 payload", 4.0, 1.0, "stored form")
    row("A", "order-0 nibble entropy H(W)", H0, H0 / 4, "static Huffman/rANS limit")

    by = (nib[:, 0::2] | (nib[:, 1::2] << 4)).ravel()   # the stored bytes
    Hb = ent.order0(by, k=256)
    print(f"byte order-0 entropy             = {Hb:.4f} bit/byte -> {Hb/2:.4f} bit/weight")
    row("A", "order-0 byte entropy", Hb / 2, Hb / 8, "pairs of nibbles as one symbol")

    Hs = ent.order0(scl, k=256)
    uniq = np.unique(scl)
    print(f"scale byte entropy               = {Hs:.4f} bit/byte over {len(uniq)} distinct values {uniq.tolist()}")
    row("A", "order-0 scale entropy", Hs / 32, Hs / 8, f"{len(uniq)} distinct E8M0 bytes; bits/weight column is the scale share")

    # real codecs on complete expert tensors
    rng = np.random.default_rng(7)
    picks = [(int(L), int(e)) for L in layers[:: max(1, len(layers) // n_codec_experts)]
             for e in rng.choice(384, 1)][:n_codec_experts]
    blobs, sblobs = [], []
    for (L, e) in picks:
        w = [ck.bytes(f"layers.{L}.ffn.experts.{e}.{m}.weight").tobytes() for m in ("w1", "w2", "w3")]
        s = [ck.bytes(f"layers.{L}.ffn.experts.{e}.{m}.scale").tobytes() for m in ("w1", "w2", "w3")]
        blobs.append(b"".join(w))
        sblobs.append(b"".join(s))
    print(f"real codecs on {len(blobs)} complete experts ({len(blobs[0])/1e6:.1f} MB payload each)")

    def measure(name, fn, data):
        t0 = time.time()
        tot_in = tot_out = 0
        for b in data:
            c = fn(b)
            tot_in += len(b)
            tot_out += len(c)
        dt = time.time() - t0
        ratio = tot_out / tot_in
        print(f"  {name:<14} ratio {ratio:.4f}   {tot_in/1e6/dt:6.1f} MB/s enc")
        return ratio, tot_in / 1e6 / dt

    for lvl in (1, 3, 9, 19):
        c = zstd.ZstdCompressor(level=lvl)
        r, mbs = measure(f"zstd-{lvl}", c.compress, blobs)
        row("A", f"zstd level {lvl} (payload)", 4 * r, r, "real codec on whole expert payload", enc_mbs=mbs)
    r, mbs = measure("lz4", lz4.frame.compress, blobs)
    row("A", "lz4 (payload)", 4 * r, r, "real codec", enc_mbs=mbs)
    r, mbs = measure("gzip-6", lambda b: gzip.compress(b, 6), blobs)
    row("A", "gzip -6 (payload)", 4 * r, r, "real codec", enc_mbs=mbs)

    cs = zstd.ZstdCompressor(level=9)
    r, _ = measure("zstd-9 scales", cs.compress, sblobs)
    row("A", "zstd level 9 (scales only)", 0.25 * r, r, "scale bytes are 5.9% of an expert")
    return H0


# --------------------------------------------------------------------------------- section B
def build_features(nib, scl, meta, layers):
    """Per-nibble feature arrays (flattened row-major over blocks x 32 positions)."""
    nblk = nib.shape[0]
    f = {}
    f["sym"] = nib.ravel()
    f["pos"] = np.tile(np.arange(32, dtype=np.uint8), nblk)
    smap = {int(v): i for i, v in enumerate(np.unique(scl))}
    scode = np.array([smap[int(v)] for v in scl], dtype=np.uint8)
    f["scale"] = np.repeat(scode, 32)
    f["n_scale"] = len(smap)
    lmap = {int(L): i for i, L in enumerate(layers)}
    f["layer"] = np.repeat(np.array([lmap[int(x)] for x in meta[:, 0]], dtype=np.uint8), 32)
    f["n_layer"] = len(lmap)
    f["mat"] = np.repeat(meta[:, 2].astype(np.uint8), 32)
    # compact expert id within the sample (layer, expert) -> 0..n-1
    key = meta[:, 0].astype(np.int64) * 1000 + meta[:, 1]
    _, einv = np.unique(key, return_inverse=True)
    f["expert"] = np.repeat(einv.astype(np.int32), 32)
    f["n_expert"] = int(einv.max() + 1)
    # running max magnitude code inside the block (decodable: depends only on earlier positions)
    mag = nib & 7
    rm = np.maximum.accumulate(mag, axis=1)
    runmax = np.zeros_like(rm)
    runmax[:, 1:] = rm[:, :-1]
    f["runmax"] = runmax.ravel()
    # previous nibbles along k within a row (blocks of a row are contiguous in the sample)
    flat = f["sym"]
    prev = np.empty_like(flat)
    prev[0] = 0
    prev[1:] = flat[:-1]
    prev2 = np.empty_like(flat)
    prev2[:2] = 0
    prev2[2:] = flat[:-2]
    prev3 = np.empty_like(flat)
    prev3[:3] = 0
    prev3[3:] = flat[:-3]
    f["prev"], f["prev2"], f["prev3"] = prev, prev2, prev3
    # valid = not within the first 3 nibbles of a row (row starts where block index == 0)
    rowstart = np.repeat(meta[:, 4] == 0, 32) & (f["pos"] < 3)
    f["valid"] = ~rowstart
    f["train"] = np.repeat((np.arange(nblk) % 2 == 0), 32)
    return f


def report_cond(name, f, parts, H0, note="", per_expert_table=False, mask=None):
    sym = f["sym"]
    if mask is None:
        ctx, nctx = ent.ctx_of(parts)
        r = ent.cond(ctx, sym, nctx, train=f["train"])
    else:
        ctx, nctx = ent.ctx_of([(a[mask], c) for a, c in parts])
        r = ent.cond(ctx, sym[mask], nctx, train=f["train"][mask])
    H, Ht = r["H_plugin"], r.get("H_test", float("nan"))
    # table cost: global tables amortize over the whole model; per-expert tables over one expert
    tb = (r["table_bits"] / f["n_expert"] / W_PER_EXPERT) if per_expert_table else (r["table_bits"] / W_MODEL)
    tot = Ht + tb
    print(f"  {name:<46} H={H:.4f}  H_test={Ht:.4f}  +table {tb:.4f} -> {tot:.4f} bit  ({tot/4:.4f}x)"
          f"   [{r['n_ctx_used']} ctx]")
    row("B", name, tot, tot / 4, note, H_plugin=H, H_test=Ht, table_bits_per_weight=tb,
        n_ctx_used=r["n_ctx_used"])
    return r


def section_b(f, H0):
    print("\n=== B. conditional entropy (H_test = held-out, +table = shipped tables) ===", flush=True)
    v = f["valid"]
    report_cond("H(W)", f, [(np.zeros(len(f["sym"]), np.uint8), 1)], H0, "order-0")
    report_cond("H(W | scale)", f, [(f["scale"], f["n_scale"])], H0, "exact E8M0 byte")
    report_cond("H(W | position_in_block)", f, [(f["pos"], 32)], H0)
    report_cond("H(W | scale, position)", f, [(f["scale"], f["n_scale"]), (f["pos"], 32)], H0)
    report_cond("H(W | layer)", f, [(f["layer"], f["n_layer"])], H0)
    report_cond("H(W | layer, matrix)", f, [(f["layer"], f["n_layer"]), (f["mat"], 3)], H0)
    report_cond("H(W | expert, matrix) [per-expert tables]", f,
                [(f["expert"], f["n_expert"]), (f["mat"], 3)], H0,
                "table cost charged per expert", per_expert_table=True)
    report_cond("H(W | layer, matrix, scale, position)", f,
                [(f["layer"], f["n_layer"]), (f["mat"], 3), (f["scale"], f["n_scale"]), (f["pos"], 32)], H0,
                "practical lower-bound candidate")
    report_cond("H(W_i | W_i-1)  Markov-1", f, [(f["prev"], 16)], H0, mask=v)
    report_cond("H(W_i | W_i-1, W_i-2)  Markov-2", f, [(f["prev"], 16), (f["prev2"], 16)], H0, mask=v)
    report_cond("H(W_i | W_i-1..3)  Markov-3", f,
                [(f["prev"], 16), (f["prev2"], 16), (f["prev3"], 16)], H0, mask=v)
    report_cond("H(W | position, running block max)", f, [(f["pos"], 32), (f["runmax"], 8)], H0,
                "sequentially decodable")
    report_cond("H(W | pos, runmax, W_i-1)", f, [(f["pos"], 32), (f["runmax"], 8), (f["prev"], 16)], H0, mask=v)
    report_cond("H(W | layer, matrix, scale, pos, runmax)", f,
                [(f["layer"], f["n_layer"]), (f["mat"], 3), (f["scale"], f["n_scale"]),
                 (f["pos"], 32), (f["runmax"], 8)], H0, "everything decodable, combined")


def section_b2(f, nib, H0):
    """Richer, still decodable, contexts: running block statistics and explicit per-block side info."""
    print("\n=== B2. richer context models (the 'learned predictor' family) ===", flush=True)
    sym, v = f["sym"], f["valid"]
    mag = (nib & 7).astype(np.int16)
    val = np.array([0, 1, 2, 3, 4, 6, 8, 12], np.int16)[mag]   # 2x the E2M1 magnitude, integer
    # running mean magnitude inside the block, quantized to 8 bins (causal: previous positions only)
    csum = np.cumsum(val, axis=1)
    run = np.zeros_like(csum)
    run[:, 1:] = csum[:, :-1]
    denom = np.arange(32, dtype=np.int16)[None, :].repeat(len(nib), 0)
    denom[:, 0] = 1
    rm = (run * 4 // np.maximum(denom, 1)).clip(0, 31).astype(np.uint8)   # quarter-unit resolution
    runmean = (rm // 4).clip(0, 7).ravel()
    report_cond("H(W | pos, running mean magnitude)", f, [(f["pos"], 32), (runmean, 8)], H0,
                "sequentially decodable")
    report_cond("H(W | scale, pos, runmax, W_i-1)", f,
                [(f["scale"], f["n_scale"]), (f["pos"], 32), (f["runmax"], 8), (f["prev"], 16)], H0, mask=v)
    report_cond("H(W | scale, pos, runmean, W_i-1)", f,
                [(f["scale"], f["n_scale"]), (f["pos"], 32), (runmean, 8), (f["prev"], 16)], H0, mask=v)
    report_cond("H(W | layer, mat, scale, pos, runmax, W_i-1)", f,
                [(f["layer"], f["n_layer"]), (f["mat"], 3), (f["scale"], f["n_scale"]),
                 (f["pos"], 32), (f["runmax"], 8), (f["prev"], 16)], H0, "largest context tried", mask=v)
    # two-pass: an explicit per-block class (quantized block mean magnitude) costs c bits per 32 weights
    bmean = val.mean(1)
    for nc in (2, 4, 8, 16):
        q = np.quantile(bmean, np.linspace(0, 1, nc + 1)[1:-1])
        cls = np.repeat(np.searchsorted(q, bmean).astype(np.uint8), 32)
        ctx, n = ent.ctx_of([(cls, nc), (f["scale"], f["n_scale"]), (f["pos"], 32)])
        r = ent.cond(ctx, sym, n, train=f["train"])
        side = np.log2(nc) / 32
        tot = r["H_test"] + side
        print(f"  {'H(W | block class(' + str(nc) + '), scale, pos)':<46} H={r['H_plugin']:.4f}  "
              f"H_test={r['H_test']:.4f}  +side {side:.4f} -> {tot:.4f} bit  ({tot/4:.4f}x)")
        row("B2", f"two-pass block class ({nc}) + scale + pos", tot, tot / 4,
            f"explicit {np.log2(nc):.0f} bit per 32-weight block of side info")


# --------------------------------------------------------------------------------- section C
def section_c(f, H0):
    print("\n=== C. bit-planes (bit3=sign, bit2:1=exponent, bit0=mantissa) ===", flush=True)
    sym = f["sym"]
    planes = {3: (sym >> 3) & 1, 2: (sym >> 2) & 1, 1: (sym >> 1) & 1, 0: sym & 1}
    names = {3: "bit3 sign", 2: "bit2 exp-hi", 1: "bit1 exp-lo", 0: "bit0 mantissa"}
    indep = 0.0
    for b in (3, 2, 1, 0):
        x = planes[b]
        p1 = float(x.mean())
        H = ent.order0(x, k=2)
        ctx, n = ent.ctx_of([(f["scale"], f["n_scale"]), (f["pos"], 32)])
        r_sp = ent.cond(ctx, x, n, train=f["train"], k=2)
        r_pos = ent.cond(f["pos"].astype(np.int64), x, 32, train=f["train"], k=2)
        ctx2, n2 = ent.ctx_of([(f["prev"], 16)])
        r_prev = ent.cond(ctx2[f["valid"]], x[f["valid"]], n2, train=f["train"][f["valid"]], k=2)
        ctx3, n3 = ent.ctx_of([(f["pos"], 32), (f["runmax"], 8)])
        r_rm = ent.cond(ctx3, x, n3, train=f["train"], k=2)
        indep += r_sp["H_test"]
        print(f"  {names[b]:<14} P(1)={p1:.4f}  H={H:.4f}  H|pos={r_pos['H_plugin']:.4f} "
              f" H|scale,pos={r_sp['H_plugin']:.4f}  H|prev nibble={r_prev['H_plugin']:.4f} "
              f" H|pos,runmax={r_rm['H_plugin']:.4f}")
        row("C", f"{names[b]}", None, None, f"P(1)={p1:.4f}", H_plane=H,
            H_given_pos=r_pos["H_plugin"], H_given_scale_pos=r_sp["H_plugin"],
            H_given_prev_nibble=r_prev["H_plugin"], H_given_pos_runmax=r_rm["H_plugin"])
    print(f"  sum of independent per-plane coders (cond. on scale,pos) = {indep:.4f} bit/weight "
          f"({indep/4:.4f}x)  vs joint H(W|scale,pos)")
    row("C", "bit-plane coders summed (each cond. scale,pos)", indep, indep / 4,
        "independent planes; >= joint symbol entropy")
    # chain rule: coding the planes in order, each conditioned on the higher planes (= exactly H(W))
    chain = 0.0
    for b in (3, 2, 1, 0):
        hi = sym >> (b + 1)
        card = 1 << (3 - b)
        r = ent.cond(hi.astype(np.int64), planes[b], card, train=f["train"], k=2)
        chain += r["H_test"]
    print(f"  chained plane coders (each cond. on higher planes)      = {chain:.4f} bit/weight (= H(W))")
    row("C", "bit-plane chain (cond. on higher planes)", chain, chain / 4, "identical to symbol entropy")


# --------------------------------------------------------------------------------- section D
def section_d(nib, f, H0):
    print("\n=== D. PRNG XOR residual ===", flush=True)
    import zstandard as zstd
    sym = f["sym"]
    for seedname, nseeds in (("per matrix", 3), ("per expert", f["n_expert"]), ("per block", None)):
        if nseeds is None:
            rng = np.random.default_rng(99)
            r = rng.integers(0, 16, size=sym.shape, dtype=np.uint8)
        else:
            g = f["mat"] if seedname == "per matrix" else f["expert"]
            r = np.empty_like(sym)
            for s in range(nseeds):
                m = g == s
                r[m] = np.random.default_rng(1000 + s).integers(0, 16, size=int(m.sum()), dtype=np.uint8)
        res = sym ^ r
        H = ent.order0(res)
        zero = float((res == 0).mean())
        b = (res[0::2] | (res[1::2] << 4)).tobytes()
        zr = len(zstd.ZstdCompressor(level=9).compress(b)) / len(b)
        print(f"  XOR with fixed PRNG ({seedname:<10}): residual H={H:.4f}  zero-nibble {zero:.4f}  zstd-9 {zr:.4f}")
        row("D", f"PRNG XOR ({seedname})", H, H / 4, f"zero-nibble rate {zero:.4f}, zstd-9 {zr:.4f}")
    base_zero = float((sym == 0).mean())
    print(f"  (baseline: zero-nibble rate {base_zero:.4f}, H={H0:.4f})")


# --------------------------------------------------------------------------------- section E
def hamming_setup(nib):
    """blocks -> +-1 float32 bit matrix [N,128] for Hamming via one matmul."""
    bits = np.unpackbits(nib.reshape(-1, 32, 1), axis=2, count=4, bitorder="little")
    return bits.reshape(len(nib), 128)


def nearest(qbits_pm, dbits_pm, chunk=1024):
    """min Hamming distance and argmin of each query row against the dictionary."""
    best = np.full(len(qbits_pm), 1 << 30, dtype=np.int32)
    arg = np.zeros(len(qbits_pm), dtype=np.int32)
    for i in range(0, len(qbits_pm), chunk):
        q = qbits_pm[i:i + chunk]
        d = (128.0 - q @ dbits_pm.T) * 0.5
        j = np.argmin(d, axis=1)
        best[i:i + chunk] = d[np.arange(len(q)), j].astype(np.int32)
        arg[i:i + chunk] = j
    return best, arg


def section_e(nib, f, H0, n_query=20000, sizes=(256, 1024, 4096, 16384, 65536)):
    print("\n=== E. block dictionary (32-weight = 128-bit blocks) ===", flush=True)
    nblk = len(nib)
    packed = np.ascontiguousarray((nib[:, 0::2] | (nib[:, 1::2] << 4)))  # [N,16] the stored bytes
    view = packed.view(np.dtype((np.void, 16))).ravel()
    uniq, cnt = np.unique(view, return_counts=True)
    dup = 1.0 - len(uniq) / nblk
    print(f"  exact duplicate blocks: {dup*100:.4f}%  ({nblk-len(uniq):,} of {nblk:,}); max multiplicity {cnt.max()}")
    row("E", "exact duplicate block rate", None, None, f"{dup*100:.4f}% of {nblk:,} blocks")

    bits = hamming_setup(nib).astype(np.float32)
    pm = bits * 2.0 - 1.0
    rng = np.random.default_rng(5)
    qi = rng.choice(nblk, n_query, replace=False)
    q = pm[qi]
    # random-pair reference distance
    ri = rng.choice(nblk, n_query, replace=False)
    rand_d = (128.0 - (q * pm[ri]).sum(1)) * 0.5
    print(f"  random pair Hamming: mean {rand_d.mean():.2f} bit (of 128), min {rand_d.min():.0f}")
    row("E", "random pair Hamming distance", None, None, f"mean {rand_d.mean():.2f}/128 bit")

    pos_ctx = np.tile(np.arange(32), n_query)
    for D in sizes:
        di = rng.choice(nblk, D, replace=False)
        d = np.ascontiguousarray(pm[di])
        t0 = time.time()
        best, arg = nearest(q, d)
        resid = nib[qi] ^ nib[di][arg]           # XOR residual nibbles
        zero = float((resid == 0).mean())
        Hr = ent.order0(resid.ravel())
        ctx, n = ent.ctx_of([(pos_ctx.astype(np.int64), 32)])
        Hrp = ent.cond(ctx, resid.ravel(), n)["H_plugin"]
        bits_blk = np.log2(D) + 32 * Hr
        bpw = bits_blk / 32
        dict_bpw = D * 128 / W_MODEL
        print(f"  D={D:<6} NN Hamming mean {best.mean():5.2f}  residual zero-nibble {zero:.4f} "
              f" H(res)={Hr:.4f}  -> {bpw + dict_bpw:.4f} bit/weight ({(bpw+dict_bpw)/4:.4f}x)  [{time.time()-t0:.1f}s]")
        row("E", f"block dictionary D={D} (random templates)", bpw + dict_bpw, (bpw + dict_bpw) / 4,
            f"NN Hamming {best.mean():.2f}/128, zero-nibble {zero:.4f}, H(res|pos)={Hrp:.4f}",
            index_bits=float(np.log2(D)), residual_bits_per_weight=Hr, dict_bits_per_weight=dict_bpw)

    # bound: nearest neighbour among ALL other sampled blocks (a 1M-entry dictionary, index cost ignored)
    t0 = time.time()
    nq = min(4000, n_query)
    mask = np.ones(nblk, bool)
    mask[qi] = False                      # never match a query against itself
    didx = np.flatnonzero(mask)
    dpm = np.ascontiguousarray(pm[didx])
    best, arg = nearest(q[:nq], dpm, chunk=256)
    resid = nib[qi[:nq]] ^ nib[didx[arg]]
    Hr = ent.order0(resid.ravel())
    zero = float((resid == 0).mean())
    bpw = (np.log2(len(didx)) + 32 * Hr) / 32
    print(f"  D={len(didx):,} (every other sampled block) NN Hamming mean {best.mean():.2f}  zero-nibble {zero:.4f} "
          f" H(res)={Hr:.4f} -> {bpw:.4f} bit/weight (dictionary storage ignored)  [{time.time()-t0:.1f}s]")
    row("E", f"nearest of all {len(didx):,} sampled blocks", bpw, bpw / 4,
        f"NN Hamming {best.mean():.2f}/128; dictionary storage NOT counted (optimistic bound)")

    # control: destroy all block structure by shuffling each nibble position independently across
    # blocks.  Same marginals, provably no structure -> if the real data matches this, there is none.
    rng2 = np.random.default_rng(11)
    shuf = np.empty_like(nib)
    for p_ in range(32):
        shuf[:, p_] = nib[rng2.permutation(nblk), p_]
    spm = hamming_setup(shuf).astype(np.float32) * 2.0 - 1.0
    sq = spm[qi[:nq]]
    di = rng.choice(nblk, 65536, replace=False)
    b_ctrl, a_ctrl = nearest(sq, np.ascontiguousarray(spm[di]))
    r_ctrl = shuf[qi[:nq]] ^ shuf[di][a_ctrl]
    H_ctrl = ent.order0(r_ctrl.ravel())
    print(f"  control (nibbles shuffled across blocks, D=65536): NN Hamming mean {b_ctrl.mean():.2f} "
          f" H(res)={H_ctrl:.4f}   <- compare with the real D=65536 line")
    row("E", "control: independence-shuffled blocks, D=65536", None, None,
        f"NN Hamming {b_ctrl.mean():.2f}/128, H(res)={H_ctrl:.4f}; matching the real data means no block structure")


# --------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--sample", default="data/sample.npz")
    ap.add_argument("--out", default="results/phase1")
    ap.add_argument("--codec-experts", type=int, default=4)
    ap.add_argument("--sections", default="ABCDE")
    a = ap.parse_args()

    d = np.load(a.sample)
    nib, scl, meta, layers = d["nib"], d["scale"], d["meta"], d["layers"]
    print(f"sample: {len(nib):,} blocks = {len(nib)*32/1e6:.1f}M weights from layers {layers.tolist()}, "
          f"{int(d['n_experts'])} experts/layer, {int(d['n_rows'])} rows/matrix")
    ck = Checkpoint(a.ckpt)
    H0 = ent.order0(nib.ravel())
    if "A" in a.sections:
        H0 = section_a(ck, nib, scl, meta, layers.tolist(), a.codec_experts)
    f = build_features(nib, scl, meta, layers.tolist())
    if "B" in a.sections:
        section_b(f, H0)
    if "2" in a.sections:
        section_b2(f, nib, H0)
    if "C" in a.sections:
        section_c(f, H0)
    if "D" in a.sections:
        section_d(nib, f, H0)
    if "E" in a.sections:
        section_e(nib, f, H0)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump({"sample": {"blocks": int(len(nib)), "weights": int(nib.size),
                          "layers": layers.tolist()}, "rows": RESULTS},
              open(a.out + ".json", "w"), indent=1)
    import csv
    keys = sorted({k for r in RESULTS for k in r})
    head = ["section", "method", "bits_per_weight", "ratio_vs_4bit", "note"]
    keys = head + [k for k in keys if k not in head]
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in RESULTS:
            w.writerow(r)
    with open(a.out + ".md", "w") as fh:
        fh.write("| section | method | bit/weight | ratio vs 4 bit | note |\n|---|---|---|---|---|\n")
        for r in RESULTS:
            b = f"{r['bits_per_weight']:.4f}" if r["bits_per_weight"] is not None else "-"
            rt = f"{r['ratio_vs_4bit']:.4f}" if r["ratio_vs_4bit"] is not None else "-"
            fh.write(f"| {r['section']} | {r['method']} | {b} | {rt} | {r['note']} |\n")
    print(f"\nwrote {a.out}.json / .csv / .md")


if __name__ == "__main__":
    main()
