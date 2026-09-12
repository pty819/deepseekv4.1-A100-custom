"""Section J: assemble the final decision table.

Combines the measured entropies (phase1), the scale study (phase1_fg / phase1_extra) and the measured
storage bandwidth (phase1_io) into one table: theoretical bit/weight, real ratio, ratio with metadata,
decode throughput, per-expert I/O + decode time, verdict.
"""
from __future__ import annotations

import csv
import json
import os
import time

W_PER_EXPERT = 3 * 2304 * 5120
SCALES_PER_EXPERT = 2 * 2304 * 160 + 5120 * 72
EXPERT_BYTES = W_PER_EXPERT // 2 + SCALES_PER_EXPERT
BASE_BITS = 8 * EXPERT_BYTES / W_PER_EXPERT           # 4.25 bit per weight, whole expert

VERDICT = [(0.90, "reject (>0.90)"), (0.80, "only if decode is ~free"), (0.65, "promising"), (0.0, "worth a GPU codec")]


def verdict(ratio: float) -> str:
    for thr, v in VERDICT:
        if ratio > thr:
            return v
    return VERDICT[-1][1]


def expert_bits(payload_bits: float, scale_bits: float) -> tuple[float, float]:
    """(bit/weight over a whole expert, ratio vs the stored 4.25 bit/weight)."""
    b = payload_bits + scale_bits * SCALES_PER_EXPERT / W_PER_EXPERT
    return b, b / BASE_BITS


def main():
    ck = "/mnt/ssd/models/DeepSeek-V4.1-Flash"
    io = json.load(open("results/phase1_io.json"))
    bw = io["bandwidth_MBs"] * 1e6
    t_base = EXPERT_BYTES / bw

    # measured decode throughput of the real codecs, on one real expert
    import lz4.frame
    import zstandard as zstd
    from st import Checkpoint
    c = Checkpoint(ck)
    payload = b"".join(c.bytes(f"layers.17.ffn.experts.5.{m}.weight").tobytes() for m in ("w1", "w2", "w3"))
    scales = b"".join(c.bytes(f"layers.17.ffn.experts.5.{m}.scale").tobytes() for m in ("w1", "w2", "w3"))
    blob = payload + scales
    dec = {}
    for name, comp, dfn in (("zstd-3", zstd.ZstdCompressor(level=3).compress, zstd.ZstdDecompressor().decompress),
                            ("zstd-9", zstd.ZstdCompressor(level=9).compress, zstd.ZstdDecompressor().decompress),
                            ("lz4", lz4.frame.compress, lz4.frame.decompress)):
        z = comp(blob)
        t0 = time.perf_counter()
        n = 5
        for _ in range(n):
            dfn(z, max_output_size=len(blob) + 64) if name.startswith("zstd") else dfn(z)
        dt = (time.perf_counter() - t0) / n
        rp = len(comp(payload)) / len(payload)
        rs = len(comp(scales)) / len(scales)
        dec[name] = (rp, rs, len(blob) / dt / 1e6)
        print(f"  {name}: payload ratio {rp:.4f}, scale ratio {rs:.4f}, whole expert {len(z)/len(blob):.4f}, "
              f"decode {dec[name][2]:.0f} MB/s (1 core)")

    # the entropy numbers measured in the other phases
    P_RAW, P_H0 = 4.0, 3.8931
    P_BEST = 3.7497            # H(W | scale, position, running block max, previous nibble), held out
    S_RAW, S_H0, S_PACK = 8.0, 1.0041, 4.0   # full-model scan: 11 distinct E8M0 values, H0 = 1.0041 bit
    rows = []

    def add(method, theo_bits, real_ratio, payload_bits, scale_bits, decode, note, decode_bytes=None):
        b, r = expert_bits(payload_bits, scale_bits)
        t_io = EXPERT_BYTES * r / bw
        t_dec = ((decode_bytes or EXPERT_BYTES) / 1e6 / decode) if decode else 0.0
        need = (EXPERT_BYTES / 1e6) / (t_base - t_io) if t_io < t_base else float("nan")
        rows.append({
            "method": method, "theoretical_bit_per_weight": theo_bits, "real_ratio_payload": real_ratio,
            "expert_bit_per_weight": b, "expert_ratio_incl_metadata": r,
            "decode_MBs": decode if decode else None,
            "io_ms": t_io * 1e3, "decode_ms": t_dec * 1e3, "total_ms": (t_io + t_dec) * 1e3,
            "vs_baseline_ms": (t_io + t_dec) * 1e3 - t_base * 1e3,
            "break_even_decode_MBs": need,
            "total_ms_overlapped": max(t_io, t_dec) * 1e3,
            "min_decode_MBs_for_overlap": (EXPERT_BYTES / 1e6) / t_io,
            "verdict": verdict(r), "note": note})

    add("raw FP4 (stored form)", 4.0, 1.0, P_RAW, S_RAW, None, "baseline: 18.8 MB, 11.28 ms from disk")
    for nm, note in (("zstd-3", "general-purpose LZ; finds nothing in the payload, only in the scales"),
                     ("zstd-9", ""), ("lz4", "expands the payload; the gain is all from the scales")):
        rp, rs, dmb = dec[nm]
        add(nm, None, rp, 4 * rp, 8 * rs, dmb, note)
    add("order-0 nibble rANS", P_H0, None, P_H0, S_RAW, None, "static 16-symbol table")
    add("H(W | scale_bin, position)", 3.7710, None, 3.7710, S_RAW, None, "192 contexts, held out")
    add("Markov-1", 3.8675, None, 3.8675, S_RAW, None, "16 contexts")
    add("Markov-2", 3.8454, None, 3.8454, S_RAW, None, "256 contexts")
    add("bit-plane coders (cond. scale,pos)", 3.8708, None, 3.8708, S_RAW, None,
        "worse than one joint symbol coder")
    add("PRNG XOR", 4.0, 1.0000, 4.0, S_RAW, None, "provably a bijection: entropy unchanged (measured 4.0000)")
    add("block dictionary 65536 + XOR residual", 3.8862, None, 3.8862, S_RAW, None,
        "NN Hamming 35/128; the independence control gives 36.7")
    add("nearest-block residual (1M dictionary)", 3.9783, None, 3.9783, S_RAW, None,
        "worse than raw once the index is paid")
    add("best context model (payload only)", P_BEST, None, P_BEST, S_RAW, None,
        "H(W | scale, pos, running block max, prev nibble), 6867 contexts, held out")
    add("scales: 4-bit packing only", None, 1.0, P_RAW, S_PACK, 20000,
        "11 distinct E8M0 values model-wide; decode is one shift, foldable into the FP4 kernels",
        decode_bytes=SCALES_PER_EXPERT)
    add("scales: entropy coded only", None, None, P_RAW, S_H0, 1000,
        "1.012 bit/scale; payload untouched, random access preserved",
        decode_bytes=SCALES_PER_EXPERT)
    add("best combined (context rANS + coded scales)", P_BEST + S_H0 / 32, None, P_BEST, S_H0, None,
        "the floor of everything measured")
    add("+-0 alias merged (NOT byte-lossless)", P_BEST - 0.1166, None, P_BEST - 0.1166, S_H0, None,
        "codes 0 and 8 both dequantize to zero; exactly P(mag=0)=0.1166 bit saved because the sign bit "
        "measures 1.0000 under every context; inference-identical, checkpoint bytes change")

    hdr = ["method", "theoretical_bit_per_weight", "real_ratio_payload", "expert_bit_per_weight",
           "expert_ratio_incl_metadata", "decode_MBs", "io_ms", "decode_ms", "total_ms",
           "vs_baseline_ms", "break_even_decode_MBs", "total_ms_overlapped",
           "min_decode_MBs_for_overlap", "verdict", "note"]
    os.makedirs("results", exist_ok=True)
    with open("results/summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    json.dump({"bandwidth_MBs": bw / 1e6, "baseline_ms": t_base * 1e3, "rows": rows},
              open("results/summary.json", "w"), indent=1)
    fmt = lambda v, n=4: ("-" if v is None else f"{v:.{n}f}")
    lines = ["| method | theoretical bit/weight | real ratio (payload) | expert bit/weight | ratio incl. metadata | decode MB/s | I/O ms | decode ms | total ms | break-even decode MB/s | overlapped ms | decode MB/s needed to overlap | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    print(f"\n{'method':<44} {'theo':>6} {'ratio':>7} {'bit/w':>7} {'ratio+md':>9} {'I/O ms':>7} {'dec ms':>7} {'tot ms':>7} {'need MB/s':>10} {'ovl ms':>7} {'ovl MB/s':>9}  verdict")
    for r in rows:
        print(f"{r['method']:<44} {fmt(r['theoretical_bit_per_weight'],3):>6} "
              f"{fmt(r['real_ratio_payload']):>7} {fmt(r['expert_bit_per_weight'],3):>7} "
              f"{fmt(r['expert_ratio_incl_metadata']):>9} {r['io_ms']:7.2f} {r['decode_ms']:7.2f} {r['total_ms']:7.2f} "
              f"{r['break_even_decode_MBs']:10.0f} {r['total_ms_overlapped']:7.2f} "
              f"{r['min_decode_MBs_for_overlap']:9.0f}  {r['verdict']}")
        dmb = "-" if not r["decode_MBs"] else f"{r['decode_MBs']:.0f}"
        lines.append(f"| {r['method']} | {fmt(r['theoretical_bit_per_weight'],3)} | {fmt(r['real_ratio_payload'])} | "
                     f"{fmt(r['expert_bit_per_weight'],3)} | {fmt(r['expert_ratio_incl_metadata'])} | {dmb} | "
                     f"{r['io_ms']:.2f} | {r['decode_ms']:.2f} | {r['total_ms']:.2f} | "
                     f"{r['break_even_decode_MBs']:.0f} | {r['total_ms_overlapped']:.2f} | "
                     f"{r['min_decode_MBs_for_overlap']:.0f} | {r['verdict']} |")
    open("results/summary.md", "w").write("\n".join(lines) + "\n")
    print("\nwrote results/summary.csv / .json / .md")


if __name__ == "__main__":
    main()
