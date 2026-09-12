"""The speed/quality Pareto frontier, solved as: minimise bits subject to a PPL budget.

Two cost axes, because they do not agree:
  stored bits       what VRAM / unified memory / the checkpoint costs   (cost_e = b_e)
  transferred bits  what the memory system actually moves per step      (cost_e = w_e(B) b_e,
                    w_e(B) = 1 - (1-P_e)^B: an expert is read only if some token in the step wants it)

At B = 1 the routing frequency cancels out of the allocation problem and uniform rates are optimal;
at large B every expert is touched anyway and the frequency-weighted allocation wins.  Each measured
PPL point is therefore placed on both axes and the frontier is reported separately.
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import torch

import lossy_alloc as LA
from dynamic_alloc import touch_weight, _bandwidth_optimal
from lossy_alloc import CURVE, allocate

RATES = [3.0, 3.25, 3.5, 4.0]


def alloc_of(tag: str, mass, Pavg, nl, E):
    """Recover the per-expert bit map of a measured configuration."""
    LA.BITS = np.array(RATES)
    LA.EPS = np.array([CURVE[b] if b in CURVE else 0.0 for b in RATES])
    if tag == "baseline":
        return np.full((nl, E), 4.0)
    if tag.startswith("vq") and tag.endswith("-all"):
        return np.full((nl, E), float(tag[2:-4]))
    if tag.startswith("mixed"):
        t = float(tag[5:].split("-")[0])
        return np.stack([allocate(mass[l], t) for l in range(nl)])
    if tag.startswith("bwopt"):
        B = int(tag.split("B")[-1])
        return _bandwidth_optimal(mass, touch_weight(Pavg, B), float(tag[5:].split("-")[0]), nl, E)
    if tag == "vq3.0-half":
        b = np.full((nl, E), 4.0)
        b[20:] = 3.0
        return b
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb-per-token", type=float, default=0.233, help="measured FP4 bytes read per token")
    ap.add_argument("--tok-s", type=float, default=9.15, help="measured FP4 tokens/s on that machine")
    ap.add_argument("--batch", type=int, default=1, help="batch size of that measurement")
    ap.add_argument("--out", default="results/pareto")
    a = ap.parse_args()

    tel = torch.load("../results/route_telemetry.pt", weights_only=False)
    mass = np.mean([v.numpy() for v in tel["mass"].values()], 0)
    mass = mass / mass.sum(1, keepdims=True)
    cnt = np.mean([v.numpy() for v in tel["count"].values()], 0)
    Pavg = cnt / cnt.sum(1, keepdims=True) * 6.0
    nl, E = mass.shape

    runs = {}
    for line in open("results/ppl.jsonl"):
        d = json.loads(line)
        if d["chunks"] == 16 and d.get("calib", "none") in ("none", ""):
            runs[d["tag"]] = d           # last run of a tag wins
    base = runs["baseline"]["ppl"]

    rows = []
    for tag, d in runs.items():
        b = alloc_of(tag, mass, Pavg, nl, E)
        if b is None:
            continue
        r = {"config": tag, "ppl": d["ppl"], "delta_ppl_pct": (d["ppl"] / base - 1) * 100,
             "stored_bits": float(b.mean())}
        for B in (1, 8, 32):
            w = touch_weight(Pavg, B)
            r[f"transferred_bits_B{B}"] = float((w * b).sum() / w.sum())
        # whole-expert ratios (payload + 4-bit scales, the lossless part of the change)
        r["stored_ratio"] = (r["stored_bits"] + 0.125) / 4.25
        r["transferred_ratio_B1"] = (r["transferred_bits_B1"] + 0.125) / 4.25
        r["transferred_ratio_B32"] = (r["transferred_bits_B32"] + 0.125) / 4.25
        rows.append(r)
    rows.sort(key=lambda r: r["stored_bits"])

    print(f"{'config':<16} {'PPL':>8} {'dPPL':>7} {'stored':>7} {'trans B=1':>10} {'trans B=32':>11} "
          f"{'ratio(st)':>10} {'ratio(B1)':>10}")
    for r in rows:
        print(f"{r['config']:<16} {r['ppl']:>8.4f} {r['delta_ppl_pct']:>6.2f}% {r['stored_bits']:>7.3f} "
              f"{r['transferred_bits_B1']:>10.3f} {r['transferred_bits_B32']:>11.3f} "
              f"{r['stored_ratio']:>10.3f} {r['transferred_ratio_B1']:>10.3f}")

    print("\n=== minimise bits subject to a PPL budget ===")
    best = []
    for budget in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        ok = [r for r in rows if r["delta_ppl_pct"] <= budget]
        if not ok:
            continue
        s = min(ok, key=lambda r: r["stored_bits"])
        t1 = min(ok, key=lambda r: r["transferred_bits_B1"])
        t32 = min(ok, key=lambda r: r["transferred_bits_B32"])
        print(f"  dPPL <= {budget:>4.1f}%:  stored-optimal {s['config']:<14} {s['stored_bits']:.3f} bit "
              f"({s['stored_ratio']:.3f}x)   |  B=1 transfer-optimal {t1['config']:<14} "
              f"{t1['transferred_bits_B1']:.3f} bit ({t1['transferred_ratio_B1']:.3f}x)   |  "
              f"B=32 {t32['config']} {t32['transferred_bits_B32']:.3f}")
        best.append({"budget_pct": budget, "stored_best": s["config"], "stored_bits": s["stored_bits"],
                     "stored_ratio": s["stored_ratio"], "b1_best": t1["config"],
                     "b1_bits": t1["transferred_bits_B1"], "b1_ratio": t1["transferred_ratio_B1"],
                     "b32_best": t32["config"], "b32_bits": t32["transferred_bits_B32"]})

    print(f"\n=== projection for a streaming machine measured at {a.gb_per_token} GB/token, "
          f"{a.tok_s} tok/s (batch {a.batch}) ===")
    print("  assumes those bytes are expert payload + scales and that time scales with bytes;")
    print("  the VQ decode is ~15 us per expert on an A100, i.e. 0.2% of one expert's NVMe read.")
    key = f"transferred_ratio_B{a.batch}" if a.batch in (1, 32) else "stored_ratio"
    print(f"  {'config':<16} {'dPPL':>7} {'GB/token':>9} {'tok/s (bytes-scaled)':>21}")
    for r in sorted(rows, key=lambda r: r[key]):
        gb = a.gb_per_token * r[key]
        print(f"  {r['config']:<16} {r['delta_ppl_pct']:>6.2f}% {gb:>9.3f} {a.tok_s / r[key]:>21.2f}")
        r["proj_gb_per_token"] = gb
        r["proj_tok_s"] = a.tok_s / r[key]

    json.dump({"rows": rows, "frontier": best}, open(a.out + ".json", "w"), indent=1)
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {a.out}.csv / .json")


if __name__ == "__main__":
    main()
