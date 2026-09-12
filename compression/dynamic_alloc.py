"""Quality per byte TRANSFERRED, not per average bit.

Two different budgets pull the allocation in opposite directions:

  storage / VRAM   cost_e = b_e                       (every expert is stored once)
  bandwidth        cost_e = w_e(B) * b_e              (an expert is read only if some token in the
                                                       step routes to it: w_e(B) = 1 - (1 - P_e)^B)

Damage is  sum_e m_e eps(b_e)^2  with m_e the routing mass.  At batch size 1, w_e = P_e ~ m_e, so the
Lagrangian becomes sum_e m_e [eps(b_e)^2 + lam b_e]: **the routing frequency cancels and the optimal
allocation is uniform**.  At large batch every expert is touched every step, w_e -> 1, the cost is flat
per expert and the frequency-weighted allocation is optimal again.  This script computes where the
crossover is for the real routing distribution, and converts every measured PPL point into
"bits per weight actually transferred" at a given batch size.
"""
from __future__ import annotations

import csv
import json

import numpy as np
import torch

import lossy_alloc as LA
from lossy_alloc import CURVE, allocate

BITS3 = np.array([3.0, 3.5, 4.0])                       # the rates we have codebooks and PPL for
EPS3 = np.array([CURVE[3.0], CURVE[3.5], 0.0])


def touch_weight(P: np.ndarray, B: int) -> np.ndarray:
    """w_e(B): probability that at least one of B tokens routes to expert e."""
    return 1.0 - np.power(1.0 - np.clip(P, 0, 1), B)


def main():
    tel = torch.load("../results/route_telemetry.pt", weights_only=False)
    tasks = list(tel["count"])
    # P_e = probability that a token routes to expert e (top-6 of 384, so sum_e P_e = 6)
    Cnt = {t: tel["count"][t].numpy() for t in tasks}
    P = {t: c / c.sum(1, keepdims=True)[:, :] * 6.0 for t, c in Cnt.items()}
    mass = {t: (m := tel["mass"][t].numpy()) / m.sum(1, keepdims=True) for t in tasks}
    Pavg = np.mean([P[t] for t in tasks], 0)
    Mavg = np.mean([mass[t] for t in tasks], 0)
    nl, E = Pavg.shape
    print(f"routing: {nl} layers x {E} experts, top-6.  sum_e P_e = {Pavg.sum(1).mean():.2f}")

    print("\n=== how many distinct experts a step touches (and the bytes that implies) ===")
    print(f"  {'batch':>6} {'distinct experts/layer':>23} {'bytes/token vs B=inf':>21}")
    for B in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        w = touch_weight(Pavg, B)
        print(f"  {B:>6} {w.sum(1).mean():>23.1f} {w.sum(1).mean()/B/6.0:>20.2f}x")

    print("\n=== the same nominal mean rate costs different BYTES depending on the allocation ===")
    rows = []
    allocs = {}
    LA.BITS, LA.EPS = BITS3, EPS3
    for target in (3.5, 3.25):
        allocs[f"mixed{target}"] = np.stack([allocate(Mavg[l], target) for l in range(nl)])
    for name in ("uniform3.5", "uniform3.0", "uniform3.75", "mixed3.5", "mixed3.25"):
        b = allocs.get(name)
        if b is None:
            b = np.full((nl, E), float(name.replace("uniform", "")))
        dmg = float(np.sqrt(np.mean([(Mavg[l] * np.interp(b[l], BITS3, EPS3) ** 2).sum() for l in range(nl)])))
        r = {"alloc": name, "storage_bits": float(b.mean()), "expert_damage_rms_pct": dmg * 100}
        for B in (1, 8, 32, 128):
            w = touch_weight(Pavg, B)
            r[f"transferred_bits_B{B}"] = float((w * b).sum() / w.sum())
        rows.append(r)
    print(f"  {'allocation':<12} {'storage bit/w':>13} " + " ".join(f"{'B=' + str(B):>8}" for B in (1, 8, 32, 128)))
    for r in rows:
        print(f"  {r['alloc']:<12} {r['storage_bits']:>13.3f} "
              + " ".join(f"{r[f'transferred_bits_B{B}']:>8.3f}" for B in (1, 8, 32, 128)))
    print("  (transferred bit/w = sum_e w_e(B) b_e / sum_e w_e(B): what the HBM/NVMe actually moves)")

    print("\n=== bandwidth-optimal vs storage-optimal allocation, per batch size ===")
    print(f"  {'batch':>6} {'budget bit/w':>12} {'uniform dmg':>12} {'freq-weighted dmg':>18} {'bw-optimal dmg':>15}")
    out2 = []
    for B in (1, 8, 32, 128, 1024):
        w = touch_weight(Pavg, B)
        budget = 3.5
        # uniform at the same transferred rate
        dmg_u = float(np.sqrt(np.mean([(Mavg[l] * np.interp(np.full(E, budget), BITS3, EPS3) ** 2).sum()
                                       for l in range(nl)])))
        # frequency-weighted (storage-optimal) allocation, rescaled to the same TRANSFERRED budget
        bf = _scale_to_transferred(Mavg, w, budget, nl, E)
        dmg_f = float(np.sqrt(np.mean([(Mavg[l] * np.interp(bf[l], BITS3, EPS3) ** 2).sum() for l in range(nl)])))
        # bandwidth-optimal: minimise sum m eps^2 + lam sum w b  (cost per expert is w_e, not 1)
        bb = _bandwidth_optimal(Mavg, w, budget, nl, E)
        dmg_b = float(np.sqrt(np.mean([(Mavg[l] * np.interp(bb[l], BITS3, EPS3) ** 2).sum() for l in range(nl)])))
        print(f"  {B:>6} {budget:>12.2f} {dmg_u*100:>11.2f}% {dmg_f*100:>17.2f}% {dmg_b*100:>14.2f}%")
        out2.append({"batch": B, "transferred_budget": budget, "uniform_pct": dmg_u * 100,
                     "freq_weighted_pct": dmg_f * 100, "bandwidth_optimal_pct": dmg_b * 100})
    json.dump({"bytes": rows, "batch": out2}, open("results/dynamic_alloc.json", "w"), indent=1)
    with open("results/dynamic_alloc.csv", "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow(r)

    print("\n=== task overlap: does a per-task allocation transfer to another task? ===")
    print(f"  {'fitted on':<13} " + " ".join(f"{t[:9]:>10}" for t in tasks) + "   uniform-routing")
    LA.BITS, LA.EPS = BITS3, EPS3
    for fit in tasks + ["average"]:
        src = Mavg if fit == "average" else mass[fit]
        b = np.stack([allocate(src[l], 3.5) for l in range(nl)])
        cells = []
        for t in tasks:
            d = float(np.sqrt(np.mean([(mass[t][l] * np.interp(b[l], BITS3, EPS3) ** 2).sum() for l in range(nl)])))
            cells.append(f"{d*100:>9.2f}%")
        anyt = float(np.sqrt(np.mean(np.interp(b, BITS3, EPS3) ** 2)))
        print(f"  {fit:<13} " + " ".join(cells) + f"   {anyt*100:>9.2f}%")
    print("  (rows: which task's routing the bit allocation was fitted on; columns: which task it is used on)")


def _scale_to_transferred(M, w, budget, nl, E):
    """Frequency-weighted allocation whose TRANSFERRED rate equals `budget`."""
    lo, hi = 2.0, 4.0
    for _ in range(40):
        mid = (lo + hi) / 2
        b = np.stack([allocate(M[l], mid) for l in range(nl)])
        t = (w * b).sum() / w.sum()
        if t > budget:
            hi = mid
        else:
            lo = mid
    return np.stack([allocate(M[l], (lo + hi) / 2) for l in range(nl)])


def _bandwidth_optimal(M, w, budget, nl, E):
    """min sum_e m_e eps(b_e)^2  s.t.  sum_e w_e b_e / sum_e w_e <= budget."""
    lo, hi = 1e-12, 1e6
    for _ in range(200):
        lam = np.sqrt(lo * hi)
        cost = M[:, :, None] * EPS3[None, None, :] ** 2 + lam * w[:, :, None] * BITS3[None, None, :]
        b = BITS3[cost.argmin(2)]
        if (w * b).sum() / w.sum() > budget:
            lo = lam
        else:
            hi = lam
    cost = M[:, :, None] * EPS3[None, None, :] ** 2 + np.sqrt(lo * hi) * w[:, :, None] * BITS3[None, None, :]
    return BITS3[cost.argmin(2)]


if __name__ == "__main__":
    main()
