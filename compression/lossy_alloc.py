"""L4: routing-aware mixed precision.

A MoE has a lever a dense model does not: expert usage is very unequal, so bits can be spent where the
routing mass is.  This takes the measured rate-distortion curve of the E2M1-snapped VQ (lossy.py, the
kernel-friendly variant) and the recorded routing telemetry, and solves

    minimise  sum_e  mass_e * eps(b_e)^2     subject to   mean_e b_e <= B

over the discrete bit choices, by a Lagrangian sweep.  The comparison is against spending the same
average rate uniformly.  Damage is reported as the mass-weighted RMS expert-output error, and then
re-evaluated under each individual task's routing distribution (the allocation is fitted on the
average, so per-task numbers are the honest ones).
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import torch

# measured on a real expert (lossy.py L3, VQ dim 4, E2M1-snapped codebook): bits -> relative RMS
# error of the full SwiGLU expert output
CURVE = {4.0: 0.0, 3.75: 0.0840, 3.5: 0.1540, 3.25: 0.2130, 3.0: 0.2677, 2.75: 0.3213, 2.5: 0.3863}
BITS = np.array(sorted(CURVE))
EPS = np.array([CURVE[b] for b in BITS])


def uniform_eps(b: float) -> float:
    return float(np.interp(b, BITS, EPS))


def allocate(mass: np.ndarray, target: float, floor: float = 0.0) -> np.ndarray:
    """Bits per expert minimising sum mass*eps^2 at mean rate `target` (Lagrangian sweep)."""
    ok = BITS >= floor
    bits, eps = BITS[ok], EPS[ok]
    lo, hi = 1e-9, 1e6
    for _ in range(200):
        lam = np.sqrt(lo * hi)
        cost = mass[:, None] * eps[None, :] ** 2 + lam * bits[None, :]
        b = bits[cost.argmin(1)]
        if b.mean() > target:
            lo = lam
        else:
            hi = lam
    cost = mass[:, None] * eps[None, :] ** 2 + np.sqrt(lo * hi) * bits[None, :]
    return bits[cost.argmin(1)]


def run(avg, M, tasks, nl, target, floor, rows):
    b = np.stack([allocate(avg[l], target, floor) for l in range(nl)])
    e = np.interp(b, BITS, EPS)
    eps_mixed = float(np.sqrt(np.mean([(avg[l] * e[l] ** 2).sum() for l in range(nl)])))
    per_task = {t: float(np.sqrt(np.mean([(M[t][l] * e[l] ** 2).sum() for l in range(nl)]))) for t in tasks}
    worst = max(per_task.values())
    any_task = float(np.sqrt((e ** 2).mean()))   # a task routing uniformly: the worst case for ANY task
    eps_u = uniform_eps(float(b.mean()))
    hist = {float(x): int((b == x).sum()) for x in BITS if (b == x).sum()}
    tot = (float(b.mean()) + 0.125) / 4.25       # with 4-bit scales
    print(f"  {b.mean():10.3f} {tot:6.3f} {eps_u*100:7.2f}% {eps_mixed*100:7.2f}% {worst*100:10.2f}% "
          f"{any_task*100:13.2f}%  {hist}")
    rows.append({"floor_bits": floor, "mean_bits": float(b.mean()), "ratio_incl_4bit_scales": tot,
                 "uniform_expert_rms_err_pct": eps_u * 100, "mixed_expert_rms_err_pct": eps_mixed * 100,
                 "worst_task_rms_err_pct": worst * 100, "any_task_bound_rms_err_pct": any_task * 100,
                 "per_task": {k: v * 100 for k, v in per_task.items()}, "bit_histogram": hist})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", default="../results/route_telemetry.pt")
    ap.add_argument("--out", default="results/lossy_alloc")
    a = ap.parse_args()
    d = torch.load(a.telemetry, weights_only=False)
    tasks = list(d["mass"])
    M = {t: d["mass"][t].numpy() for t in tasks}          # [layers, experts] routing weight mass
    nl, E = M[tasks[0]].shape
    for t in tasks:
        M[t] = M[t] / M[t].sum(1, keepdims=True)
    avg = np.mean([M[t] for t in tasks], 0)
    print(f"telemetry: {nl} layers x {E} experts, tasks {tasks}")
    print("\n=== routing concentration (share of routing mass, mean over layers) ===")
    print(f"  {'top-k':>6} {'average':>9} " + " ".join(f"{t[:9]:>9}" for t in tasks))
    for k in (16, 32, 64, 96, 128, 192):
        cov_avg = np.mean(np.sort(avg, 1)[:, ::-1][:, :k].sum(1))
        cols = []
        for t in tasks:
            # the hot set is chosen on the AVERAGE distribution, then charged at the task's own mass
            hot = np.argsort(-avg, 1)[:, :k]
            cols.append(np.mean(np.take_along_axis(M[t], hot, 1).sum(1)))
        print(f"  {k:>6} {cov_avg*100:8.2f}% " + " ".join(f"{c*100:8.2f}%" for c in cols))

    rows = []
    for floor in (2.5, 3.0):
        print(f"\n=== mixed precision vs uniform, same average rate (per-expert floor {floor} bit) ===")
        print(f"  {'mean bit/w':>10} {'ratio':>6} {'uniform':>8} {'mixed':>8} {'worst task':>11} "
              f"{'any-task bound':>14}  bit histogram")
        for target in (3.75, 3.5, 3.25, 3.0):
            run(avg, M, tasks, nl, target, floor, rows)
    json.dump(rows, open(a.out + ".json", "w"), indent=1)

    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["floor_bits", "mean_bits", "ratio_incl_4bit_scales",
                                           "uniform_expert_rms_err_pct", "mixed_expert_rms_err_pct",
                                           "worst_task_rms_err_pct", "any_task_bound_rms_err_pct",
                                           "bit_histogram", "per_task"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {a.out}.json / .csv")
    print("\nNOTE: eps is the relative RMS error of one expert's output, not model quality.  The FP4")
    print("checkpoint itself sits at roughly 20% by the same measure (11.5% weight RMS x sqrt(3)),")
    print("so these numbers say how much noise is ADDED on top of what the model already tolerates.")
    print("Only a perplexity / benchmark run on the real model can turn them into a quality answer.")


if __name__ == "__main__":
    main()
