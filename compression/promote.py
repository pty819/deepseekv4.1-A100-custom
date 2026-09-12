"""Items 4 and 5 of the roadmap: runtime promotion of hot experts, and per-token precision.

A. promotion curve   base b_lo for every expert + the top-K of the current task promoted to 4 bit:
                     damage, stored bits and transferred bits as a function of K.
B. switching cost    how much of the top-K set changes when the task changes (= promotion traffic).
C. per-token signal  is there anything for a per-token precision decision to key on?  The quantization
                     residual of a VQ is close to white, so ||dW x|| ~ ||dW||_F ||x|| / sqrt(dim) with
                     very little token-to-token spread; this measures that spread on a real expert.
"""
from __future__ import annotations

import csv
import json

import numpy as np
import torch

from lossy_alloc import CURVE

BITS3 = np.array([3.0, 3.5, 4.0])
EPS3 = np.array([CURVE[3.0], CURVE[3.5], 0.0])
EXPERT_MB = (3 * 2304 * 5120 // 2 + 2 * 2304 * 160 + 5120 * 72) / 1e6


def touch_weight(P, B):
    return 1.0 - np.power(1.0 - np.clip(P, 0, 1), B)


def main():
    tel = torch.load("../results/route_telemetry.pt", weights_only=False)
    tasks = list(tel["mass"])
    mass = {t: (m := tel["mass"][t].numpy()) / m.sum(1, keepdims=True) for t in tasks}
    cnt = {t: (c := tel["count"][t].numpy()) / c.sum(1, keepdims=True) * 6.0 for t in tasks}
    Mavg = np.mean([mass[t] for t in tasks], 0)
    Pavg = np.mean([cnt[t] for t in tasks], 0)
    nl, E = Mavg.shape
    rows = []

    print("=== A. promotion: base 3.0 bit everywhere + top-K per layer promoted to 4 bit ===")
    print(f"  {'K':>4} {'stored (2 copies)':>17} {'stored (in place)':>17} {'transf B=1':>11} "
          f"{'transf B=32':>12} {'damage: fitted task':>20} {'other tasks':>12}")
    for K in (0, 8, 16, 32, 64, 96, 128, 192, 384):
        dmg_fit, dmg_other = [], []
        for t in tasks:
            hot = np.argsort(-mass[t], 1)[:, :K]
            b = np.full((nl, E), 3.0)
            np.put_along_axis(b, hot, 4.0, 1)
            e = np.interp(b, BITS3, EPS3)
            dmg_fit.append(np.sqrt(np.mean([(mass[t][l] * e[l] ** 2).sum() for l in range(nl)])))
            for u in tasks:
                if u != t:
                    dmg_other.append(np.sqrt(np.mean([(mass[u][l] * e[l] ** 2).sum() for l in range(nl)])))
        b = np.full((nl, E), 3.0)
        np.put_along_axis(b, np.argsort(-Mavg, 1)[:, :K], 4.0, 1)
        st2 = 3.0 + 4.0 * K / E          # both copies resident
        st1 = float(b.mean())            # promoted expert replaces its low-bit copy
        tr1 = float((touch_weight(Pavg, 1) * b).sum() / touch_weight(Pavg, 1).sum())
        tr32 = float((touch_weight(Pavg, 32) * b).sum() / touch_weight(Pavg, 32).sum())
        print(f"  {K:>4} {st2:>17.3f} {st1:>17.3f} {tr1:>11.3f} {tr32:>12.3f} "
              f"{np.mean(dmg_fit)*100:>19.2f}% {np.mean(dmg_other)*100:>11.2f}%")
        rows.append({"K": K, "stored_two_copies": st2, "stored_in_place": st1,
                     "transferred_B1": tr1, "transferred_B32": tr32,
                     "damage_fitted_task_pct": float(np.mean(dmg_fit)) * 100,
                     "damage_other_tasks_pct": float(np.mean(dmg_other)) * 100})

    print("\n=== B. task switch: how much of the top-K set has to be swapped ===")
    print(f"  {'K':>4} " + " ".join(f"{t[:9]:>10}" for t in tasks) + f"  {'traffic GB':>11} {'seconds @1.67GB/s':>18}")
    for K in (16, 32, 64, 128):
        tops = {t: np.argsort(-mass[t], 1)[:, :K] for t in tasks}
        worst = 0.0
        cells = []
        for t in tasks:
            ov = []
            for u in tasks:
                if u == t:
                    continue
                inter = np.mean([len(np.intersect1d(tops[t][l], tops[u][l])) for l in range(nl)])
                ov.append(inter / K)
            cells.append(f"{np.mean(ov)*100:>9.1f}%")
            worst = max(worst, 1 - min(ov))
        gb = worst * K * nl * EXPERT_MB / 1e3
        print(f"  {K:>4} " + " ".join(cells) + f"  {gb:>11.1f} {gb*1e3/1670:>18.1f}")
        rows.append({"K": K, "switch_worst_fraction": worst, "switch_traffic_GB": gb})

    print("\n  (columns: mean overlap of this task's top-K with the other tasks' top-K)")

    print("\n=== C. is there a per-token signal?  spread of ||dW x|| / ||x|| over tokens ===")
    from st import Checkpoint
    from lossy import FP4
    ck = Checkpoint("/mnt/ssd/models/DeepSeek-V4.1-Flash")
    d = np.load("results/vq_3.0.npz")
    dev = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    L, e = 17, 5
    b = ck.bytes(f"layers.{L}.ffn.experts.{e}.w1.weight")
    s = ck.bytes(f"layers.{L}.ffn.experts.{e}.w1.scale")
    c = np.empty((b.shape[0], b.shape[1] * 2), np.uint8)
    c[:, 0::2], c[:, 1::2] = b & 15, b >> 4
    u16 = b[:, 0::2].astype(np.int64) | (b[:, 1::2].astype(np.int64) << 8)
    q = np.empty_like(b)
    q[:, 0::2], q[:, 1::2] = d["lut_lo"][u16], d["lut_hi"][u16]
    cq = np.empty_like(c)
    cq[:, 0::2], cq[:, 1::2] = q & 15, q >> 4
    sc = np.repeat(np.exp2(s.astype(np.float32) - 127.0), 32, axis=1)
    W = torch.from_numpy(FP4[c].astype(np.float32) * sc).to(dev)
    Wq = torch.from_numpy(FP4[cq].astype(np.float32) * sc).to(dev)
    dW = W - Wq
    rng = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(4096, W.shape[1], device=dev, generator=rng)
    r = (x @ dW.T).norm(dim=1) / x.norm(dim=1)
    print(f"  expert w1, VQ 3.0 bit: ||dW x||/||x|| mean {r.mean():.5f}, std {r.std():.5f} "
          f"({100*r.std()/r.mean():.2f}% of the mean), p1 {r.quantile(0.01):.5f}, p99 {r.quantile(0.99):.5f}")
    print(f"  => the per-token spread is {100*r.std()/r.mean():.2f}%: a per-token 3/4-bit decision has "
          f"almost nothing to key on from the residual itself.")
    rows.append({"per_token_rel_spread_pct": float(100 * r.std() / r.mean())})
    json.dump(rows, open("results/promote.json", "w"), indent=1)
    keys = sorted({k for r in rows for k in r})
    with open("results/promote.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote results/promote.{json,csv}")


if __name__ == "__main__":
    main()
