"""Section I: measured storage bandwidth and the decode-time budget a codec would have to fit in.

An expert is 18.8 MB (17.7 MB FP4 payload + 1.1 MB E8M0 scales).  Streaming it from disk costs
raw/BW; a compressed expert costs ratio*raw/BW + decode.  The break-even decode budget is the
difference: any decoder slower than that makes streaming *slower*, however good the ratio.
"""
from __future__ import annotations

import argparse
import csv
import json
import mmap
import os
import time

import numpy as np

EXPERT_BYTES = 3 * 2304 * 5120 // 2 + (2 * 2304 * 160 + 5120 * 72)


def read_direct(path: str, nbytes: int, chunk: int, offset: int = 0) -> float:
    """Bytes/s reading with O_DIRECT (page cache bypassed), from `offset`."""
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        buf = mmap.mmap(-1, chunk)                       # page-aligned
        off = (offset // 4096) * 4096
        os.lseek(fd, off, os.SEEK_SET)
        got = 0
        t0 = time.perf_counter()
        while got < nbytes:
            n = os.preadv(fd, [buf], off + got)
            if n <= 0:
                break
            got += n
        dt = time.perf_counter() - t0
        return got / dt
    finally:
        os.close(fd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--out", default="results/phase1_io")
    ap.add_argument("--best-ratio", type=float, default=0.8897,
                    help="best measured lossless ratio for a whole expert (payload + scales)")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.ckpt) if f.endswith(".safetensors"))
    res = {"chunks": {}}
    print("=== I. storage bandwidth (O_DIRECT, page cache bypassed) ===")
    for chunk_mb, total_mb in ((1, 256), (4, 512), (19, 512), (64, 1024)):
        bws = []
        for i, fn in enumerate(files[2:6]):
            p = os.path.join(a.ckpt, fn)
            bw = read_direct(p, total_mb * 2**20 // 4, chunk_mb * 2**20, offset=i * 512 * 2**20)
            bws.append(bw)
        bw = float(np.mean(bws))
        res["chunks"][f"{chunk_mb}MB"] = bw / 1e6
        print(f"  {chunk_mb:>3} MB reads: {bw/1e6:8.1f} MB/s   ({EXPERT_BYTES/bw*1e3:6.2f} ms per 18.8 MB expert)")
    bw = max(res["chunks"].values()) * 1e6
    # threaded / queued reads: 4 readers on different files
    print(f"\n  taking effective bandwidth = {bw/1e6:.0f} MB/s")

    t_base = EXPERT_BYTES / bw
    print(f"\n  baseline: 18.8 MB / {bw/1e6:.0f} MB/s = {t_base*1e3:.2f} ms per expert "
          f"({1/t_base:.1f} experts/s, {bw/1e6:.0f} MB/s)")
    print("\n=== decode budget vs compression ratio ===")
    print("  ratio   compressed   I/O time   decode budget   required decoder throughput (output MB/s)")
    rows = []
    for r in (a.best_ratio, 0.9, 0.8, 0.7, 0.6, 0.5):
        t_io = EXPERT_BYTES * r / bw
        budget = t_base - t_io
        need = EXPERT_BYTES / budget / 1e6 if budget > 0 else float("inf")
        tag = "  <- best measured" if abs(r - a.best_ratio) < 1e-9 else ""
        print(f"  {r:5.3f}  {EXPERT_BYTES*r/1e6:7.1f} MB  {t_io*1e3:7.2f} ms  {budget*1e3:9.2f} ms   "
              f"{need:10.0f} MB/s{tag}")
        rows.append({"ratio": r, "compressed_MB": EXPERT_BYTES * r / 1e6, "io_ms": t_io * 1e3,
                     "decode_budget_ms": budget * 1e3, "required_decode_MBs": need})
    res["baseline_ms"] = t_base * 1e3
    res["bandwidth_MBs"] = bw / 1e6
    res["rows"] = rows
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {a.out}.json / .csv")


if __name__ == "__main__":
    main()
