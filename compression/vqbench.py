"""Driver for cuda/vqbench.cu: read / decode / MMA time per storage format.

Answers the only question the rate choice now hangs on: does the byte saving of a 3.0 / 3.5 bit VQ
survive the LUT decode, or does the unpack eat it?  Every variant walks the same number of weights in
the same warp/lane structure as dsv41/cuda/fp4_tc.cu, so the columns are comparable.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsv41 import cukern                                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NVCC = os.environ.get("NVCC", "nvcc")
WARPS = 4
# bytes per 64 weights (payload only; every format also carries 2 E8M0 scale bytes per 64 weights)
FORMATS = {"fp4": (32, 12), "vq12": (24, 12), "vq14": (28, 14)}


def build() -> bytes:
    src = os.path.join(HERE, "cuda", "vqbench.cu")
    out = os.path.join(HERE, "cuda", ".vqbench.sm80.cubin")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
        subprocess.run([NVCC, "-cubin", "-arch=sm_80", "-O3", "-o", out, src], check=True)
    return open(out, "rb").read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=4)
    ap.add_argument("--experts", type=int, default=16, help="how many expert-sized weight blocks to sweep")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default="results/vqbench")
    a = ap.parse_args()
    dev = torch.device(f"cuda:{a.device}")
    torch.cuda.set_device(dev)
    image = build()
    mod = ctypes.c_void_p()
    cukern._check(cukern._cuda.cuModuleLoadData(ctypes.byref(mod), image), "cuModuleLoadData")

    N, K = 4608, 5120                       # one expert's w13
    E = a.experts
    W_PER_EXPERT = N * K
    lo = torch.randint(0, 256, (E, N, K // 4), dtype=torch.uint8, device=dev)   # 1 byte per group
    lo_fp4 = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
    hi12 = torch.randint(0, 256, (E, N, K // 8), dtype=torch.uint8, device=dev)
    hi14 = torch.randint(0, 256, (E, N, 3 * K // 16), dtype=torch.uint8, device=dev)
    S = torch.randint(118, 126, (E, N, K // 32), dtype=torch.uint8, device=dev)
    lut12 = torch.randint(0, 65536, (4096,), dtype=torch.int32, device=dev).to(torch.uint16)
    lut14 = torch.randint(0, 65536, (16384,), dtype=torch.int32, device=dev).to(torch.uint16)
    out = torch.zeros(N, dtype=torch.float32, device=dev)
    print(f"cuda:{a.device}  {E} expert-sized blocks of [{N}, {K}] = {E*W_PER_EXPERT/1e6:.0f}M weights\n"
          f"FP4 payload {E*W_PER_EXPERT/2/1e6:.0f} MB, VQ12 {E*W_PER_EXPERT*3/8/1e6:.0f} MB, "
          f"VQ14 {E*W_PER_EXPERT*3.5/8/1e6:.0f} MB (+ scales {E*N*K//32/1e6:.0f} MB)")

    def run(name, fmt, smem_bytes, n_iter):
        f = ctypes.c_void_p()
        cukern._check(cukern._cuda.cuModuleGetFunction(ctypes.byref(f), mod, name.encode()), "getFunction")
        if smem_bytes > 48 * 1024:
            cukern._cuda.cuFuncSetAttribute(f, 8, ctypes.c_int(smem_bytes))   # MAX_DYNAMIC_SHARED_SIZE_BYTES
        wlo = lo_fp4 if fmt == "fp4" else lo
        whi = hi12 if fmt == "vq12" else hi14
        lut = lut12 if fmt != "vq14" else lut14
        args = [ctypes.c_void_p(wlo.data_ptr()), ctypes.c_void_p(whi.data_ptr()),
                ctypes.c_void_p(S.data_ptr()), ctypes.c_void_p(lut.data_ptr()),
                ctypes.c_void_p(out.data_ptr()), ctypes.c_int(N), ctypes.c_int(K), ctypes.c_int(E),
                ctypes.c_longlong(wlo.stride(0)), ctypes.c_longlong(whi.stride(0)),
                ctypes.c_longlong(S.stride(0))]
        grid = (N // (WARPS * 8), E, 1)
        for _ in range(3):
            cukern.launch(f, grid, (WARPS * 32, 1, 1), args, dev, shared=smem_bytes)
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            cukern.launch(f, grid, (WARPS * 32, 1, 1), args, dev, shared=smem_bytes)
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) / n_iter

    rows = []
    print(f"\n  {'format':<14} {'bytes/expert':>13} {'read':>9} {'+decode':>9} {'+MMA':>9} "
          f"{'read GB/s':>10} {'total vs FP4':>13}")
    base_total = None
    for suffix in ("", "_k128"):
      for fmt, (b64, lutbits) in FORMATS.items():
        smem = (4096 * 4) if lutbits == 12 else (16384 * 2)
        smem = 0 if fmt == "fp4" else smem
        t_r = run(f"{fmt}_read{suffix}", fmt, smem, a.iters)
        t_d = run(f"{fmt}_dec{suffix}", fmt, smem, a.iters)
        t_f = run(f"{fmt}_full{suffix}", fmt, smem, a.iters)
        payload = W_PER_EXPERT * b64 / 64
        total_bytes = (payload + N * K // 32) * E
        gbs = total_bytes / t_r / 1e9
        if base_total is None:
            base_total = t_f
        name = fmt + (suffix or "_k64")
        print(f"  {name:<14} {(payload + N*K//32)/1e6:>12.2f}M {t_r*1e3:>8.3f}ms {t_d*1e3:>8.3f}ms "
              f"{t_f*1e3:>8.3f}ms {gbs:>9.0f} {t_f/base_total:>12.3f}x")
        rows.append({"format": name, "bits_per_weight": b64 * 8 / 64,
                     "bytes_per_expert_MB": (payload + N * K // 32) / 1e6,
                     "read_ms": t_r * 1e3, "read_decode_ms": t_d * 1e3, "total_ms": t_f * 1e3,
                     "decode_ms": (t_d - t_r) * 1e3, "mma_ms": (t_f - t_d) * 1e3,
                     "read_GBs": gbs, "total_vs_fp4": t_f / base_total,
                     "smem_lut_bytes": smem})
    os.makedirs("results", exist_ok=True)
    json.dump(rows, open(a.out + ".json", "w"), indent=1)
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\n  (read = loads only, +decode = LUT and bit placement, +MMA = the tensor-core issue on top;"
          "\n   the mma count per weight matches fp4_tc.cu)")
    print(f"wrote {a.out}.csv / .json")


if __name__ == "__main__":
    main()
