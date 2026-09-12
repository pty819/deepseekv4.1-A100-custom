"""Perplexity of DeepSeek-V4.1-Flash on wikitext-2, with the routed experts optionally re-quantized.

The VQ built by vq_build.py maps a packed FP4 byte pair to another packed FP4 byte pair, so a rate can
be evaluated by rewriting the expert tensors in place and running the **unmodified** kernels.  The
checkpoint on disk is never touched; only the GPU copies are rewritten after load.

usage:
  python ppl.py --devices 4,5,6,7 --ep --chunks 16                       # baseline
  python ppl.py --devices 4,5,6,7 --ep --chunks 16 --vq results/vq_3.5.npz
  python ppl.py --devices 4,5,6,7 --ep --chunks 16 --vq results/vq_3.5.npz --vq-layers 20-39
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dsv41.load import load_model          # noqa: E402
from dsv41.model import rmsnorm            # noqa: E402


SHRINK = 1024.0   # pseudo-tokens of shrinkage toward the layer-pooled Hessian (--calib-shrink)


# --------------------------------------------------------------------------- re-quantization
def apply_lut(t: torch.Tensor, lut_lo: torch.Tensor, lut_hi: torch.Tensor, chunk_bytes=1 << 24):
    """In-place uint16 -> uint16 lookup over a packed FP4 tensor (4 weights = 2 bytes = one VQ group)."""
    flat = t.view(-1)
    n = flat.numel()
    assert n % 2 == 0
    for i in range(0, n, chunk_bytes * 2):
        s = flat[i:i + chunk_bytes * 2]
        lo, hi = s[0::2], s[1::2]
        idx = lo.long() | (hi.long() << 8)
        lo.copy_(lut_lo[idx])
        hi.copy_(lut_hi[idx])



def shrink_hessian(H: torch.Tensor, seen: torch.Tensor, lam: float) -> torch.Tensor:
    """Per-expert diagonal Hessians shrunk toward the layer-pooled mean.

    H[l, e] is a SUM over the n_e tokens routed to expert e, so the per-token mean is H/n_e.  A cold
    expert can see far fewer tokens than the input dimension, which makes its own estimate noisier than
    the layer average; James-Stein style shrinkage with `lam` pseudo-tokens fixes that:
        mean_e = (H_e + lam * m_layer) / (n_e + lam),   m_layer = sum_e H_e / sum_e n_e
    """
    n = seen.to(torch.float32)                                   # [L, E]
    m = H.sum(1) / n.sum(1).clamp_min(1)[:, None]                # [L, dim] per-token layer mean
    return (H + lam * m[:, None, :]) / (n + lam)[:, :, None]


def unpack(w: torch.Tensor) -> torch.Tensor:
    """uint8 [.., Kh] packed -> uint8 [.., 2*Kh] nibbles (low nibble = even k)."""
    out = torch.empty(*w.shape[:-1], w.shape[-1] * 2, dtype=torch.uint8, device=w.device)
    out[..., 0::2] = w & 0x0F
    out[..., 1::2] = w >> 4
    return out


def pack(c: torch.Tensor) -> torch.Tensor:
    return c[..., 0::2] | (c[..., 1::2] << 4)


def requantize_calibrated(model, vq_path: str, calib_path: str, layers: set[int]) -> dict:
    """Activation-weighted encoding with the same codebook (calib_collect.py -> vq_encode.py)."""
    from dsv41.quant import tile_fp4, untile_fp4
    from vq_encode import CalibEncoder
    d = np.load(vq_path)
    hess = torch.load(calib_path, weights_only=False)
    seen = hess["tokens"]
    H = {"w13": shrink_hessian(hess["H13"], seen, SHRINK), "w2": shrink_hessian(hess["H2"], seen, SHRINK)}
    stats = {"vq": vq_path, "calib": calib_path, "shrink": SHRINK, "bits": float(d["bits"]),
             "vq_layers": sorted(layers), "bytes": 0, "fallback_experts": 0}
    encs, luts = {}, {}
    t0 = time.time()
    for blk in model.blocks:
        if blk.layer_id not in layers:
            continue
        moe = blk.ffn
        shards = moe.ep if moe.ep else [{"w13": moe.w13, "w2": moe.w2, "start": 0, "device": moe.device}]
        for sh in shards:
            dev = sh["w13"].device
            if dev not in encs:
                encs[dev] = CalibEncoder(d["codebook"], dev)
                luts[dev] = (torch.from_numpy(d["lut_lo"]).to(dev), torch.from_numpy(d["lut_hi"]).to(dev))
            enc = encs[dev]
            for e in range(sh["w13"].shape[0]):
                ge = sh["start"] + e
                for key in ("w13", "w2"):
                    t = sh[key][e:e + 1]
                    w = untile_fp4(t) if moe.tiled else t
                    h = H[key][blk.layer_id, ge].to(dev)
                    if float(seen[blk.layer_id, ge]) == 0 or float(h.sum()) == 0:
                        stats["fallback_experts"] += 1
                        apply_lut(w, *luts[dev])
                    else:
                        c = unpack(w[0])
                        w[0] = pack(enc.encode(c, h))
                    sh[key][e:e + 1] = tile_fp4(w) if moe.tiled else w
                    stats["bytes"] += t.numel()
        print(f"  layer {blk.layer_id} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"calibrated re-quantization of {stats['bytes']/1e9:.2f} GB in {len(layers)} layers "
          f"({time.time()-t0:.0f}s, {stats['fallback_experts']} unseen expert-matrices fell back to plain VQ)",
          flush=True)
    return stats


def requantize_mixed(model, vq_paths: dict, telemetry: str, target: float, layers: set[int],
                     calib_path: str = "", task: str = "average", bw_batch: int = 0) -> dict:
    """Routing-aware mixed precision: bits per expert chosen from the available codebooks so that the
    mean rate is `target` and sum(routing mass * expert output error^2) is minimised (lossy_alloc.py)."""
    import torch as T
    from lossy_alloc import CURVE, allocate
    import numpy as _np
    tel = T.load(telemetry, weights_only=False)
    mass = _np.mean([v.numpy() for v in tel["mass"].values()], 0)
    mass = mass / mass.sum(1, keepdims=True)
    rates = sorted(vq_paths)                      # e.g. [3.0, 3.5]; 4.0 means "leave alone"
    import lossy_alloc as LA
    LA.BITS = _np.array(rates + [4.0])
    LA.EPS = _np.array([CURVE[b] for b in rates] + [0.0])
    if task and task != "average":
        mass = tel["mass"][task].numpy()
        mass = mass / mass.sum(1, keepdims=True)
    balloc = None
    if bw_batch:                      # budget the TRANSFERRED bits, not the stored mean
        from dynamic_alloc import touch_weight, _bandwidth_optimal
        cnt = _np.mean([v.numpy() for v in tel["count"].values()], 0)
        P = cnt / cnt.sum(1, keepdims=True) * 6.0
        w = touch_weight(P, bw_batch)
        balloc = _bandwidth_optimal(mass, w, target, mass.shape[0], mass.shape[1])
        print(f"bandwidth-optimal allocation at batch {bw_batch}: transferred "
              f"{(w*balloc).sum()/w.sum():.3f} bit/weight, stored {balloc.mean():.3f}", flush=True)
    luts, stats = {}, {"vq": "mixed", "target_bits": target, "vq_layers": sorted(layers), "bytes": 0,
                      "calib": calib_path or "none", "alloc_task": task, "bw_batch": bw_batch}
    tables = {b: _np.load(p) for b, p in vq_paths.items()}
    if calib_path:
        from dsv41.quant import tile_fp4, untile_fp4
        from vq_encode import CalibEncoder
        hess = T.load(calib_path, weights_only=False)
        seen = hess["tokens"]
        H = {"w13": shrink_hessian(hess["H13"], seen, SHRINK),
             "w2": shrink_hessian(hess["H2"], seen, SHRINK)}
        stats["shrink"] = SHRINK
        encs: dict = {}
        stats["fallback_experts"] = 0
    hist: dict = {}
    t0 = time.time()
    for blk in model.blocks:
        if blk.layer_id not in layers:
            continue
        b = balloc[blk.layer_id] if bw_batch else allocate(mass[blk.layer_id], target)
        moe = blk.ffn
        for sh in (moe.ep if moe.ep else [{"w13": moe.w13, "w2": moe.w2, "start": 0}]):
            for e in range(sh["w13"].shape[0]):
                rate = float(b[sh["start"] + e])
                hist[rate] = hist.get(rate, 0) + 1
                if rate >= 4.0:
                    continue
                dev = sh["w13"].device
                key = (dev, rate)
                if key not in luts:
                    d = tables[rate]
                    luts[key] = (torch.from_numpy(d["lut_lo"]).to(dev), torch.from_numpy(d["lut_hi"]).to(dev))
                for k in ("w13", "w2"):
                    t = sh[k][e:e + 1]
                    stats["bytes"] += t.numel()
                    ge = sh["start"] + e
                    if not calib_path or float(seen[blk.layer_id, ge]) == 0:
                        if calib_path:
                            stats["fallback_experts"] += 1
                        apply_lut(t, *luts[key])
                        continue
                    if key not in encs:
                        encs[key] = CalibEncoder(tables[rate]["codebook"], dev)
                    w = untile_fp4(t) if moe.tiled else t
                    h = H[k][blk.layer_id, ge].to(dev)
                    if float(h.sum()) == 0:
                        apply_lut(w, *luts[key])
                    else:
                        w[0] = pack(encs[key].encode(unpack(w[0]), h))
                    sh[k][e:e + 1] = tile_fp4(w) if moe.tiled else w
    stats["bit_histogram"] = hist
    stats["mean_bits"] = sum(k * v for k, v in hist.items()) / sum(hist.values())
    print(f"mixed precision: mean {stats['mean_bits']:.3f} bit/weight, histogram {hist} ({time.time()-t0:.0f}s)",
          flush=True)
    return stats


def requantize_cb3(model, bits: int, layers: set[int]) -> dict:
    """The DGX Spark project's CB3/CB2 format: per matrix ROW the best 2^bits-of-16 subset of the E2M1
    grid (scale^2-weighted, exhaustive over all C(16, 2^bits) subsets).  Like the VQ, the result is
    still valid FP4 codes, so it is scored with the unmodified kernels."""
    from cb3_sim import CodebookSim
    from dsv41.quant import tile_fp4, tile_fp4_scales, untile_fp4
    stats = {"vq": f"cb{bits}", "bits": float(bits), "vq_layers": sorted(layers), "bytes": 0}
    sims, t0 = {}, time.time()
    for blk in model.blocks:
        if blk.layer_id not in layers:
            continue
        moe = blk.ffn
        for sh in (moe.ep if moe.ep else [{"w13": moe.w13, "s13": moe.s13, "w2": moe.w2, "s2": moe.s2}]):
            dev = sh["w13"].device
            if dev not in sims:
                sims[dev] = CodebookSim(bits, str(dev))
            for wk, sk in (("w13", "s13"), ("w2", "s2")):
                for e in range(sh[wk].shape[0]):
                    t, sc = sh[wk][e:e + 1], sh[sk][e:e + 1]
                    w = untile_fp4(t)[0] if moe.tiled else t[0]
                    s_ = (untile_fp4_scales(sc)[0] if moe.tiled else sc[0])
                    q = sims[dev].requant_packed(w, s_)
                    out = q[None]
                    sh[wk][e:e + 1] = tile_fp4(out) if moe.tiled else out
                    stats["bytes"] += t.numel()
        print(f"  layer {blk.layer_id} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"CB{bits} re-quantization of {stats['bytes']/1e9:.2f} GB ({time.time()-t0:.0f}s)", flush=True)
    return stats


def untile_fp4_scales(s):
    """Inverse of quant.tile_fp4_scales (scales tiled as [N/16][K/128][16][4])."""
    E, N, Kb = s.shape
    return s.view(E, N // 16, Kb // 4, 16, 4).permute(0, 1, 3, 2, 4).reshape(E, N, Kb).contiguous()


def requantize(model, vq_path: str, layers: set[int]) -> dict:
    d = np.load(vq_path)
    stats = {"vq": vq_path, "bits": float(d["bits"]), "vq_layers": sorted(layers), "bytes": 0}
    luts = {}
    t0 = time.time()
    for blk in model.blocks:
        if blk.layer_id not in layers:
            continue
        moe = blk.ffn
        shards = moe.ep if moe.ep else [{"w13": moe.w13, "w2": moe.w2}]
        for sh in shards:
            for key in ("w13", "w2"):
                t = sh[key] if isinstance(sh, dict) else getattr(sh, key)
                dev = t.device
                if dev not in luts:
                    luts[dev] = (torch.from_numpy(d["lut_lo"]).to(dev), torch.from_numpy(d["lut_hi"]).to(dev))
                apply_lut(t, *luts[dev])
                stats["bytes"] += t.numel()
    print(f"re-quantized {stats['bytes']/1e9:.2f} GB of expert payload in {len(layers)} layers "
          f"({time.time()-t0:.1f}s)", flush=True)
    return stats


# --------------------------------------------------------------------------- all-position forward
@torch.inference_mode()
def forward_all(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Same as Transformer.forward but returns the hidden state of every position (pre-head)."""
    dev0 = model.blocks[0].device
    input_ids = input_ids.to(dev0)
    hashes = model.engram_hash(input_ids, 0) if model.engram_hash is not None else None
    h = F.embedding(input_ids, model.embed)
    h = h.unsqueeze(2).repeat(1, 1, model.hc, 1)
    pre_mix = h.new_zeros(h.size(0), h.size(1), model.hc, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    for blk in model.blocks:
        if h.device != blk.device:
            h = h.to(blk.device, non_blocking=True)
            pre_mix = pre_mix.to(blk.device, non_blocking=True)
        if blk.engram is not None:
            h = blk.engram(h, hashes[:, :, blk.engram.layer_hash_index, :])
        h, pre_mix = blk(h, 0, pre_mix)
    h = model.blocks[-1].hc_pre(h, pre_mix)
    return rmsnorm(h, model.norm_w, model.args.norm_eps)


@torch.inference_mode()
def chunk_nll(model, ids: torch.Tensor, head_chunk: int = 256) -> tuple[float, int]:
    """Sum of -log p(next token) over one context window, and the number of scored tokens."""
    h = forward_all(model, ids)[0][:-1]               # [s-1, dim]: position t predicts token t+1
    tgt = ids[0, 1:].to(h.device)
    tot, n = 0.0, 0
    for i in range(0, len(tgt), head_chunk):
        hh = h[i:i + head_chunk]
        lg = F.linear(hh, model.head).float()
        tot += float(F.cross_entropy(lg, tgt[i:i + len(hh)], reduction="sum"))
        n += len(hh)
    return tot, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--devices", default="4,5,6,7")
    ap.add_argument("--ep", action="store_true")
    ap.add_argument("--ep-shards", default="")
    ap.add_argument("--data", default="../data/wikitext-2-raw/wiki.test.raw")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--chunks", type=int, default=16)
    ap.add_argument("--vq", default="")
    ap.add_argument("--calib", default="", help="results/calib_hess.pt: use activation-weighted encoding")
    ap.add_argument("--calib-shrink", type=float, default=1024.0,
                    help="pseudo-tokens of shrinkage of the per-expert Hessian toward the layer mean")
    ap.add_argument("--cb", type=int, default=0, help="the Spark project's CB3/CB2 row-codebook format")
    ap.add_argument("--mixed", type=float, default=0.0, help="routing-aware mixed precision at this mean bit rate")
    ap.add_argument("--telemetry", default="../results/route_telemetry.pt")
    ap.add_argument("--alloc-task", default="average", help="fit the bit allocation on one task's routing")
    ap.add_argument("--bw-batch", type=int, default=0,
                    help="budget TRANSFERRED bits at this batch size instead of stored bits")
    ap.add_argument("--vq-layers", default="all", help="all | 20-39 | 0,5,11")
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="results/ppl.jsonl")
    a = ap.parse_args()

    global SHRINK
    SHRINK = a.calib_shrink
    devs = [int(x) for x in a.devices.split(",")]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    t0 = time.time()
    model = load_model(a.ckpt, devs, max_seq_len=a.ctx + 8, max_batch=1, engram=True, tokenizer=tok,
                       ep=a.ep, n_layers=a.n_layers, bf16_copies=False,
                       ep_shards=[int(v) for v in a.ep_shards.split(",")] if a.ep_shards else None)
    print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

    nl = len(model.blocks)
    if a.vq or a.mixed or a.cb:
        if a.vq_layers == "all":
            layers = set(range(nl))
        elif "-" in a.vq_layers:
            lo, hi = a.vq_layers.split("-")
            layers = set(range(int(lo), int(hi) + 1))
        else:
            layers = {int(x) for x in a.vq_layers.split(",")}
        sel = layers & set(range(nl))
        if a.cb:
            vq_stats = requantize_cb3(model, a.cb, sel)
        elif a.mixed:
            vq_stats = requantize_mixed(model, {3.0: "results/vq_3.0.npz", 3.25: "results/vq_3.25.npz",
                                                3.5: "results/vq_3.5.npz"},
                                        a.telemetry, a.mixed, sel, a.calib, a.alloc_task, a.bw_batch)
        elif a.calib:
            vq_stats = requantize_calibrated(model, a.vq, a.calib, sel)
        else:
            vq_stats = requantize(model, a.vq, sel)
    else:
        vq_stats = {"vq": "none"}

    text = open(a.data, encoding="utf-8").read()
    ids = tok.encode(text)
    print(f"wikitext-2 test: {len(ids):,} tokens, {a.chunks} x {a.ctx} scored", flush=True)
    tot, n, t0 = 0.0, 0, time.time()
    for c in range(a.chunks):
        s = c * a.ctx
        if s + a.ctx + 1 > len(ids):
            break
        x = torch.tensor([ids[s:s + a.ctx]], dtype=torch.long)
        nll, k = chunk_nll(model, x)
        tot += nll
        n += k
        print(f"  chunk {c+1}/{a.chunks}: ppl so far {np.exp(tot/n):.4f}  ({time.time()-t0:.0f}s)", flush=True)
    ppl = float(np.exp(tot / n))
    res = {"tag": a.tag or (a.vq or "baseline"), "ppl": ppl, "nll": tot / n, "tokens": n,
           "ctx": a.ctx, "chunks": a.chunks, "n_layers": nl, **vq_stats}
    print(f"\nPPL = {ppl:.4f}  (nll {tot/n:.4f} over {n:,} tokens)")
    with open(a.out, "a") as fh:
        fh.write(json.dumps(res) + "\n")
    print(f"appended to {a.out}")


if __name__ == "__main__":
    main()
