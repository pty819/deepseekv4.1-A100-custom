"""Collect per-expert activation statistics for calibration-aware quantization.

For a weight matrix W whose input is x, the quantization objective that matters is ||dW x||^2, not
||dW||^2.  The diagonal of the layer Hessian, h_j = sum_t x_tj^2, is what turns a plain MSE quantizer
into an activation-weighted one (the AWQ/GPTQ family without the off-diagonal terms).

In a MoE every expert sees only the tokens routed to it, so the statistics are per expert:
  H13[layer, expert, j] = sum over routed tokens of  gate^2 * x_j^2     (input of w1 and w3, dim)
  H2 [layer, expert, j] = sum over routed tokens of  hq_j^2             (input of w2, inter;
                                                                         hq already carries the gate)
The MoE forward is *wrapped*, not replaced: the original path still runs untouched and the wrapper
re-runs the w13 GEMM to get the w2 input.  Nothing in dsv41/ is modified and the checkpoint is only read.

usage: python calib_collect.py --devices 4,5,6,7 --ep --seqs 32 --ctx 2048
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dsv41.model as M                                          # noqa: E402
from dsv41.fused import swiglu_quant                             # noqa: E402
from dsv41.load import load_model                                # noqa: E402
from dsv41.moe_kernels import GroupedPairs, grouped_fp4_gemm     # noqa: E402

ACC: dict = {}          # (layer, "13"/"2") -> fp32 tensor [E_total, dim or inter] on the shard device
TOKENS: dict = {}       # (layer,) -> per-expert routed-token counts


def install_wrapper(model):
    orig = M.MoE._forward_ep
    for blk in model.blocks:
        blk.ffn._layer_id = blk.layer_id

    def wrapped(self, xq, eid, tok, weights, n_tok, n_pairs):
        y = orig(self, xq, eid, tok, weights, n_tok, n_pairs)
        L = getattr(self, "_layer_id", -1)
        wflat = weights.flatten().float()
        for sh in (self.ep or []):
            sel = ((eid >= sh["start"]) & (eid < sh["start"] + sh["n"])).nonzero().flatten()
            if sel.numel() == 0:
                continue
            d = sh["device"]
            xs = xq.to(d)
            le = (eid[sel] - sh["start"]).to(d).long()
            tk = tok[sel].to(d).long()
            wt = wflat[sel].to(d)
            m = sel.numel()
            k13 = (L, "13", d)
            if k13 not in ACC:
                ACC[k13] = torch.zeros(sh["n"], self.dim, dtype=torch.float32, device=d)
                ACC[(L, "2", d)] = torch.zeros(sh["n"], self.inter, dtype=torch.float32, device=d)
                TOKENS[(L, d)] = torch.zeros(sh["n"], dtype=torch.float64, device=d)
                ACC[(L, "start", d)] = sh["start"]
            ACC[k13].index_add_(0, le, (wt[:, None] ** 2) * xs[tk].float() ** 2)
            TOKENS[(L, d)].index_add_(0, le, torch.ones(m, dtype=torch.float64, device=d))
            # the w2 input: re-run the w13 GEMM + SwiGLU exactly as the original path does
            local_rows = torch.arange(m, device=d, dtype=torch.int32)
            ones = torch.ones(m, device=d)
            p1 = GroupedPairs(le.to(torch.int32), tk.to(torch.int32), local_rows, ones, 64)
            gu = grouped_fp4_gemm(xs, sh["w13"], sh["s13"], p1, m, tiled=self.tiled)
            hq = swiglu_quant(gu, wt.contiguous(), self.inter, self.swiglu_limit)
            ACC[(L, "2", d)].index_add_(0, le, hq.float() ** 2)
        return y

    M.MoE._forward_ep = wrapped
    return orig


def calib_text(sources: str, extra: str) -> list[str]:
    docs = []
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for s in sources.split(","):
        s = s.strip()
        if s == "wiki":
            docs.append(open(os.path.join(base, "data/wikitext-2-raw/wiki.train.raw"), encoding="utf-8").read()[:2_000_000])
        elif s == "code":
            buf = []
            for pat in ("dsv41/*.py", "dsv41/cuda/*.cu", "dsv41/cpu/*.cpp", "compression/*.py"):
                for f in sorted(glob.glob(os.path.join(base, pat))):
                    buf.append(open(f, encoding="utf-8", errors="ignore").read())
            docs.append("\n\n".join(buf))
        elif s == "docs":
            buf = []
            for f in sorted(glob.glob(os.path.join(base, "*.md")) + glob.glob(os.path.join(base, "compression/*.md"))):
                buf.append(open(f, encoding="utf-8", errors="ignore").read())
            docs.append("\n\n".join(buf))
    for f in [x for x in extra.split(",") if x]:
        docs.append(open(f, encoding="utf-8", errors="ignore").read())
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/mnt/ssd/models/DeepSeek-V4.1-Flash")
    ap.add_argument("--devices", default="4,5,6,7")
    ap.add_argument("--ep", action="store_true")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--seqs", type=int, default=24, help="calibration windows (ctx tokens each)")
    ap.add_argument("--sources", default="wiki,code,docs")
    ap.add_argument("--extra-files", default="", help="comma separated extra text files (e.g. a Japanese corpus)")
    ap.add_argument("--out", default="results/calib_hess.pt")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    docs = calib_text(a.sources, a.extra_files)
    ids = []
    per = max(1, a.seqs // max(len(docs), 1))
    for dtext in docs:
        t = tok.encode(dtext)
        for i in range(per):
            s = i * a.ctx
            if s + a.ctx <= len(t):
                ids.append(t[s:s + a.ctx])
    ids = ids[:a.seqs]
    print(f"calibration: {len(ids)} windows x {a.ctx} tokens from {a.sources} {a.extra_files}", flush=True)

    model = load_model(a.ckpt, [int(x) for x in a.devices.split(",")], max_seq_len=a.ctx + 8, max_batch=1,
                       engram=True, tokenizer=tok, ep=a.ep, bf16_copies=False)
    install_wrapper(model)
    t0 = time.time()
    for i, w in enumerate(ids):
        model.forward(torch.tensor([w], dtype=torch.long), 0)
        print(f"  window {i+1}/{len(ids)}  ({time.time()-t0:.0f}s)", flush=True)

    # gather per layer into [n_layers, E, dim] on the CPU
    nl, E = len(model.blocks), model.args.n_routed_experts
    dim, inter = model.args.dim, model.blocks[0].ffn.inter
    H13 = torch.zeros(nl, E, dim, dtype=torch.float32)
    H2 = torch.zeros(nl, E, inter, dtype=torch.float32)
    cnt = torch.zeros(nl, E, dtype=torch.float64)
    for (L, kind, d), t in ACC.items():
        if kind == "start":
            continue
        s = ACC[(L, "start", d)]
        (H13 if kind == "13" else H2)[L, s:s + t.shape[0]] = t.cpu()
    for (L, d), t in TOKENS.items():
        s = ACC[(L, "start", d)]
        cnt[L, s:s + t.shape[0]] = t.cpu()
    torch.save({"H13": H13, "H2": H2, "tokens": cnt, "ctx": a.ctx, "seqs": len(ids),
                "sources": a.sources + "," + a.extra_files}, a.out)
    live = int((cnt.sum(0) > 0).sum())
    print(f"\nsaved {a.out}: H13 {tuple(H13.shape)}, H2 {tuple(H2.shape)}; "
          f"{live}/{E} experts saw at least one token; "
          f"routed tokens per expert min/median/max "
          f"{cnt.min():.0f}/{cnt.median():.0f}/{cnt.max():.0f}")


if __name__ == "__main__":
    main()
