"""Full scan of every expert scale tensor: the model-wide E8M0 alphabet and per-tensor ranges.
Read-only; ~17 GB of reads."""
import json, time
import numpy as np
from st import Checkpoint

ck = Checkpoint("/mnt/ssd/models/DeepSeek-V4.1-Flash")
hist = np.zeros(256, np.int64)
lo, hi, span = 255, 0, 0
t0 = time.time()
for L in range(40):
    for e in range(384):
        for m in ("w1", "w2", "w3"):
            s = ck.bytes(f"layers.{L}.ffn.experts.{e}.{m}.scale").ravel()
            hist += np.bincount(s, minlength=256)
            a, b = int(s.min()), int(s.max())
            lo, hi, span = min(lo, a), max(hi, b), max(span, b - a + 1)
    print(f"layer {L}: values {lo}..{hi}, widest single tensor {span}, {time.time()-t0:.0f}s", flush=True)
nz = np.flatnonzero(hist)
p = hist[nz] / hist.sum()
H = float(-(p * np.log2(p)).sum())
out = {"values": nz.tolist(), "probs": p.tolist(), "H0": H, "n_bytes": int(hist.sum()),
       "global_min": lo, "global_max": hi, "widest_tensor_span": span}
json.dump(out, open("results/scale_alphabet_full.json", "w"), indent=1)
print(f"\nALL {hist.sum():,} scale bytes: {len(nz)} distinct {nz.tolist()}, H0={H:.4f} bit, "
      f"range {lo}..{hi}, widest per-tensor span {span} -> {int(np.ceil(np.log2(span)))} bit with a per-tensor base")
