# DGX Spark side

What runs on `gx10-b872` (GB10, 121 GiB unified, one NVMe), against the project at
`~/dsv41-spark/work` (a clone of github.com/0xBakeer/deepseek-v41-flash-spark). That clone is
someone else's repository and had another agent's uncommitted work in it, so nothing here was
committed there: these are the files as they were copied to `~/dsv41-spark/a100-vq/`, plus the one
patch the engine needs.

## The problem

The box streams routed experts from NVMe. Unpruned it measured **9.16 tok/s at 0.233 GB/tok**,
hit 0.9585; the project's fast configuration reaches 18.98 tok/s only by dropping 60 % of the
experts, which their own notes show degenerates free generation.

Switching the arena to their 3-bit CB3 format collapses the traffic -- the same 94 GB holds 6,500
CB3 slots instead of 5,000 FP4 ones, so hit goes 0.9585 -> 0.9872 and NVMe 0.233 -> 0.069 GB/tok --
and still **loses**, 148.5 ms/tok against 108.7, because every miss packs FP4 -> CB3 on the GPU
(measured 20.8 ms/expert against the 4.6 ms the read itself takes).

## The fix

Pre-pack the experts that actually stream, and read the packed slot directly on a miss.

| | FP4 | CB3 + this store |
|---|---|---|
| run 1 (cold) | 6.93 tok/s, 0.369 GB/tok, hit 0.9289 | 4.04, 0.116, 0.9426 |
| run 2 | 9.13, 0.233, 0.9585 | 8.33, 0.036, 0.985 |
| **run 3** | **9.71, 0.213, 0.9625** | **18.28, 0.003, 1.000** |

FP4 plateaus at hit 0.9625 because the working set does not fit in 5,000 slots. Same prompt, same
`ARENA_GB=94 DSV41_BLOCK=1`, no pruning.

## Files

| file | what |
|---|---|
| `pack_store.py` | builds the store: the 12 slot tensors of `cb3_moe.CB3ArenaV2` concatenated per expert at a fixed 14,454,784 B stride (a multiple of 4096, so a record is one O_DIRECT pread), in the engine's own trace ranking order, resumable. 6,087 experts / 88.0 GB / 8.2 min. |
| `cb3_store.py` | the reader the engine calls on a miss. Opt-in: only attaches when `DSV41_CB3_STORE` names a store. |
| `experts.py.a100-vq.patch` | the engine change: 10 lines at the top of `ExpertStore._load_into_slot`. |
| `fast_fill.py` | a bit-identical `cb3.fp4_to_cb3_v2` that stays in the packed byte domain (the original materialises the [N, K] nibbles as int64 twice, 189 MB per w13). 1.5x; used by the packer. |
| `fast_requant.py` | the same treatment for `CodebookSim.requant_packed` (the simulation path). 2.6x, bit-identical. |
| `cb3store_bench.sh`, `fp4ctl_bench.sh` | the A/B above. |
| `gen_ab.sh`, `prompts.txt`, `genfmt.py` | greedy generation A/B on five prompts. |
| `tf_eval.sh` | teacher-forced NLL on the project's held-out corpus. **Not completed** -- one arm ran 41 min under streaming without finishing and was stopped. |

## Running it

```
DSV41_CB3_STORE=/home/shi3z/dsv41-spark/models/cb3_store \
DSV41_BLOCK=1 EXPERT_FORMAT=cb3 ARENA_GB=94 TRANSIENT_SLOTS=64 KEEP_FREE_GB=12 ./start.sh
```

## Quality

Generation is correct on all five prompts (Japanese prose, Python, translation, a proof, an English
essay) with no degeneration; the FP4 arm errored on two of them with `transient ring exhausted`,
a `TRANSIENT_SLOTS` limit the CB3 arena does not reach. CB3's own cost, measured on the A100 over
all 40 layers unpruned, is **+8.54 % wikitext PPL / +2.17 % code**. A dim-4 VQ codebook halves that
(+4.23 % / +0.97 %) at identical bytes and identical slot geometry -- the open work is the Triton
decode (a per-row codebook shift becomes one 4096-entry LUT lookup).
