#!/bin/bash
# Pareto sweep: minimise bits subject to a PPL budget.
set -u
P=../.venv-lc/bin/python
$P ppl.py --devices 4,5,6,7 --ep --chunks 16 --vq results/vq_3.25.npz --tag "vq3.25-all"
for t in 3.45 3.40 3.35 3.30; do
  $P ppl.py --devices 4,5,6,7 --ep --chunks 16 --mixed $t --vq results/vq_3.5.npz --tag "mixed$t"
done
