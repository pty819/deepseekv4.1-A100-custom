#!/bin/bash
set -u
P=../.venv-lc/bin/python
for t in 3.5 3.25; do
  $P ppl.py --devices 4,5,6,7 --ep --chunks 16 --mixed $t --vq results/vq_3.5.npz --tag "mixed$t"
done
