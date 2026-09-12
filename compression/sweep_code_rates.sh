#!/bin/bash
set -u
P=../.venv-lc/bin/python
C="--data data/eval_code.txt --chunks 12"
for b in 3.75 3.25 3.0; do
  $P ppl.py --devices 4,5,6,7 --ep $C --vq results/vq_$b.npz --tag "code-vq$b-all"
done
