#!/bin/bash
set -u
P=../.venv-lc/bin/python
for b in 3.0 3.5; do
  $P ppl.py --devices 4,5,6,7 --ep --chunks 16 --vq results/vq_$b.npz --calib results/calib_hess.pt --tag "vq$b-calib"
done
