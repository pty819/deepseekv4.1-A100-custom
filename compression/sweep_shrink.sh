#!/bin/bash
set -u
P=../.venv-lc/bin/python
$P ppl.py --devices 4,5,6,7 --ep --chunks 16 --mixed 3.5 --vq results/vq_3.5.npz --calib results/calib_hess.pt --calib-shrink 1024 --tag "mixed3.5-calib-shrink1024"
$P ppl.py --devices 4,5,6,7 --ep --chunks 16 --vq results/vq_3.0.npz --calib results/calib_hess.pt --calib-shrink 1024 --tag "vq3.0-calib-shrink1024"
