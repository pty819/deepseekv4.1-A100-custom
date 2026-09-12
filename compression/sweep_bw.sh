#!/bin/bash
# waits for the dynamic sweep to finish, then runs the bandwidth-budgeted allocation
set -u
while pgrep -f sweep_dynamic.sh > /dev/null; do sleep 30; done
../.venv-lc/bin/python ppl.py --devices 4,5,6,7 --ep --chunks 16 --mixed 3.5 --bw-batch 32 \
  --vq results/vq_3.5.npz --tag "bwopt3.5-B32"
../.venv-lc/bin/python ppl.py --devices 4,5,6,7 --ep --chunks 16 --mixed 3.5 --bw-batch 1 \
  --vq results/vq_3.5.npz --tag "bwopt3.5-B1"
