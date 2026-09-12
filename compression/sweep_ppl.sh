#!/bin/bash
# PPL sweep: baseline and each VQ rate, same protocol (wikitext-2, 16 x 2048)
set -u
P=../.venv-lc/bin/python
D=${DEVICES:-4,5,6,7}
C=${CHUNKS:-16}
$P ppl.py --devices $D --ep --chunks $C --tag baseline
for b in 3.75 3.5 3.0; do
  $P ppl.py --devices $D --ep --chunks $C --vq results/vq_$b.npz --tag "vq$b-all"
done
$P ppl.py --devices $D --ep --chunks $C --vq results/vq_3.0.npz --vq-layers 20-39 --tag "vq3.0-half"
