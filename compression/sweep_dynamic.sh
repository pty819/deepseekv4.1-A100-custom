#!/bin/bash
# (a) bandwidth-budgeted allocation on wikitext; (b) task-conditioned allocation on a code corpus
set -u
P=../.venv-lc/bin/python
D=4,5,6,7
# (a) same TRANSFERRED bits as uniform 3.5 at batch 32
$P ppl.py --devices $D --ep --chunks 16 --mixed 3.5 --bw-batch 32 --vq results/vq_3.5.npz --tag "bwopt3.5-B32"
# (b) code corpus: baseline, uniform, global-mixed, code-fitted mixed, japanese-fitted mixed
C="--data data/eval_code.txt --chunks 12"
$P ppl.py --devices $D --ep $C --tag "code-baseline"
$P ppl.py --devices $D --ep $C --vq results/vq_3.5.npz --tag "code-vq3.5-all"
$P ppl.py --devices $D --ep $C --mixed 3.5 --vq results/vq_3.5.npz --tag "code-mixed3.5-avg"
$P ppl.py --devices $D --ep $C --mixed 3.5 --alloc-task coding --vq results/vq_3.5.npz --tag "code-mixed3.5-fitcode"
$P ppl.py --devices $D --ep $C --mixed 3.5 --alloc-task japanese --vq results/vq_3.5.npz --tag "code-mixed3.5-fitja"
