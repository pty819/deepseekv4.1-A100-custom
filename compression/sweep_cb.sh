#!/bin/bash
set -u
P=../.venv-lc/bin/python
$P ppl.py --devices 4,5,6,7 --ep --chunks 16 --cb 3 --tag "cb3-spark-format"
$P ppl.py --devices 4,5,6,7 --ep --data data/eval_code.txt --chunks 12 --cb 3 --tag "code-cb3-spark-format"
