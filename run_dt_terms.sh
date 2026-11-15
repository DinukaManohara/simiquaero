#!/usr/bin/env bash
# Decision-tree term-restricted query vs number of terms K (rows 32000, cols 17).
set -euo pipefail
OUT="dt_terms.csv"
TVD="${1:-0.15}"                # optional TVD threshold, default 0.15

python dt_terms.py --rows 32000 --cols 17 --terms 1 2 4 8 --tvd "$TVD" --out "$OUT"
echo "done -> $OUT"
