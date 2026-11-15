#!/usr/bin/env bash
# Decision-tree term-restricted query vs number of terms K (rows 32000, cols 17).
set -euo pipefail
OUT="dt_terms.csv"
TERMS=(1 2 4 8)
TVDS=(0.3 0.25 0.2 0.15)        # TVD threshold for each K, in the same order

python dt_terms.py --rows 32000 --cols 17 --terms "${TERMS[@]}" --tvd "${TVDS[@]}" --out "$OUT"
echo "done -> $OUT"
