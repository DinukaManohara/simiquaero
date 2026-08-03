#!/usr/bin/env bash
# Decision-tree: column scaling (n_rows fixed at 32000), both query types.
set -euo pipefail
ROWS=32000
COLS="3 5 9 17 33 65"

for c in $COLS; do
    echo ">>> dt rows=$ROWS cols=$c"
    python dt_rule_length_agnostic.py --rows "$ROWS" --cols "$c" \
        --out dt_rule_length_agnostic_cols.csv
    python dt_rule_length_fixed.py    --rows "$ROWS" --cols "$c" --boxes 4 \
        --out dt_rule_length_fixed_cols.csv
done
echo "done -> dt_rule_length_agnostic_cols.csv, dt_rule_length_fixed_cols.csv"
