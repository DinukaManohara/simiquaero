#!/usr/bin/env bash
# Decision-tree: row scaling (n_cols fixed at 16), both query types.
set -euo pipefail
COLS=16
ROWS="1024 2048 4096 8192 16384 32768 65536 131072 262144 524288 1024000"

for n in $ROWS; do
    echo ">>> dt rows=$n cols=$COLS"
    python dt_rule_length_agnostic.py --rows "$n" --cols "$COLS" \
        --out dt_rule_length_agnostic_rows.csv
    python dt_rule_length_fixed.py    --rows "$n" --cols "$COLS" --boxes 4 \
        --out dt_rule_length_fixed_rows.csv
done
echo "done -> dt_rule_length_agnostic_rows.csv, dt_rule_length_fixed_rows.csv"
