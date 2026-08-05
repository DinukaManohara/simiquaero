#!/usr/bin/env bash
# IP exact island-minimisation: row scaling (n_cols fixed at 3).
set -euo pipefail
OUT="ip_results.csv"
TIME_LIMIT="${1:-300}"          # optional solver time-limit (s), default 300

for n in 128 256 512 1024 2048; do
    echo ">>> ip.py rows=$n"
    python ip.py "$n" --time-limit "$TIME_LIMIT" --out "$OUT"
done
echo "done -> $OUT"
