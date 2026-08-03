"""dt_rule_length_fixed.py — decision-tree query, RULE-LENGTH-FIXED.

This is the "query by number of boxes" mode: fix K = number of bounding boxes
(disjoint subtrees, one DNF conjunct each -> rule length is FIXED at K) and find
the K boxes whose union's bin-distribution is closest (min TVD) to the target.
We record the achieved TVD (the quality at the fixed rule length).

Run one (rows, cols) config per invocation and append to a CSV:
  python dt_rule_length_fixed.py --rows R [--cols C] [--boxes K]

Sweeps to reproduce (drive with a shell loop):
  rows in 1024,2048,...,1024000  with  --cols 16   (row scaling)
  cols in 3,5,9,17,33,65         with  --rows 32000 (column scaling)
"""
import argparse
import time

from bench_common import (make_dataset, build_index, gaussian_target,
                          find_best_boxes, append_csv)

N_BINS = 8        # fixed number of target bins (the queried distribution's support)


def run(n_rows, n_cols, n_boxes, seed):
    df, target_col = make_dataset(n_rows, n_cols, seed=seed)
    n_bins = N_BINS

    t0 = time.perf_counter()
    ix = build_index(df, target_col, n_bins=n_bins)          # Stage 1 (once per dataset)
    build_sec = time.perf_counter() - t0

    target = gaussian_target(df, target_col, n_bins)
    t1 = time.perf_counter()
    res = find_best_boxes(ix, target, n_boxes=n_boxes)       # fixed-K box query
    query_sec = time.perf_counter() - t1

    selected = sum(sum(ix["node_cnt"][v].values()) for v in res["chosen"])
    return dict(approach="dt_rule_length_fixed", n_rows=n_rows, n_cols=n_cols,
                n_bins=n_bins, tree_nodes=ix["n"], leaves=len(ix["leaves"]),
                n_boxes=res["n_boxes"], rule_length=res["n_boxes"], selected=selected,
                tvd=round(res["tvd"], 4),
                build_sec=round(build_sec, 4), query_sec=round(query_sec, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=32000)
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--boxes", type=int, default=4, help="fixed number of bounding boxes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="dt_rule_length_fixed_results.csv")
    args = ap.parse_args()

    res = run(args.rows, args.cols, args.boxes, args.seed)
    append_csv(args.out, res)
    print(f"[dt-fixed] rows={args.rows} cols={args.cols} boxes={res['n_boxes']}  "
          f"TVD={res['tvd']}  build={res['build_sec']}s query={res['query_sec']}s  -> {args.out}")


if __name__ == "__main__":
    main()
