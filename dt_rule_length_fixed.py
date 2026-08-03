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


def run(n_rows, n_cols, n_boxes, seed, warmup, repeats):
    df, target_col = make_dataset(n_rows, n_cols, seed=seed)

    t0 = time.perf_counter()
    ix = build_index(df, target_col)                         # Stage 1; n_bins via Scott's rule
    build_sec = time.perf_counter() - t0
    n_bins = ix["n_bins"]

    target = gaussian_target(df, target_col, n_bins)

    # repeat only the fixed-K box query; build stays out of the timed loop
    query_secs, res = [], None
    for r in range(warmup + repeats):
        t1 = time.perf_counter()
        res = find_best_boxes(ix, target, n_boxes=n_boxes)   # fixed-K box query
        dt = time.perf_counter() - t1
        if r >= warmup:
            query_secs.append(dt)

    selected = sum(sum(ix["node_cnt"][v].values()) for v in res["chosen"])
    base = dict(approach="dt_rule_length_fixed", n_rows=n_rows, n_cols=n_cols,
                seed=seed, n_bins=n_bins, tree_nodes=ix["n"], leaves=len(ix["leaves"]),
                n_boxes=res["n_boxes"], rule_length=res["n_boxes"], selected=selected,
                tvd=round(res["tvd"], 4), build_sec=round(build_sec, 4))
    return base, query_secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=32000)
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--boxes", type=int, default=4, help="fixed number of bounding boxes")
    ap.add_argument("--warmup", type=int, default=2, help="unrecorded warm-up queries")
    ap.add_argument("--repeats", type=int, default=3, help="recorded queries (one CSV row each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="dt_rule_length_fixed_results.csv")
    args = ap.parse_args()

    base, query_secs = run(args.rows, args.cols, args.boxes, args.seed,
                           args.warmup, args.repeats)
    for rep, sec in enumerate(query_secs):                 # one row per recorded query
        append_csv(args.out, dict(rep=rep, query_sec=round(sec, 4), **base))
    mean = sum(query_secs) / len(query_secs) if query_secs else float("nan")
    print(f"[dt-fixed] rows={args.rows} cols={args.cols} boxes={base['n_boxes']}  "
          f"TVD={base['tvd']}  build={base['build_sec']}s "
          f"query_mean={mean:.4f}s over {len(query_secs)} run(s)  -> {args.out}")


if __name__ == "__main__":
    main()
