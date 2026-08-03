"""dt_rule_length_agnostic.py — decision-tree query, RULE-LENGTH-AGNOSTIC.

This is the "find subsets and sort by rule length" mode: fix a cardinality, match
the target bin-distribution with the subtree-first greedy, then reduce to a rule
whose length is whatever it is (an OUTPUT, not a constraint).  We record the
rule length (number of DNF terms -- the decision-tree analogue of "islands").

Run one (rows, cols) config per invocation and append to a CSV:
  python dt_rule_length_agnostic.py --rows R [--cols C]

Sweeps to reproduce (drive with a shell loop):
  rows in 1024,2048,...,1024000  with  --cols 16   (row scaling)
  cols in 3,5,9,17,33,65         with  --rows 32000 (column scaling)
"""
import argparse
import time

from bench_common import (make_dataset, build_index, gaussian_target,
                          select_subtrees, reduce_labeling, tv_distance, append_csv)

N_BINS = 8        # fixed number of target bins (the queried distribution's support)


def run(n_rows, n_cols, pct, seed, warmup, repeats):
    df, target_col = make_dataset(n_rows, n_cols, seed=seed)
    n_bins = N_BINS

    t0 = time.perf_counter()
    ix = build_index(df, target_col, n_bins=n_bins)          # Stage 1 (once per dataset)
    build_sec = time.perf_counter() - t0

    target = gaussian_target(df, target_col, n_bins)
    total = round(pct * n_rows)

    # repeat only the query (Stages 2-3); build stays out of the timed loop
    query_secs, got, tc, red, labels = [], None, None, None, None
    for r in range(warmup + repeats):
        t1 = time.perf_counter()
        labels, got, tc = select_subtrees(ix, target, total=total)            # Stage 2
        red = reduce_labeling(ix, labels)                                     # Stage 3
        dt = time.perf_counter() - t1
        if r >= warmup:
            query_secs.append(dt)

    base = dict(approach="dt_rule_length_agnostic", n_rows=n_rows, n_cols=n_cols,
                seed=seed, n_bins=n_bins, tree_nodes=ix["n"], leaves=len(ix["leaves"]),
                selected=sum(labels.values()), rule_length=red["n_terms"],
                tvd=round(tv_distance(got, tc), 4), build_sec=round(build_sec, 4))
    return base, query_secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=32000)
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--pct", type=float, default=0.30, help="cardinality fraction")
    ap.add_argument("--warmup", type=int, default=2, help="unrecorded warm-up queries")
    ap.add_argument("--repeats", type=int, default=3, help="recorded queries (one CSV row each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="dt_rule_length_agnostic_results.csv")
    args = ap.parse_args()

    base, query_secs = run(args.rows, args.cols, args.pct, args.seed,
                           args.warmup, args.repeats)
    for rep, sec in enumerate(query_secs):                 # one row per recorded query
        append_csv(args.out, dict(rep=rep, query_sec=round(sec, 4), **base))
    mean = sum(query_secs) / len(query_secs) if query_secs else float("nan")
    print(f"[dt-agnostic] rows={args.rows} cols={args.cols}  rule_length={base['rule_length']}  "
          f"build={base['build_sec']}s query_mean={mean:.4f}s over {len(query_secs)} run(s)  -> {args.out}")


if __name__ == "__main__":
    main()
