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
                          select_subtrees, reduce_labeling, dist_metrics, append_csv)
from quaero_core import simplify_rule


def run(n_rows, n_cols, pct, seed, warmup, repeats):
    df, target_col = make_dataset(n_rows, n_cols, seed=seed)

    t0 = time.perf_counter()
    ix = build_index(df, target_col)                         # Stage 1; n_bins via Scott's rule
    build_sec = time.perf_counter() - t0
    n_bins = ix["n_bins"]

    target = gaussian_target(df, target_col, n_bins)
    total = round(pct * n_rows)

    # repeat only the query (Stages 2-3) and, timed separately, the sympy rule
    # simplification; build stays out of the timed loop
    query_secs, simp_secs = [], []
    got = tc = red = labels = simple = None
    for r in range(warmup + repeats):
        t1 = time.perf_counter()
        labels, got, tc = select_subtrees(ix, target, total=total)            # Stage 2
        red = reduce_labeling(ix, labels)                                     # Stage 3
        dq = time.perf_counter() - t1
        t2 = time.perf_counter()
        simple = simplify_rule(red["dnf"])                                    # sympy reduction
        ds = time.perf_counter() - t2
        if r >= warmup:
            query_secs.append(dq); simp_secs.append(ds)

    base = dict(approach="dt_rule_length_agnostic", n_rows=n_rows, n_cols=n_cols,
                seed=seed, n_bins=n_bins, tree_nodes=ix["n"], leaves=len(ix["leaves"]),
                selected=sum(labels.values()), rule_length=red["n_terms"],
                simp_terms=simple["n_terms"], literals=simple["n_literals"],
                **dist_metrics(got, tc), build_sec=round(build_sec, 4))
    return base, query_secs, simp_secs


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

    base, query_secs, simp_secs = run(args.rows, args.cols, args.pct, args.seed,
                                      args.warmup, args.repeats)
    for rep, (q, s) in enumerate(zip(query_secs, simp_secs)):   # one row per recorded query
        append_csv(args.out, dict(rep=rep, query_sec=round(q, 4),
                                  simplify_sec=round(s, 4), **base))
    mean = sum(query_secs) / len(query_secs) if query_secs else float("nan")
    smean = sum(simp_secs) / len(simp_secs) if simp_secs else float("nan")
    print(f"[dt-agnostic] rows={args.rows} cols={args.cols}  rule_length={base['rule_length']} "
          f"-> {base['simp_terms']} simplified  build={base['build_sec']}s "
          f"query_mean={mean:.4f}s simplify_mean={smean:.4f}s over {len(query_secs)} run(s)"
          f"  -> {args.out}")


if __name__ == "__main__":
    main()
