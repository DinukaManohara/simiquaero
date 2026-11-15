"""dt_rule_length_agnostic.py — decision-tree query, RULE-LENGTH-AGNOSTIC.

This is the "find subsets and sort by rule length" mode: fix a cardinality, match
the target bin-distribution with the subtree-first greedy, then reduce to a rule
whose length is whatever it is (an OUTPUT, not a constraint).  Disjoint subsets
are extracted to exhaustion (until nothing more can be selected or the next
subset's TVD exceeds --tvd); query_sec times that whole loop.  The subsets are
then simplified (sympy) and ranked terms -> literals -> TVD, timed separately as
simplify_sec; the CSV row reports the number of subsets and the top-ranked
subset's rule length (number of DNF terms -- the decision-tree analogue of
"islands") and distribution metrics, plus the mean/std of each metric across
all subsets found ({tvd,jsd,kld,hellinger,bc}_{mean,std}).

Run one (rows, cols) config per invocation and append to a CSV:
  python dt_rule_length_agnostic.py --rows R [--cols C] [--pct P] [--tvd T]

Sweeps to reproduce (drive with a shell loop):
  rows in 1024,2048,...,1024000  with  --cols 16   (row scaling)
  cols in 3,5,9,17,33,65         with  --rows 32000 (column scaling)
"""
import argparse
import time

import numpy as np

from bench_common import (make_dataset, build_index, gaussian_target,
                          find_multiple_subsets, dist_metrics, append_csv)
from quaero_core import simplify_rule

METRICS = ("tvd", "jsd", "kld", "hellinger", "bc")


def run(n_rows, n_cols, pct, tvd_threshold, seed, warmup, repeats):
    df, target_col = make_dataset(n_rows, n_cols, seed=seed)

    t0 = time.perf_counter()
    ix = build_index(df, target_col)                         # Stage 1; n_bins via Scott's rule
    build_sec = time.perf_counter() - t0
    n_bins = ix["n_bins"]

    target = gaussian_target(df, target_col, n_bins)
    total = round(pct * n_rows)

    # timed: the full query = every disjoint subset (Stages 2-3) until exhaustion;
    # then, timed separately, simplifying + ranking all of them
    query_secs, simp_secs, ranked = [], [], []
    for r in range(warmup + repeats):
        t1 = time.perf_counter()
        found = find_multiple_subsets(ix, target, total, tvd_threshold)
        dq = time.perf_counter() - t1
        t2 = time.perf_counter()
        for s in found:
            s["simple"] = simplify_rule(s["red"]["dnf"])     # sympy reduction
        ranked = sorted(found, key=lambda s: (s["simple"]["n_terms"],
                                              s["simple"]["n_literals"], s["tvd"]))
        ds = time.perf_counter() - t2
        if r >= warmup:
            query_secs.append(dq); simp_secs.append(ds)

    base = dict(approach="dt_rule_length_agnostic", n_rows=n_rows, n_cols=n_cols,
                seed=seed, n_bins=n_bins, tree_nodes=ix["n"], leaves=len(ix["leaves"]),
                pct=pct, tvd_threshold=tvd_threshold, n_subsets=len(ranked),
                total_selected=sum(sum(s["labels"].values()) for s in ranked),
                build_sec=round(build_sec, 4))
    if ranked:                                    # the top-ranked subset's rule + metrics
        best = ranked[0]
        base.update(selected=sum(best["labels"].values()), rule_length=best["red"]["n_terms"],
                    simp_terms=best["simple"]["n_terms"],
                    literals=best["simple"]["n_literals"],
                    **dist_metrics(best["got"], best["tc"]))
    else:
        base.update(selected=None, rule_length=None, simp_terms=None, literals=None,
                    tvd=None, jsd=None, kld=None, hellinger=None, bc=None)

    # accuracy across ALL subsets found (identical in every rep: computed once, untimed);
    # std is left empty when there is a single subset
    per_subset = [dist_metrics(s["got"], s["tc"]) for s in ranked]
    for key in METRICS:
        vals = np.array([m[key] for m in per_subset], dtype=float)
        base[f"{key}_mean"] = round(float(vals.mean()), 4) if len(vals) else None
        base[f"{key}_std"] = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else None
    return base, query_secs, simp_secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=32000)
    ap.add_argument("--cols", type=int, default=16)
    ap.add_argument("--pct", type=float, default=0.30, help="cardinality fraction")
    ap.add_argument("--tvd", type=float, default=0.15,
                    help="TVD threshold: keep finding disjoint subsets until exceeded")
    ap.add_argument("--warmup", type=int, default=2, help="unrecorded warm-up queries")
    ap.add_argument("--repeats", type=int, default=3, help="recorded queries (one CSV row each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="dt_rule_length_agnostic_results.csv")
    args = ap.parse_args()

    base, query_secs, simp_secs = run(args.rows, args.cols, args.pct, args.tvd,
                                      args.seed, args.warmup, args.repeats)
    for rep, (q, s) in enumerate(zip(query_secs, simp_secs)):   # one row per recorded query
        append_csv(args.out, dict(rep=rep, query_sec=round(q, 4),
                                  simplify_sec=round(s, 4), **base))
    mean = sum(query_secs) / len(query_secs) if query_secs else float("nan")
    smean = sum(simp_secs) / len(simp_secs) if simp_secs else float("nan")
    print(f"[dt-agnostic] rows={args.rows} cols={args.cols} pct={args.pct} tvd<={args.tvd}  "
          f"subsets={base['n_subsets']}  best: {base['rule_length']} -> {base['simp_terms']} "
          f"terms, TVD={base['tvd']}  build={base['build_sec']}s query_mean={mean:.4f}s "
          f"simplify_mean={smean:.4f}s over {len(query_secs)} run(s)  -> {args.out}")


if __name__ == "__main__":
    main()
