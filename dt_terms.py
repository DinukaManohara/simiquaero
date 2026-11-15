"""dt_terms.py — decision-tree term-restricted query vs the number of terms K.

Fixed dataset size (default 32000 rows x 17 columns: 16 descriptors + target);
the index is built once and the term-restricted query is run, to exhaustion,
for each K.  query_sec times the full query (all disjoint K-box subsets),
simplify_sec the sympy simplification + ranking (terms -> literals -> TVD).
Each CSV row (one per K and recorded rep) holds the number of subsets, the
top-ranked subset's rule length and distribution metrics, and the mean/std of
each metric across all subsets found ({tvd,jsd,kld,hellinger,bc}_{mean,std}).

  python dt_terms.py [--rows 32000] [--cols 17] [--terms 1 2 4 8]
                     [--tvd T | --tvd T1 T2 ...]      (one threshold, or one per K)
"""
import argparse
import time

import numpy as np

from bench_common import (make_dataset, build_index, gaussian_target,
                          find_multiple_box_subsets, dist_metrics, append_csv)
from quaero_core import simplify_rule

METRICS = ("tvd", "jsd", "kld", "hellinger", "bc")


def run_k(ix, target, k, tvd_threshold, warmup, repeats):
    """Full query for K boxes, warmup + repeats times; returns the ranked
    subsets of the last run and the recorded query / simplify times."""
    query_secs, simp_secs, ranked = [], [], []
    for r in range(warmup + repeats):
        t1 = time.perf_counter()
        found = find_multiple_box_subsets(ix, target, k, tvd_threshold)
        dq = time.perf_counter() - t1
        t2 = time.perf_counter()
        for s in found:
            s["simple"] = simplify_rule(s["dnf"])
        ranked = sorted(found, key=lambda s: (s["simple"]["n_terms"],
                                              s["simple"]["n_literals"], s["tvd"]))
        ds = time.perf_counter() - t2
        if r >= warmup:
            query_secs.append(dq); simp_secs.append(ds)
    return ranked, query_secs, simp_secs


def summarize(ix, ranked):
    """Subset count, top-ranked subset's rule + metrics, and mean/std of each
    metric across all subsets (std empty for a single subset)."""
    rows_of = lambda s: sum(sum(ix["node_cnt"][v].values()) for v in s["chosen"])
    out = dict(n_subsets=len(ranked), total_selected=sum(rows_of(s) for s in ranked))
    if ranked:
        best = ranked[0]
        out.update(selected=rows_of(best), rule_length=best["n_boxes"],
                   simp_terms=best["simple"]["n_terms"],
                   literals=best["simple"]["n_literals"],
                   **dist_metrics(best["got"], best["tc"]))
    else:
        out.update(selected=None, rule_length=None, simp_terms=None, literals=None,
                   **{m: None for m in METRICS})
    per_subset = [dist_metrics(s["got"], s["tc"]) for s in ranked]
    for m in METRICS:
        vals = np.array([d[m] for d in per_subset], dtype=float)
        out[f"{m}_mean"] = round(float(vals.mean()), 4) if len(vals) else None
        out[f"{m}_std"] = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=32000)
    ap.add_argument("--cols", type=int, default=17, help="descriptors + 1 target")
    ap.add_argument("--terms", type=int, nargs="+", default=[1, 2, 4, 8],
                    help="numbers of terms (boxes) K to run")
    ap.add_argument("--tvd", type=float, nargs="+", default=[0.15],
                    help="TVD threshold(s): one for every K, or one per K in --terms order")
    ap.add_argument("--warmup", type=int, default=2, help="unrecorded warm-up queries")
    ap.add_argument("--repeats", type=int, default=3, help="recorded queries (one CSV row each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="dt_terms.csv")
    args = ap.parse_args()
    if len(args.tvd) not in (1, len(args.terms)):
        ap.error(f"--tvd takes 1 value or {len(args.terms)} (one per K); got {len(args.tvd)}")
    thresholds = args.tvd * len(args.terms) if len(args.tvd) == 1 else args.tvd

    df, target_col = make_dataset(args.rows, args.cols, seed=args.seed)
    t0 = time.perf_counter()
    ix = build_index(df, target_col)                 # one index for every K
    build_sec = time.perf_counter() - t0
    target = gaussian_target(df, target_col, ix["n_bins"])

    for k, thr in zip(args.terms, thresholds):
        ranked, query_secs, simp_secs = run_k(ix, target, k, thr,
                                              args.warmup, args.repeats)
        base = dict(approach="dt_terms", n_rows=args.rows, n_cols=args.cols,
                    seed=args.seed, n_bins=ix["n_bins"], tree_nodes=ix["n"],
                    leaves=len(ix["leaves"]), n_boxes=k, tvd_threshold=thr,
                    build_sec=round(build_sec, 4), **summarize(ix, ranked))
        for rep, (q, s) in enumerate(zip(query_secs, simp_secs)):
            append_csv(args.out, dict(rep=rep, query_sec=round(q, 4),
                                      simplify_sec=round(s, 4), **base))
        mean = sum(query_secs) / len(query_secs) if query_secs else float("nan")
        print(f"[dt-terms] rows={args.rows} cols={args.cols} K={k} tvd<={thr}  "
              f"subsets={base['n_subsets']}  best TVD={base['tvd']}  "
              f"query_mean={mean:.2f}s  -> {args.out}")


if __name__ == "__main__":
    main()
