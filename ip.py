"""ip.py — solve the island-minimisation INTEGER PROGRAM exactly (no rounding).

For a dataset sorted by one column, choose x_i in {0,1} to minimise the number
of contiguous "islands of 1s"  sum_i x_i (1 - x_{i+1})  (x_n = 0)  subject to a
per-bin cardinality  sum_{i in B_k} x_i = b_k.  Solved exactly as a MILP: the
bilinear objective is linearised with w_i = x_i x_{i+1} (McCormick), which is
tight because x is binary (so this is exact, not the LP-relaxation heuristic).

Usage:  python ip.py <n_rows> [--bins 4] [--frac 0.5] [--sort-col f1]
                       [--time-limit SECONDS] [--seed 0] [--out ip_results.csv]

Records n_rows, n_cols, n_bins, total_selected, islands, build/solve seconds and
solver status to a CSV (appended), for later plotting.  Row size is the CLI arg;
column count is fixed at 3 (2 descriptors + 1 target) -- column-wise sorting only
needs one column.
"""
import argparse
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds

from bench_common import make_dataset, append_csv

N_COLS = 3        # 2 descriptors + 1 target (fixed; only one sort column needed)


def solve_islands_ip(df, target_col, sort_col, n_bins, frac, time_limit,
                     warmup, repeats):
    tgt = df[target_col].to_numpy()
    bin_id = np.asarray(pd.cut(tgt, bins=n_bins, labels=range(n_bins),
                               include_lowest=True).astype(int))
    order = np.argsort(df[sort_col].to_numpy(), kind="mergesort")   # sort by one column
    bin_sorted = bin_id[order]                                      # bin per sorted position
    n = len(order)

    # per-bin target counts b_k (a fraction of each bin -> always feasible)
    b = {k: int((bin_sorted == k).sum() * frac) for k in range(n_bins)}

    t_build = time.perf_counter()
    # variables z = [x_0..x_{n-1}, w_0..w_{n-2}];  minimise sum x - sum w  (= islands)
    nv = n + (n - 1)
    c = np.concatenate([np.ones(n), -np.ones(n - 1)])

    # McCormick inequalities:  w_i<=x_i,  w_i<=x_{i+1},  x_i+x_{i+1}-w_i<=1
    rows, cols, data, bub = [], [], [], []
    r = 0
    for i in range(n - 1):
        wi = n + i
        rows += [r, r];       cols += [wi, i];        data += [1, -1]; bub.append(0); r += 1
        rows += [r, r];       cols += [wi, i + 1];    data += [1, -1]; bub.append(0); r += 1
        rows += [r, r, r];    cols += [i, i + 1, wi]; data += [1, 1, -1]; bub.append(1); r += 1
    A_ub = sparse.coo_matrix((data, (rows, cols)), shape=(r, nv)).tocsr()
    b_ub = np.array(bub, dtype=float)

    # per-bin equalities  sum_{i in B_k} x_i = b_k
    er, ec, ed = [], [], []
    for k in range(n_bins):
        for pos in np.where(bin_sorted == k)[0]:
            er.append(k); ec.append(int(pos)); ed.append(1)
    A_eq = sparse.coo_matrix((ed, (er, ec)), shape=(n_bins, nv)).tocsr()
    b_eq = np.array([b[k] for k in range(n_bins)], dtype=float)

    integrality = np.concatenate([np.ones(n), np.zeros(n - 1)])     # x integer, w continuous
    constraints = [LinearConstraint(A_ub, -np.inf, b_ub),
                   LinearConstraint(A_eq, b_eq, b_eq)]
    build_sec = time.perf_counter() - t_build

    options = {"time_limit": time_limit} if time_limit else {}
    # solve the SAME MILP warmup+repeats times; keep only the recorded solve times
    # (the optimum is deterministic, so only the solve wall-clock varies)
    solve_secs, res = [], None
    for r in range(warmup + repeats):
        t_solve = time.perf_counter()
        res = milp(c=c, constraints=constraints, integrality=integrality,
                   bounds=Bounds(0, 1), options=options)
        dt = time.perf_counter() - t_solve
        if r >= warmup:
            solve_secs.append(dt)

    islands = int(round(res.fun)) if res.fun is not None else None
    return dict(n_bins=n_bins, total_selected=sum(b.values()), islands=islands,
                build_sec=round(build_sec, 4), solve_secs=solve_secs,
                status=int(res.status), optimal=bool(res.status == 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_rows", type=int, help="number of rows (dataset size)")
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--frac", type=float, default=0.5, help="fraction of each bin to select")
    ap.add_argument("--sort-col", default="f1", help="column to sort by (independent of target)")
    ap.add_argument("--time-limit", type=float, default=None, help="solver time limit (s)")
    ap.add_argument("--warmup", type=int, default=2, help="unrecorded warm-up solves")
    ap.add_argument("--repeats", type=int, default=3, help="recorded solves (one CSV row each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ip_results.csv")
    args = ap.parse_args()

    df, target_col = make_dataset(args.n_rows, N_COLS, seed=args.seed)
    res = solve_islands_ip(df, target_col, args.sort_col, args.bins,
                           args.frac, args.time_limit, args.warmup, args.repeats)
    solve_secs = res.pop("solve_secs")
    for rep, sec in enumerate(solve_secs):                 # one row per recorded solve
        append_csv(args.out, dict(approach="ip_exact", n_rows=args.n_rows,
                                  n_cols=N_COLS, seed=args.seed, rep=rep,
                                  solve_sec=round(sec, 4), **res))
    mean = sum(solve_secs) / len(solve_secs) if solve_secs else float("nan")
    print(f"[ip] n_rows={args.n_rows}  islands={res['islands']}  "
          f"solve_mean={mean:.4f}s over {len(solve_secs)} run(s)  "
          f"optimal={res['optimal']}  -> {args.out}")


if __name__ == "__main__":
    main()
