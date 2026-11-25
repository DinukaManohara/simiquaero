"""mdl_synth.py — synthetic experiments for the MDL branch and bound (mdl.py).

Random data with K planted boxes: descriptors and the background target are
uniform on [0, 1]; inside the planted boxes the target is redrawn from a planted
distribution (default N(0.5, 0.05)).  The query is that distribution and the
subset size M is the number of planted rows, so the K boxes are the answer.

Each box constrains 2 columns.  To keep K boxes disjoint for any K they share
one "slot" column, split into K disjoint slots (box k gets the central 80% of
slot k, width 0.8/K), and each box also constrains a second, random column to an
interval of width 0.3.  The planted share is therefore about 0.8 * 0.3 = 24% for
every K.

  python mdl_synth.py [--rows 4000] [--desc 8] [--boxes 1 2 4 8] [--seeds 0 1 2]
                      [--eps1 0.5] [--out mdl_synth.csv]
"""
import argparse
import time

import numpy as np
import pandas as pd

import mdl
from bench_common import append_csv, dist_metrics


def make_planted(n_rows=4000, n_desc=8, n_boxes=1, width=0.3, gap=0.1,
                 loc=0.5, sd=0.05, seed=0):
    """Returns (df with columns x0.. and y, planted row mask, planted boxes)."""
    rng = np.random.default_rng(seed)
    X = rng.random((n_rows, n_desc))
    y = rng.random(n_rows)
    slot_col = int(rng.integers(n_desc))
    planted, boxes = np.zeros(n_rows, dtype=bool), []
    for k in range(n_boxes):
        s_lo, s_hi = (k + gap) / n_boxes, (k + 1 - gap) / n_boxes
        other = int(rng.choice([c for c in range(n_desc) if c != slot_col]))
        o_lo = float(rng.uniform(0, 1 - width))
        inside = ((X[:, slot_col] >= s_lo) & (X[:, slot_col] < s_hi)
                  & (X[:, other] >= o_lo) & (X[:, other] < o_lo + width))
        planted |= inside
        boxes.append({slot_col: (s_lo, s_hi), other: (o_lo, o_lo + width)})
    y[planted] = rng.normal(loc, sd, planted.sum())
    df = pd.DataFrame(X, columns=[f"x{j}" for j in range(n_desc)])
    df["y"] = y
    return df, planted, boxes


def run(n_rows, n_desc, n_boxes, seed, eps1, max_bins, time_limit, max_depth, loc, sd):
    df, planted, _ = make_planted(n_rows, n_desc, n_boxes, loc=loc, sd=sd, seed=seed)
    query = [("gaussian", {"loc": loc, "scale": sd}, 1.0)]
    prep = mdl.prepare(df, "y", query, size=int(planted.sum()), max_bins=max_bins)
    t0 = time.perf_counter()
    best, stats = mdl.search(prep, eps1=eps1, max_bins=max_bins, time_limit=time_limit,
                             max_depth=max_depth)
    sec = time.perf_counter() - t0
    row = dict(n_rows=n_rows, n_desc=n_desc, planted_boxes=n_boxes, seed=seed, eps1=eps1,
               planted_rows=int(planted.sum()), target_bins=len(prep["p"]),
               zero_quota_bins=int((prep["b"] == 0).sum()), search_sec=round(sec, 3),
               levels=len(stats), solved=sum(s["solved"] for s in stats),
               infeasible=sum(s["infeasible"] for s in stats),
               pruned=sum(s["pruned"] for s in stats))
    if best is None:
        row.update(found_boxes=None, depth=None, score=None, kl_bits=None, jaccard=None, tvd=None)
    else:
        sel = np.zeros(n_rows, dtype=bool)
        sel[mdl.selected_rows(best.boxes)] = True
        counts = np.bincount(prep["ybin"][sel], minlength=len(prep["p"]))
        row.update(found_boxes=len(best.boxes), depth=best.depth, score=round(best.score, 4),
                   kl_bits=round(best.kl, 4),
                   jaccard=round(float((sel & planted).sum() / (sel | planted).sum()), 4),
                   tvd=dist_metrics({k: int(v) for k, v in enumerate(counts)},
                                    {k: float(v) for k, v in enumerate(prep["p"])})["tvd"])
    return row, best, prep, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=4000)
    ap.add_argument("--desc", type=int, default=8)
    ap.add_argument("--boxes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eps1", type=float, default=0.5)
    ap.add_argument("--max-bins", type=int, default=128)
    ap.add_argument("--time-limit", type=float, default=60)
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--loc", type=float, default=0.5)
    ap.add_argument("--sd", type=float, default=0.05)
    ap.add_argument("--out", default="mdl_synth.csv")
    args = ap.parse_args()
    for k in args.boxes:
        for seed in args.seeds:
            row, best, prep, _ = run(args.rows, args.desc, k, seed, args.eps1, args.max_bins,
                                     args.time_limit, args.max_depth, args.loc, args.sd)
            append_csv(args.out, row)
            print(f"[mdl-synth] K={k} seed={seed}  found={row['found_boxes']} "
                  f"depth={row['depth']} score={row['score']} jaccard={row['jaccard']} "
                  f"infeasible={row['infeasible']}/{row['solved']}  {row['search_sec']}s")
            if best is not None:
                print("  " + mdl.rule_str(best, prep["columns"]).replace("\n", "\n  "))


if __name__ == "__main__":
    main()
