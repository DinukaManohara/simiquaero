"""test_mdl.py — invariant checks for the MDL branch and bound (python test_mdl.py).

Runs a small search with loose quotas (so the tree has several feasible levels)
and checks, on every node:
  - the quotas of its level are met,
  - its boxes are disjoint, non-empty, and their bounds select exactly their rows,
  - no box was dropped: every parent box contains >= 1 child box,
  - #boxes and the lower bound never decrease along a path; score >= LB,
  - children with no new bounds add nothing to sum gamma(#pred + 1);
and that pruning does not change the best score.
"""
import math

import numpy as np
import pandas as pd

import mdl


def small_problem(seed=0, n=600, n_desc=4):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.random((n, n_desc)), columns=[f"x{j}" for j in range(n_desc)])
    # target follows x0 (plus noise) and the query is bimodal, so matching it on
    # x0 needs two separate intervals -> nodes with several boxes
    df["y"] = 0.8 * df["x0"] + 0.2 * rng.random(n)
    # (+ a small uniform component so that no target bin has a zero quota; a zero
    # quota caps that bin at 0 for every eps)
    query = [("gaussian", {"loc": 0.25, "scale": 0.08}, 1.0),
             ("gaussian", {"loc": 0.75, "scale": 0.08}, 1.0),
             ("uniform", {"loc": 0.0, "scale": 1.0}, 0.6)]
    return mdl.prepare(df, "y", query, pct=0.3, max_bins=12)


def check_node(node, prep):
    X, ybin, b = prep["X"], prep["ybin"], prep["b"]
    upper = np.floor(b * (1 + node.eps)).astype(int)
    counts = np.bincount(ybin[mdl.selected_rows(node.boxes)], minlength=len(b))
    assert np.all(counts >= b) and np.all(counts <= upper), "quotas violated"

    seen = set()
    for bx in node.boxes:
        assert len(bx.rows) > 0, "empty box"
        rows = set(bx.rows.tolist())
        assert not (rows & seen), "boxes overlap"
        seen |= rows
        assert np.array_equal(np.flatnonzero(bx.mask(X)), np.sort(bx.rows)), \
            "bounds do not select exactly the box's rows"

    assert node.score >= node.lb - 1e-9
    assert math.isclose(node.lb, mdl.structure_cost(node.boxes))

    parent = node.parent
    if parent is not None and parent.path:                     # not a child of the root
        assert len(node.boxes) >= len(parent.boxes), "box count decreased"
        assert node.lb >= parent.lb - 1e-9, "lower bound decreased"
        child_rows = [set(bx.rows.tolist()) for bx in node.boxes]
        for pb in parent.boxes:
            prow = set(pb.rows.tolist())
            inside = [cr for cr in child_rows if cr <= prow]
            assert inside, "a parent box was dropped"
        for bx in node.boxes:                                   # bounds only ever added
            pb = next(p for p in parent.boxes if set(bx.rows.tolist()) <= set(p.rows.tolist()))
            assert bx.n_preds() >= pb.n_preds()
            for c, bnd in pb.bounds.items():
                assert bx.bounds[c] == bnd


def test_gamma():
    assert [mdl.gamma_len(n) for n in (1, 2, 3, 4, 7, 8)] == [1, 3, 3, 5, 5, 7]


def test_invariants():
    prep = small_problem()
    nodes = []
    # eps1 = 6: about the tightest start at which level 1 is feasible here (see the
    # note on per-bin quotas in the module); the BIPs still have to trim and split
    best, stats = mdl.search(prep, eps1=6.0, max_bins=12, max_depth=3, record=nodes)
    assert best is not None and len(nodes) >= 5, (len(nodes), stats)
    assert max(n.depth for n in nodes) >= 2, "need at least two levels to test refinement"
    deep = [n for n in nodes if n.depth >= 2]
    assert any(n.lb > n.parent.lb for n in deep), "no refinement ever added bounds"
    assert any(len(n.boxes) > 1 for n in nodes), "no node with several boxes"
    for node in nodes:
        check_node(node, prep)
    return stats, len(nodes)


def test_pruning_is_exact():
    prep = small_problem(seed=1)
    b1, s1 = mdl.search(prep, eps1=6.0, max_bins=12, max_depth=3, prune=True)
    b2, s2 = mdl.search(prep, eps1=6.0, max_bins=12, max_depth=3, prune=False)
    assert math.isclose(b1.score, b2.score), (b1.score, b2.score)
    return sum(s["pruned"] for s in s1), sum(s["solved"] for s in s1), sum(s["solved"] for s in s2)


if __name__ == "__main__":
    test_gamma()
    stats, n = test_invariants()
    for s in stats:
        print(s)
    print(f"invariants hold on all {n} nodes")
    pruned, solved_p, solved_np = test_pruning_is_exact()
    print(f"pruning does not change the best score ({pruned} nodes pruned; "
          f"{solved_p} BIPs solved vs {solved_np} without pruning)")
    print("all tests passed")
