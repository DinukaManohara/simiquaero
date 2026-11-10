"""Shared helpers for the benchmark scripts (ip.py, dt_*.py).

Dataset: descriptors ~ U(0,1)^(n_cols-1) and a target ~ U(0,1) drawn
INDEPENDENTLY of the descriptors.  Because the target does not depend on the
descriptors, each target bin's rows scatter across descriptor space and the
purity-grown tree has many fine leaves per bin -- the realistic
subgroup-discovery case where a distributional query selects a *subset* of each
bin's rows (fractional selection is possible).  A target that were a function of
the descriptors would collapse each bin to a single leaf and make that
impossible.
"""
import os
import csv
import math
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
def make_dataset(n_rows, n_cols, seed=0):
    """Return (df, target_col).  n_cols = descriptors + 1 target column.

    The target is INDEPENDENT of the descriptors, so each target bin's rows are
    scattered across descriptor space -> the purity-grown tree has many fine
    leaves per bin (the realistic subgroup-discovery case, where a query selects
    a *subset* of each bin's rows).  A target that were a function of the
    descriptors would collapse each bin to a single leaf and make fractional
    selection impossible.
    """
    rng = np.random.default_rng(seed)
    n_desc = max(2, n_cols - 1)
    X = rng.random((n_rows, n_desc))
    target = rng.random(n_rows)                              # independent of descriptors
    df = pd.DataFrame(X, columns=[f"f{j}" for j in range(n_desc)])
    df["target"] = target
    return df, "target"


def append_csv(path, row):
    """Append one dict row to a CSV, writing the header if the file is new."""
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


# ----------------------------------------------------------------------------
# Binning + target pdf
# ----------------------------------------------------------------------------
def bin_target(df, target_col, n_bins=None):
    """Discretise the target into n_bins (fixed) or via Scott's rule if None."""
    x = df[target_col].dropna().values
    if n_bins is None:
        std = np.std(x, ddof=1)
        h = 3.5 * std * (len(x) ** (-1 / 3))
        n_bins = max(1, math.ceil((x.max() - x.min()) / h))
    ids = pd.cut(df[target_col], bins=n_bins, labels=range(n_bins),
                 include_lowest=True).astype(int).to_numpy()
    return ids, n_bins


def _bin_centers(df, target_col, n_bins):
    lo, hi = df[target_col].min(), df[target_col].max()
    e = np.linspace(lo, hi, n_bins + 1)
    return (e[:-1] + e[1:]) / 2


def gaussian_target(df, target_col, n_bins):
    """A Gaussian target distribution over the bins (proportions summing to 1)."""
    ctr = _bin_centers(df, target_col, n_bins)
    mu = float(df[target_col].mean()); sig = float(df[target_col].std()) / 2 or 1.0
    p = np.exp(-0.5 * ((ctr - mu) / sig) ** 2)
    p = p / p.sum() if p.sum() > 0 else np.ones(n_bins) / n_bins
    return {b: float(p[b]) for b in range(n_bins)}


# ----------------------------------------------------------------------------
# Decision-tree index + selection (from quaero_tree)
# ----------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier


def build_index(df, target_col, desc_cols=None, n_bins=None):
    if desc_cols is None:
        desc_cols = [c for c in df.columns if c != target_col]
    y, n_bins = bin_target(df, target_col, n_bins)
    X = df[desc_cols].to_numpy()
    clf = DecisionTreeClassifier(criterion="gini", random_state=0).fit(X, y)
    t = clf.tree_
    n = t.node_count
    cl, cr, feat, thr = t.children_left, t.children_right, t.feature, t.threshold
    is_leaf = cl == -1
    parent = np.full(n, -1, int); depth = np.zeros(n, int); st = [(0, 0)]
    while st:
        v, d = st.pop(); depth[v] = d
        if not is_leaf[v]:
            parent[cl[v]] = parent[cr[v]] = v
            st += [(cl[v], d + 1), (cr[v], d + 1)]
    leaves = [v for v in range(n) if is_leaf[v]]
    leaf_of_row = clf.apply(X)
    rows_of_leaf = {}
    for i, lf in enumerate(leaf_of_row):
        rows_of_leaf.setdefault(int(lf), []).append(i)
    node_cnt = {v: {} for v in range(n)}
    for lf in leaves:
        c = {}
        for i in rows_of_leaf.get(lf, []):
            b = int(y[i]); c[b] = c.get(b, 0) + 1
        node_cnt[lf] = c
    for v in sorted(range(n), key=lambda v: -depth[v]):
        if not is_leaf[v]:
            m = {}
            for ch in (cl[v], cr[v]):
                for b, k in node_cnt[ch].items():
                    m[b] = m.get(b, 0) + k
            node_cnt[v] = m
    leaf_bin = {v: max(node_cnt[v], key=node_cnt[v].get) for v in leaves}

    def path_predicate(v):
        preds, cur = [], v
        while parent[cur] != -1:
            p = parent[cur]
            op = "<=" if cl[p] == cur else ">"
            preds.append((desc_cols[feat[p]], op, float(thr[p]))); cur = p
        return list(reversed(preds))

    return dict(clf=clf, desc_cols=desc_cols, n_bins=n_bins, n=n, is_leaf=is_leaf,
                cl=cl, cr=cr, parent=parent, depth=depth, node_cnt=node_cnt,
                leaves=leaves, leaf_bin=leaf_bin, rows_of_leaf=rows_of_leaf,
                path_predicate=path_predicate)


def leaves_under(ix, v):
    out, st = [], [v]
    while st:
        u = st.pop()
        (out.append(u) if ix["is_leaf"][u] else st.extend([ix["cl"][u], ix["cr"][u]]))
    return out


def tv_distance(got, tc):
    gs, ts = sum(got.values()), sum(tc.values())
    if gs == 0 or ts == 0:
        return 1.0
    bins = set(got) | set(tc)
    return 0.5 * sum(abs(got.get(b, 0) / gs - tc.get(b, 0) / ts) for b in bins)


def dist_metrics(got, tc, eps=1e-12):
    """Distances between the found subset's bin distribution (`got`) and the
    target (`tc`).  Both may be counts or proportions; they are normalised over
    the union of their bins.  KLD is KL(target || subset), eps-smoothed so a
    bin the subset misses gives a large finite value instead of inf."""
    bins = sorted(set(got) | set(tc))
    q = np.array([got.get(b, 0.0) for b in bins], dtype=float)   # subset
    p = np.array([tc.get(b, 0.0) for b in bins], dtype=float)    # target
    if q.sum() <= 0 or p.sum() <= 0:
        return dict(tvd=1.0, jsd=1.0, kld=float("inf"), hellinger=1.0, bc=0.0)
    p, q = p / p.sum(), q / q.sum()

    tvd = 0.5 * np.abs(p - q).sum()
    bc = float(np.sqrt(p * q).sum())
    hellinger = float(np.sqrt(max(0.0, 1.0 - bc)))

    ps, qs = (p + eps) / (p + eps).sum(), (q + eps) / (q + eps).sum()
    kld = float(np.sum(ps * np.log(ps / qs)))                    # nats
    m = 0.5 * (ps + qs)
    jsd = float(0.5 * np.sum(ps * np.log2(ps / m)) +
                0.5 * np.sum(qs * np.log2(qs / m)))              # bits, in [0, 1]

    return dict(tvd=round(float(tvd), 4), jsd=round(jsd, 4), kld=round(kld, 4),
                hellinger=round(hellinger, 4), bc=round(bc, 4))


def select_subtrees(ix, target, total=None, excluded=None, cap_slack=0.02):
    excluded = excluded or set()
    bins = sorted(ix["node_cnt"][0])
    s = sum(target.values())
    if total is None and s <= 1.5:
        total = min(ix["node_cnt"][0].get(b, 0) / target[b] for b in bins if target.get(b, 0) > 0)
    tc = {b: target.get(b, 0) * (total if s <= 1.5 else 1) for b in bins}
    cap = {b: tc[b] * (1 + cap_slack) for b in bins}
    selected, got = set(), {b: 0.0 for b in bins}
    internal = [v for v in range(ix["n"]) if not ix["is_leaf"][v]]
    tot = lambda v: sum(ix["node_cnt"][v].values())
    for v in sorted(internal, key=lambda v: (ix["depth"][v], -tot(v))):
        lu = leaves_under(ix, v)
        if any(l in selected or l in excluded for l in lu):
            continue
        c = ix["node_cnt"][v]
        if all(got[b] + c.get(b, 0) <= cap[b] + 1e-9 for b in bins):
            selected.update(lu)
            for b, k in c.items():
                got[b] += k
    for lf in sorted(ix["leaves"], key=lambda l: -ix["node_cnt"][l].get(ix["leaf_bin"][l], 0)):
        if lf in selected or lf in excluded:
            continue
        b = ix["leaf_bin"][lf]; k = ix["node_cnt"][lf].get(b, 0)
        if got[b] + k <= cap[b] + 1e-9 and got[b] < tc[b] - 1e-9:
            selected.add(lf); got[b] += k
    labels = {lf: int(lf in selected) for lf in ix["leaves"]}
    return labels, got, tc


def reduce_labeling(ix, labels):
    n, parent = ix["n"], ix["parent"]
    lab = np.full(n, -1, int)
    for v in sorted(range(n), key=lambda v: -ix["depth"][v]):
        if ix["is_leaf"][v]:
            lab[v] = labels[v]
        else:
            a, b = lab[ix["cl"][v]], lab[ix["cr"][v]]
            lab[v] = a if (a == b and a != -1) else -1
    A = [v for v in range(n) if lab[v] == 1 and (parent[v] == -1 or lab[parent[v]] != 1)]
    dnf = [ix["path_predicate"](v) for v in A]
    return dict(node_label=lab, antichain=A, dnf=dnf, n_terms=len(A))


def find_multiple_subsets(ix, target, total, tvd_threshold, max_subsets=50):
    excluded, results = set(), []
    while len(results) < max_subsets:
        labels, got, tc = select_subtrees(ix, target, total, excluded=excluded)
        if sum(labels.values()) == 0:
            break
        tvd = tv_distance(got, tc)
        if tvd > tvd_threshold:
            break
        red = reduce_labeling(ix, labels)
        results.append(dict(labels=labels, got=got, tc=tc, tvd=tvd, red=red))
        excluded |= {lf for lf, l in labels.items() if l}
    return results


def _leaf_intervals(ix):
    cl, cr, is_leaf = ix["cl"], ix["cr"], ix["is_leaf"]
    leaf_index, order, st = {}, [0], [0]
    while st:
        v = st.pop()
        if is_leaf[v]:
            leaf_index[v] = order[0]; order[0] += 1
        else:
            st.append(cr[v]); st.append(cl[v])
    lo, hi = {}, {}
    for v in sorted(range(ix["n"]), key=lambda v: -ix["depth"][v]):
        if is_leaf[v]:
            lo[v] = hi[v] = leaf_index[v]
        else:
            lo[v] = min(lo[cl[v]], lo[cr[v]]); hi[v] = max(hi[cl[v]], hi[cr[v]])
    return lo, hi


def find_best_boxes(ix, target, n_boxes, beam=16, cand_cap=3000):
    """Query by fixed number of boxes: n_boxes disjoint subtrees whose union's
    bin distribution is closest (min TVD) to `target`.  Beam search.

    On large (near-O(n)-node) trees the candidate set is capped to the
    `cand_cap` heaviest nodes so the per-step cost stays bounded."""
    lo, hi = _leaf_intervals(ix)
    nodes = [v for v in range(ix["n"]) if sum(ix["node_cnt"][v].values()) > 0]
    if len(nodes) > cand_cap:                                    # keep the heaviest nodes
        nodes = sorted(nodes, key=lambda v: -sum(ix["node_cnt"][v].values()))[:cand_cap]
    states = [(1.0, (), [], {})]
    for _ in range(n_boxes):
        cand = {}
        for _tvd, chosen, occ, got in states:
            for v in nodes:
                if v in chosen:
                    continue
                a, b = lo[v], hi[v]
                if any(not (b < c or d < a) for (c, d) in occ):
                    continue
                g2 = dict(got)
                for bb, k in ix["node_cnt"][v].items():
                    g2[bb] = g2.get(bb, 0) + k
                key = frozenset(chosen + (v,))
                t = tv_distance(g2, target)
                if key not in cand or t < cand[key][0]:
                    cand[key] = (t, chosen + (v,), occ + [(a, b)], g2)
        if not cand:
            break
        states = sorted(cand.values(), key=lambda s: s[0])[:beam]
    tvd, chosen, occ, got = states[0]
    dnf = [ix["path_predicate"](v) for v in chosen]
    return dict(chosen=list(chosen), got=got, tc=dict(target), tvd=tvd,
                dnf=dnf, n_boxes=len(chosen))
