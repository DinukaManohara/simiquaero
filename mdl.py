"""mdl.py — MDL branch-and-bound search for distributional subset queries.

Finds a set of disjoint axis-aligned boxes (a DNF rule over the descriptors)
whose selected rows follow a query distribution over the target, scoring every
candidate with a two-part MDL code (in bits):

    score = gamma(#boxes) + sum_i gamma(#pred_i + 1) + KL(selected || query)

gamma(n) = 2*floor(log2 n) + 1 is the Elias-gamma code length; #pred_i counts
the bounds of box i (one per bound; a bound at the edge of the box's own rows is
not written, so a column contributes 0, 1 or 2).

Search tree
-----------
The root holds all rows as a single box.  A node is expanded by every column not
yet on its path (ordered: a->b and b->a are different nodes).  Expanding by a
column: inside every box (island) the column is discretised with the
Freedman-Diaconis rule over that box's rows, and a binary integer program picks
bins to minimise the total number of islands, subject to

    b_k <= sum_{boxes, bins} count_k(bin) * x  <= floor(b_k * (1 + eps_d))   (every target bin k)
    every box keeps >= 1 non-empty bin                                      (no box is dropped)

Each run of chosen bins becomes a child box (the parent's bounds plus an
interval on the column).  eps halves at every level, eps_d = eps1 / 2^(d-1); the
level at which every cap equals b_k is the last one.

Levels are processed breadth-first: every node of a level is solved and scored
(updating the best score), then nodes whose lower bound

    LB = gamma(#boxes) + sum_i gamma(#pred_i + 1)

exceeds the best score are pruned and the survivors are expanded.  LB is valid
for every descendant: boxes are never dropped and bounds are only ever added, so
both terms are non-decreasing along a path, and KL >= 0.

The target is discretised with the Freedman-Diaconis rule too; the quotas are
b_k = round(p_k * M) for query proportions p and subset size M.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import milp, LinearConstraint, Bounds

from quaero_core import target_props


# ----------------------------------------------------------------------------
# Codes and discretisation
# ----------------------------------------------------------------------------
def gamma_len(n):
    """Elias-gamma code length of a positive integer: 2*floor(log2 n) + 1."""
    n = int(n)
    if n < 1:
        raise ValueError(f"Elias-gamma needs n >= 1, got {n}")
    return 2 * (n.bit_length() - 1) + 1


def fd_edges(v, max_bins=128):
    """Freedman-Diaconis bin edges over the values v.  IQR = 0 falls back to one
    bin per distinct value; the bin count is capped at max_bins and at the number
    of distinct values.  A constant column gives a single bin."""
    v = np.asarray(v, dtype=float)
    lo, hi = float(v.min()), float(v.max())
    if len(v) < 2 or hi == lo:
        return np.array([lo, hi])
    q25, q75 = np.percentile(v, [25, 75])
    h = 2.0 * (q75 - q25) * len(v) ** (-1.0 / 3.0)
    n_unique = len(np.unique(v))
    n_bins = math.ceil((hi - lo) / h) if h > 0 else n_unique
    n_bins = max(1, min(n_bins, max_bins, n_unique))
    return np.linspace(lo, hi, n_bins + 1)


def assign_bins(v, edges):
    """Bin index of each value: bin i is [e_i, e_{i+1}), the last bin is closed."""
    n_bins = len(edges) - 1
    return np.clip(np.searchsorted(edges, v, side="right") - 1, 0, n_bins - 1)


# ----------------------------------------------------------------------------
# Boxes and nodes
# ----------------------------------------------------------------------------
@dataclass
class Box:
    rows: np.ndarray                              # row indices in the box
    bounds: dict = field(default_factory=dict)    # column -> (lo, hi); None = unbounded

    def n_preds(self):
        return sum((lo is not None) + (hi is not None) for lo, hi in self.bounds.values())

    def mask(self, X):
        """Rows of X satisfying the box's bounds (lo <= x < hi)."""
        m = np.ones(len(X), dtype=bool)
        for c, (lo, hi) in self.bounds.items():
            if lo is not None:
                m &= X[:, c] >= lo
            if hi is not None:
                m &= X[:, c] < hi
        return m


@dataclass
class Node:
    path: tuple                 # columns used, in order
    boxes: list
    eps: float                  # eps of the level that produced it
    lb: float                   # gamma(#boxes) + sum gamma(#pred + 1)
    kl: float                   # KL(selected || query), bits
    score: float                # lb + kl
    optimal: bool = True        # BIP solved to optimality (False: time limit)
    parent: object = None

    @property
    def depth(self):
        return len(self.path)


def structure_cost(boxes):
    """gamma(#boxes) + sum_i gamma(#pred_i + 1): the score's structural part and
    the lower bound for the node and all its descendants."""
    return gamma_len(len(boxes)) + sum(gamma_len(bx.n_preds() + 1) for bx in boxes)


def selected_rows(boxes):
    return np.concatenate([bx.rows for bx in boxes]) if boxes else np.array([], dtype=int)


def kl_bits(boxes, ybin, p):
    """KL(selected || query) in bits over the target bins."""
    counts = np.bincount(ybin[selected_rows(boxes)], minlength=len(p)).astype(float)
    q = counts / counts.sum()
    nz = q > 0
    if np.any(p[nz] <= 0):
        return math.inf
    return float(np.sum(q[nz] * np.log2(q[nz] / p[nz])))


# ----------------------------------------------------------------------------
# The expansion BIP
# ----------------------------------------------------------------------------
def expand(boxes, col, X, ybin, n_tbins, lower, upper, max_bins=128, time_limit=None):
    """Refine every box on column `col`.  Per box, FD-bin its rows on the column;
    choose bins (x) to minimise the total number of islands subject to the
    target-bin quotas  lower_k <= selected count_k <= upper_k  and to every box
    keeping at least one non-empty bin.

    Returns (child boxes, n_islands, optimal) or None if infeasible / no solution."""
    blocks, nx = [], 0
    for bx in boxes:
        v = X[bx.rows, col]
        edges = fd_edges(v, max_bins)
        bins = assign_bins(v, edges)
        m = len(edges) - 1
        C = np.zeros((m, n_tbins), dtype=np.int64)        # C[i, k]: rows in bin i, target bin k
        np.add.at(C, (bins, ybin[bx.rows]), 1)
        blocks.append((bx, edges, bins, C, nx))
        nx += m
    nw = sum(len(blk[1]) - 2 for blk in blocks)            # m - 1 links per box
    nv = nx + nw

    # objective: sum x - sum w = number of islands (w_i = x_i * x_{i+1} within a box)
    cost = np.concatenate([np.ones(nx), -np.ones(nw)])

    # w <= x_i and w <= x_{i+1} (w is maximised, so this makes w = x_i * x_{i+1})
    r_idx, c_idx, vals, r, w_off = [], [], [], 0, nx
    for bx, edges, bins, C, xo in blocks:
        m = len(edges) - 1
        for i in range(m - 1):
            w = w_off + i
            r_idx += [r, r, r + 1, r + 1]
            c_idx += [w, xo + i, w, xo + i + 1]
            vals += [1.0, -1.0, 1.0, -1.0]
            r += 2
        w_off += m - 1
    constraints = []
    if r:
        A = sparse.csr_matrix((vals, (r_idx, c_idx)), shape=(r, nv))
        constraints.append(LinearConstraint(A, -np.inf, 0.0))

    # quotas per target bin
    q_r, q_c, q_v = [], [], []
    for bx, edges, bins, C, xo in blocks:
        ii, kk = np.nonzero(C)
        q_r.append(kk); q_c.append(xo + ii); q_v.append(C[ii, kk])
    Q = sparse.csr_matrix((np.concatenate(q_v).astype(float),
                           (np.concatenate(q_r), np.concatenate(q_c))), shape=(n_tbins, nv))
    constraints.append(LinearConstraint(Q, lower, upper))

    # no box is dropped: each keeps >= 1 non-empty bin
    k_r, k_c = [], []
    for l, (bx, edges, bins, C, xo) in enumerate(blocks):
        nonempty = np.nonzero(C.sum(axis=1))[0]
        k_r.append(np.full(len(nonempty), l)); k_c.append(xo + nonempty)
    K = sparse.csr_matrix((np.ones(sum(len(a) for a in k_r)),
                           (np.concatenate(k_r), np.concatenate(k_c))), shape=(len(blocks), nv))
    constraints.append(LinearConstraint(K, 1.0, np.inf))

    integrality = np.concatenate([np.ones(nx), np.zeros(nw)])
    options = {"time_limit": time_limit} if time_limit else {}
    res = milp(cost, constraints=constraints, integrality=integrality,
               bounds=Bounds(0, 1), options=options)
    if res.x is None:
        return None
    x = res.x[:nx] > 0.5

    # each run of chosen bins -> a child box
    children, n_islands = [], 0
    for bx, edges, bins, C, xo in blocks:
        m = len(edges) - 1
        sel = x[xo:xo + m]
        i = 0
        while i < m:
            if not sel[i]:
                i += 1
                continue
            j = i
            while j + 1 < m and sel[j + 1]:
                j += 1
            n_islands += 1
            in_run = (bins >= i) & (bins <= j)
            if in_run.any():
                lo = float(edges[i]) if i > 0 else None          # edge of the box's rows: no bound
                hi = float(edges[j + 1]) if j < m - 1 else None
                bounds = dict(bx.bounds)
                if lo is not None or hi is not None:
                    bounds[col] = (lo, hi)
                children.append(Box(bx.rows[in_run], bounds))
            i = j + 1
    return children, n_islands, res.status == 0


# ----------------------------------------------------------------------------
# Preparation and search
# ----------------------------------------------------------------------------
def prepare(df, target_col, components, size=None, pct=None, max_bins=128):
    """Arrays for the search.  The target is FD-binned; the query mixture
    `components` gives proportions p per bin; quotas b = round(p * M) with
    M = size, or pct * N."""
    desc = [c for c in df.columns if c != target_col]
    X = df[desc].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    tedges = fd_edges(y, max_bins)
    ybin = assign_bins(y, tedges)
    centers = (tedges[:-1] + tedges[1:]) / 2
    p = target_props(components, centers)
    if p is None:
        raise ValueError("the query puts no mass on the target's range")
    if size is None:
        if pct is None:
            raise ValueError("give size or pct")
        size = pct * len(df)
    b = np.round(p * size).astype(int)
    return dict(X=X, columns=desc, y=y, tedges=tedges, ybin=ybin, p=np.asarray(p, float),
                b=b, size=int(round(size)), target=target_col)


def search(prep, eps1=0.5, max_bins=128, time_limit=60, max_depth=None, prune=True,
           verbose=False, record=None):
    """Level-wise branch and bound.  Returns (best node or None, per-level stats).
    If `record` is a list, every feasible node created is appended to it."""
    X, ybin, p, b = prep["X"], prep["ybin"], prep["p"], prep["b"]
    n_tbins, n_cols = len(p), X.shape[1]
    root = Node(path=(), boxes=[Box(np.arange(len(X)))], eps=math.inf,
                lb=0.0, kl=math.inf, score=math.inf)
    best, level, d, stats = None, [root], 1, []
    while level:
        eps = eps1 / 2 ** (d - 1)
        upper = np.floor(b * (1 + eps)).astype(int)
        t0 = time.perf_counter()
        children, solved, infeasible = [], 0, 0
        for node in level:
            for col in range(n_cols):
                if col in node.path:
                    continue
                solved += 1
                out = expand(node.boxes, col, X, ybin, n_tbins, b, upper, max_bins, time_limit)
                if out is None:
                    infeasible += 1
                    continue
                boxes, _, optimal = out
                lb = float(structure_cost(boxes))
                kl = kl_bits(boxes, ybin, p)
                child = Node(node.path + (col,), boxes, eps, lb, kl, lb + kl, optimal, node)
                children.append(child)
                if record is not None:
                    record.append(child)
                if best is None or child.score < best.score:
                    best = child
        expandable = [c for c in children if c.depth < n_cols]
        if prune and best is not None:
            survivors = [c for c in expandable if c.lb <= best.score]
        else:
            survivors = expandable
        last = bool((upper == b).all()) or (max_depth is not None and d >= max_depth)
        stats.append(dict(level=d, eps=eps, expanded=len(level), solved=solved,
                          feasible=len(children), infeasible=infeasible,
                          pruned=len(expandable) - len(survivors),
                          best_score=None if best is None else round(best.score, 4),
                          sec=round(time.perf_counter() - t0, 3), last=last))
        if verbose:
            print(stats[-1])
        if last:
            break
        level = survivors
        d += 1
    return best, stats


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def rule_str(node, columns, prec=4):
    """The node's boxes as a DNF rule, one box per line."""
    terms = []
    for bx in node.boxes:
        parts = []
        for c, (lo, hi) in sorted(bx.bounds.items()):
            name = columns[c]
            if lo is not None and hi is not None:
                parts.append(f"{lo:.{prec}g} <= {name} < {hi:.{prec}g}")
            elif lo is not None:
                parts.append(f"{name} >= {lo:.{prec}g}")
            elif hi is not None:
                parts.append(f"{name} < {hi:.{prec}g}")
        terms.append("(" + (" AND ".join(parts) if parts else "TRUE") + ")")
    return "\n  OR ".join(terms)


def score_parts(node):
    """The three parts of the score."""
    return dict(boxes_bits=gamma_len(len(node.boxes)),
                preds_bits=sum(gamma_len(bx.n_preds() + 1) for bx in node.boxes),
                kl_bits=node.kl, score=node.score)
