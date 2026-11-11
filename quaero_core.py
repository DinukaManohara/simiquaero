"""quaero_core.py — the decision-tree distributional-query pipeline, no GUI.

Shared by quaero_ui.py (Tk app) and case_study_dt.ipynb (notebook), so both run
exactly the same preprocessing, indexing and query code.  Importing this module
does not touch tkinter or the matplotlib backend.

  load:   read_dataset -> target_candidates -> prepare_frame
  index:  build_index  (Scott's-rule target bins, purity-grown decision tree)
  query:  find_multiple_subsets (by cardinality)  |
          find_multiple_box_subsets / find_best_boxes (by #boxes)
  rule:   reduce_labeling -> simplify_rule (sympy) -> rule_to_str
  rank:   rank_by_rule_length (shortest simplified rule first)
"""
import os
import math
import importlib.util
import numpy as np
import pandas as pd
import sympy as sp
from scipy import stats as scipy_stats
from sklearn.tree import DecisionTreeClassifier

from bench_common import dist_metrics      # TVD / JSD / KLD / Hellinger / BC
from bench_common import free_nodes        # O(1) per-node test against used leaves


# ----------------------------------------------------------------------------
# Pipeline (embedded from quaero_tree)
# ----------------------------------------------------------------------------
def bin_target(df, target_col, max_bins=128):
    """Scott's-rule binning of the target, guarded for real-world columns:
    zero-variance, tiny, or discrete targets would otherwise blow up n_bins."""
    x = pd.to_numeric(df[target_col], errors="coerce").dropna().to_numpy(dtype=float)
    spread = float(x.max() - x.min()) if len(x) else 0.0
    std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    h = 3.5 * std * (len(x) ** (-1 / 3)) if std > 0 else 0.0
    n_bins = max(1, math.ceil(spread / h)) if h > 0 and spread > 0 else 1
    n_bins = min(n_bins, max_bins, max(1, len(np.unique(x))))   # never out-bin the data
    ids = pd.cut(df[target_col], bins=n_bins, labels=range(n_bins),
                 include_lowest=True).astype(int).to_numpy()
    return ids, n_bins


def _bin_centers(df, target_col, n_bins):
    lo, hi = df[target_col].min(), df[target_col].max()
    e = np.linspace(lo, hi, n_bins + 1)
    return (e[:-1] + e[1:]) / 2


DISTRIBUTIONS = {
    "gaussian": scipy_stats.norm, "normal": scipy_stats.norm,
    "exponential": scipy_stats.expon, "uniform": scipy_stats.uniform,
    "beta": scipy_stats.beta,
    "weibull": scipy_stats.weibull_min,                  # shape k, scale lambda, loc
    "lognormal": scipy_stats.lognorm, "log-normal": scipy_stats.lognorm,
}


def _component(name, params):
    """(scipy distribution, scipy kwargs) for one mixture component.  Weibull
    takes shape/scale/loc; Log-Normal takes mu/sigma/loc with
    ln(X - loc) ~ N(mu, sigma^2).  scipy's own names (c, s) pass through."""
    key = name.lower()
    if key not in DISTRIBUTIONS:
        raise ValueError(f"unknown distribution {name!r}; choose from "
                         f"{', '.join(sorted(DISTRIBUTIONS))}")
    bad = [k for k in ("scale", "shape", "sigma", "c", "s", "a", "b")
           if k in params and params[k] <= 0]                # report the names as typed
    if bad:
        raise ValueError(f"{name}: {', '.join(bad)} must be > 0")
    p = dict(params)
    if key == "weibull" and "shape" in p:
        p["c"] = p.pop("shape")
    if key in ("lognormal", "log-normal"):
        if "sigma" in p:
            p["s"] = p.pop("sigma")
        if "mu" in p:
            p["scale"] = math.exp(p.pop("mu"))
    return DISTRIBUTIONS[key], p


def make_pdf(spec):
    if isinstance(spec, tuple):
        d, p = _component(*spec)
        return lambda x: d.pdf(x, **p)
    tot = sum(w for *_, w in spec) or 1.0
    comp = [(*_component(n, p), w / tot) for n, p, w in spec]
    return lambda x: sum(w * d.pdf(x, **p) for d, p, w in comp)


def build_index(df, target_col, desc_cols=None):
    if desc_cols is None:
        desc_cols = [c for c in df.columns if c != target_col]
    y, n_bins = bin_target(df, target_col)
    X = df[desc_cols].to_numpy()
    clf = DecisionTreeClassifier(criterion="gini", random_state=0).fit(X, y)
    t, C = clf.tree_, clf.classes_
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


def select_subtrees(ix, target, total, excluded=None, cap_slack=0.02):
    """Select leaves matching `target` at cardinality `total`, ignoring any leaf
    in `excluded` (already used by an earlier subset)."""
    excluded = excluded or set()
    bins = sorted(ix["node_cnt"][0])
    tc = {b: target.get(b, 0) * total for b in bins}
    cap = {b: tc[b] * (1 + cap_slack) for b in bins}
    selected, got = set(), {b: 0.0 for b in bins}
    internal = [v for v in range(ix["n"]) if not ix["is_leaf"][v]]
    tot = lambda v: sum(ix["node_cnt"][v].values())
    for v in sorted(internal, key=lambda v: (ix["depth"][v], -tot(v))):
        lu = leaves_under(ix, v)
        if any(l in selected or l in excluded for l in lu):   # subtree must be fully available
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


def tv_distance(got, tc):
    gs, ts = sum(got.values()), sum(tc.values())
    if gs == 0 or ts == 0:
        return 1.0
    bins = set(got) | set(tc)
    return 0.5 * sum(abs(got.get(b, 0) / gs - tc.get(b, 0) / ts) for b in bins)


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
    reduced = [v for v in range(n) if parent[v] == -1 or lab[parent[v]] == -1]
    dnf = [ix["path_predicate"](v) for v in A]
    return dict(node_label=lab, antichain=A, dnf=dnf,
                reduced_size=len(reduced), n_terms=len(A))


def selected_rows(ix, labels):
    rows = []
    for lf, l in labels.items():
        if l:
            rows += ix["rows_of_leaf"].get(lf, [])
    return rows


def find_multiple_subsets(ix, target, total, tvd_threshold):
    """Extract disjoint subsets that each match `target` at cardinality `total`,
    run to exhaustion.

    Find the first subset (+ its rule), exclude its leaves, find the next from the
    remaining leaves, and so on -- stopping once the best achievable subset exceeds
    `tvd_threshold` or nothing more can be selected.  Returns a list of dicts with
    labels, got, tc, tvd, red (rule), and rows.
    """
    excluded, results = set(), []
    while True:
        labels, got, tc = select_subtrees(ix, target, total, excluded=excluded)
        if sum(labels.values()) == 0:
            break
        tvd = tv_distance(got, tc)
        if tvd > tvd_threshold:
            break
        red = reduce_labeling(ix, labels)
        results.append(dict(labels=labels, got=got, tc=tc, tvd=tvd, red=red,
                            rows=selected_rows(ix, labels)))
        excluded |= {lf for lf, l in labels.items() if l}
    return results


def dnf_to_str(dnf, max_terms=12):
    terms = dnf[:max_terms]
    lines = []
    for term in terms:
        lines.append("(" + " AND ".join(f"{f} {op} {t:.3f}" for f, op, t in term) + ")"
                     if term else "(TRUE)")
    out = "\n  OR ".join(lines)
    if len(dnf) > max_terms:
        out += f"\n  ... (+{len(dnf) - max_terms} more terms)"
    return out


# ----------------------------------------------------------------------------
# Rule simplification (sympy interval algebra) + ranking by rule length
# ----------------------------------------------------------------------------
_FULL = sp.Interval(-sp.oo, sp.oo)


def _term_box(term):
    """One DNF conjunct -> {feature: sympy Interval}.  Tree splits are
    x <= t  -> (-oo, t]  and  x > t -> (t, oo); repeats are intersected."""
    box = {}
    for f, op, t in term:
        iv = (sp.Interval.Lopen(-sp.oo, sp.Float(t)) if op == "<="
              else sp.Interval.open(sp.Float(t), sp.oo))
        box[f] = box.get(f, _FULL) & iv
    return box


def _merge_boxes(boxes):
    """Merge boxes that agree on every feature but one and whose intervals on
    that feature form a single interval.  Repeats until nothing merges.
    The union of the boxes is unchanged."""
    boxes = [dict(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for f in sorted({g for b in boxes for g in b}):
            groups = {}
            for b in boxes:
                key = frozenset((g, iv) for g, iv in b.items() if g != f and iv != _FULL)
                groups.setdefault(key, []).append(b)
            merged = []
            for grp in groups.values():
                grp.sort(key=lambda b: b.get(f, _FULL).inf)
                cur = grp[0]
                for b in grp[1:]:
                    u = sp.Union(cur.get(f, _FULL), b.get(f, _FULL))
                    if isinstance(u, sp.Interval):             # touching/overlapping
                        cur, changed = {**cur, f: u}, True
                    else:
                        merged.append(cur); cur = b
                merged.append(cur)
            boxes = merged
    return [{g: iv for g, iv in b.items() if iv != _FULL} for b in boxes]


def _box_expr(box):
    lits = []
    for f, iv in sorted(box.items()):
        x = sp.Symbol(f)
        if iv.inf != -sp.oo:
            lits.append(x > iv.inf if iv.left_open else x >= iv.inf)
        if iv.sup != sp.oo:
            lits.append(x < iv.sup if iv.right_open else x <= iv.sup)
    return sp.And(*lits), len(lits)


def simplify_rule(dnf):
    """Reduce a tree DNF with sympy interval algebra.  Returns the boxes, the
    sympy expression, and the reduced n_terms / n_literals."""
    boxes = _merge_boxes([_term_box(t) for t in dnf])
    parts = [_box_expr(b) for b in boxes]
    return dict(boxes=boxes,
                expr=sp.Or(*[e for e, _ in parts]) if parts else sp.false,
                n_terms=len(boxes), n_literals=sum(k for _, k in parts))


def rule_to_str(simple, max_terms=20, prec=3):
    """Readable form: one box per line, chained bounds like 0.938 < x <= 1.509."""
    def iv_str(f, iv):
        if iv.sup == sp.oo:                                    # lower bound only: x > lo
            return f"{f} {'>' if iv.left_open else '>='} {float(iv.inf):.{prec}f}"
        lo = "" if iv.inf == -sp.oo else f"{float(iv.inf):.{prec}f} {'<' if iv.left_open else '<='} "
        return f"{lo}{f} {'<' if iv.right_open else '<='} {float(iv.sup):.{prec}f}"
    terms = ["(" + " AND ".join(iv_str(f, iv) for f, iv in sorted(b.items())) + ")"
             if b else "(TRUE)" for b in simple["boxes"][:max_terms]]
    out = "\n  OR ".join(terms) if terms else "(FALSE)"
    if simple["n_terms"] > max_terms:
        out += f"\n  ... (+{simple['n_terms'] - max_terms} more terms)"
    return out


def rank_by_rule_length(results, by="terms"):
    """Attach each result's simplified rule (res["simple"]) and sort shortest
    first: terms, then literals, then TVD (by="literals" swaps the first two)."""
    for r in results:
        r["simple"] = simplify_rule(r["red"]["dnf"])
    if by == "terms":
        key = lambda r: (r["simple"]["n_terms"], r["simple"]["n_literals"], r["tvd"])
    else:
        key = lambda r: (r["simple"]["n_literals"], r["simple"]["n_terms"], r["tvd"])
    return sorted(results, key=key)


def _leaf_intervals(ix):
    """Contiguous [lo, hi] leaf-index interval per node (left-to-right DFS
    numbering).  Two nodes' subtrees are disjoint iff their intervals don't
    overlap -- an O(1) disjointness test for the box search."""
    cl, cr, is_leaf = ix["cl"], ix["cr"], ix["is_leaf"]
    leaf_index, order, st = {}, [0], [0]
    while st:                                   # pre-order DFS, left before right
        v = st.pop()
        if is_leaf[v]:
            leaf_index[v] = order[0]; order[0] += 1
        else:
            st.append(cr[v]); st.append(cl[v])
    lo, hi = {}, {}
    for v in sorted(range(ix["n"]), key=lambda v: -ix["depth"][v]):   # children first
        if is_leaf[v]:
            lo[v] = hi[v] = leaf_index[v]
        else:
            lo[v] = min(lo[cl[v]], lo[cr[v]]); hi[v] = max(hi[cl[v]], hi[cr[v]])
    return lo, hi


def find_best_boxes(ix, target, n_boxes, beam=16, excluded=()):
    """Query by NUMBER OF BOXES: find `n_boxes` disjoint subtrees (bounding boxes,
    one DNF conjunct each) whose union's bin distribution is CLOSEST (min TVD) to
    `target` proportions.  Beam search over disjoint node combinations.

    Unlike the cardinality query, there is no size target -- the objective is the
    distributional match of exactly n_boxes boxes in disjunction.

    `excluded`: (lo, hi) leaf-index intervals already used by earlier subsets;
    no chosen box may overlap them.
    """
    if "leaf_iv" not in ix:
        ix["leaf_iv"] = _leaf_intervals(ix)
    lo, hi = ix["leaf_iv"]
    nodes = free_nodes(ix, list(excluded),
                       [v for v in range(ix["n"]) if sum(ix["node_cnt"][v].values()) > 0])
    states = [(1.0, (), [], {})]                # (tvd, chosen, occupied intervals, got)
    for _ in range(n_boxes):
        cand = {}
        for _tvd, chosen, occ, got in states:
            for v in nodes:
                if v in chosen:
                    continue
                a, b = lo[v], hi[v]
                if any(not (b < c or d < a) for (c, d) in occ):   # overlaps a chosen box
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
    labels = {lf: 0 for lf in ix["leaves"]}
    for v in chosen:
        for lf in leaves_under(ix, v):
            labels[lf] = 1
    dnf = [ix["path_predicate"](v) for v in chosen]
    red = dict(dnf=dnf, n_terms=len(chosen), antichain=list(chosen),
               reduced_size=len(chosen), node_label=None)
    return dict(labels=labels, got=got, tc=dict(target), tvd=tvd, red=red,
                n_boxes=len(chosen), intervals=[(lo[v], hi[v]) for v in chosen])


def find_multiple_box_subsets(ix, target, n_boxes, tvd_threshold, beam=16):
    """Disjoint K-box subsets, run to exhaustion: the best `n_boxes` boxes, then
    the best among the leaves not yet used, and so on, until the next result's
    TVD exceeds `tvd_threshold` or fewer than `n_boxes` disjoint boxes remain.
    Returns the same dicts as find_best_boxes, plus "rows"."""
    excluded, results = [], []
    while True:
        res = find_best_boxes(ix, target, n_boxes, beam=beam, excluded=excluded)
        if res["n_boxes"] < n_boxes or res["tvd"] > tvd_threshold:
            break
        res["rows"] = selected_rows(ix, res["labels"])
        results.append(res)
        excluded += res["intervals"]
    return results


# ----------------------------------------------------------------------------
# Default dataset (swap in your own df + target_col at the bottom)
# ----------------------------------------------------------------------------
def default_dataset(N=1500, seed=42):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0, 1, size=(4, 3))
    assign = rng.integers(0, 4, size=N)
    coords = centers[assign] + rng.normal(0, 0.08, size=(N, 3))
    df = pd.DataFrame(coords, columns=["feature_a", "feature_b", "feature_c"])
    df["target"] = rng.random(N)
    cond = assign == 0
    df.loc[cond, "target"] = rng.normal(0.5, 0.15, size=int(cond.sum()))
    return df, "target"


# ----------------------------------------------------------------------------
# Loading user-selected CSVs, using datasets/metadata.py when it knows the file
# ----------------------------------------------------------------------------
def load_metadata():
    """Return the uci_datasets_info dict from datasets/metadata.py ({} if absent)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "datasets", "metadata.py")
    if not os.path.exists(path):
        return {}
    try:
        spec = importlib.util.spec_from_file_location("quaero_dataset_metadata", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "uci_datasets_info", {})
    except Exception:
        return {}


METADATA = load_metadata()


def read_dataset(path):
    """Read a CSV by path.  If datasets/metadata.py knows this file's stem, use
    its delimiter; otherwise let pandas sniff the separator.
    Returns (raw_df, meta, name)."""
    name = os.path.splitext(os.path.basename(path))[0]
    meta = METADATA.get(name, {})
    if meta.get("delim"):
        raw = pd.read_csv(path, sep=meta["delim"])
    else:
        raw = pd.read_csv(path, sep=None, engine="python")   # sniff , ; \t
    return raw, meta, name


def target_candidates(raw, meta):
    """Targets declared in the metadata, else every column that could be one."""
    cands = [c for c in meta.get("target_variable", []) if c in raw.columns]
    if cands:
        return cands
    ids = set(meta.get("id_variables", []))
    return [c for c in raw.columns if c not in ids]


def prepare_frame(raw, target_col, meta=None):
    """Turn a raw CSV frame into a modelling frame for the pipeline:
    drop id columns and the dataset's *other* declared targets (so they cannot
    leak in as descriptors), integer-encode any non-numeric column, and drop
    rows with missing/inf values."""
    meta = meta or {}
    drop = [c for c in meta.get("id_variables", []) if c in raw.columns]
    drop += [c for c in meta.get("target_variable", [])
             if c in raw.columns and c != target_col]
    out = raw.drop(columns=drop, errors="ignore").copy()
    for c in out.columns:                      # categoricals -> integer codes
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.factorize(out[c])[0].astype(float)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Query helpers shared by the UI and the notebook
# ----------------------------------------------------------------------------
def target_props(components, centers):
    """Target proportions per bin from mixture components [(name, params, w)]:
    the mixture pdf evaluated at each bin centre, normalised.  None if empty."""
    if not components:
        return None
    pdf = make_pdf([(n, p, w) for n, p, w in components])
    vals = np.array([max(pdf(c), 0.0) for c in centers], dtype=float)
    s = vals.sum()
    return (vals / s) if s > 0 else None


def feasibility(p, avail, N, pct):
    """Can a subset of pct% of N rows follow proportions p, given `avail` rows
    per bin?  Returns (feasible, max_feasible_pct, worst_bin, need, have)."""
    total = pct / 100.0 * N
    need = {b: total * p[b] for b in range(len(p))}
    bad = [(b, need[b], avail.get(b, 0)) for b in need if need[b] > avail.get(b, 0) + 1e-9]
    limits = [avail.get(b, 0) / p[b] for b in range(len(p)) if p[b] > 0]
    max_pct = 100.0 * min(limits) / N if limits else 0.0
    if bad:
        b, nd, av = max(bad, key=lambda t: t[1] - t[2])
        return False, max_pct, b, nd, av
    return True, max_pct, None, None, None
