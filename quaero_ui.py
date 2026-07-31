"""quaero_ui.py — Tk UI to formulate distributional queries and visualise results.

Left panel: build a target distribution from mixture components, set a cardinality
percentage (with a live feasibility check), and run the query.
Right panel: live pdf preview, result histogram, and the induced DNF rule.

Backend: the quaero_tree pipeline (decision-tree index -> subtree-first selection
-> lossless reduction to a DNF rule), embedded below so the app is standalone.
"""
import math
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.tree import DecisionTreeClassifier

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import ttk, scrolledtext


# ----------------------------------------------------------------------------
# Pipeline (embedded from quaero_tree)
# ----------------------------------------------------------------------------
def bin_target(df, target_col):
    x = df[target_col].dropna().values
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


def make_pdf(spec):
    DIST = {"gaussian": scipy_stats.norm, "normal": scipy_stats.norm,
            "exponential": scipy_stats.expon, "uniform": scipy_stats.uniform,
            "beta": scipy_stats.beta}
    if isinstance(spec, tuple):
        name, p = spec; d = DIST[name.lower()]
        return lambda x: d.pdf(x, **p)
    tot = sum(w for *_, w in spec) or 1.0
    comp = [(DIST[n.lower()], p, w / tot) for n, p, w in spec]
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


def find_multiple_subsets(ix, target, total, tvd_threshold, max_subsets=50):
    """Extract disjoint subsets that each match `target` at cardinality `total`.

    Find the first subset (+ its rule), exclude its leaves, find the next from the
    remaining leaves, and so on -- stopping once the best achievable subset exceeds
    `tvd_threshold` (or nothing more can be selected).  Returns a list of dicts with
    labels, got, tc, tvd, red (rule), and rows.
    """
    excluded, results = set(), []
    while len(results) < max_subsets:
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


def find_best_boxes(ix, target, n_boxes, beam=16):
    """Query by NUMBER OF BOXES: find `n_boxes` disjoint subtrees (bounding boxes,
    one DNF conjunct each) whose union's bin distribution is CLOSEST (min TVD) to
    `target` proportions.  Beam search over disjoint node combinations.

    Unlike the cardinality query, there is no size target -- the objective is the
    distributional match of exactly n_boxes boxes in disjunction.
    """
    lo, hi = _leaf_intervals(ix)
    nodes = [v for v in range(ix["n"]) if sum(ix["node_cnt"][v].values()) > 0]
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
                n_boxes=len(chosen))


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
# Distribution component specs (label -> default) per family
# ----------------------------------------------------------------------------
DIST_PARAMS = {
    "gaussian":    [("loc", "0.5"),  ("scale", "0.15")],
    "exponential": [("loc", "0.0"),  ("scale", "0.20")],
    "uniform":     [("loc", "0.0"),  ("scale", "1.00")],
    "beta":        [("a",   "2.0"),  ("b",     "2.0")],
}


class QueryUI:
    def __init__(self, root, df, target_col):
        self.df, self.target_col = df, target_col
        self.N = len(df)
        self.bin_ids, self.n_bins = bin_target(df, target_col)
        self.centers = _bin_centers(df, target_col, self.n_bins)
        self.avail = {b: int((self.bin_ids == b).sum()) for b in range(self.n_bins)}
        self.tmin = float(df[target_col].min()); self.tmax = float(df[target_col].max())
        self.components = []                       # list of (name, params_dict, weight)

        root.title("quaero — distributional query builder")
        self._build(root)
        root.after(50, self._build_index_async)   # build index just after UI shows

    # ---- index (built once) ----
    def _build_index_async(self):
        self.status.set("Building tree index …")
        self.root.update_idletasks()
        self.ix = build_index(self.df, self.target_col)
        self.status.set(f"Index ready: {len(self.ix['leaves'])} leaves, "
                        f"{self.ix['n_bins']} bins, N={self.N}")
        self._refresh()

    # ---- layout ----
    def _build(self, root):
        self.root = root
        left = ttk.Frame(root, padding=8);  left.grid(row=0, column=0, sticky="ns")
        right = ttk.Frame(root, padding=8); right.grid(row=0, column=1, sticky="nsew")
        root.columnconfigure(1, weight=1); root.rowconfigure(0, weight=1)

        # component builder
        ttk.Label(left, text="Add distribution component", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w")
        self.kind = tk.StringVar(value="gaussian")
        cb = ttk.Combobox(left, textvariable=self.kind, values=list(DIST_PARAMS),
                          state="readonly", width=14)
        cb.grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._sync_param_labels())

        self.p1lab = ttk.Label(left, text="loc"); self.p1lab.grid(row=2, column=0, sticky="w")
        self.p1 = ttk.Entry(left, width=10); self.p1.grid(row=2, column=1, sticky="w")
        self.p2lab = ttk.Label(left, text="scale"); self.p2lab.grid(row=3, column=0, sticky="w")
        self.p2 = ttk.Entry(left, width=10); self.p2.grid(row=3, column=1, sticky="w")
        ttk.Label(left, text="weight").grid(row=4, column=0, sticky="w")
        self.wt = ttk.Entry(left, width=10); self.wt.grid(row=4, column=1, sticky="w")
        self.wt.insert(0, "1.0")
        self._sync_param_labels()

        ttk.Button(left, text="Add component", command=self._add).grid(
            row=5, column=0, pady=4, sticky="w")
        ttk.Button(left, text="Clear all", command=self._clear).grid(
            row=5, column=1, pady=4, sticky="w")

        ttk.Label(left, text="Mixture components").grid(row=6, column=0, columnspan=2, sticky="w")
        self.comp_list = tk.Listbox(left, height=6, width=32)
        self.comp_list.grid(row=7, column=0, columnspan=2, sticky="we")

        # cardinality + feasibility
        ttk.Separator(left, orient="horizontal").grid(row=8, column=0, columnspan=2,
                                                       sticky="we", pady=6)
        ttk.Label(left, text="Cardinality (% of rows)").grid(row=9, column=0, columnspan=2, sticky="w")
        self.pct = tk.DoubleVar(value=30.0)
        sc = ttk.Scale(left, from_=1, to=100, variable=self.pct, command=lambda e: self._refresh())
        sc.grid(row=10, column=0, columnspan=2, sticky="we")
        self.pct_lab = ttk.Label(left, text="30%"); self.pct_lab.grid(row=11, column=0, sticky="w")
        self.feas = ttk.Label(left, text="", font=("", 9, "bold")); self.feas.grid(
            row=11, column=1, sticky="w")

        ttk.Label(left, text="Max TVD threshold").grid(row=12, column=0, sticky="w")
        self.tvd_thr = ttk.Entry(left, width=10); self.tvd_thr.grid(row=12, column=1, sticky="w")
        self.tvd_thr.insert(0, "0.10")

        ttk.Label(left, text="# bounding boxes").grid(row=13, column=0, sticky="w")
        self.n_boxes = tk.IntVar(value=2)
        ttk.Spinbox(left, from_=1, to=10, textvariable=self.n_boxes, width=8).grid(
            row=13, column=1, sticky="w")

        ttk.Button(left, text="Query by cardinality (subsets)", command=self._process).grid(
            row=14, column=0, columnspan=2, pady=(8, 2), sticky="we")
        ttk.Button(left, text="Query by # boxes (best match)", command=self._process_boxes).grid(
            row=15, column=0, columnspan=2, pady=(0, 8), sticky="we")

        ttk.Label(left, text="Results (select to view)").grid(
            row=16, column=0, columnspan=2, sticky="w")
        self.subset_list = tk.Listbox(left, height=6, width=32)
        self.subset_list.grid(row=17, column=0, columnspan=2, sticky="we")
        self.subset_list.bind("<<ListboxSelect>>", self._on_subset_select)
        self.subsets = []

        self.status = tk.StringVar(value="starting …")
        ttk.Label(left, textvariable=self.status, foreground="#555",
                  wraplength=240).grid(row=18, column=0, columnspan=2, sticky="w")

        # right: figures + rule text
        self.fig = Figure(figsize=(6.5, 5.2), dpi=100)
        self.ax_pdf = self.fig.add_subplot(211)
        self.ax_hist = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        ttk.Label(right, text="Induced rule (DNF over descriptors)").pack(anchor="w")
        self.rule = scrolledtext.ScrolledText(right, height=8, width=80, wrap="word")
        self.rule.pack(fill="both", expand=False)

    # ---- component helpers ----
    def _sync_param_labels(self):
        (l1, d1), (l2, d2) = DIST_PARAMS[self.kind.get()]
        self.p1lab.config(text=l1); self.p2lab.config(text=l2)
        for e, d in ((self.p1, d1), (self.p2, d2)):
            e.delete(0, "end"); e.insert(0, d)

    def _add(self):
        try:
            (l1, _), (l2, _) = DIST_PARAMS[self.kind.get()]
            params = {l1: float(self.p1.get()), l2: float(self.p2.get())}
            w = float(self.wt.get())
        except ValueError:
            self.status.set("Invalid number in a parameter/weight field."); return
        self.components.append((self.kind.get(), params, w))
        ps = ", ".join(f"{k}={v}" for k, v in params.items())
        self.comp_list.insert("end", f"{self.kind.get()}({ps})  w={w}")
        self._refresh()

    def _clear(self):
        self.components.clear(); self.comp_list.delete(0, "end")
        self.rule.delete("1.0", "end"); self.ax_hist.clear(); self.canvas.draw()
        self._refresh()

    # ---- target distribution over bins from the current mixture ----
    def _target_props(self):
        if not self.components:
            return None
        pdf = make_pdf([(n, p, w) for n, p, w in self.components])
        vals = np.array([max(pdf(c), 0.0) for c in self.centers], dtype=float)
        s = vals.sum()
        return (vals / s) if s > 0 else None

    # ---- redraw pdf preview + feasibility ----
    def _refresh(self, *_):
        self.pct_lab.config(text=f"{self.pct.get():.0f}%")
        # pdf preview
        self.ax_pdf.clear()
        if self.components:
            pdf = make_pdf([(n, p, w) for n, p, w in self.components])
            xs = np.linspace(self.tmin, self.tmax, 400)
            ys = [pdf(x) for x in xs]
            self.ax_pdf.plot(xs, ys, color="navy")
            self.ax_pdf.fill_between(xs, ys, color="navy", alpha=0.15)
            self.ax_pdf.set_title("Specified target pdf")
        else:
            self.ax_pdf.set_title("Specified target pdf — add a component")
        self.ax_pdf.set_xlabel(self.target_col); self.ax_pdf.set_yticks([])
        self.fig.tight_layout(pad=2.0); self.canvas.draw()
        self._feasibility()

    def _feasibility(self):
        p = self._target_props()
        if p is None:
            self.feas.config(text="—", foreground="#555"); return
        total = self.pct.get() / 100.0 * self.N
        need = {b: total * p[b] for b in range(self.n_bins)}
        bad = [(b, need[b], self.avail[b]) for b in range(self.n_bins)
               if need[b] > self.avail[b] + 1e-9]
        limits = [self.avail[b] / p[b] for b in range(self.n_bins) if p[b] > 0]
        max_pct = 100.0 * min(limits) / self.N if limits else 0.0
        if bad:
            b, nd, av = max(bad, key=lambda t: t[1] - t[2])
            self.feas.config(text=f"INFEASIBLE (bin {b}: need {nd:.0f} > {av})   "
                                  f"max ≈ {max_pct:.0f}%", foreground="#c62828")
        else:
            self.feas.config(text=f"feasible   (max ≈ {max_pct:.0f}%)", foreground="#2e7d32")

    # ---- run the query: find MULTIPLE disjoint subsets ----
    def _process(self):
        if not hasattr(self, "ix"):
            self.status.set("Index still building — try again in a moment."); return
        p = self._target_props()
        if p is None:
            self.status.set("Add at least one distribution component first."); return
        try:
            thr = float(self.tvd_thr.get())
        except ValueError:
            self.status.set("Invalid TVD threshold."); return
        target = {b: float(p[b]) for b in range(self.n_bins)}
        total = round(self.pct.get() / 100.0 * self.N)

        self.status.set("Searching for subsets …"); self.root.update_idletasks()
        self.subsets = find_multiple_subsets(self.ix, target, total, thr)

        self.subset_list.delete(0, "end")
        for i, s in enumerate(self.subsets):
            self.subset_list.insert(
                "end", f"#{i + 1}: {len(s['rows'])} rows, TVD={s['tvd']:.3f}, "
                       f"{s['red']['n_terms']} terms")
        if self.subsets:
            self.subset_list.selection_clear(0, "end")
            self.subset_list.selection_set(0)
            self._show_subset(0)
            self.status.set(f"Found {len(self.subsets)} disjoint subset(s) under TVD {thr:.3f}.")
        else:
            self.ax_hist.clear(); self.canvas.draw()
            self.rule.delete("1.0", "end")
            self.status.set(f"No subset achievable under TVD {thr:.3f}.")

    # ---- run the query: best N bounding boxes (by count, min-TVD match) ----
    def _process_boxes(self):
        if not hasattr(self, "ix"):
            self.status.set("Index still building — try again in a moment."); return
        p = self._target_props()
        if p is None:
            self.status.set("Add at least one distribution component first."); return
        target = {b: float(p[b]) for b in range(self.n_bins)}
        k = int(self.n_boxes.get())
        self.status.set(f"Searching best {k} box(es) …"); self.root.update_idletasks()
        res = find_best_boxes(self.ix, target, k)
        res["rows"] = selected_rows(self.ix, res["labels"])
        self.subsets = [res]
        self.subset_list.delete(0, "end")
        self.subset_list.insert(
            "end", f"{res['n_boxes']} box(es): {len(res['rows'])} rows, TVD={res['tvd']:.3f}")
        self.subset_list.selection_clear(0, "end"); self.subset_list.selection_set(0)
        self._show_subset(0)
        self.status.set(f"Best {res['n_boxes']} box(es): TVD={res['tvd']:.3f}.")

    def _on_subset_select(self, event):
        sel = self.subset_list.curselection()
        if sel:
            self._show_subset(sel[0])

    def _show_subset(self, i):
        """Draw the histogram and rule for the i-th found subset."""
        s = self.subsets[i]
        got, tc, red = s["got"], s["tc"], s["red"]
        sel = self.df[self.target_col].to_numpy()[s["rows"]]
        full = self.df[self.target_col].to_numpy()
        pdf = make_pdf([(n, pp, w) for n, pp, w in self.components])

        self.ax_hist.clear()
        self.ax_hist.hist(full, bins=self.n_bins, density=True, alpha=0.35,
                          color="steelblue", label="full dataset")
        if len(sel):
            self.ax_hist.hist(sel, bins=self.n_bins, density=True, alpha=0.75,
                              color="tomato", label=f"subset #{i + 1} (n={len(sel)})")
        xs = np.linspace(self.tmin, self.tmax, 400)
        self.ax_hist.plot(xs, [pdf(x) for x in xs], "k--", lw=1.3, label="specified pdf")
        self.ax_hist.set_title(f"Subset #{i + 1} — TVD={s['tvd']:.3f}, "
                               f"{red['n_terms']} rule terms")
        self.ax_hist.set_xlabel(self.target_col); self.ax_hist.set_yticks([])
        self.ax_hist.legend(fontsize=7)
        self.fig.tight_layout(pad=2.0); self.canvas.draw()

        self.rule.delete("1.0", "end")
        self.rule.insert("end",
            f"subset #{i + 1}   rows: {len(sel)} / {self.N}   TVD: {s['tvd']:.4f}   "
            f"DNF terms: {red['n_terms']}   reduced size: {red['reduced_size']}\n\n")
        self.rule.insert("end", dnf_to_str(red["dnf"], max_terms=12))


if __name__ == "__main__":
    df, target_col = default_dataset()      # swap in your own DataFrame + target column here
    root = tk.Tk()
    QueryUI(root, df, target_col)
    root.mainloop()
