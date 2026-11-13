"""quaero_ui.py — Tk UI to formulate distributional queries and visualise results.

Left panel: build a target distribution from mixture components, set a cardinality
percentage (with a live feasibility check), and run the query.
Right panel: live pdf preview, result histogram, and the induced DNF rule.

Backend: the decision-tree pipeline (index -> subtree-first selection -> lossless
reduction to a DNF rule) lives in quaero_core.py, shared with case_study_dt.ipynb.
"""
from datetime import datetime
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, scrolledtext, filedialog, messagebox


from quaero_core import (bin_target, _bin_centers, make_pdf, build_index,
                         find_multiple_subsets, find_multiple_box_subsets,
                         default_dataset, read_dataset, target_candidates,
                         prepare_frame, rule_to_str, rank_by_rule_length)
from plot_style import CB_palette          # palette only; the UI keeps Courier


# ----------------------------------------------------------------------------
# UI typography: Courier, sized so window captures stay legible in the paper.
# A capture of the WINDOW_PX-wide window is placed at \textwidth (DOC_WIDTH_PT);
# fonts are sized so text then prints at PRINT_PT (\footnotesize at 10pt).
#   printed size = font pt x (screen px per pt) x DOC_WIDTH_PT / window px
# The window opens at a fixed WINDOW_PX so this holds; resizing it changes it.
# ----------------------------------------------------------------------------
DOC_WIDTH_PT = 516.0          # \textwidth of the paper
PRINT_PT = 8                  # \footnotesize at 10pt: printed text size we want
WINDOW_PX = (1200, 960)       # fixed window size (only the width sets the font)
UI_FONT_FAMILIES = ["Courier", "Courier New", "Nimbus Mono PS", "Liberation Mono"]
MPL_DPI = 100                 # dpi of the embedded matplotlib figure


def ui_font_pt(root):
    """Tk font size (points) that prints at PRINT_PT once a WINDOW_PX-wide
    screenshot is scaled to DOC_WIDTH_PT."""
    px_per_pt = root.winfo_fpixels("1p")
    return max(1, round(PRINT_PT * WINDOW_PX[0] / (DOC_WIDTH_PT * px_per_pt)))


def mpl_font_pt(dpi):
    """Matplotlib font size for the embedded plots at the given figure dpi."""
    return PRINT_PT * WINDOW_PX[0] / (DOC_WIDTH_PT * dpi / 72)


_MPL_PT = mpl_font_pt(MPL_DPI)
matplotlib.rcParams.update({
    "font.family": "monospace",
    "font.monospace": UI_FONT_FAMILIES + ["DejaVu Sans Mono"],
    "font.size": _MPL_PT,
    "axes.titlesize": _MPL_PT,
    "axes.labelsize": _MPL_PT,
    "xtick.labelsize": _MPL_PT,
    "ytick.labelsize": _MPL_PT,
    "legend.fontsize": _MPL_PT,
})


def apply_ui_font(root):
    """Point every Tk named font at Courier, sized for legible captures.
    Returns (family, size_pt)."""
    available = set(tkfont.families(root))
    family = next((f for f in UI_FONT_FAMILIES if f in available), "Courier")
    size = ui_font_pt(root)
    for name in ("TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont",
                 "TkHeadingFont", "TkCaptionFont", "TkSmallCaptionFont",
                 "TkIconFont", "TkTooltipFont"):
        tkfont.nametofont(name, root=root).configure(family=family, size=size)
    root.option_add("*TCombobox*Listbox.font", (family, size))
    return family, size



# ----------------------------------------------------------------------------
# Distribution component specs (label -> default) per family
# ----------------------------------------------------------------------------
DIST_PARAMS = {
    "gaussian":    [("loc", "0.5"),   ("scale", "0.15")],
    "exponential": [("loc", "0.0"),   ("scale", "0.20")],
    "uniform":     [("loc", "0.0"),   ("scale", "1.00")],
    "beta":        [("a",   "2.0"),   ("b",     "2.0")],
    "weibull":     [("shape", "1.5"), ("scale", "0.30"), ("loc", "0.0")],
    "lognormal":   [("mu",  "-1.0"),  ("sigma", "0.5"),  ("loc", "0.0")],
    "pareto":      [("alpha", "3.0"), ("scale", "0.10"), ("loc", "0.0")],
}
MAX_PARAMS = max(len(v) for v in DIST_PARAMS.values())


class QueryUI:
    def __init__(self, root, df, target_col, name="synthetic (default)"):
        self.components = []                       # list of (name, params_dict, weight)
        self.font_family, self.ui_pt = apply_ui_font(root)
        root.geometry(f"{WINDOW_PX[0]}x{WINDOW_PX[1]}")
        root.title("quaero — distributional query builder")
        self._build(root)
        self._set_dataset(df, target_col, name)

    # ---- (re)point the UI at a dataset, then rebuild the index ----
    def _set_dataset(self, df, target_col, name):
        self.df, self.target_col = df, target_col
        self.N = len(df)
        self.bin_ids, self.n_bins = bin_target(df, target_col)
        self.centers = _bin_centers(df, target_col, self.n_bins)
        self.avail = {b: int((self.bin_ids == b).sum()) for b in range(self.n_bins)}
        self.tmin = float(df[target_col].min()); self.tmax = float(df[target_col].max())

        self.ds_lab.config(text=f"{name}\nN={self.N}, {df.shape[1] - 1} descriptors, "
                                f"{self.n_bins} bins")
        self.subsets = []                          # results belong to the old dataset
        self.subset_list.delete(0, "end")
        self.rule.delete("1.0", "end")
        self.ax_hist.clear()
        if hasattr(self, "ix"):
            del self.ix
        self.root.after(50, self._build_index_async)

    # ---- pick a CSV from the file system ----
    def _load_csv(self):
        path = filedialog.askopenfilename(
            title="Select a dataset CSV",
            filetypes=[("CSV / TSV", "*.csv *.tsv *.txt"), ("All files", "*.*")])
        if not path:
            return
        try:
            raw, meta, name = read_dataset(path)
        except Exception as e:
            messagebox.showerror("Could not read file", str(e)); return
        cands = target_candidates(raw, meta)
        if not cands:
            messagebox.showerror("No usable columns", "That file has no target column.")
            return
        self._raw, self._meta, self._name = raw, meta, name
        known = "metadata" if meta else "no metadata — sniffed"
        self.status.set(f"Loaded {name} ({known}); {len(raw)} rows, {raw.shape[1]} cols.")
        self.tgt_cb.config(values=cands)
        self.tgt_var.set(cands[0])
        self._apply_target()

    def _apply_target(self, *_):
        """Build the modelling frame for the currently selected target column."""
        if not hasattr(self, "_raw"):
            return
        tcol = self.tgt_var.get()
        try:
            frame = prepare_frame(self._raw, tcol, self._meta)
            if tcol not in frame.columns:
                raise ValueError(f"'{tcol}' did not survive cleaning (all-NaN?).")
            if frame.shape[1] < 2:
                raise ValueError("need at least one descriptor column besides the target.")
            if len(frame) < 10:
                raise ValueError(f"only {len(frame)} complete rows after dropping NaNs.")
            self._set_dataset(frame, tcol, self._name)
        except Exception as e:
            messagebox.showerror("Could not use this target", str(e))

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

        # ---- dataset picker (own CSV from the file system) ----
        ds = ttk.LabelFrame(left, text="Dataset", padding=6)
        ds.grid(row=0, column=0, sticky="we", pady=(0, 8))
        ds.columnconfigure(1, weight=1)
        ttk.Button(ds, text="Load CSV …", command=self._load_csv).grid(
            row=0, column=0, sticky="w")
        self.ds_lab = ttk.Label(ds, text="—", wraplength=280, foreground="#333")
        self.ds_lab.grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(ds, text="target column").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.tgt_var = tk.StringVar()
        self.tgt_cb = ttk.Combobox(ds, textvariable=self.tgt_var, state="readonly", width=20)
        self.tgt_cb.grid(row=1, column=1, sticky="we", pady=(4, 0))
        self.tgt_cb.bind("<<ComboboxSelected>>", self._apply_target)

        body = ttk.Frame(left); body.grid(row=1, column=0, sticky="nsew")

        # component builder
        ttk.Label(body, text="Add distribution component",
                  font=(self.font_family, self.ui_pt, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w")
        self.kind = tk.StringVar(value="gaussian")
        cb = ttk.Combobox(body, textvariable=self.kind, values=list(DIST_PARAMS),
                          state="readonly", width=14)
        cb.grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._sync_param_labels())

        # parameter fields (as many as the family needs) + weight, in one sub-frame
        # spanning the rows the old fixed two-parameter layout used
        pf = ttk.Frame(body)
        pf.grid(row=2, column=0, rowspan=3, columnspan=2, sticky="w")
        self.plabs, self.pents = [], []
        for j in range(MAX_PARAMS):
            lab = ttk.Label(pf, text="");  lab.grid(row=j, column=0, sticky="w")
            ent = ttk.Entry(pf, width=10); ent.grid(row=j, column=1, sticky="w")
            self.plabs.append(lab); self.pents.append(ent)
        ttk.Label(pf, text="weight").grid(row=MAX_PARAMS, column=0, sticky="w")
        self.wt = ttk.Entry(pf, width=10); self.wt.grid(row=MAX_PARAMS, column=1, sticky="w")
        self.wt.insert(0, "1.0")
        self._sync_param_labels()

        ttk.Button(body, text="Add component", command=self._add).grid(
            row=5, column=0, pady=4, sticky="w")
        ttk.Button(body, text="Clear all", command=self._clear).grid(
            row=5, column=1, pady=4, sticky="w")

        ttk.Label(body, text="Mixture components").grid(row=6, column=0, columnspan=2, sticky="w")
        self.comp_list = tk.Listbox(body, height=4, width=32)
        self.comp_list.grid(row=7, column=0, columnspan=2, sticky="we")

        # cardinality + feasibility
        ttk.Separator(body, orient="horizontal").grid(row=8, column=0, columnspan=2,
                                                       sticky="we", pady=6)
        ttk.Label(body, text="Cardinality (% of rows)").grid(row=9, column=0, columnspan=2, sticky="w")
        self.pct = tk.DoubleVar(value=30.0)
        sc = ttk.Scale(body, from_=1, to=100, variable=self.pct, command=lambda e: self._refresh())
        sc.grid(row=10, column=0, columnspan=2, sticky="we")
        self.pct_lab = ttk.Label(body, text="30%"); self.pct_lab.grid(row=11, column=0, sticky="w")
        self.feas = ttk.Label(body, text="", font=(self.font_family, self.ui_pt, "bold"))
        self.feas.grid(
            row=11, column=1, sticky="w")

        ttk.Label(body, text="Max TVD threshold").grid(row=12, column=0, sticky="w")
        self.tvd_thr = ttk.Entry(body, width=10); self.tvd_thr.grid(row=12, column=1, sticky="w")
        self.tvd_thr.insert(0, "0.10")

        ttk.Label(body, text="# bounding boxes").grid(row=13, column=0, sticky="w")
        self.n_boxes = tk.IntVar(value=2)
        ttk.Spinbox(body, from_=1, to=10, textvariable=self.n_boxes, width=8).grid(
            row=13, column=1, sticky="w")

        ttk.Button(body, text="Query by cardinality (subsets)", command=self._process).grid(
            row=14, column=0, columnspan=2, pady=(8, 2), sticky="we")
        ttk.Button(body, text="Query by # boxes (subsets)", command=self._process_boxes).grid(
            row=15, column=0, columnspan=2, pady=(0, 8), sticky="we")

        ttk.Label(body, text="Results (select to view)").grid(
            row=16, column=0, columnspan=2, sticky="w")
        self.subset_list = tk.Listbox(body, height=6, width=32)
        self.subset_list.grid(row=17, column=0, columnspan=2, sticky="we")
        self.subset_list.bind("<<ListboxSelect>>", self._on_subset_select)
        self.subsets = []

        self.status = tk.StringVar(value="starting …")
        ttk.Label(body, textvariable=self.status, foreground="#555",
                  wraplength=420).grid(row=18, column=0, columnspan=2, sticky="w")

        # right: figures (top) + rule text (bottom).  Grid with row weights so the
        # rule area claims more room than before, at the expense of the plots.
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=2)          # figures  (was the whole panel)
        right.rowconfigure(2, weight=3)          # rule     (now the larger share)

        self.fig = Figure(figsize=(6.5, 3.6), dpi=MPL_DPI)
        self.ax_pdf = self.fig.add_subplot(211)
        self.ax_hist = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=2.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # rule header: label on the left, a very small capture button on the right
        hdr = ttk.Frame(right); hdr.grid(row=1, column=0, sticky="we", pady=(4, 0))
        hdr.columnconfigure(0, weight=1)
        ttk.Label(hdr, text="Induced rule (DNF over descriptors)").grid(
            row=0, column=0, sticky="w")
        self.cap_btn = ttk.Button(hdr, text="⤓", width=2, command=self._capture)
        self.cap_btn.grid(row=0, column=1, sticky="e")

        self.rule = scrolledtext.ScrolledText(right, height=16, width=60, wrap="word")
        self.rule.grid(row=2, column=0, sticky="nsew")

    # ---- component helpers ----
    def _sync_param_labels(self):
        spec = DIST_PARAMS[self.kind.get()]
        for j, (lab, ent) in enumerate(zip(self.plabs, self.pents)):
            ent.delete(0, "end")
            if j < len(spec):
                lab.config(text=spec[j][0]); ent.insert(0, spec[j][1])
                lab.grid(); ent.grid()
            else:
                lab.grid_remove(); ent.grid_remove()

    def _add(self):
        kind = self.kind.get()
        try:
            params = {name: float(e.get())
                      for (name, _), e in zip(DIST_PARAMS[kind], self.pents)}
            w = float(self.wt.get())
            make_pdf((kind, params))                 # e.g. rejects sigma <= 0
        except ValueError as e:
            self.status.set(f"Invalid component: {e}"); return
        self.components.append((kind, params, w))
        ps = ", ".join(f"{k}={v}" for k, v in params.items())
        self.comp_list.insert("end", f"{kind}({ps})  w={w}")
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
            self.ax_pdf.plot(xs, ys, color=CB_palette["CBdarkblue"])
            self.ax_pdf.fill_between(xs, ys, color=CB_palette["CBblue"], alpha=0.2)
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
        self.subsets = rank_by_rule_length(find_multiple_subsets(self.ix, target, total, thr))

        self.subset_list.delete(0, "end")
        for i, s in enumerate(self.subsets):
            self.subset_list.insert(
                "end", f"#{i + 1}: {len(s['rows'])} rows, TVD={s['tvd']:.3f}, "
                       f"{s['simple']['n_terms']} terms / {s['simple']['n_literals']} lits")
        if self.subsets:
            self.subset_list.selection_clear(0, "end")
            self.subset_list.selection_set(0)
            self._show_subset(0)
            self.status.set(f"Found {len(self.subsets)} disjoint subset(s) under TVD {thr:.3f}, "
                            f"sorted by rule length.")
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
        try:
            thr = float(self.tvd_thr.get())
        except ValueError:
            self.status.set("Invalid TVD threshold."); return
        target = {b: float(p[b]) for b in range(self.n_bins)}
        k = int(self.n_boxes.get())
        self.status.set(f"Searching {k}-box subsets …"); self.root.update_idletasks()
        self.subsets = rank_by_rule_length(find_multiple_box_subsets(self.ix, target, k, thr))

        self.subset_list.delete(0, "end")
        for i, s in enumerate(self.subsets):
            self.subset_list.insert(
                "end", f"#{i + 1}: {k} box(es) → {s['simple']['n_terms']} terms / "
                       f"{s['simple']['n_literals']} lits: {len(s['rows'])} rows, "
                       f"TVD={s['tvd']:.3f}")
        if self.subsets:
            self.subset_list.selection_clear(0, "end"); self.subset_list.selection_set(0)
            self._show_subset(0)
            self.status.set(f"Found {len(self.subsets)} disjoint {k}-box subset(s) under "
                            f"TVD {thr:.3f}, sorted by rule length.")
        else:
            self.ax_hist.clear(); self.canvas.draw()
            self.rule.delete("1.0", "end")
            self.status.set(f"No {k}-box subset achievable under TVD {thr:.3f}.")

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
                          color=CB_palette["CBblue"], label="full dataset")
        if len(sel):
            self.ax_hist.hist(sel, bins=self.n_bins, density=True, alpha=0.75,
                              color=CB_palette["CBred"], label=f"subset #{i + 1} (n={len(sel)})")
        xs = np.linspace(self.tmin, self.tmax, 400)
        self.ax_hist.plot(xs, [pdf(x) for x in xs], "--", color=CB_palette["CBdarkgrey"],
                          lw=1.3, label="specified pdf")
        simple = s["simple"]
        self.ax_hist.set_title(f"Subset #{i + 1} — TVD={s['tvd']:.3f}, "
                               f"{simple['n_terms']} terms / {simple['n_literals']} lits")
        self.ax_hist.set_xlabel(self.target_col); self.ax_hist.set_yticks([])
        self.ax_hist.legend()
        self.fig.tight_layout(pad=2.0); self.canvas.draw()

        self.rule.delete("1.0", "end")
        self.rule.insert("end",
            f"subset #{i + 1}   rows: {len(sel)} / {self.N}   TVD: {s['tvd']:.4f}   "
            f"terms: {simple['n_terms']} (tree: {red['n_terms']})   "
            f"literals: {simple['n_literals']}\n\n")
        self.rule.insert("end", rule_to_str(simple, max_terms=20))

    # ---- capture the WHOLE Tk window to a high-quality PDF ----
    def _capture(self, *_):
        self.root.update_idletasks()
        fname = f"quaero_capture_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        try:
            from PIL import ImageGrab
        except ImportError:
            self.status.set("Capture needs Pillow — `pip install pillow`."); return
        x, y = self.root.winfo_rootx(), self.root.winfo_rooty()
        w, h = self.root.winfo_width(), self.root.winfo_height()
        try:
            img = ImageGrab.grab(bbox=(x, y, x + w, y + h)).convert("RGB")
            # embed at the screen's pixel density so the PDF stays crisp
            img.save(fname, "PDF", resolution=float(self.root.winfo_fpixels("1i")))
            printed = self.ui_pt * self.root.winfo_fpixels("1p") * DOC_WIDTH_PT / w
            self.status.set(f"Saved full-window PDF → {fname}  (text prints at "
                            f"{printed:.1f}pt at {DOC_WIDTH_PT:.0f}pt width)")
        except Exception as e:
            self.status.set(f"Capture failed: {e}")


if __name__ == "__main__":
    df, target_col = default_dataset()      # swap in your own DataFrame + target column here
    root = tk.Tk()
    QueryUI(root, df, target_col)
    root.mainloop()
