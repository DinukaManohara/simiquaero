# quaero — distributional subset queries

Find subsets of a tabular dataset whose target-variable distribution matches a
distribution you specify, and describe each subset as a readable rule
(a DNF over the descriptor columns), using a decision-tree index.

## Setup

```bash
pip install numpy pandas scipy scikit-learn matplotlib sympy pillow jinja2
```

The UI also needs Tk (`sudo apt install python3-tk` on Debian/Ubuntu).
Put your CSVs in `datasets/`. If a file's name (without `.csv`) has an entry
in `datasets/metadata.py`, its delimiter, target(s), id columns and
categorical columns are taken from there; otherwise the separator is sniffed.

## Case study notebook (`case_study_dt.ipynb`)

Run it top to bottom from the repository root. Edit the cells marked ✏️:

1. **Dataset cell** — `DATASET_PATH` (e.g. `datasets/factory.csv`), `TARGET`
   (`None` = first declared target), `MAX_ROWS` (optional subsample),
   `SAVE_PDF` / `FIG_DIR` / `TITLE` (figure saving).
2. **Query cell** — `COMPONENTS`: the target distribution as a mixture of
   `(family, params, weight)`; families: `gaussian`, `exponential`, `uniform`,
   `beta`, `weibull`, `lognormal`, `pareto` (parameters are listed in the
   notebook). Also `CARDINALITY_PCT`, `TVD_THRESHOLD`, `N_BOXES`, `RANK_BY`.

The notebook then preprocesses the data, builds the index, plots the query,
and runs both queries:

- **Cardinality query** — disjoint subsets of `CARDINALITY_PCT`% of the rows,
  each within `TVD_THRESHOLD` of the query.
- **Term-restricted query** — disjoint subsets described by exactly `N_BOXES`
  boxes (rule terms).

Results are simplified (sympy) and ranked by rule length (terms → literals →
TVD), with TVD / JSD / KLD / Hellinger / BC. The last sections save
full-width figures (best cardinality subset + 1–3 term subsets, with their
rules in a `.txt`) and a Weibull vs log-normal comparison to `FIG_DIR`.

## UI (`quaero_ui.py`)

```bash
python quaero_ui.py
```

1. **Load CSV …** and pick the **target column** (a synthetic dataset is
   loaded at start).
2. **Add distribution component(s)** — choose a family, set its parameters
   and weight, **Add component**; the specified pdf is plotted live.
3. Set **Cardinality (% of rows)** (a feasibility check is shown),
   **Max TVD threshold** and **# bounding boxes**.
4. **Query by cardinality (subsets)** or **Query by # boxes (subsets)**.
5. Select a result in the list to see its histogram against the full data
   and the query, and its simplified rule below.
6. The small **⤓** button saves the whole window as a PDF
   (`quaero_capture_<time>.pdf`); fonts are sized to stay legible when the
   capture is placed at full text width in a paper.

Large datasets make the index build slow (the window freezes meanwhile).
