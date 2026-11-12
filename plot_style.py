"""plot_style.py — the paper's plotting design language, shared by every notebook.

Edit values here; bench_plots.ipynb and case_study_dt.ipynb import them.
quaero_ui.py imports only the palette (the UI keeps its own Courier font).
Importing this module changes nothing; call apply_style() to set the fonts.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "fonts", "LinLibertineTTF", "LinLibertine_Rah.ttf")

inches_per_pt = 1.0 / 72.27
ASPECT = 0.77
fig_width = 252.0 * inches_per_pt          # \columnwidth: single plot
fig_height = fig_width * ASPECT
text_width = 516.0 * inches_per_pt         # \textwidth: full-width figures

CB_palette = {
    # Base
    'CBblue': (68/255, 119/255, 170/255),
    'CBcyan': (102/255, 204/255, 238/255),
    'CBgreen': (34/255, 136/255, 51/255),
    'CByellow': (204/255, 187/255, 68/255),
    'CBred': (238/255, 102/255, 119/255),
    'CBpurple': (170/255, 51/255, 119/255),
    'CBgrey': (187/255, 187/255, 187/255),
    'CBorange': (213/255, 94/255, 0/255),
    'CBsand': (221/255, 204/255, 119/255),
    'CBteal': (68/255, 170/255, 153/255),
    'CBolive': (170/255, 170/255, 0/255),
    # Pale
    'CBpaleblue': (187/255, 204/255, 238/255),
    'CBpalecyan': (204/255, 238/255, 255/255),
    'CBpalegreen': (204/255, 221/255, 170/255),
    'CBpaleyellow': (238/255, 238/255, 187/255),
    'CBpalered': (255/255, 204/255, 204/255),
    'CBpalegrey': (221/255, 221/255, 221/255),
    # Dark
    'CBdarkblue': (34/255, 34/255, 85/255),
    'CBdarkcyan': (34/255, 85/255, 85/255),
    'CBdarkgreen': (34/255, 85/255, 34/255),
    'CBdarkyellow': (102/255, 102/255, 51/255),
    'CBdarkred': (102/255, 51/255, 51/255),
    'CBdarkgrey': (85/255, 85/255, 85/255),
    'CBdarkindigo': (51/255, 34/255, 102/255),
}
cb_color_list = list(CB_palette.values())


def apply_style():
    """Load Linux Libertine (falls back to the default font if the TTF is
    absent) and set the design-language rcParams.  Returns the font name."""
    if os.path.exists(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        font = fm.FontProperties(fname=FONT_PATH).get_name()
    else:
        font = plt.rcParams['font.family'][0]
        print(f"[warn] font not found at {FONT_PATH}; using '{font}'")
    plt.rcParams.update({
        'font.family': font,
        'font.size': 12,
        'axes.titlesize': 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 3,
        'lines.markersize': 5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    return font
