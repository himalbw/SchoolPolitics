"""Project workflow figure: DistrictView -> pre-processing -> two branches.
Supervised: bag of words + hand-built dictionaries -> theme trends, higher criticism on phrase groups, LLM scoring.
Unsupervised: LDA topic model -> topic trends, clustering of districts, higher criticism on all words.

Writes output/figures/workflow.png and workflow.svg (white background). Run from the repo root:
  python3 source/analysis/workflow_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUTDIR_FIGURES = Path("output/figures")
# (fill, stroke, title text, body text) from a light-fill / dark-text ramp per branch
COLORS = {
    "gray": ("#F1EFE8", "#5F5E5A", "#2C2C2A", "#444441"),
    "teal": ("#E1F5EE", "#0F6E56", "#04342C", "#085041"),
    "purple": ("#EEEDFE", "#534AB7", "#26215C", "#3C3489"),
}
LABEL_X, X0 = 20, 140
COL_W, GAP, MID_GAP = 140, 14, 40                  # six final-analysis columns, wider gap between branches
COLS = [X0 + i * (COL_W + GAP) + (MID_GAP - GAP if i >= 3 else 0) for i in range(6)]
X1 = COLS[5] + COL_W
FIG_W, FIG_H = X1 + 20, 790

def Main():
    OUTDIR_FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(FIG_W / 72 * 1.4, FIG_H / 72 * 1.4), facecolor="white")
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(FIG_H, 0)
    ax.axis("off")
    ax.set_facecolor("white")
    for text, y in [("Dataset", 64), ("Pre-processing", 202), ("Representation", 386), ("Final analysis", 612)]:
        ax.text(LABEL_X, y, text, fontsize=10, color="#6B6B6B", va="center", ha="left")
    Box(ax, X0, 20, X1 - X0, 88, "gray", "DistrictView",
        ["Corpus of 80,063 school board meeting transcripts", "with YouTube-generated captions"])
    Box(ax, X0, 148, X1 - X0, 108, "gray", "Pre-processing and cleaning",
        ["Filter 2018-2023, deduplicate, keep school board meetings only,", "tokenize, drop stop words",
         "Append county-level partisanship (2020 vote share)"])
    lx, lw = COLS[0], COLS[2] + COL_W - COLS[0]
    rx, rw = COLS[3], X1 - COLS[3]
    xl, xr = lx + lw / 2, rx + rw / 2
    for x, key, name in [(xl, "teal", "Supervised"), (xr, "purple", "Unsupervised")]:
        ax.text(x + 10, 291, name, fontsize=10, fontweight="bold", color=COLORS[key][2], ha="left", va="center")
    Box(ax, lx, 326, lw, 120, "teal", "Bag of words",
        ["Term counts by district and period, reweighted by TF-IDF", "Hand-built dictionaries: 17 themes (4 controls,",
         "13 political) and 6 contested phrase groups"])
    Box(ax, rx, 326, rw, 120, "purple", "Topic model (LDA)", ["K = 10, 20, 30 topics", "fit on ~1,000-word chunks"])
    final = [("teal", "Theme trends", ["Dictionary mention", "rates over time,", "R vs D counties"]),
             ("teal", "Higher criticism", ["Did each phrase", "group's share", "change, 2018-19", "vs 2022-23?"]),
             ("teal", "LLM scoring", ["Positions and", "hostility rated", "0-100 by GABRIEL,", "mirroring themes"]),
             ("purple", "Topic trends", ["Topic prevalence", "over time,", "2018-2023"]),
             ("purple", "Higher criticism", ["Did any word", "shares change,", "2018-19 vs", "2022-23?"]),
             ("purple", "Clustering", ["Spectral clusters", "of districts on", "topic weights vs", "partisanship"])]
    for x, (key, title, body) in zip(COLS, final):
        Box(ax, x, 530, COL_W, 160, key, title, body)
    Arrow(ax, (X0 + X1) / 2, 108, (X0 + X1) / 2, 148)
    Arrow(ax, xl, 256, xl, 326)
    Arrow(ax, xr, 256, xr, 326)
    for x in COLS:
        Arrow(ax, x + COL_W / 2, 446, x + COL_W / 2, 530)
    for (x, name, key) in [(140, "Shared steps", "gray"), (330, "Supervised branch (bag of words)", "teal"),
                           (620, "Unsupervised branch (topic model)", "purple")]:
        fill, stroke, _, _ = COLORS[key]
        ax.add_patch(FancyBboxPatch((x, 736), 14, 14, boxstyle="round,pad=0,rounding_size=3", fc=fill, ec=stroke, lw=0.8))
        ax.text(x + 22, 743, name, fontsize=9, color="#6B6B6B", va="center")
    for ext in ("png", "svg"):
        fig.savefig(OUTDIR_FIGURES / f"workflow.{ext}", dpi=200, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    print(f"Saved {OUTDIR_FIGURES / 'workflow.png'}")
    return

def Box(ax, x, y, w, h, key, title, lines, dashed=False):
    fill, stroke, c_title, c_body = COLORS[key]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=8", fc=fill, ec=stroke, lw=0.9,
                                ls=(0, (4, 3)) if dashed else "-"))
    n_lines = len(lines)
    top = y + h / 2 - (18 * n_lines + 28) / 2 + 9   # centre title + body block vertically
    ax.text(x + w / 2, top, title, fontsize=11, fontweight="bold", color=c_title, ha="center", va="center")
    for i, line in enumerate(lines):
        ax.text(x + w / 2, top + 26 + 18 * i, line, fontsize=9.5, color=c_body, ha="center", va="center")
    return

def Arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0 + 2), (x1, y1 - 2), arrowstyle="-|>", mutation_scale=10, color="#6B6B6B", lw=1))
    return

if __name__ == "__main__":
    Main()
