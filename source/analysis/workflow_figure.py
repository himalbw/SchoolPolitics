"""Project workflow figure: DistrictView -> pre-processing -> bag of words / LDA -> filtering -> final analysis.

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
LABEL_X, X0, X1 = 20, 130, 640
LEFT = (130, 280)    # x, width of the bag-of-words column
RIGHT = (430, 210)   # x, width of the LDA column
FIG_W, FIG_H = 680, 800

def Main():
    OUTDIR_FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(FIG_W / 72 * 1.4, FIG_H / 72 * 1.4), facecolor="white")
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(FIG_H, 0)
    ax.axis("off")
    ax.set_facecolor("white")
    for text, y in [("Dataset", 64), ("Pre-processing", 202), ("Representation", 346), ("Filtering", 516), ("Final analysis", 692)]:
        ax.text(LABEL_X, y, text, fontsize=10, color="#6B6B6B", va="center", ha="left")
    Box(ax, X0, 20, X1 - X0, 88, "gray", "DistrictView",
        ["Corpus of 80,063 school board meeting transcripts", "with YouTube-generated captions"])
    Box(ax, X0, 148, X1 - X0, 108, "gray", "Pre-processing and cleaning",
        ["Filter 2018-2023, deduplicate, keep school board meetings only,", "tokenize, drop stop words",
         "Append county-level partisanship (2024 vote share)"])
    Box(ax, *LEFT[:1], 296, LEFT[1], 96, "teal", "Bag of words", ["Generate 17 word clusters from", "TF-IDF representation"])
    Box(ax, RIGHT[0], 296, RIGHT[1], 96, "purple", "Topic model (LDA)", ["K = 10, 20, 30 topics", "fit on ~1,000-word chunks"])
    Box(ax, LEFT[0], 432, LEFT[1], 150, "teal", "Conflict clusters",
        ["Hand-picked from BoW (LLM-assisted):", "meeting conflict, legal conflict, plus", "11 culture-war topics: parental rights,",
         "books, CRT/race, gender/LGBTQ, choice,", "COVID, equity/SEL, guns, unions, more"], dashed=True)
    Box(ax, LEFT[0], 622, LEFT[1], 104, "teal", "LLM scoring", ["Selected attributes,", "rated 0-100 by GABRIEL"])
    Box(ax, RIGHT[0], 622, RIGHT[1], 104, "purple", "Topic prevalence tests",
        ["Higher criticism: did", "vocabulary change,", "2018-19 vs 2022-23?"])
    xl, xr = LEFT[0] + LEFT[1] / 2, RIGHT[0] + RIGHT[1] / 2
    Arrow(ax, (X0 + X1) / 2, 108, (X0 + X1) / 2, 148)
    Arrow(ax, xl, 256, xl, 296)
    Arrow(ax, xr, 256, xr, 296)
    Arrow(ax, xl, 392, xl, 432)
    Arrow(ax, xl, 582, xl, 622)
    Arrow(ax, xr, 392, xr, 622)
    for (x, name, key) in [(130, "Shared steps", "gray"), (260, "Bag-of-words branch", "teal"), (430, "Topic-model branch", "purple")]:
        fill, stroke, _, _ = COLORS[key]
        ax.add_patch(FancyBboxPatch((x, 758), 14, 14, boxstyle="round,pad=0,rounding_size=3", fc=fill, ec=stroke, lw=0.8))
        ax.text(x + 22, 765, name, fontsize=9, color="#6B6B6B", va="center")
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
