"""Project workflow figure: DistrictView -> pre-processing -> supervised (bag of words -> HC, dictionary -> LLM) / unsupervised (LDA -> trends, clustering).

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
LABEL_X, X0, X1 = 20, 130, 790
LEFT = (130, 400)    # x, width of the supervised (bag-of-words) column
RIGHT = (560, 230)   # x, width of the unsupervised (topic-model) column
HC = (130, 185)      # x, width of the higher-criticism box
LLM = (345, 185)     # x, width of the LLM-scoring box (sits under the dictionary)
FIG_W, FIG_H = 810, 860

def Main():
    OUTDIR_FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(FIG_W / 72 * 1.4, FIG_H / 72 * 1.4), facecolor="white")
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(FIG_H, 0)
    ax.axis("off")
    ax.set_facecolor("white")
    for text, y in [("Dataset", 64), ("Pre-processing", 202), ("Representation", 376), ("Filtering", 536), ("Final analysis", 712)]:
        ax.text(LABEL_X, y, text, fontsize=10, color="#6B6B6B", va="center", ha="left")
    Box(ax, X0, 20, X1 - X0, 88, "gray", "DistrictView",
        ["Corpus of 80,063 school board meeting transcripts", "with YouTube-generated captions"])
    Box(ax, X0, 148, X1 - X0, 108, "gray", "Pre-processing and cleaning",
        ["Filter 2018-2023, deduplicate, keep school board meetings only,", "tokenize, drop stop words",
         "Append county-level partisanship (2020 vote share)"])
    ax.text(LEFT[0] + LEFT[1] / 2 + 10, 291, "Supervised", fontsize=10, fontweight="bold", color=COLORS["teal"][2], ha="left", va="center")
    ax.text(RIGHT[0] + RIGHT[1] / 2 + 10, 291, "Unsupervised", fontsize=10, fontweight="bold", color=COLORS["purple"][2], ha="left", va="center")
    Box(ax, LEFT[0], 326, LEFT[1], 100, "teal", "Bag of words", ["Term counts by district and period,", "reweighted by TF-IDF"])
    Box(ax, RIGHT[0], 326, RIGHT[1], 100, "purple", "Topic model (LDA)", ["K = 10, 20, 30 topics", "fit on ~1,000-word chunks"])
    Box(ax, LLM[0], 466, LLM[1], 140, "teal", "Dictionary of 17 themes",
        ["Hand-built word and", "phrase lists: 4 controls,", "13 political themes", "(e.g. CRT, books, masks)"], dashed=True)
    Box(ax, HC[0], 646, HC[1], 140, "teal", "Higher criticism",
        ["Did word shares change,", "2018-19 vs 2022-23?", "Follow-up on contested", "phrase dictionaries"])
    Box(ax, LLM[0], 646, LLM[1], 140, "teal", "LLM scoring", ["Attributes mirroring", "the themes, rated", "0-100 by GABRIEL"])
    Box(ax, RIGHT[0], 646, RIGHT[1], 140, "purple", "Topic trends and clustering",
        ["Topic prevalence over time", "Spectral clustering of", "districts on topic weights,", "compared with partisanship"])
    xl, xr = LEFT[0] + LEFT[1] / 2, RIGHT[0] + RIGHT[1] / 2
    xhc, xllm = HC[0] + HC[1] / 2, LLM[0] + LLM[1] / 2
    Arrow(ax, (X0 + X1) / 2, 108, (X0 + X1) / 2, 148)
    Arrow(ax, xl, 256, xl, 326)
    Arrow(ax, xr, 256, xr, 326)
    Arrow(ax, xhc, 426, xhc, 646)
    Arrow(ax, xllm, 426, xllm, 466)
    Arrow(ax, xllm, 606, xllm, 646)
    Arrow(ax, xr, 426, xr, 646)
    for (x, name, key) in [(130, "Shared steps", "gray"), (290, "Supervised branch (bag of words)", "teal"),
                           (560, "Unsupervised branch (topic model)", "purple")]:
        fill, stroke, _, _ = COLORS[key]
        ax.add_patch(FancyBboxPatch((x, 818), 14, 14, boxstyle="round,pad=0,rounding_size=3", fc=fill, ec=stroke, lw=0.8))
        ax.text(x + 22, 825, name, fontsize=9, color="#6B6B6B", va="center")
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
