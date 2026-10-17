"""Figures for the bag-of-words analysis, from the tables written by gst_polarization.py and partisan_lexicon.py.

Writes output/figures/gst_polarization.png, lexicon_category_gap.png, lexicon_category_trend.png. Run from the repo root.
Colour: party is a fixed meaning (R red, D blue); everything else is grey/neutral. Text stays in ink colours.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bow_data import OUTDIR_FIGURES, OUTDIR_TABLES

RED, BLUE, INK, MUTED, GRID = "#C0392B", "#2C6FAE", "#222222", "#6B6B6B", "#E4E4E4"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})

def Main():
    OUTDIR_FIGURES.mkdir(parents=True, exist_ok=True)
    PlotGst()
    PlotCategoryGap()
    PlotCategoryTrend()
    return

def PlotGst():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
    ax = axes[0]
    specs = [("gst_year_balanced_thinned_national.csv", "Balanced panel, national vocabulary", INK),
             ("gst_year_balanced_thinned.csv", "Balanced panel, all vocabulary", MUTED),
             ("gst_year.csv", "All districts (sample changes in 2020)", "#B5B5B5")]
    for fname, label, color in specs:
        d = pd.read_csv(OUTDIR_TABLES / fname)
        ax.plot(d.label, d.pi, marker="o", ms=5, lw=2, color=color, label=label)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--")
    ax.text(2018, 0.5002, "no partisanship (0.5)", color=MUTED, fontsize=8, va="bottom")
    ax.set_title("GST partisanship of language, by year", loc="left", fontsize=11)
    ax.set_ylabel("leave-out pi (prob. of guessing party from one word)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylim(0.4995, 0.5075)
    ax = axes[1]
    d = pd.read_csv(OUTDIR_TABLES / "gst_quarter.csv")
    x = range(len(d))
    ax.plot(x, d.pi, color=INK, lw=2, marker="o", ms=4, label="data (R vs D districts)")
    ax.plot(x, d.within_state_mean, color=MUTED, lw=1.5, ls=":", label="labels shuffled within state")
    ax.plot(x, d.random_null_mean, color=MUTED, lw=1.5, ls="--", label="labels shuffled at random")
    ax.set_xticks(list(x)[::2])
    ax.set_xticklabels(d.label[::2], rotation=45, ha="right")
    ax.set_title("By quarter, all districts (sample size jumps in 2020Q2)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTDIR_FIGURES / "gst_polarization.png", dpi=160)
    plt.close(fig)

def PlotCategoryGap():
    d = pd.read_csv(OUTDIR_TABLES / "lexicon_category_gap.csv")
    d = d[d.period == "post"].sort_values("rate_total_t")
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    colors = [RED if t > 3 else BLUE if t < -3 else "#B5B5B5" for t in d.rate_total_t]
    ax.barh(d.category, d.rate_total_t, color=colors, height=0.6)
    ax.axvline(0, color=MUTED, lw=1)
    for v in (-3, 3):
        ax.axvline(v, color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("t-statistic of R - D district mention rates, 2022-23   (right = more in Republican-county districts)")
    ax.set_title("Which agendas are partisan? (|t| > 3 coloured)", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUTDIR_FIGURES / "lexicon_category_gap.png", dpi=160)
    plt.close(fig)

def PlotCategoryTrend():
    d = pd.read_csv(OUTDIR_TABLES / "lexicon_category_trend_quarter.csv")
    cats = ["covid_mandates", "equity_sel", "gender_lgbtq", "book_content", "parental_rights", "school_choice", "crt_race", "guns_safety"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), sharex=True)
    quarters = sorted(d.label.unique())
    for ax, cat in zip(axes.ravel(), cats):
        g = d[d.category == cat].set_index("label").reindex(quarters)
        ax.plot(range(len(quarters)), g.R, color=RED, lw=2, label="R districts")
        ax.plot(range(len(quarters)), g.D, color=BLUE, lw=2, label="D districts")
        ax.set_title(cat, loc="left", fontsize=10)
        ax.set_xticks(range(0, len(quarters), 4))
        ax.set_xticklabels([q[:4] for q in quarters[::4]])
    axes[0, 0].set_ylabel("mentions per 10k words")
    axes[1, 0].set_ylabel("mentions per 10k words")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR_FIGURES / "lexicon_category_trend.png", dpi=160)
    plt.close(fig)

if __name__ == "__main__":
    Main()
