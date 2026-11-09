"""Step 3 + 4a + 5: adoption of each tracked phrase across districts, and its correlation with partisanship / locale.

Adoption = a district's first month with >= 1 mention (procedural/excluded hits do not count).
Sample-composition fixes:
  * "panel" districts = >= 1 kept meeting in the PRE years AND >= 1 in the POST years (globals.json). Adoption curves and
    the timing regressions use only these, so a district cannot "adopt" just because it starts posting videos in 2021,
    and a pre-period mention is visible. Curves for all districts are kept in the CSV for comparison.
  * Meeting shares are district-weighted: each district's share of its own meetings in the quarter that mention the
    phrase, averaged over districts that posted that quarter (so heavy posters do not dominate).
Writes output/tables/diffusion/approach2/{phrase_table.csv, adoption_monthly.csv, meeting_share_quarterly.csv,
       timing_regressions.csv} and output/figures/diffusion/approach2/adoption_curves.png
Run from the repo root (after count_mentions.py):  python3 source/analysis/diffusion/approach2/adoption.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from phrases import PHRASES  # noqa: E402

N_PLOT = 8
# second figure: narrower, nationally coordinated talking points (fewer adopters, so curves are noisier)
TALKING_POINTS = ["parents bill of rights", "moms for liberty", "unmask our children", "co-parent with government",
                  "medical freedom", "groomer / grooming kids", "sexually explicit", "gender ideology"]
COL_R, COL_D = "#c0392b", "#2c6fbb"


def Main():
    docs, dm, dist = Load()
    first = FirstMention(dm, dist)
    months = pd.period_range(f"{cm.SAMPLE_START}-01", f"{cm.SAMPLE_END}-12", freq="M")
    curves = AdoptionCurves(first, dist, months)
    shares = MeetingShares(docs, dm)
    table = PhraseTable(dm, first, dist, docs)
    timing = TimingRegressions(first, dist, docs)
    cm.OUTDIR_TABLES.mkdir(parents=True, exist_ok=True)
    curves.to_csv(cm.OUTDIR_TABLES / "adoption_monthly.csv", index=False, float_format="%.4f")
    shares.to_csv(cm.OUTDIR_TABLES / "meeting_share_quarterly.csv", index=False, float_format="%.5f")
    table.to_csv(cm.OUTDIR_TABLES / "phrase_table.csv", index=False, float_format="%.3f")
    timing.to_csv(cm.OUTDIR_TABLES / "timing_regressions.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(table.to_string())
    print(timing.loc[timing.term.isin(["gop_z"])].to_string())
    top = table.sort_values("panel_adopters", ascending=False).phrase.head(N_PLOT).tolist()
    PlotCurves(curves, top, "adoption_curves.png")
    PlotCurves(curves, TALKING_POINTS, "adoption_curves_talking_points.png")
    return


def Load():
    docs = pd.read_parquet(cm.OUTDIR_DATA / "docs.parquet")
    docs["month"] = docs.date.dt.to_period("M")
    docs["year"] = docs.date.dt.year
    dm = pd.read_parquet(cm.OUTDIR_DATA / "mentions.parquet")
    dm = dm.loc[dm["count"] > 0].copy()
    dm["month"] = dm.date.dt.to_period("M")
    st = cm.LoadStatic()
    g = docs.groupby("leaid")
    dist = pd.DataFrame({"n_docs": g.size(), "n_pre": g.year.apply(lambda y: y.isin(cm.PRE_YEARS).sum()),
                         "n_post": g.year.apply(lambda y: y.isin(cm.POST_YEARS).sum()),
                         "words": g.n_words.sum(), "first_month": g.month.min()}).reset_index()
    dist["panel"] = (dist.n_pre > 0) & (dist.n_post > 0)
    dist = dist.merge(st, on="leaid", how="left", validate="1:1")
    return docs, dm, dist


def FirstMention(dm, dist):
    first = dm.groupby(["phrase", "leaid"]).month.min().rename("adopt_month").reset_index()
    return first.merge(dist[["leaid", "panel", "rep", "state", "gop_2p_2020", "locale4"]], on="leaid", how="left")


def AdoptionCurves(first, dist, months):
    rows = []
    for sample, dd in (("panel", dist.loc[dist.panel]), ("all", dist)):
        denom = dd.groupby("rep").size().to_dict()
        denom["all"] = len(dd)
        ff = first.loc[first.leaid.isin(dd.leaid)]
        for ph, g in ff.groupby("phrase"):
            for grp in ("all", "R", "D"):
                gg = g if grp == "all" else g.loc[g.rep == grp]
                cnt = gg.adopt_month.value_counts().reindex(months, fill_value=0).cumsum()
                rows.append(pd.DataFrame({"sample": sample, "phrase": ph, "group": grp, "month": months.astype(str),
                                          "n_adopted": cnt.values, "share_adopted": cnt.values / denom[grp]}))
    return pd.concat(rows, ignore_index=True)


def MeetingShares(docs, dm):
    docs = docs.loc[~docs.is_empty].copy()
    docs["quarter"] = docs.date.dt.to_period("Q")
    rows = []
    for ph, g in dm.groupby("phrase"):
        d = docs.assign(hit=docs.video_id.isin(g.video_id))
        per = d.groupby(["quarter", "leaid"]).hit.mean().groupby("quarter")
        rows.append(pd.DataFrame({"phrase": ph, "quarter": per.mean().index.astype(str),
                                  "district_weighted_share": per.mean().values, "n_districts": per.size().values,
                                  "pooled_share": d.groupby("quarter").hit.mean().values}))
    return pd.concat(rows, ignore_index=True)


def PhraseTable(dm, first, dist, docs):
    panel = dist.loc[dist.panel]
    r_share_all = (dist.rep == "R").mean()
    rows = []
    pre_docs = docs.loc[docs.year.isin(cm.PRE_YEARS) & docs.leaid.isin(panel.leaid)]
    post_docs = docs.loc[docs.year.isin(cm.POST_YEARS) & docs.leaid.isin(panel.leaid)]
    raw = pd.read_parquet(cm.OUTDIR_DATA / "mentions.parquet")
    for ph, meta in PHRASES.items():
        f = first.loc[first.phrase == ph]
        fp = f.loc[f.panel]
        g = dm.loc[dm.phrase == ph]
        rr = raw.loc[raw.phrase == ph]
        pre_hit = g.loc[g.video_id.isin(pre_docs.video_id)]
        post_hit = g.loc[g.video_id.isin(post_docs.video_id)]
        med = fp.adopt_month.sort_values().iloc[len(fp) // 2] if len(fp) else None
        rows.append({
            "phrase": ph, "source": meta["source"], "side": meta["side"],
            "n_districts": f.leaid.nunique(), "n_states": f.state.nunique(), "n_meetings": g.video_id.nunique(),
            "mentions": int(g["count"].sum()), "excluded_share": rr.count_excluded.sum() / max(1, rr[["count", "count_excluded"]].sum().sum()),
            "first_month": str(f.adopt_month.min()) if len(f) else "",
            "panel_adopters": len(fp), "panel_adopt_share": len(fp) / len(panel),
            "panel_median_adopt_month": str(med) if med is not None else "",
            "panel_pre_adopters": int((fp.adopt_month.dt.year.isin(cm.PRE_YEARS)).sum()),
            "meet_share_pre": pre_hit.video_id.nunique() / len(pre_docs),
            "meet_share_post": post_hit.video_id.nunique() / len(post_docs),
            "r_share_adopters": (f.rep == "R").mean(), "r_share_baseline": r_share_all,
        })
    return pd.DataFrame(rows).sort_values("n_districts", ascending=False)


def TimingRegressions(first, dist, docs):
    """Panel districts, per phrase and pooled: LPM of ever adopting in POST (excluding PRE adopters), and among POST
    adopters months from the start of POST to adoption, on standardized county GOP share + locale + log meetings."""
    panel = dist.loc[dist.panel & dist.gop_2p_2020.notna()].copy()
    panel["gop_z"] = (panel.gop_2p_2020 - panel.gop_2p_2020.mean()) / panel.gop_2p_2020.std()
    panel["log_post"] = np.log(panel.n_post)
    post0 = pd.Period(f"{cm.POST_YEARS[0]}-01", "M")
    out = []
    frames = []
    for ph in PHRASES:
        f = first.loc[first.phrase == ph, ["leaid", "adopt_month"]]
        d = panel.merge(f, on="leaid", how="left")
        d = d.loc[~(d.adopt_month.notna() & (d.adopt_month < post0))].copy()  # drop PRE adopters
        d["adopt"] = d.adopt_month.notna().astype(float)
        d["months_to"] = [(m - post0).n if pd.notna(m) else np.nan for m in d.adopt_month]
        d["phrase"] = ph
        frames.append(d)
        if d.adopt.sum() >= 30:
            out += Fit(d, "adopt ~ gop_z + C(locale4) + log_post", ph, "ever_adopt_lpm")
            out += Fit(d.loc[d.adopt == 1], "months_to ~ gop_z + C(locale4) + log_post", ph, "months_to_adopt")
    pooled = pd.concat(frames)
    out += Fit(pooled, "adopt ~ gop_z + C(locale4) + log_post + C(phrase)", "POOLED", "ever_adopt_lpm")
    out += Fit(pooled.loc[pooled.adopt == 1], "months_to ~ gop_z + C(locale4) + log_post + C(phrase)", "POOLED",
               "months_to_adopt")
    return pd.DataFrame(out)


def Fit(d, formula, ph, outcome):
    res = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d.leaid)[0]})
    keep = [t for t in res.params.index if t == "gop_z" or t.startswith("C(locale4)")]
    return [{"phrase": ph, "outcome": outcome, "term": t, "coef": res.params[t], "se": res.bse[t], "p": res.pvalues[t],
             "n": int(res.nobs), "mean_y": d[formula.split(" ~")[0]].mean()} for t in keep]


def PlotCurves(curves, top, fname):
    c = curves.loc[(curves["sample"] == "panel") & curves.phrase.isin(top)].copy()
    c["month"] = pd.PeriodIndex(c.month, freq="M").to_timestamp()
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
    for ax, ph in zip(axes.flat, top):
        for grp, col, lab in (("R", COL_R, "R county (GOP 2020 > 50%)"), ("D", COL_D, "D county")):
            g = c.loc[(c.phrase == ph) & (c.group == grp)]
            ax.plot(g.month, g.share_adopted, color=col, lw=1.8, label=lab)
        ax.axvline(pd.Timestamp(f"{cm.POST_YEARS[0]}-03-01"), color="0.6", lw=0.8, ls=":")
        ax.set_title(ph, fontsize=10)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=None))
    for ax in axes[:, 0]:
        ax.set_ylabel("cumulative share of\npanel districts adopted")
    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("First mention of tracked phrases, panel districts (meetings posted both 2018-19 and 2020-23); "
                 "dotted line = Mar 2020", fontsize=10)
    fig.tight_layout()
    cm.OUTDIR_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(cm.OUTDIR_FIGURES / fname, dpi=150)
    print(f"figure -> {cm.OUTDIR_FIGURES / fname}")


if __name__ == "__main__":
    Main()
