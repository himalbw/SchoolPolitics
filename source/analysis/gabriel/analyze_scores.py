"""Descriptive analysis of the GABRIEL scores (no API calls).

Run from the repo root:  micromamba run -n stat236 python source/analysis/gabriel/analyze_scores.py
Inputs: derived/gabriel/{scores_full,scores_terra,public_comment_share}.parquet, excerpts{,_extension,_extra_random}.parquet
+ district_static.parquet. `analyze_scores.py variance` prints only the variance split (step 3 decision).
Years 2018-2024 (2020 and 2024 added 2026-10-28; 2024 is outside globals.json's SAMPLE_END, see RESULTS.md).
Random and flagged excerpts are analysed separately throughout and never pooled. R / D = county gop_2p_2020 > / <= 0.5
(a county proxy for district partisanship). District-year means weight districts equally; CIs bootstrap over districts.
Outputs: output/figures/gabriel_*.png, output/tables/gabriel_*.csv, derived/gabriel/gabriel_district_year.parquet.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bow"))
from conflict_map import TILES  # noqa: E402

INDIR = Path("Datastore/DistrictView/derived/gabriel/")
INDIR_STATIC = Path("Datastore/District Covariates/processed/district_static.parquet")
OUTDIR_FIGURES = Path("output/figures/")
OUTDIR_TABLES = Path("output/tables/")

GLOBALS = json.loads(Path("globals.json").read_text())
PRE_YEARS = list(range(GLOBALS["SAMPLE_START"], 2020))
END_YEAR = 2024   # GABRIEL extension runs past globals.json SAMPLE_END (2023) by one year
POST_YEARS = list(range(2021, END_YEAR + 1))
YEARS = PRE_YEARS + [2020] + POST_YEARS   # 2020 is in the figures and event studies, not in pre/post contrasts
HALVES = ["2020H1", "2020H2", "2021H1", "2021H2"]
EXCERPT_FILES = ["excerpts.parquet", "excerpts_extension.parquet", "excerpts_extra_random.parquet"]
PC_CUTOFF = 50   # public_comment_share >= PC_CUTOFF -> mostly public comment
ATTRS = ["hostility", "culture_war_share", "parental_rights_framing", "covid_contention", "progressive_positions",
         "conservative_positions"]
SCORES = ATTRS + ["stance"]
CATEGORIES = ["crt_race", "parental_rights", "book_content", "gender_lgbtq", "covid_mandates", "equity_sel"]
LOCALE_POST = ["suburb", "town", "rural"]     # locale x post dummies, city = reference
SEED = 236
N_BOOT = 1000
MIN_IDENTIFYING = 50   # fewer districts observed both pre and post -> estimate flagged "thin"
INK, MUTED, GRID = "#222222", "#6B6B6B", "#E4E4E4"
COLORS = {"D": "#2B6CB0", "R": "#C53030"}

# Place / school names that survived name-stripping (heuristic): "city of X", "X county", "X High" (cased captions);
# X must not be a stop word or a generic school/place word.
from spacy.lang.en.stop_words import STOP_WORDS  # noqa: E402

GENERIC = sorted(STOP_WORDS | {"neighboring", "surrounding", "entire", "whole", "same", "rural", "local", "home",
                               "virtual", "summer", "online", "tech", "technical", "career", "executive", "early",
                               "charter", "alternative", "preparatory", "prep", "leadership", "regional", "district",
                               "school", "schools", "education", "community", "middle", "high", "junior", "senior",
                               "public", "private", "new", "old", "own", "another", "adjacent", "different", "large",
                               "small", "big", "largest", "smallest", "poorest", "richest", "state", "nearby"},
                 key=len, reverse=True)
NOT_GENERIC = r"(?!(?:" + "|".join(map(re.escape, GENERIC)) + r")\b)"
LEAK_RE = re.compile(r"\b(?:city|town|village|borough|township) of " + NOT_GENERIC + r"[a-z]{3,}"
                     r"|\b" + NOT_GENERIC + r"[a-z]{3,} (?:county|township|parish)\b", re.I)
LEAK_CASED_RE = re.compile(r"\b" + NOT_GENERIC.replace("(?!(?:", "(?!(?i:") + r"[A-Z][a-z]{2,} "
                           r"(?:High|Middle|Elementary|Academy)\b")


LABELS = {"hostility": "Hostility", "culture_war_share": "Share of discussion on culture-war issues",
          "parental_rights_framing": "Parental-rights framing", "covid_contention": "Contention over COVID policy",
          "progressive_positions": "Progressive positions argued", "conservative_positions": "Conservative positions argued",
          "stance": "Stance (conservative minus progressive)"}
CAT_LABELS = {"crt_race": "CRT / race", "parental_rights": "Parental rights", "book_content": "Books / content",
              "gender_lgbtq": "Gender / LGBTQ", "covid_mandates": "COVID mandates", "equity_sel": "Equity / SEL"}
MIN_STATE_DISTRICTS = 5
HEADLINE = ["conservative_positions", "parental_rights_framing"]


def Main():
    OUTDIR_FIGURES.mkdir(parents=True, exist_ok=True)
    OUTDIR_TABLES.mkdir(parents=True, exist_ok=True)
    df = Load()
    if sys.argv[1:] == ["variance"]:
        return VarianceSplit(df)
    VarianceSplit(df)
    ExtraRandomCI(df)
    TerraCheck(df)
    panel = Panel(df)
    trends = Trends(panel)
    WithinTopic(panel)
    gaps = GapByYear(panel)
    Headline(trends, gaps)
    ChangeMap(panel)
    reg = pd.concat([Regressions(panel, "all"), Regressions(panel[panel.year != 2024], "all, post = 2021-23")])
    reg.to_csv(OUTDIR_TABLES / "gabriel_regressions.csv", index=False)
    EventStudy(panel)
    EventStudyHalfYear(df)
    EventStudyBySpeaker(df)
    DiagnoseDip(df, panel)
    LeakCheck(df)
    SplitCheck(panel)
    ModelCheck(df)
    return


def Load():
    scores = pd.read_parquet(INDIR / "scores_full.parquet")
    excerpts = pd.concat([pd.read_parquet(INDIR / f).drop(columns=["batch", "half_year"], errors="ignore")
                          for f in EXCERPT_FILES if (INDIR / f).exists()], ignore_index=True)
    static = pd.read_parquet(INDIR_STATIC, columns=["leaid", "latitude", "longitude", "state_location"])
    static["leaid"] = static.leaid.astype(str).str.zfill(7)
    df = scores.merge(excerpts, on="excerpt_id", how="inner", validate="one_to_one")
    df = df.merge(static, on="leaid", how="left", validate="many_to_one")
    if (INDIR / "public_comment_share.parquet").exists():
        pc = pd.read_parquet(INDIR / "public_comment_share.parquet", columns=["excerpt_id", "public_comment_share"])
        df = df.merge(pc, on="excerpt_id", how="left", validate="one_to_one")
        print(f"public_comment_share missing for {df.public_comment_share.isna().sum():,} excerpts")
    print("Excerpts by batch and type:\n" + pd.crosstab(df.batch, df.excerpt_type).to_string())
    print(f"Scores {len(scores):,}, excerpts {len(excerpts):,}, joined {len(df):,}; "
          f"lat/lon missing {df.latitude.isna().sum()}, GOP share missing {df.gop_2p_2020.isna().sum()}")
    df["party"] = np.where(df.gop_2p_2020 > 0.5, "R", "D")
    df["post"] = (df.period == "post").astype(int)
    df["gop10"] = df.gop_2p_2020 * 10      # coefficients are per 10 percentage points of GOP share
    return df


def TerraCheck(df):
    terra = pd.read_parquet(INDIR / "scores_terra.parquet")
    both = terra.merge(df, on="excerpt_id", suffixes=("_terra", ""), validate="one_to_one")
    rows = []
    for kind, g in both.groupby("excerpt_type"):
        for a in SCORES:
            rows.append({"excerpt_type": kind, "attribute": a, "n": len(g), "corr": g[f"{a}_terra"].corr(g[a]),
                         "spearman": g[f"{a}_terra"].corr(g[a], method="spearman"),
                         "mean_luna": g[a].mean(), "mean_terra": g[f"{a}_terra"].mean(),
                         "mean_diff_terra_minus_luna": (g[f"{a}_terra"] - g[a]).mean()})
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR_TABLES / "gabriel_terra_check.csv", index=False)
    print("\nTerra vs Luna (1,000 shared excerpts):\n" + out.round(3).to_string(index=False))
    return


def Panel(df):
    # Random: leaid x year means. Flagged: leaid x year x lexicon_category means.
    keys = ["leaid", "year", "period", "post", "split", "gop_2p_2020", "gop10", "party", "locale", "latitude",
            "longitude", "state_location"]
    parts = []
    for kind, extra in [("random", []), ("flagged", ["lexicon_category"])]:
        g = df[df.excerpt_type == kind].groupby(keys + extra, dropna=False)
        p = (g[SCORES].mean().join(g.size().rename("n_excerpts")).join(g.video_id.nunique().rename("n_meetings"))
             .reset_index())
        parts.append(p.assign(excerpt_type=kind))
    panel = pd.concat(parts, ignore_index=True)
    panel.to_parquet(INDIR / "gabriel_district_year.parquet", index=False)
    panel.to_csv(OUTDIR_TABLES / "gabriel_district_year.csv", index=False)
    r = panel[panel.excerpt_type == "random"]
    print(f"\nPanel: {len(r):,} random district-years ({r.leaid.nunique()} districts), "
          f"{(panel.excerpt_type == 'flagged').sum():,} flagged district-year-categories")
    return panel


def BootMeans(pivot, rng):
    # pivot: districts x columns (NaN = missing). Mean across districts and a 95% bootstrap over districts.
    x = pivot.to_numpy(dtype=float)
    boots = np.nanmean(x[rng.integers(0, len(x), (N_BOOT, len(x)))], axis=1)
    return np.nanmean(x, axis=0), boots


def YearTable(r, a, rng):
    rows = []
    boots = {}
    for party in ["D", "R"]:
        pv = r[r.party == party].pivot_table(index="leaid", columns="year", values=a).reindex(columns=YEARS)
        mean, boots[party] = BootMeans(pv, rng)
        for i, y in enumerate(YEARS):
            rows.append({"attribute": a, "party": party, "year": y, "mean": mean[i],
                         "lo": np.percentile(boots[party][:, i], 2.5), "hi": np.percentile(boots[party][:, i], 97.5)})
    return pd.DataFrame(rows), boots


def PlotTrend(ax, t):
    for party, s in t.groupby("party"):
        ax.plot(s.year, s["mean"], "-o", color=COLORS[party], ms=4, lw=1.6,
                label=f"{'Republican' if party == 'R' else 'Democratic'}-leaning counties")
        ax.fill_between(s.year, s.lo, s.hi, color=COLORS[party], alpha=0.15, lw=0)
    ax.axvspan(2019.5, 2020.5, color=GRID, alpha=0.6, lw=0)   # 2020: meetings often virtual, public comment restricted
    ax.set_title(LABELS[t.attribute.iloc[0]], loc="left", fontsize=10)
    ax.set_ylabel("Mean score (0-100)", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_xticks(YEARS)


def Trends(panel):
    rng = np.random.default_rng(SEED)
    r = panel[panel.excerpt_type == "random"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    rows = []
    for ax, a in zip(axes.ravel(), ATTRS):
        t, _ = YearTable(r, a, rng)
        rows.append(t)
        PlotTrend(ax, t)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Typical meeting content (random excerpts): yearly mean by county partisanship, "
                 "95% bootstrap over districts; grey = 2020 (often virtual meetings)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR_FIGURES / "gabriel_trends_random.png", dpi=160)
    plt.close(fig)
    trends = pd.concat(rows)
    trends.to_csv(OUTDIR_TABLES / "gabriel_trends_random.csv", index=False)
    return trends


def WithinTopic(panel):
    # Flagged: district mean per category x period (mean of its district-year means), then mean across districts.
    rng = np.random.default_rng(SEED)
    f = panel[panel.excerpt_type == "flagged"]
    d = f.groupby(["leaid", "party", "lexicon_category", "period"])[["stance", "hostility"]].mean().reset_index()
    rows = []
    for (cat, period, party), g in d.groupby(["lexicon_category", "period", "party"]):
        for a in ["stance", "hostility"]:
            mean, boots = BootMeans(g[[a]], rng)
            rows.append({"lexicon_category": cat, "period": period, "party": party, "attribute": a, "n_districts": len(g),
                         "mean": mean[0], "lo": np.percentile(boots[:, 0], 2.5), "hi": np.percentile(boots[:, 0], 97.5)})
    t = pd.DataFrame(rows)
    t.to_csv(OUTDIR_TABLES / "gabriel_topic_period.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    offsets = {("D", "pre"): -0.27, ("D", "post"): -0.09, ("R", "pre"): 0.09, ("R", "post"): 0.27}
    for ax, a in zip(axes, ["stance", "hostility"]):
        for i, cat in enumerate(CATEGORIES):
            for (party, period), off in offsets.items():
                s = t[(t.attribute == a) & (t.lexicon_category == cat) & (t.party == party) & (t.period == period)]
                if s.empty:
                    continue
                s = s.iloc[0]
                ax.errorbar(s["mean"], i + off, xerr=[[s["mean"] - s.lo], [s.hi - s["mean"]]], fmt="o", ms=5,
                            color=COLORS[party], mfc=COLORS[party] if period == "post" else "white", lw=1.2,
                            label=f"{party}, {'2021-24' if period == 'post' else '2018-19'}" if i == 0 else None)
        if a == "stance":
            ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(range(len(CATEGORIES)), [CAT_LABELS[c] for c in CATEGORIES])
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_title(LABELS[a], loc="left", fontsize=10)
        ax.set_xlabel("Mean score" + (" (negative = progressive, positive = conservative)" if a == "stance" else " (0-100)"),
                      fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("When a contested topic comes up (flagged excerpts): mean by topic and period, Republican- vs "
                 "Democratic-leaning counties (95% bootstrap over districts)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR_FIGURES / "gabriel_topic_period.png", dpi=160)
    plt.close(fig)
    return


def GapByYear(panel):
    rng = np.random.default_rng(SEED)
    r = panel[panel.excerpt_type == "random"]
    rows = []
    for a in SCORES:
        t, boots = YearTable(r, a, rng)
        gap = boots["R"] - boots["D"]
        means = t.pivot(index="year", columns="party", values="mean")
        for i, y in enumerate(YEARS):
            rows.append({"attribute": a, "year": y, "gap_R_minus_D": means.loc[y, "R"] - means.loc[y, "D"],
                         "lo": np.percentile(gap[:, i], 2.5), "hi": np.percentile(gap[:, i], 97.5)})
    t = pd.DataFrame(rows)
    t.to_csv(OUTDIR_TABLES / "gabriel_gap_by_year.csv", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
    for ax, a in zip(axes.ravel(), SCORES):
        PlotGap(ax, t[t.attribute == a])
    axes.ravel()[-1].set_axis_off()
    fig.suptitle("Typical meeting content (random excerpts): Republican- minus Democratic-leaning county mean by year "
                 "(95% bootstrap over districts)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR_FIGURES / "gabriel_gap_by_year.png", dpi=160)
    plt.close(fig)
    return t


def PlotGap(ax, g):
    ax.plot(g.year, g.gap_R_minus_D, "-o", color=INK, ms=4, lw=1.4)
    ax.fill_between(g.year, g.lo, g.hi, color=INK, alpha=0.12, lw=0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.axvspan(2019.5, 2020.5, color=GRID, alpha=0.6, lw=0)
    ax.set_title(LABELS[g.attribute.iloc[0]], loc="left", fontsize=10)
    ax.set_ylabel("R minus D (points)", fontsize=9)
    ax.set_xticks(YEARS)
    ax.grid(axis="y", color=GRID, lw=0.6)


def Headline(trends, gaps):
    # Slide figure: 2 x 2, columns = the two headline attributes, rows = R vs D trends and the R - D gap.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for j, a in enumerate(HEADLINE):
        PlotTrend(axes[0, j], trends[trends.attribute == a])
        PlotGap(axes[1, j], gaps[gaps.attribute == a])
        axes[1, j].set_title(f"{LABELS[a]}: Republican minus Democratic", loc="left", fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Typical meeting content (random excerpts), 95% bootstrap over districts; "
                 "grey = 2020 (often virtual meetings)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR_FIGURES / "gabriel_headline.png", dpi=180)
    plt.close(fig)
    return


MAP_VARS = [("d_hostility_random", "hostility", "Hostility"),
            ("d_conservative_random", "conservative_positions", "Conservative positions argued")]


def ChangeMap(panel):
    # Per district, random excerpts: mean over 2021-24 district-years minus mean over 2018-19 district-years.
    r = panel[panel.excerpt_type == "random"]
    geo = panel.drop_duplicates("leaid").set_index("leaid")[["latitude", "longitude", "state_location", "party", "gop_2p_2020"]]
    d = geo.copy()
    for col, attr, _ in MAP_VARS + [("d_progressive_random", "progressive_positions", "")]:
        p = r.pivot_table(index="leaid", columns="period", values=attr)
        d[col] = p["post"] - p["pre"]
        if attr != "hostility":
            d[f"{attr}_pre"], d[f"{attr}_post"] = p["pre"], p["post"]
    d["n_meetings"] = r.groupby("leaid").n_meetings.sum()
    d = d.join(pd.read_parquet(INDIR / "districts.parquet", columns=["leaid", "place_name", "lea_name"]).set_index("leaid"))
    d.reset_index().to_csv(OUTDIR_TABLES / "gabriel_change_district.csv", index=False)
    # Appendix: district points + state tiles. Slide: state tiles only.
    fig = plt.figure(figsize=(15, 9))
    slide, slide_axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for row, (col, attr, label) in enumerate(MAP_VARS):
        x = d.dropna(subset=[col])
        x = x[x.longitude.between(-125, -66) & x.latitude.between(24, 50)]
        state = x.groupby("state_location")[col].agg(["mean", "size"])
        lim = max(np.nanpercentile(np.abs(x[col]), 95), 1e-6)
        title = f"State mean change in {label.lower()}, 2018-19 to 2021-24 (n = districts; grey = fewer than {MIN_STATE_DISTRICTS})"
        ax = fig.add_axes([0.02, 0.53 - 0.48 * row, 0.56, 0.42])
        sc = ax.scatter(x.longitude, x.latitude, c=x[col], cmap="RdBu_r", vmin=-lim, vmax=lim, s=22,
                        edgecolor="white", linewidth=0.3)
        ax.set_aspect(1.25)
        ax.set_axis_off()
        ax.set_title(f"{label} (random excerpts): change 2018-19 to 2021-24, {len(x)} districts", loc="left", fontsize=11)
        fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.01).set_label("2021-24 minus 2018-19 (points)")
        DrawTiles(fig.add_axes([0.62, 0.53 - 0.48 * row, 0.37, 0.42]), state, "State mean (n = districts)", -lim, lim)
        DrawTiles(slide_axes[row], state, f"{label}: change 2018-19 to 2021-24", -lim, lim)
    fig.savefig(OUTDIR_FIGURES / "gabriel_change_map_appendix.png", dpi=160)
    plt.close(fig)
    slide.suptitle(f"Typical meeting content (random excerpts): state mean of district changes, points on a 0-100 "
                   f"scale; n = districts; grey = fewer than {MIN_STATE_DISTRICTS}", x=0.01, ha="left", fontsize=11)
    slide.tight_layout(rect=[0, 0, 1, 0.93])
    slide.savefig(OUTDIR_FIGURES / "gabriel_change_map.png", dpi=180)
    plt.close(slide)
    return


def DrawTiles(ax, state, title, vmin, vmax, cmap="RdBu_r"):
    # Same state tile grid as source/analysis/bow/conflict_map.py (no boundary files offline), diverging colours.
    # `state` has columns mean and size; states with fewer than MIN_STATE_DISTRICTS districts are grey.
    cm = plt.get_cmap(cmap)
    for st, (x, y) in TILES.items():
        n = int(state["size"].get(st, 0))
        v = state["mean"].get(st, np.nan) if n >= MIN_STATE_DISTRICTS else np.nan
        z = (v - vmin) / (vmax - vmin) if not pd.isna(v) else np.nan
        ax.add_patch(plt.Rectangle((x, -y), 0.94, 0.94, color="#DDDDDD" if pd.isna(v) else cm(np.clip(z, 0, 1)), lw=0))
        dark = not pd.isna(v) and abs(z - 0.5) > 0.3
        ax.text(x + 0.47, -y + 0.68, st, ha="center", va="center", fontsize=8, color="white" if dark else INK)
        if not pd.isna(v):
            ax.text(x + 0.47, -y + 0.42, f"{v:+.1f}", ha="center", va="center", fontsize=7,
                    color="white" if dark else INK)
        if n:
            ax.text(x + 0.47, -y + 0.17, f"n={n}", ha="center", va="center", fontsize=6,
                    color="white" if dark else MUTED)
    ax.set_xlim(-0.2, 11)
    ax.set_ylim(-7.3, 1.2)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, loc="left", fontsize=10)


def FitFE(data, y, extra_fe=""):
    # y ~ gop10:post + district FE + year FE + locale x post; SEs clustered by district. 2020 is neither pre nor post.
    d = data[data.period != "covid"].dropna(subset=[y]).copy()
    d["gop10_post"] = d.gop10 * d.post
    for loc in LOCALE_POST:
        d[f"{loc}_post"] = (d.locale == loc).astype(int) * d.post
    formula = f"{y} ~ gop10_post + C(leaid) + C(year) + " + " + ".join(f"{l}_post" for l in LOCALE_POST) + extra_fe
    groups = pd.factorize(d.leaid)[0]
    fit = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": groups})
    b, se = fit.params["gop10_post"], fit.bse["gop10_post"]
    # Only districts observed both pre and post identify gop x post (district FE absorb the rest).
    both = int((d.groupby("leaid").post.nunique() == 2).sum())
    return {"attribute": y, "coef_gop10_x_post": b, "se": se, "ci_lo": b - 1.96 * se, "ci_hi": b + 1.96 * se,
            "p": fit.pvalues["gop10_post"], "n_obs": int(fit.nobs), "n_districts": d.leaid.nunique(),
            "n_districts_pre_and_post": both, "thin": both < MIN_IDENTIFYING, "mean_pre": d.loc[d.post == 0, y].mean()}


def Regressions(panel, sample):
    rows = []
    r = panel[panel.excerpt_type == "random"]
    for a in SCORES:
        rows.append({"sample": sample, "excerpt_type": "random", "lexicon_category": "", **FitFE(r, a)})
    f = panel[panel.excerpt_type == "flagged"]
    for cat in CATEGORIES:
        rows.append({"sample": sample, "excerpt_type": "flagged", "lexicon_category": cat,
                     **FitFE(f[f.lexicon_category == cat], "stance")})
    out = pd.DataFrame(rows)
    print(f"\nRegressions ({sample}): coefficient on GOP share (per 10 pp) x post\n"
          + out[["excerpt_type", "lexicon_category", "attribute", "coef_gop10_x_post", "se", "p", "n_districts_pre_and_post",
                 "n_districts"]].round(3).to_string(index=False))
    return out


def EventStudyFit(r, attrs):
    # score ~ sum_y gop10 x 1[year = y] (2019 omitted) + district FE + year FE + locale x year; SEs clustered by district.
    years = [y for y in YEARS if y != 2019]
    rows = []
    for a in attrs:
        d = r.dropna(subset=[a]).copy()
        terms = []
        for y in years:
            d[f"g{y}"] = d.gop10 * (d.year == y)
            terms.append(f"g{y}")
            for loc in LOCALE_POST:
                d[f"{loc}_{y}"] = ((d.locale == loc) & (d.year == y)).astype(int)
                terms.append(f"{loc}_{y}")
        fit = smf.ols(f"{a} ~ " + " + ".join(terms) + " + C(leaid) + C(year)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(d.leaid)[0]})
        rows.append({"attribute": a, "year": 2019, "coef_gop10": 0.0, "se": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p": np.nan})
        for y in years:
            b, se = fit.params[f"g{y}"], fit.bse[f"g{y}"]
            rows.append({"attribute": a, "year": y, "coef_gop10": b, "se": se, "ci_lo": b - 1.96 * se,
                         "ci_hi": b + 1.96 * se, "p": fit.pvalues[f"g{y}"], "n_obs": int(fit.nobs)})
    return pd.DataFrame(rows).sort_values(["attribute", "year"])


def EventStudy(panel):
    # Random excerpts, coefficients per 10 pp of GOP share relative to 2019.
    t = EventStudyFit(panel[panel.excerpt_type == "random"], SCORES)
    t.to_csv(OUTDIR_TABLES / "gabriel_event_study.csv", index=False)
    print("\nEvent study (GOP share per 10 pp x year, 2019 = 0):\n"
          + t.pivot(index="attribute", columns="year", values="coef_gop10").round(2).to_string())
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
    for ax, a in zip(axes.ravel(), SCORES):
        g = t[(t.attribute == a) & (t.year != 2019)]
        ax.errorbar(g.year, g.coef_gop10, yerr=[g.coef_gop10 - g.ci_lo, g.ci_hi - g.coef_gop10], fmt="o", color=INK,
                    ms=5, lw=1.2, capsize=0)
        ax.plot([2019], [0], "o", ms=5, mfc="white", mec=INK, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvspan(2019.5, 2020.5, color=GRID, alpha=0.6, lw=0)
        ax.set_title(LABELS[a], loc="left", fontsize=10)
        ax.set_ylabel("Coefficient per 10 pp GOP share", fontsize=9)
        ax.set_xticks(YEARS)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes.ravel()[-1].set_axis_off()
    fig.suptitle("Event study, random excerpts: GOP share x year (2019 = reference, open dot), district and year FE, "
                 "locale x year; 95% CI clustered by district", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR_FIGURES / "gabriel_event_study.png", dpi=160)
    plt.close(fig)
    return


def VarianceSplit(df, attr="conservative_positions"):
    # Sums of squares of excerpt-level random scores: between districts, between years within a district, and
    # within district-year (excerpts of the same district and year; includes excerpt sampling noise).
    r = df[df.excerpt_type == "random"].dropna(subset=[attr])
    x, m = r[attr], r[attr].mean()
    m_d = r.groupby("leaid")[attr].transform("mean")
    m_dy = r.groupby(["leaid", "year"])[attr].transform("mean")
    ss = {"between_districts": ((m_d - m) ** 2).sum(), "between_years_within_district": ((m_dy - m_d) ** 2).sum(),
          "within_district_year": ((x - m_dy) ** 2).sum()}
    total = ((x - m) ** 2).sum()
    out = pd.DataFrame([{"attribute": attr, "component": k, "sum_sq": v, "share": v / total} for k, v in ss.items()])
    out["n_excerpts"], out["batches"] = len(r), ", ".join(sorted(r.batch.unique()))
    out.to_csv(OUTDIR_TABLES / "gabriel_variance_split.csv", index=False)
    print(f"\nVariance split, {attr}, random excerpts ({len(r):,}; batches {out.batches[0]}):\n"
          + out[["component", "share"]].round(3).to_string(index=False))
    return out


def RandomPanel(df):
    # District-year means of random excerpts (no files written).
    r = df[df.excerpt_type == "random"]
    return r.groupby(["leaid", "year", "party", "gop10", "locale"])[SCORES].mean().reset_index()


def ExtraRandomCI(df):
    # Width of the R - D gap CI (headline attributes) with and without the extra random excerpts.
    rows = []
    for label, d in [("without extra_random", df[df.batch != "extra_random"]), ("with extra_random", df)]:
        if label == "with extra_random" and "extra_random" not in set(df.batch):
            continue
        r = RandomPanel(d)
        for a in HEADLINE:
            t, boots = YearTable(r, a, np.random.default_rng(SEED))
            gap = boots["R"] - boots["D"]
            for i, y in enumerate(YEARS):
                lo, hi = np.percentile(gap[:, i], [2.5, 97.5])
                rows.append({"sample": label, "attribute": a, "year": y, "lo": lo, "hi": hi, "width": hi - lo,
                             "mean_random_excerpts_per_district_year":
                                 d[(d.excerpt_type == "random") & (d.year == y)].groupby("leaid").size().mean()})
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR_TABLES / "gabriel_extra_random_ci.csv", index=False)
    print("\nR - D gap CI width (headline figure), 2021 and 2023:\n"
          + out[out.year.isin([2021, 2023])].pivot_table(index=["attribute", "year"], columns="sample",
                                                         values="width").round(2).to_string())
    return


def EventStudyHalfYear(df):
    # Random excerpts at district x period, where period = half-year in 2020-21 and calendar year otherwise.
    # score ~ sum_t gop10 x 1[t] (2019 omitted) + district FE + period FE + locale x period; clustered by district.
    r = df[df.excerpt_type == "random"].copy()
    half = r.meeting_date.dt.year.astype(str) + np.where(r.meeting_date.dt.month <= 6, "H1", "H2")
    r["t"] = np.where(r.year.isin([2020, 2021]), half, r.year.astype(str))
    d = r.groupby(["leaid", "t", "gop10", "locale"])[SCORES].mean().reset_index()
    periods = [str(y) for y in PRE_YEARS if y != 2019] + HALVES + [str(y) for y in POST_YEARS if y != 2021]
    xpos = {t: (int(t[:4]) + (0.25 if t.endswith("H1") else 0.75) if "H" in t else int(t) + 0.5) for t in periods}
    rows = []
    for a in SCORES:
        x = d.dropna(subset=[a]).copy()
        terms = []
        for t in periods:
            k = t.replace("H", "h")
            x[f"g{k}"] = x.gop10 * (x.t == t)
            terms.append(f"g{k}")
            for loc in LOCALE_POST:
                x[f"{loc}_{k}"] = ((x.locale == loc) & (x.t == t)).astype(int)
                terms.append(f"{loc}_{k}")
        fit = smf.ols(f"{a} ~ " + " + ".join(terms) + " + C(leaid) + C(t)", data=x).fit(
            cov_type="cluster", cov_kwds={"groups": pd.factorize(x.leaid)[0]})
        rows.append({"attribute": a, "period": "2019", "x": 2019.5, "coef_gop10": 0.0, "se": 0.0, "ci_lo": 0.0,
                     "ci_hi": 0.0, "p": np.nan})
        for t in periods:
            k = t.replace("H", "h")
            b, se = fit.params[f"g{k}"], fit.bse[f"g{k}"]
            rows.append({"attribute": a, "period": t, "x": xpos[t], "coef_gop10": b, "se": se, "ci_lo": b - 1.96 * se,
                         "ci_hi": b + 1.96 * se, "p": fit.pvalues[f"g{k}"]})
    t = pd.DataFrame(rows).sort_values(["attribute", "x"])
    t.to_csv(OUTDIR_TABLES / "gabriel_event_study_halfyear.csv", index=False)
    print("\nHalf-year event study (2019 = 0):\n"
          + t.pivot(index="attribute", columns="period", values="coef_gop10").round(2).to_string())
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharex=True)
    for ax, a in zip(axes.ravel(), SCORES):
        g = t[(t.attribute == a) & (t.period != "2019")]
        for is_half, color in [(False, MUTED), (True, INK)]:
            h = g[g.period.str.contains("H") == is_half]
            ax.errorbar(h.x, h.coef_gop10, yerr=[h.coef_gop10 - h.ci_lo, h.ci_hi - h.coef_gop10], fmt="o",
                        color=color, ms=5 if is_half else 4, lw=1.2, capsize=0,
                        label="half-year (2020-21)" if is_half else "calendar year")
        ax.plot([2019.5], [0], "o", ms=5, mfc="white", mec=INK, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvspan(2020, 2021, color=GRID, alpha=0.6, lw=0)
        ax.set_title(LABELS[a], loc="left", fontsize=10)
        ax.set_ylabel("Coefficient per 10 pp GOP share", fontsize=9)
        ax.set_xticks([y + 0.5 for y in YEARS], [str(y) for y in YEARS])
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes.ravel()[-1].set_axis_off()
    fig.suptitle("Event study, random excerpts: GOP share x half-year in 2020-21 (calendar years otherwise; 2019 = "
                 "reference, open dot), district and period FE, locale x period; 95% CI clustered by district",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR_FIGURES / "gabriel_event_study_halfyear.png", dpi=160)
    plt.close(fig)
    return


def EventStudyBySpeaker(df):
    # Yearly event study on random excerpts split by public_comment_share (>= PC_CUTOFF = mostly public comment).
    if "public_comment_share" not in df:
        print("No public_comment_share yet; skipping the by-speaker event study")
        return
    attrs = ["conservative_positions", "hostility"]
    labels = {"public": f"Mostly public comment (share >= {PC_CUTOFF})", "board": f"Mostly board / staff (share < {PC_CUTOFF})"}
    r = df[(df.excerpt_type == "random") & df.public_comment_share.notna()]
    parts = []
    for grp, d in [("public", r[r.public_comment_share >= PC_CUTOFF]), ("board", r[r.public_comment_share < PC_CUTOFF])]:
        p = d.groupby(["leaid", "year", "gop10", "locale"])[attrs].mean().reset_index()
        t = EventStudyFit(p, attrs).assign(speaker=grp)
        n = p.groupby("year").leaid.nunique()
        t["n_districts_year"] = t.year.map(n)
        t["n_excerpts_year"] = t.year.map(d.groupby("year").size())
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    t.to_csv(OUTDIR_TABLES / "gabriel_event_study_by_speaker.csv", index=False)
    print("\nEvent study by speaker type (random excerpts):\n"
          + t.pivot_table(index=["attribute", "speaker"], columns="year", values="coef_gop10").round(2).to_string())
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    for ax, a in zip(axes, attrs):
        for grp, off, color in [("board", -0.12, MUTED), ("public", 0.12, INK)]:
            g = t[(t.attribute == a) & (t.speaker == grp) & (t.year != 2019)]
            ax.errorbar(g.year + off, g.coef_gop10, yerr=[g.coef_gop10 - g.ci_lo, g.ci_hi - g.coef_gop10], fmt="o",
                        color=color, ms=5, lw=1.2, capsize=0, label=labels[grp])
        ax.plot([2019], [0], "o", ms=5, mfc="white", mec=INK, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.axvspan(2019.5, 2020.5, color=GRID, alpha=0.6, lw=0)
        ax.set_title(LABELS[a], loc="left", fontsize=10)
        ax.set_ylabel("Coefficient per 10 pp GOP share", fontsize=9)
        ax.set_xticks(YEARS)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Event study by who is speaking, random excerpts: GOP share x year (2019 = reference), district and "
                 "year FE, locale x year; 95% CI clustered by district", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTDIR_FIGURES / "gabriel_event_study_by_speaker.png", dpi=160)
    plt.close(fig)
    return


def DiagnoseDip(df, panel):
    # Is 2022 different in composition? Random excerpts, 2021-23.
    r = df[(df.excerpt_type == "random") & df.year.isin(POST_YEARS)].copy()
    r["quarter"] = "Q" + r.meeting_date.dt.quarter.astype(str)
    rows = []
    for y, g in r.groupby("year"):
        row = {"year": y, "n_excerpts": len(g), "n_districts": g.leaid.nunique(), "n_meetings": g.video_id.nunique(),
               "meetings_per_district": g.groupby("leaid").video_id.nunique().mean(), "mean_words": g.n_words.mean(),
               "share_R": (g.party == "R").mean()}
        for col in ["locale", "gop_tercile", "quarter"]:
            row |= {f"share_{col}_{k}": v for k, v in g[col].value_counts(normalize=True).sort_index().items()}
        rows.append(row)
    comp = pd.DataFrame(rows).set_index("year").T
    comp.to_csv(OUTDIR_TABLES / "gabriel_dip_composition.csv")
    print("\nRandom-excerpt composition, 2021-23:\n" + comp.round(3).to_string())
    # 2022 R - D gap for conservative_positions, by split.
    rng = np.random.default_rng(SEED)
    rows = []
    for split in ["A", "B"]:
        t, boots = YearTable(panel[(panel.excerpt_type == "random") & (panel.split == split)], "conservative_positions", rng)
        means = t.pivot(index="year", columns="party", values="mean")
        gap = boots["R"] - boots["D"]
        for i, y in enumerate(YEARS):
            rows.append({"split": split, "year": y, "mean_R": means.loc[y, "R"], "mean_D": means.loc[y, "D"],
                         "gap_R_minus_D": means.loc[y, "R"] - means.loc[y, "D"],
                         "lo": np.percentile(gap[:, i], 2.5), "hi": np.percentile(gap[:, i], 97.5)})
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR_TABLES / "gabriel_gap_conservative_by_split.csv", index=False)
    print("\nConservative positions, R - D gap by split:\n" + out.round(2).to_string(index=False))
    return


def LeakCheck(df):
    # Excerpt-level stance ~ GOP share (+ year FE; + category FE for flagged), with vs without a detected leftover name.
    df = df.copy()
    df["leak_match"] = [m.group(0) if (m := LEAK_RE.search(t) or LEAK_CASED_RE.search(t)) else "" for t in df.text]
    df["leak"] = df.leak_match != ""
    rows = []
    for kind, g in df.groupby("excerpt_type"):
        for leak, h in g.groupby("leak"):
            formula = "stance ~ gop10 + C(year)" + (" + C(lexicon_category)" if kind == "flagged" else "")
            fit = smf.ols(formula, data=h).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(h.leaid)[0]})
            rows.append({"excerpt_type": kind, "leak_detected": leak, "n_excerpts": len(h),
                         "slope_stance_per_gop10": fit.params["gop10"], "se": fit.bse["gop10"], "p": fit.pvalues["gop10"]})
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR_TABLES / "gabriel_leak_check.csv", index=False)
    print("\nLeak check (stance on GOP share per 10 pp, excerpt level):\n" + out.round(3).to_string(index=False))
    print("Most common detected leftover names:", df.leak_match[df.leak].str.lower().value_counts().head(15).to_dict())
    return


def SplitCheck(panel):
    a = Regressions(panel[panel.split == "A"], "A")
    b = Regressions(panel[panel.split == "B"], "B")
    keys = ["excerpt_type", "lexicon_category", "attribute"]
    cols = ["coef_gop10_x_post", "se", "p", "n_obs", "n_districts", "n_districts_pre_and_post", "thin"]
    out = a[keys + cols].merge(b[keys + cols], on=keys, suffixes=("_A", "_B"))
    out.to_csv(OUTDIR_TABLES / "gabriel_regressions_split.csv", index=False)
    return


def ModelCheck(df):
    # Item 7 spec on the 1,000 Terra excerpts, with Luna and with Terra scores. Flagged excerpts are too sparse for one
    # regression per category here, so flagged uses a single regression with category FE (flagged only, never pooled
    # with random).
    terra = pd.read_parquet(INDIR / "scores_terra.parquet")
    sub = df[df.excerpt_id.isin(terra.excerpt_id)]
    rows = []
    for model, d in [("luna", sub), ("terra", sub.drop(columns=SCORES).merge(terra[["excerpt_id"] + SCORES],
                                                                             on="excerpt_id"))]:
        panel = Panel_(d)
        for a in ["stance", "conservative_positions", "hostility"]:
            r = panel[panel.excerpt_type == "random"]
            rows.append({"model": model, "excerpt_type": "random", **FitFE(r, a)})
            f = panel[panel.excerpt_type == "flagged"]
            rows.append({"model": model, "excerpt_type": "flagged (category FE)",
                         **FitFE(f, a, extra_fe=" + C(lexicon_category)")})
    t = pd.DataFrame(rows)
    keys = ["excerpt_type", "attribute"]
    cols = ["coef_gop10_x_post", "se", "p", "n_obs", "n_districts", "n_districts_pre_and_post", "thin"]
    out = t[t.model == "luna"][keys + cols].merge(t[t.model == "terra"][keys + cols], on=keys, suffixes=("_luna", "_terra"))
    out.to_csv(OUTDIR_TABLES / "gabriel_model_check.csv", index=False)
    print("\nModel check (1,000 Terra excerpts):\n" + out.round(3).to_string(index=False))
    return


def Panel_(df):
    # Panel() without writing files (for the model check).
    keys = ["leaid", "year", "period", "post", "split", "gop10", "locale"]
    parts = []
    for kind, extra in [("random", []), ("flagged", ["lexicon_category"])]:
        p = df[df.excerpt_type == kind].groupby(keys + extra)[SCORES].mean().reset_index()
        parts.append(p.assign(excerpt_type=kind))
    return pd.concat(parts, ignore_index=True)


if __name__ == "__main__":
    Main()
