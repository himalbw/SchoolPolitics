"""Do closures and educational outcomes predict what boards talk about, once partisanship and district type are held fixed?

District regressions on mentions per 10k words (closure_data.CategoryRatePanel):
  1. Levels by period, including the pre-period placebo (closures cannot cause 2018-19 language).
  2. Long differences within district, pre -> covid and pre -> post.
  3. Closure x GOP share, for the idea that conflict is highest where policy and local politics disagree.
  4. Event study by quarter with district and state x quarter fixed effects.
  5. Within-district timing, using the weekly learning-model panel for 2020Q3-2021Q2.
All right-hand-side variables are z-scored; standard errors are clustered by state.
Writes output/tables/closures_screened/language_*.csv. Run from the repo root (after count_closure_phrases.py).
"""
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats

from closure_data import AddZScores, CategoryRatePanel, LANGUAGE_CATEGORIES, LoadCovariates, OUTDIR_TABLES, RHS_Z

PERIODS = {2018: "pre", 2019: "pre", 2020: "covid", 2021: "covid", 2022: "post", 2023: "post"}
MIN_WORDS_PERIOD = 20000
MIN_WORDS_QUARTER = 5000
BASE = "z_closure + z_gop + z_ach + z_ses + z_ln_enroll + C(locale)"
REPORT = ["z_closure", "z_gop", "z_ach", "z_ses"]
EVENT_CATEGORIES = ["closure_talk", "learning_loss", "covid_mandates", "meeting_conflict", "parental_rights", "crt_race", "equity_sel", "ctrl_budget"]
EVENT_REF = "2019Q4"
HUB_WEEKLY = "Datastore/Covid School Data Hub/processed/hub_learning_model_weekly_dv.parquet"

def Main():
    OUTDIR_TABLES.mkdir(parents=True, exist_ok=True)
    cov = AddZScores(LoadCovariates())
    panel = CategoryRatePanel("period", MIN_WORDS_PERIOD)
    Levels(panel, cov)
    LongDifference(panel, cov)
    Mismatch(panel, cov)
    quarter = CategoryRatePanel("quarter", MIN_WORDS_QUARTER).merge(cov, left_on="leaid", right_on="leaid")
    quarter = quarter.rename(columns={"label": "quarter"})
    EventStudy(quarter)
    Timing(quarter)
    return

def Tidy(m, y, **extra):
    t = m.tidy()
    return [{"category": y, "term": k, "b": t.loc[k, "Estimate"], "se": t.loc[k, "Std. Error"], "p": t.loc[k, "Pr(>|t|)"], "n": m._N, **extra} for k in t.index if k in REPORT or k.startswith("z_closure:")]

def BH(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    out = np.empty(len(p))
    out[order] = np.minimum.accumulate(q[::-1])[::-1].clip(max=1)
    return out

def Levels(panel, cov):
    d = panel.merge(cov, on="leaid")
    rows = []
    for period in ["pre", "covid", "post"]:
        g = d[d.label == period]
        for y in LANGUAGE_CATEGORIES:
            for spec, fml in {"no state FE": f"{y} ~ {BASE}", "state FE": f"{y} ~ {BASE} | state"}.items():
                m = pf.feols(fml, g, vcov={"CRV1": "state"})
                rows += Tidy(m, y, period=period, spec=spec, mean_y=g[y].mean())
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR_TABLES / "language_levels.csv", index=False)
    show = out[(out.spec == "state FE") & (out.term.isin(["z_closure", "z_gop", "z_ach"]))].assign(t=lambda x: x.b / x.se)
    print("LEVELS, state FE, t-stats\n", show.pivot_table(index="category", columns=["term", "period"], values="t").reindex(LANGUAGE_CATEGORIES).round(1).to_string())
    print("n by period", show.groupby("period").n.max().to_dict())

def LongDifference(panel, cov):
    wide = panel.pivot(index="leaid", columns="label", values=LANGUAGE_CATEGORIES)
    rows = []
    for later in ["covid", "post"]:
        d = pd.DataFrame({y: wide[y][later] - wide[y]["pre"] for y in LANGUAGE_CATEGORIES}).dropna().reset_index()
        pre = {y: wide[y]["pre"].mean() for y in LANGUAGE_CATEGORIES}
        d = d.merge(cov, on="leaid", suffixes=("", "_cov"))
        for y in LANGUAGE_CATEGORIES:
            d[y] = d[y].clip(*d[y].quantile([.01, .99]))
            for spec, fml in {"no state FE": f"{y} ~ {BASE}", "state FE": f"{y} ~ {BASE} | state", "state FE, gop only": f"{y} ~ z_gop + z_ach + z_ses + z_ln_enroll + C(locale) | state",
                              "state FE, enrollment": f"{y} ~ {BASE} + z_d_enroll_21 | state", "state FE, virtual": f"{y} ~ z_virtual + z_gop + z_ach + z_ses + z_ln_enroll + C(locale) | state"}.items():
                m = pf.feols(fml, d, vcov={"CRV1": "state"})
                t = m.tidy()
                keep = [k for k in t.index if k.startswith("z_") and k != "z_ln_enroll"]
                rows += [{"category": y, "later": later, "spec": spec, "term": k, "b": t.loc[k, "Estimate"], "se": t.loc[k, "Std. Error"], "p": t.loc[k, "Pr(>|t|)"], "n": m._N,
                          "mean_change": d[y].mean(), "sd_change": d[y].std(), "pre_mean": pre[y], "r2_within": getattr(m, "_r2_within", np.nan)} for k in keep]
    out = pd.DataFrame(rows)
    out["q"] = np.nan
    for _, g in out.groupby(["later", "spec", "term"]):
        out.loc[g.index, "q"] = BH(g.p)
    out.to_csv(OUTDIR_TABLES / "language_long_difference.csv", index=False)
    show = out[(out.spec == "state FE") & out.term.isin(["z_closure", "z_gop", "z_ach", "z_ses"])]
    for later in ["covid", "post"]:
        s = show[show.later == later]
        print(f"\nLONG DIFF pre -> {later}, state FE, n={s.n.max()}; b per SD (per 10k words), stars = BH q<.05")
        s = s.assign(txt=s.b.map("{:+.2f}".format) + " (" + s.se.map("{:.2f}".format) + ")" + np.where(s.q < .05, "*", ""))
        print(s.pivot(index="category", columns="term", values="txt").reindex(LANGUAGE_CATEGORIES).join(s.drop_duplicates("category").set_index("category")[["pre_mean", "mean_change", "sd_change"]].round(2)).to_string())

def Mismatch(panel, cov):
    wide = panel.pivot(index="leaid", columns="label", values=LANGUAGE_CATEGORIES)
    rows = []
    for later in ["covid", "post"]:
        d = pd.DataFrame({y: wide[y][later] - wide[y]["pre"] for y in LANGUAGE_CATEGORIES}).dropna().reset_index().merge(cov, on="leaid")
        for y in ["meeting_conflict", "conflict_legal", "parental_rights", "covid_mandates", "crt_race"]:
            d[y] = d[y].clip(*d[y].quantile([.01, .99]))
            m = pf.feols(f"{y} ~ z_closure * z_gop + z_ach + z_ses + z_ln_enroll + C(locale) | state", d, vcov={"CRV1": "state"})
            t = m.tidy()
            rows.append({"category": y, "later": later, "n": m._N, **{f"b_{k}": t.loc[k, "Estimate"] for k in ["z_closure", "z_gop", "z_closure:z_gop"]}, "se_interaction": t.loc["z_closure:z_gop", "Std. Error"], "p_interaction": t.loc["z_closure:z_gop", "Pr(>|t|)"]})
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR_TABLES / "language_mismatch.csv", index=False)
    print("\nMISMATCH (closure x GOP)\n", out.round(3).to_string(index=False))

def EventStudy(quarter):
    rows, tests = [], []
    d = quarter.dropna(subset=["z_closure", "z_gop", "z_ach", "z_ses"]).copy()
    pre_q = [q for q in sorted(d.quarter.unique()) if q < EVENT_REF]
    for y in EVENT_CATEGORIES:
        d["yy"] = d[y].clip(upper=d[y].quantile(.995))
        m = pf.feols(f"yy ~ i(quarter, z_closure, ref='{EVENT_REF}') + i(quarter, z_gop, ref='{EVENT_REF}') + i(quarter, z_ach, ref='{EVENT_REF}') + i(quarter, z_ses, ref='{EVENT_REF}') | leaid + state^quarter", d, vcov={"CRV1": "state"})
        t = m.tidy()
        for var in ("z_closure", "z_gop", "z_ach"):
            for name, row in t[t.index.str.contains(var)].iterrows():
                q = name.split("::")[1].split(":")[0].replace("]", "")
                rows.append({"category": y, "var": var, "quarter": q, "b": row["Estimate"], "se": row["Std. Error"], "n": m._N})
        names = [n for n in t.index if "z_closure" in n and any(p in n for p in pre_q)]
        idx = [list(t.index).index(n) for n in names]
        beta = t.loc[names, "Estimate"].to_numpy()
        vcov = np.asarray(m._vcov)[np.ix_(idx, idx)]
        wald = beta @ np.linalg.pinv(vcov) @ beta
        tests.append({"category": y, "n_pre_quarters": len(names), "wald": wald, "p_pretrend_chi2": 1 - stats.chi2.cdf(wald, len(names)), "n": m._N})
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR_TABLES / "language_event_study.csv", index=False)
    tests = pd.DataFrame(tests)
    tests.to_csv(OUTDIR_TABLES / "language_event_pretrend.csv", index=False)
    print("\nEVENT STUDY closure x quarter\n", out[out["var"] == "z_closure"].assign(txt=lambda x: x.b.map("{:+.2f}".format) + "/" + x.se.map("{:.2f}".format)).pivot(index="quarter", columns="category", values="txt").to_string())
    print(tests.round(3).to_string(index=False))

def WeeklyExposure():
    """District x quarter share of school days not fully in person (hybrid = .5). Periods are spread over quarters by day."""
    w = pd.read_parquet(HUB_WEEKLY)
    w["start"] = pd.to_datetime(w.TimePeriodStart, format="mixed", errors="coerce")
    w["end"] = pd.to_datetime(w.TimePeriodEnd, format="mixed", errors="coerce")
    w["score"] = w.LearningModel.str.lower().map({"in-person": 0, "hybrid": .5, "virtual": 1, "closed": 1})
    w = w.dropna(subset=["start", "end", "score"])
    rows = []
    for q in pd.period_range("2020Q3", "2021Q2", freq="Q"):
        lo, hi = max(q.start_time, pd.Timestamp("2020-09-01")), q.end_time
        days = (np.minimum(w.end, hi) - np.maximum(w.start, lo)).dt.days.clip(lower=0) + 1
        hit = (w.start <= hi) & (w.end >= lo)
        g = w[hit].assign(days=days[hit]).groupby("leaid")
        rows.append(pd.DataFrame({"exposure": g.apply(lambda x: np.average(x.score, weights=x.days), include_groups=False)}).reset_index().assign(quarter=str(q)))
    return pd.concat(rows, ignore_index=True)

def Timing(quarter):
    exp = WeeklyExposure()
    d = quarter.merge(exp, on=["leaid", "quarter"])
    print(f"\nTIMING panel: {d.leaid.nunique()} districts, {len(d)} district-quarters, exposure sd {d.exposure.std():.2f}, within-district sd {(d.exposure - d.groupby('leaid').exposure.transform('mean')).std():.2f}")
    rows = []
    for y in LANGUAGE_CATEGORIES:
        for spec, fe in {"district + quarter FE": "leaid + quarter", "district + state x quarter FE": "leaid + state^quarter"}.items():
            m = pf.feols(f"{y} ~ exposure | {fe}", d, vcov={"CRV1": "state"})
            t = m.tidy().loc["exposure"]
            rows.append({"category": y, "spec": spec, "b": t["Estimate"], "se": t["Std. Error"], "p": t["Pr(>|t|)"], "n": m._N, "mean_y": d[y].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR_TABLES / "language_timing.csv", index=False)
    print(out.round(3).to_string(index=False))

if __name__ == "__main__":
    Main()
