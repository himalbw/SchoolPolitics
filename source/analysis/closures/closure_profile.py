"""What do the closure and achievement measures capture, and do they line up with partisanship? Also: first-stage checks.

Writes output/tables/closures_screened/{profile_*.csv, weekly_vs_share.csv, assessment_by_access.csv}. Run from the repo root.
"""
import glob

import numpy as np
import pandas as pd
import pyfixest as pf

from closure_data import AddZScores, LoadCovariates, OUTDIR_TABLES, INDIR_DQ

HUB = "Datastore/Covid School Data Hub/"

def Main():
    OUTDIR_TABLES.mkdir(parents=True, exist_ok=True)
    units = pd.read_parquet(INDIR_DQ / "units.parquet")
    df = LoadCovariates()
    df = AddZScores(df[df.leaid.isin(units.leaid)].copy())
    print(f"Screened sample districts: {len(df)}; closure known: {df.closure.notna().sum()} ({df.closure.notna().mean():.1%}); "
          f"missing states: {sorted(df[df.closure.isna()].state.value_counts().head(5).index)}")
    Variance(df)
    Correlations(df)
    StateTable(df)
    FirstStage(df)
    WeeklyVsShare(df)
    Assessment()
    return

def Variance(df):
    """Share of each measure's variance that lies between states, and what predicts closures."""
    rows = []
    for c in ["closure", "virtual", "inperson", "gop", "ach", "ses", "d_enroll_21", "d_enroll_23"]:
        d = df[[c, "state"]].dropna()
        rows.append({"measure": c, "n": len(d), "mean": d[c].mean(), "sd": d[c].std(), "between_state_share": 1 - (d[c] - d.groupby("state")[c].transform("mean")).var() / d[c].var()})
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR_TABLES / "profile_variance.csv", index=False)
    print(out.round(3).to_string(index=False))
    specs = {"gop": "z_closure ~ z_gop", "gop+state": "z_closure ~ z_gop | state", "ach+ses": "z_closure ~ z_ach + z_ses", "ach+ses+state": "z_closure ~ z_ach + z_ses | state",
             "all": "z_closure ~ z_gop + z_ach + z_ses + z_ln_enroll + C(locale)", "all+state": "z_closure ~ z_gop + z_ach + z_ses + z_ln_enroll + C(locale) | state"}
    rows = []
    for name, f in specs.items():
        m = pf.feols(f, df, vcov={"CRV1": "state"})
        t = m.tidy()
        rows.append({"spec": name, "n": m._N, "r2": m._r2, **{f"b_{k}": t.loc[k, "Estimate"] for k in t.index if k.startswith("z_")}, **{f"se_{k}": t.loc[k, "Std. Error"] for k in t.index if k.startswith("z_")}})
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR_TABLES / "profile_closure_predictors.csv", index=False)
    print(out.round(3).T.to_string())

def Correlations(df):
    cols = ["closure", "virtual", "gop", "ach", "ach_net_ses", "learn_rate", "ses", "d_enroll_21", "d_enroll_23"]
    raw = df[cols].corr()
    within = df[cols].sub(df.groupby("state")[cols].transform("mean")).corr()
    raw.to_csv(OUTDIR_TABLES / "profile_corr_raw.csv")
    within.to_csv(OUTDIR_TABLES / "profile_corr_within_state.csv")
    print("raw\n", raw.round(2).to_string(), "\nwithin state\n", within.round(2).to_string())

def StateTable(df):
    g = df.groupby("state_location").agg(n=("leaid", "size"), n_closure=("closure", "count"), closure=("closure", "mean"), sd_within=("closure", "std"), distinct=("closure", "nunique"),
                                         gop=("gop", "mean"), ach=("ach", "mean")).reset_index()
    g.to_csv(OUTDIR_TABLES / "profile_state.csv", index=False)
    usable = g[(g.n_closure >= 5)]
    print(f"\nStates with >=5 districts: {len(usable)}. Closure sd within state < .05 in {(usable.sd_within < .05).sum()}, <=3 distinct values in {(usable.distinct <= 3).sum()}. "
          f"Districts in those: {usable[(usable.sd_within < .05) | (usable.distinct <= 3)].n_closure.sum()} of {usable.n_closure.sum()}")
    m = g[g.n_closure >= 5]
    print(f"State-mean r(closure, gop) = {m.closure.corr(m.gop):.2f}, n = {len(m)}")

def FirstStage(df):
    """Do closures show up in an outcome they should affect? Log enrollment change, with a pre-2019 placebo."""
    rows = []
    for y in ["d_enroll_pre", "d_enroll_21", "d_enroll_23"]:
        for name, rhs in {"raw": "z_closure", "controls": "z_closure + z_gop + z_ach + z_ses + z_ln_enroll + C(locale)", "controls+state": "z_closure + z_gop + z_ach + z_ses + z_ln_enroll + C(locale) | state"}.items():
            m = pf.feols(f"{y} ~ {rhs}", df, vcov={"CRV1": "state"})
            t = m.tidy().loc["z_closure"]
            rows.append({"outcome": y, "spec": name, "n": m._N, "b_per_sd_closure": t["Estimate"], "se": t["Std. Error"], "p": t["Pr(>|t|)"]})
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR_TABLES / "profile_enrollment_first_stage.csv", index=False)
    print("\n", out.round(4).to_string(index=False))

def WeeklyVsShare(df):
    w = pd.read_parquet(HUB + "processed/hub_learning_model_weekly_dv.parquet")
    w["start"] = pd.to_datetime(w.TimePeriodStart, format="mixed", errors="coerce")
    w["end"] = pd.to_datetime(w.TimePeriodEnd, format="mixed", errors="coerce")
    w["days"] = (w.end - w.start).dt.days.clip(lower=1)
    w["score"] = w.LearningModel.str.lower().map({"in-person": 0, "hybrid": .5, "virtual": 1, "closed": 1})
    w = w[(w.start >= "2020-09-01") & (w.end <= "2021-06-30")]   # "Closed" in Aug is often before term starts
    agg = w.groupby("leaid").apply(lambda g: np.average(g.score, weights=g.days), include_groups=False).rename("weekly_closure").reset_index().merge(df, on="leaid")
    print(f"\nWeekly-derived closure vs hub share: r = {agg.weekly_closure.corr(agg.closure):.3f}, n = {agg.closure.notna().sum()}")
    agg[["leaid", "state", "weekly_closure", "closure"]].to_csv(OUTDIR_TABLES / "weekly_vs_share.csv", index=False)

def Assessment():
    """State x in-person-access-bin change in pass rates (not district level). Exact zeros look like empty cells, so they are dropped."""
    rows = []
    for f in sorted(glob.glob(HUB + "raw/AssmtOutcomes*")):
        x = pd.read_csv(f)
        x = x[[c for c in x.columns if not c.startswith("Unnamed")]].copy()
        x = x[x.level.isin(["Very low", "Low", "Moderate", "High", "Very high"])].drop_duplicates("level")
        x["state"] = f.split("_")[1]
        rows.append(x)
    a = pd.concat(rows, ignore_index=True)
    a.to_csv(OUTDIR_TABLES / "assessment_by_access.csv", index=False)
    long = a.melt(id_vars=["state", "level"], value_vars=["ela_2022", "math_2022"], var_name="subject").dropna()
    print("\nShare of 2022 cells that are exactly 0:", (long.value == 0).mean().round(3))
    nz = long[long.value != 0]
    print(nz.groupby(["subject", "level"]).value.agg(["mean", "count"]).round(2).unstack(0).to_string())
    long["hi"] = long.level.isin(["High", "Very high"])
    long["lo"] = long.level.isin(["Very low", "Low"])
    nz = long[long.value != 0]
    gap = nz[nz.hi].groupby(["state", "subject"]).value.mean() - nz[nz.lo].groupby(["state", "subject"]).value.mean()
    print("High minus low access, by subject (pass-rate points):", gap.groupby("subject").agg(["mean", "sem", "count"]).round(2).to_dict("index"))

if __name__ == "__main__":
    Main()
