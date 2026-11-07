"""Exit and voice: do districts that lost more students in the pandemic talk more about contested topics afterward?

Long difference pre -> post on enrollment change fall 2019 -> fall 2021, holding closure, partisanship, achievement, SES, size, locale and state
fixed, with the 2017 -> 2019 enrollment trend as a control and as a placebo, plus pre-period levels and leave-one-state-out ranges.
Writes output/tables/closures_screened/exit_voice.csv. Run from the repo root.
"""
import numpy as np
import pandas as pd
import pyfixest as pf

from closure_data import AddZScores, CategoryRatePanel, LANGUAGE_CATEGORIES, LoadCovariates, OUTDIR_TABLES

CONTROLS = "z_closure + z_gop + z_ach + z_ses + z_ln_enroll + C(locale)"
MIN_WORDS_PERIOD = 20000

def Main():
    cov = AddZScores(LoadCovariates(), cols=["closure", "gop", "ach", "ses", "ln_enroll", "d_enroll_21", "d_enroll_23", "d_enroll_pre"])
    panel = CategoryRatePanel("period", MIN_WORDS_PERIOD)
    wide = panel.pivot(index="leaid", columns="label", values=LANGUAGE_CATEGORIES)
    d = pd.DataFrame({y: wide[y]["post"] - wide[y]["pre"] for y in LANGUAGE_CATEGORIES}).dropna().reset_index().merge(cov, on="leaid")
    for y in LANGUAGE_CATEGORIES:
        d[y] = d[y].clip(*d[y].quantile([.01, .99]))
    specs = {"exit 19-21": ("z_d_enroll_21", f"z_d_enroll_21 + {CONTROLS}"), "exit 19-21 | pre-trend": ("z_d_enroll_21", f"z_d_enroll_21 + z_d_enroll_pre + {CONTROLS}"),
             "exit 19-23 | pre-trend": ("z_d_enroll_23", f"z_d_enroll_23 + z_d_enroll_pre + {CONTROLS}"), "placebo: pre-trend alone": ("z_d_enroll_pre", f"z_d_enroll_pre + {CONTROLS}"),
             "no controls but state": ("z_d_enroll_21", "z_d_enroll_21")}
    rows = []
    for y in LANGUAGE_CATEGORIES:
        for name, (term, rhs) in specs.items():
            m = pf.feols(f"{y} ~ {rhs} | state", d, vcov={"CRV1": "state"})
            t = m.tidy().loc[term]
            rows.append({"category": y, "spec": name, "b": t["Estimate"], "se": t["Std. Error"], "p": t["Pr(>|t|)"], "n": m._N, "mean_change": d[y].mean(), "sd_change": d[y].std()})
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR_TABLES / "exit_voice.csv", index=False)
    print(out.assign(txt=out.b.map("{:+.3f}".format) + " (" + out.se.map("{:.3f}".format) + ")").pivot(index="category", columns="spec", values="txt").reindex(LANGUAGE_CATEGORIES).to_string())
    pre = panel[panel.label == "pre"].merge(cov, on="leaid")
    for y in ["parental_rights", "gender_lgbtq"]:
        m = pf.feols(f"{y} ~ z_d_enroll_21 + z_d_enroll_pre + {CONTROLS} | state", pre, vcov={"CRV1": "state"})
        t = m.tidy().loc["z_d_enroll_21"]
        print(f"Placebo level in 2018-19, {y}: {t['Estimate']:+.3f} ({t['Std. Error']:.3f}), n={m._N}")
        loo = []
        for s in d.state.unique():
            r = pf.feols(f"{y} ~ z_d_enroll_21 + z_d_enroll_pre + {CONTROLS} | state", d[d.state != s], vcov={"CRV1": "state"}).tidy().loc["z_d_enroll_21"]
            loo.append((r["Estimate"], r["Pr(>|t|)"], s))
        print(f"  leave one state out, {y}: b from {min(loo)[0]:+.3f} (dropping {min(loo)[2]}) to {max(loo)[0]:+.3f} (dropping {max(loo)[2]}), max p {max(x[1] for x in loo):.3f}")
    print("sd of d_enroll_21:", round(cov.d_enroll_21.std(), 3))

if __name__ == "__main__":
    Main()
