"""District covariates (closures, achievement, enrollment change) and word-list rates for the closure analysis.

closure = 1 - in-person access = hub_share_virtual + 0.5 * hub_share_hybrid, for school year 2020-21. Achievement is SEDA's pooled
2009-19 mean, so it is measured before the pandemic. `ach_net_ses` is its residual on SES within state. The one district-level
post-pandemic education outcome in the Datastore is CCD enrollment (d_enroll_21, d_enroll_23 = log change from fall 2019).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "source/analysis/bow")
from bow_data import AddStrata, CollapseTo, GLOBALS, INDIR_DERIVED, INDIR_DQ, INDIR_STATIC, LoadDQ, LoadPhraseCounts, WithAnchor
from closure_lexicon import CLOSURE_LEXICON
from lexicon import LEXICON

OUTDIR_TABLES = Path("output/tables/closures_screened")
OUTDIR_FIGURES = Path("output/figures/closures_screened")
RHS_Z = ["closure", "gop", "ach", "ses", "ln_enroll"]
LANGUAGE_CATEGORIES = ["closure_talk", "learning_loss", "enrollment_loss", "covid_mandates", "meeting_conflict", "conflict_legal", "parental_rights",
                       "book_content", "crt_race", "gender_lgbtq", "school_choice", "equity_sel", "ctrl_academics", "ctrl_budget", "ctrl_facilities", "ctrl_athletics"]

def WithinResidual(df, y, x, by):
    """Residual of y on x with `by` fixed effects (NaN where any input is missing)."""
    ok = df[[y, x, by]].notna().all(axis=1)
    yd = df.loc[ok, y] - df.loc[ok].groupby(by)[y].transform("mean")
    xd = df.loc[ok, x] - df.loc[ok].groupby(by)[x].transform("mean")
    return (yd - xd * (yd @ xd) / (xd @ xd)).reindex(df.index)

def LoadCovariates():
    df = pd.read_parquet(INDIR_STATIC)
    df["leaid"] = df.leaid.astype(str).str.zfill(7)
    df["state"] = df.leaid.str[:2]
    df["closure"] = df.hub_share_virtual + 0.5 * df.hub_share_hybrid
    df["virtual"] = df.hub_share_virtual
    df["inperson"] = df.hub_share_inperson
    df["gop"] = df.gop_2p_2020
    df["is_r"] = np.where(df.gop.notna(), df.gop > 0.5, np.nan)
    df["ach"] = df.seda_cs_mn_avg_eb
    df["learn_rate"] = df.seda_cs_mn_lrn_eb
    df["ses"] = df.sedacov_sesavgall
    df["ach_net_ses"] = WithinResidual(df, "ach", "ses", "state")
    df["locale"] = df.urban_centric_locale.map(lambda c: {1: "city", 2: "suburb", 3: "town", 4: "rural"}.get(int(c) // 10) if pd.notna(c) else None)
    panel = pd.read_parquet(INDIR_STATIC.with_name("district_year.parquet"), columns=["leaid", "year", "enrollment"])
    panel["leaid"] = panel.leaid.astype(str).str.zfill(7)
    wide = panel[panel.enrollment > 0].pivot(index="leaid", columns="year", values="enrollment")
    df = df.merge(pd.DataFrame({"enroll19": wide[2019], "d_enroll_21": np.log(wide[2021] / wide[2019]), "d_enroll_23": np.log(wide[2023] / wide[2019]),
                                "d_enroll_pre": np.log(wide[2019] / wide[2017])}).reset_index(), on="leaid", how="left")
    df["ln_enroll"] = np.log(df.enroll19)
    for c in ("d_enroll_21", "d_enroll_23", "d_enroll_pre"):
        df.loc[df[c].abs() > 0.5, c] = np.nan   # consolidations and recoded LEAIDs, not attrition
    return df

def AddZScores(df, cols=RHS_Z + ["ach_net_ses", "d_enroll_21", "d_enroll_23", "virtual", "learn_rate"]):
    df = df.copy()
    for c in cols:
        df[f"z_{c}"] = (df[c] - df[c].mean()) / df[c].std()
    return df

def LoadClosurePhraseCounts(units):
    """Closure phrase counts aligned to `units` (joined on file, summed to district x quarter)."""
    docs = pd.read_parquet(INDIR_DQ / "docs.parquet", columns=["file", "leaid", "quarter"])
    counts = docs.merge(pd.read_parquet(INDIR_DERIVED / "bow_phrases" / "closure_phrase_counts.parquet"), on="file", how="left", validate="1:1")
    assert counts.iloc[:, 3].notna().all(), "closure phrase counts are out of sync; re-run count_closure_phrases.py"
    by_unit = counts.drop(columns="file").groupby(["leaid", "quarter"]).sum()
    return by_unit.reindex(pd.MultiIndex.from_frame(units[["leaid", "quarter"]])).fillna(0).reset_index(drop=True)

def CategoryRatePanel(level, min_words):
    """Long frame district x `level` x category of mentions per 10k words, combining both lexicons."""
    dtm, vocab, units = LoadDQ()
    vocab_pos = {t: i for i, t in enumerate(vocab)}
    phrases, closure_phrases = LoadPhraseCounts(units), LoadClosurePhraseCounts(units)
    key = pd.MultiIndex.from_arrays([units.leaid, units[level]])
    codes, uniq = pd.factorize(key)
    sums = lambda values: np.bincount(codes, weights=values, minlength=len(uniq))
    rows = pd.DataFrame(list(uniq), columns=["leaid", "label"])
    rows["n_words"] = sums(units.n_words.to_numpy())
    for cat in LANGUAGE_CATEGORIES:
        spec, table = (CLOSURE_LEXICON[cat], closure_phrases) if cat in CLOSURE_LEXICON else (LEXICON[cat], phrases)
        idx = [vocab_pos[t] for t in spec["terms"] if t in vocab_pos]
        term_counts = np.asarray(dtm[:, idx].sum(1)).ravel() if idx else np.zeros(len(units))
        rows[cat] = sums(term_counts + table[spec["phrases"]].sum(1).to_numpy()) / rows.n_words * 1e4
    return rows[rows.n_words >= min_words].reset_index(drop=True)
