"""How much of the GST partisanship series do closures and educational outcomes account for?

pi - 0.5 is exactly half the R-D gap in a district's leave-out slant, s_i = sum_j q_ij (rho_j^-i - 0.5), so regressing s_i on the party
label with controls gives a covariate-adjusted pi with the same leave-out correction. Three exercises, on the balanced panel, thinned,
national vocabulary (the comparable-over-time spec in bow/README.md), with delete-20% jackknife standard errors over districts:
  1. Adjusted pi by year, adding closure, achievement, SES, locale x size and state one at a time.
  2. Pi with pandemic vocabulary removed: is the 2021 peak carried by words about closures and COVID?
  3. The same leave-out statistic with other labels (high vs low closure, achievement, SES) in place of party.
Writes output/tables/closures_screened/gst_*.csv. Run from the repo root (about 10 min).
"""
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp

from closure_data import AddStrata, AddZScores, CollapseTo, GLOBALS, LoadCovariates, LoadDQ, OUTDIR_TABLES, WithAnchor
from closure_lexicon import CLOSURE_LEXICON
from gst_polarization import LeaveOutPi, MAX_TOP_STATE_SHARE, PermuteWithin, Thin, THIN_TO
from lexicon import LEXICON

N_THIN = 3
N_JACK = 200
JACK_KEEP = 0.8
SEED = 236
BASE_YEARS = [2018, 2019]
N_PANDEMIC_TERMS = 500
MIN_TERM_TOKENS = 500
COVID_TERMS = set(LEXICON["covid_mandates"]["terms"]) | set(CLOSURE_LEXICON["closure_talk"]["terms"]) | {"covid", "coronavirus", "pandemic", "virtual", "virtually", "remote", "remotely", "zoom"}
CONTROLS = {
    "party only": [],
    "+ closure": ["closure"],
    "+ achievement": ["ach"],
    "+ SES": ["ses"],
    "+ achievement + SES": ["ach", "ses"],
    "+ locale x size": ["stratum"],
    "+ closure + achievement + SES": ["closure", "ach", "ses"],
    "+ all above": ["closure", "ach", "ses", "stratum"],
    "+ state": ["state"],
    "+ state + closure": ["state", "closure"],
    "+ state + all above": ["state", "closure", "ach", "ses", "stratum"],
}

def Main(tag, balanced):
    OUTDIR_TABLES.mkdir(parents=True, exist_ok=True)
    reps, df, vocab, rng = LoadSample(balanced)
    AdjustedPi(reps, df, rng).to_csv(OUTDIR_TABLES / f"gst_adjusted_pi{tag}.csv", index=False)
    if balanced:
        WithoutPandemicVocabulary(reps, df, vocab, rng).to_csv(OUTDIR_TABLES / f"gst_drop_pandemic_terms{tag}.csv", index=False)
        OtherLabels(reps, df, rng).to_csv(OUTDIR_TABLES / f"gst_other_labels{tag}.csv", index=False)
    return

def LoadSample(balanced):
    """Thinned national-vocabulary counts by district x year for districts with every covariate."""
    dtm, vocab, units = LoadDQ()
    dtm, units = WithAnchor(dtm, units)
    units = AddStrata(units)
    conc = pd.read_parquet(f"output/tables/{GLOBALS['BOW_OUTPUT_TAG']}/term_state_concentration.parquet")
    national = np.flatnonzero((conc.top_state_share <= MAX_TOP_STATE_SHARE).to_numpy())
    dtm, vocab = dtm[:, national], vocab[national]
    counts, df, _ = CollapseTo(dtm, units, "year")
    ok = df.n_tokens >= THIN_TO
    cov = AddZScores(LoadCovariates())
    df = df.merge(cov[["leaid", "closure", "ach", "ses", "ach_net_ses", "z_closure", "z_ach", "z_ses"]], on="leaid", how="left")
    if balanced:
        full = df[ok].groupby("leaid").label.nunique() == df.label.nunique()
        ok &= df.leaid.isin(full[full].index)
    print(f"Districts present: {df[ok].leaid.nunique()} before requiring covariates")
    ok &= df[["closure", "ach", "ses", "stratum"]].notna().all(axis=1)
    rows = np.flatnonzero(ok)
    counts, df = counts[rows], df.iloc[rows].reset_index(drop=True)
    print(f"With closure, achievement, SES and strata: {df.leaid.nunique()} districts, {len(df)} district-years; R {df.drop_duplicates('leaid').is_r.sum()}, D {(~df.drop_duplicates('leaid').is_r).sum()}")
    print(df.groupby("label").size().to_dict())
    rng = np.random.default_rng(SEED)
    return [Thin(counts, rng, THIN_TO) for _ in range(N_THIN)], df, vocab, rng

def Slant(mat, is_r):
    """s_i for every row of `mat` using leave-out word partisanship. 0.5 * (mean_R s - mean_D s) equals LeaveOutPi - 0.5."""
    m = np.asarray(mat.sum(1)).ravel()
    sR, sD = np.asarray(mat[is_r].sum(0)).ravel(), np.asarray(mat[~is_r].sum(0)).ravel()
    mR, mD = m[is_r].sum(), m[~is_r].sum()
    coo = mat.tocoo()
    i, j, c = coo.row, coo.col, coo.data.astype(float)
    r_i = is_r[i]
    qR = np.where(r_i, (sR[j] - c) / (mR - m[i]), sR[j] / mR)
    qD = np.where(r_i, sD[j] / mD, (sD[j] - c) / (mD - m[i]))
    denom = qR + qD
    rho = np.divide(qR, denom, out=np.full_like(denom, 0.5), where=denom > 0)
    return np.bincount(i, weights=c / m[i] * (rho - 0.5), minlength=mat.shape[0])

def Design(meta, controls):
    """Party label first, then controls; categorical controls enter as dummies."""
    cols = [meta.is_r.to_numpy(float), np.ones(len(meta))]
    for c in controls:
        if c in ("state", "stratum"):
            d = pd.get_dummies(meta[c], drop_first=True).to_numpy(float)
            cols.extend(d.T)
        else:
            cols.append(meta[f"z_{c}"].to_numpy(float))
    return np.column_stack(cols)

def PartyGap(s, meta, controls):
    """Half the adjusted R-D gap in slant, i.e. adjusted pi - 0.5."""
    return np.linalg.lstsq(Design(meta, controls), s, rcond=None)[0][0] / 2

def AdjustedPi(reps, df, rng):
    years = sorted(df.label.unique())
    ids = df.leaid.unique()
    def Compute(keep_ids):
        out = {}
        for y in years:
            sel = np.flatnonzero((df.label == y) & df.leaid.isin(keep_ids))
            meta = df.iloc[sel]
            is_r = meta.is_r.to_numpy(bool)
            s = np.mean([Slant(r[sel], is_r) for r in reps], axis=0)
            for name, controls in CONTROLS.items():
                out[(name, y)] = PartyGap(s, meta, controls)
        return pd.Series(out)
    point = Compute(ids)
    check_y = years[0]
    sel = np.flatnonzero(df.label == check_y)
    print(f"Check: {check_y} pi from LeaveOutPi {np.mean([LeaveOutPi(r[sel], df.iloc[sel].is_r.to_numpy(bool))[0] for r in reps]):.5f}, from slant {0.5 + point[('party only', check_y)]:.5f}")
    jack = pd.DataFrame([Compute(rng.choice(ids, int(JACK_KEEP * len(ids)), replace=False)) for _ in range(N_JACK)])
    scale = JACK_KEEP / (1 - JACK_KEEP)
    rows = []
    for name in CONTROLS:
        series = {y: point[(name, y)] for y in years}
        base = np.mean([series[y] for y in BASE_YEARS])
        jbase = jack[[(name, y) for y in BASE_YEARS]].mean(axis=1)
        row = {"spec": name, **{str(y): series[y] * 1e4 for y in years}, "n_districts": len(ids)}
        for y in years:
            row[f"se_{y}"] = np.sqrt(scale * jack[(name, y)].var(ddof=1)) * 1e4
        for y in (2021, 2023):
            row[f"d{y}"] = (series[y] - base) * 1e4
            row[f"se_d{y}"] = np.sqrt(scale * (jack[(name, y)] - jbase).var(ddof=1)) * 1e4
        rows.append(row)
    out = pd.DataFrame(rows)
    for y in (2021, 2023):
        out[f"d{y}_vs_party_only"] = out[f"d{y}"] / out.loc[0, f"d{y}"]
    print(out.round(2).T.to_string())
    return out

def PandemicTerms(reps, df, vocab):
    """Terms that rose most from 2018-19 to 2020-21 (national token share, enough usage), a data-driven pandemic vocabulary."""
    pre = np.flatnonzero(df.label.isin(BASE_YEARS))
    cov = np.flatnonzero(df.label.isin([2020, 2021]))
    a, b = np.asarray(reps[0][pre].sum(0)).ravel(), np.asarray(reps[0][cov].sum(0)).ravel()
    lift = np.log((b + 5) / b.sum()) - np.log((a + 5) / a.sum())
    lift[(a + b) < MIN_TERM_TOKENS] = -np.inf
    top = np.argsort(-lift)[:N_PANDEMIC_TERMS]
    print("Top risers:", ", ".join(vocab[top[:50]]))
    return sorted(vocab[top])

def WithoutPandemicVocabulary(reps, df, vocab, rng):
    risers = PandemicTerms(reps, df, vocab)
    sets = {"all national vocabulary": set(), "drop COVID and closure word list": COVID_TERMS, f"drop {N_PANDEMIC_TERMS} biggest 2020-21 risers": risers, "drop both": COVID_TERMS | set(risers)}
    years = sorted(df.label.unique())
    ids = df.leaid.unique()
    rows = []
    for name, drop in sets.items():
        keep = np.flatnonzero(~np.isin(vocab, list(drop)))
        sub = [r[:, keep] for r in reps]
        def Compute(keep_ids):
            res = {}
            for y in years:
                sel = np.flatnonzero((df.label == y) & df.leaid.isin(keep_ids))
                res[y] = np.mean([LeaveOutPi(r[sel], df.iloc[sel].is_r.to_numpy(bool))[0] for r in sub]) - 0.5
            return pd.Series(res)
        point = Compute(ids)
        jack = pd.DataFrame([Compute(rng.choice(ids, int(JACK_KEEP * len(ids)), replace=False)) for _ in range(N_JACK // 2)])
        scale = JACK_KEEP / (1 - JACK_KEEP)
        row = {"spec": name, "n_dropped": len(drop) if drop else 0, **{str(y): point[y] * 1e4 for y in years}}
        for y in (2021, 2023):
            row[f"d{y}"] = (point[y] - point[BASE_YEARS].mean()) * 1e4
            row[f"se_d{y}"] = np.sqrt(scale * (jack[y] - jack[BASE_YEARS].mean(axis=1)).var(ddof=1)) * 1e4
        rows.append(row)
    out = pd.DataFrame(rows)
    print(out.round(2).T.to_string())
    return out

def OtherLabels(reps, df, rng):
    """Leave-out pi with labels other than party, net of label shuffles: how well does language separate high from low closure, etc.?"""
    static = df.drop_duplicates("leaid").copy()
    labels = {"party": static.is_r.to_numpy(bool)}
    for c, name in [("closure", "closure"), ("ach", "achievement"), ("ses", "SES"), ("ach_net_ses", "achievement net of SES")]:
        labels[f"{name} (above national median)"] = static[c].to_numpy() > static[c].median()
        labels[f"{name} (above state median)"] = (static[c] > static.groupby("state")[c].transform("median")).to_numpy()
    years = sorted(df.label.unique())
    rows = []
    ids = static.leaid.to_numpy()
    for name, lab in labels.items():
        by_id = dict(zip(ids, lab))
        def Compute(keep_ids):
            res = {}
            for y in years:
                sel = np.flatnonzero((df.label == y) & df.leaid.isin(keep_ids))
                res[y] = np.mean([LeaveOutPi(r[sel], df.iloc[sel].leaid.map(by_id).to_numpy(bool))[0] for r in reps]) - 0.5
            return pd.Series(res)
        point = Compute(ids)
        states = dict(zip(ids, static.state))
        null = {y: [] for y in years}
        for k in range(10):
            for y in years:
                sel = np.flatnonzero(df.label == y)
                meta = df.iloc[sel]
                shuffled = PermuteWithin(meta.leaid.map(by_id).to_numpy(bool), meta.leaid.map(states).to_numpy(), rng)
                null[y].append(LeaveOutPi(reps[k % len(reps)][sel], shuffled)[0] - 0.5)
        jack = pd.DataFrame([Compute(rng.choice(ids, int(JACK_KEEP * len(ids)), replace=False)) for _ in range(N_JACK // 4)])
        scale = JACK_KEEP / (1 - JACK_KEEP)
        net = pd.Series({y: point[y] - np.mean(null[y]) for y in years})
        row = {"label": name, **{f"pi_{y}": point[y] * 1e4 for y in years}, **{f"net_{y}": net[y] * 1e4 for y in years}}
        for y in (2021, 2023):
            row[f"d{y}"] = (point[y] - point[BASE_YEARS].mean()) * 1e4
            row[f"se_d{y}"] = np.sqrt(scale * (jack[y] - jack[BASE_YEARS].mean(axis=1)).var(ddof=1)) * 1e4
            row[f"net_d{y}"] = (net[y] - net[BASE_YEARS].mean()) * 1e4
        rows.append(row)
    out = pd.DataFrame(rows)
    print(out.round(2).T.to_string())
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unbalanced", action="store_true", help="every district-year with enough words instead of the balanced panel")
    args = parser.parse_args()
    Main("_unbalanced" if args.unbalanced else "", not args.unbalanced)
