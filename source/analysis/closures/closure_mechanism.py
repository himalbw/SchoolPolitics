"""What is behind the 2021 partisanship peak? Follow-ups to closure_gst.py on the same panel.

  1. Paired test: how much of the 2018-19 -> 2021 rise does each control remove, and how much of that it already removes before 2020?
  2. Closure slope in slant by year (party held fixed): does closure's link to language appear after closures, or was it always there?
  3. Fixed word weights: rho from 2018-19 only, so pandemic-era new partisan words carry no weight. Same words more divided, or new words?
  4. Meeting format: does remote-meeting vocabulary line up with party and closures, and does it account for the peak?
     FORMAT_STRICT was picked for naming the platform; FORMAT_BROAD adds words seen among the top movers, so it is exploratory.
Writes output/tables/closures_screened/mech_*.csv. Run from the repo root.
"""
import numpy as np
import pandas as pd

from closure_data import OUTDIR_TABLES
from closure_gst import BASE_YEARS, Design, JACK_KEEP, LeaveOutPi, LoadSample, N_JACK, PartyGap, Slant

FORMAT_STRICT = ["zoom", "unmute", "muted", "mute", "webinar", "webex", "chat"]
FORMAT_BROAD = FORMAT_STRICT + ["screen", "slide", "slides", "presentation", "comment", "comments", "question", "questions"]
SCALE = JACK_KEEP / (1 - JACK_KEEP)

def Main():
    OUTDIR_TABLES.mkdir(parents=True, exist_ok=True)
    reps, df, vocab, rng = LoadSample(True)
    years = sorted(df.label.unique())
    ids = df.leaid.unique()
    draws = [rng.choice(ids, int(JACK_KEEP * len(ids)), replace=False) for _ in range(N_JACK)]
    PairedReduction(reps, df, years, ids, draws).to_csv(OUTDIR_TABLES / "mech_paired_reduction.csv", index=False)
    ClosureSlope(reps, df, years, ids, draws).to_csv(OUTDIR_TABLES / "mech_closure_slope.csv", index=False)
    FixedWeights(reps, df, years, ids, draws).to_csv(OUTDIR_TABLES / "mech_fixed_weights.csv", index=False)
    Format(reps, df, vocab, years, ids, draws)
    return

def Summarise(point, jack, name, years):
    """Row with the series and changes from 2018-19 to 2021 and 2023, jackknife SEs."""
    base, jbase = point[BASE_YEARS].mean(), jack[BASE_YEARS].mean(axis=1)
    row = {"spec": name, **{str(y): point[y] * 1e4 for y in years}, **{f"se_{y}": np.sqrt(SCALE * jack[y].var(ddof=1)) * 1e4 for y in years}}
    for y in (2021, 2023):
        row[f"d{y}"], row[f"se_d{y}"] = (point[y] - base) * 1e4, np.sqrt(SCALE * (jack[y] - jbase).var(ddof=1)) * 1e4
    return row

def SlantByYear(reps, df, years, keep):
    out = {}
    for y in years:
        sel = np.flatnonzero((df.label == y) & df.leaid.isin(keep))
        meta = df.iloc[sel]
        out[y] = (np.mean([Slant(r[sel], meta.is_r.to_numpy(bool)) for r in reps], axis=0), meta)
    return out

def PairedReduction(reps, df, years, ids, draws):
    """Gap removed by each control, by year. A control that removes as much before 2020 as after explains level, not the rise."""
    specs = {"closure": ["closure"], "achievement + SES": ["ach", "ses"], "closure | state": ["closure", "state"], "locale x size": ["stratum"], "closure | locale x size": ["closure", "stratum"]}
    bases = {"closure": [], "achievement + SES": [], "closure | state": ["state"], "locale x size": [], "closure | locale x size": ["stratum"]}
    def Compute(keep):
        s = SlantByYear(reps, df, years, keep)
        return {name: pd.Series({y: PartyGap(s[y][0], s[y][1], bases[name]) - PartyGap(s[y][0], s[y][1], c) for y in years}) for name, c in specs.items()}
    point = Compute(ids)
    jacks = [Compute(d) for d in draws]
    rows = [Summarise(point[n], pd.DataFrame([j[n] for j in jacks]), n, years) for n in specs]
    out = pd.DataFrame(rows)
    print("\nGAP REMOVED by control (pi x 1e4)\n", out.round(2).T.to_string())
    return out

def ClosureSlope(reps, df, years, ids, draws):
    """Coefficient on z_closure in slant, party held fixed. Units are slant x 1e4 per SD of closure."""
    specs = {"party": ["closure"], "party + state": ["closure", "state"], "party + locale x size": ["closure", "stratum"]}
    def Compute(keep):
        s = SlantByYear(reps, df, years, keep)
        return {n: pd.Series({y: np.linalg.lstsq(Design(s[y][1], c), s[y][0], rcond=None)[0][2] for y in years}) for n, c in specs.items()}
    point = Compute(ids)
    jacks = [Compute(d) for d in draws]
    out = pd.DataFrame([Summarise(point[n], pd.DataFrame([j[n] for j in jacks]), n, years) for n in specs])
    print("\nCLOSURE SLOPE\n", out.round(2).T.to_string())
    return out

def FixedSlant(mat, is_r_year, pre, is_r_pre, row_in_pre):
    """Slant of `mat` rows using word weights from `pre` (pooled 2018-19), leaving out each district's own pre-period counts."""
    m_pre = np.asarray(pre.sum(1)).ravel()
    sR, sD = np.asarray(pre[is_r_pre].sum(0)).ravel(), np.asarray(pre[~is_r_pre].sum(0)).ravel()
    mR, mD = m_pre[is_r_pre].sum(), m_pre[~is_r_pre].sum()
    m = np.asarray(mat.sum(1)).ravel()
    coo = mat.tocoo()
    i, j, c = coo.row, coo.col, coo.data.astype(float)
    k = row_in_pre[i]
    own = np.asarray(pre[k, j]).ravel().astype(float)
    r_i = is_r_year[i]
    qR = np.where(r_i, (sR[j] - own) / (mR - m_pre[k]), sR[j] / mR)
    qD = np.where(r_i, sD[j] / mD, (sD[j] - own) / (mD - m_pre[k]))
    denom = qR + qD
    rho = np.divide(qR, denom, out=np.full_like(denom, 0.5), where=denom > 0)
    return np.bincount(i, weights=c / m[i] * (rho - 0.5), minlength=mat.shape[0])

def FixedWeights(reps, df, years, ids, draws):
    """Half the R-D gap in slant with 2018-19 weights, by year. 2018-19 itself is only comparable under leave-out, which this uses."""
    def Compute(keep):
        res = {}
        for r in reps[:1]:
            d = df[df.leaid.isin(keep)]
            base_idx = d.index[d.label.isin(BASE_YEARS)].to_numpy()
            names, codes = np.unique(df.leaid.to_numpy()[base_idx], return_inverse=True)
            import scipy.sparse as sp
            group = sp.csr_matrix((np.ones(len(base_idx)), (codes, np.arange(len(base_idx)))), shape=(len(names), len(base_idx)))
            pre = (group @ r[base_idx]).tocsr()
            party = df.drop_duplicates("leaid").set_index("leaid").is_r
            is_r_pre = party.reindex(names).to_numpy(bool)
            pos = {n: k for k, n in enumerate(names)}
            for y in years:
                sel = d.index[d.label == y].to_numpy()
                meta = df.iloc[sel]
                is_r = meta.is_r.to_numpy(bool)
                s = FixedSlant(r[sel], is_r, pre, is_r_pre, meta.leaid.map(pos).to_numpy())
                res[y] = (s[is_r].mean() - s[~is_r].mean()) / 2
        return pd.Series(res)
    point = Compute(ids)
    jack = pd.DataFrame([Compute(d) for d in draws[:100]])
    out = pd.DataFrame([Summarise(point, jack, "fixed 2018-19 weights", years)])
    print("\nFIXED WEIGHTS\n", out.round(2).T.to_string())
    return out

def Format(reps, df, vocab, years, ids, draws):
    """Remote-meeting vocabulary by party and closure, and the peak with that vocabulary controlled or removed."""
    pos = {t: i for i, t in enumerate(vocab)}
    d = df.copy()
    mat = reps[0]
    m = np.asarray(mat.sum(1)).ravel()
    for name, terms in {"strict": FORMAT_STRICT, "broad": FORMAT_BROAD}.items():
        idx = [pos[t] for t in terms if t in pos]
        print(name, "terms found:", [t for t in terms if t in pos])
        d[f"format_{name}"] = np.asarray(mat[:, idx].sum(1)).ravel() / m * 1e4
        d[f"z_format_{name}"] = d.groupby("label")[f"format_{name}"].transform(lambda x: (x - x.mean()) / x.std())
    tab = d.groupby(["label", "is_r"]).format_strict.mean().unstack().rename(columns={True: "R", False: "D"})
    tab["corr_closure"] = d.groupby("label").apply(lambda g: g.format_strict.corr(g.closure), include_groups=False)
    tab["corr_gop"] = d.groupby("label").apply(lambda g: g.format_strict.corr(g.gop), include_groups=False)
    tab.to_csv(OUTDIR_TABLES / "mech_format_by_party.csv")
    print("\nFORMAT words per 10k, by party\n", tab.round(2).to_string())
    specs = {"party only": [], "+ format strict": ["format_strict"], "+ format broad": ["format_broad"], "+ closure": ["closure"], "+ closure + format strict": ["closure", "format_strict"], "+ closure + format broad": ["closure", "format_broad"]}
    def Compute(keep):
        s = SlantByYear(reps, d, years, keep)
        return {n: pd.Series({y: PartyGap(s[y][0], s[y][1], c) for y in years}) for n, c in specs.items()}
    point = Compute(ids)
    jacks = [Compute(x) for x in draws]
    rows = [Summarise(point[n], pd.DataFrame([j[n] for j in jacks]), n, years) for n in specs]
    drops = {"drop format strict": FORMAT_STRICT, "drop format broad": FORMAT_BROAD}
    for name, terms in drops.items():
        keep_terms = np.flatnonzero(~np.isin(vocab, terms))
        sub = [r[:, keep_terms] for r in reps]
        def ComputeDrop(keep):
            return pd.Series({y: np.mean([LeaveOutPi(r[np.flatnonzero((df.label == y) & df.leaid.isin(keep))], df[(df.label == y) & df.leaid.isin(keep)].is_r.to_numpy(bool))[0] for r in sub]) - 0.5 for y in years})
        rows.append(Summarise(ComputeDrop(ids), pd.DataFrame([ComputeDrop(x) for x in draws[:100]]), name, years))
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR_TABLES / "mech_format_gap.csv", index=False)
    print(out.round(2).T.to_string())

if __name__ == "__main__":
    Main()
