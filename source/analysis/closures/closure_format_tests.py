"""Is the remote-meeting result real, or would any partisan-loaded word list do the same?

  1. Placebo: control for random sets of 7 terms drawn from the biggest 2020-21 risers, matched on their R-D gap and frequency in 2021,
     instead of FORMAT_STRICT. Reports where the real reduction in the rise falls. COVID health words are shown as a named comparison.
  2. Within district x year: a meeting is `remote` if it has >= REMOTE_MIN FORMAT_STRICT words per transcript. Each transcript's slant uses
     that year's word weights with its own district left out, and FE compare meetings of the same district in the same year.
     Then implied gap = delta x (R-D gap in remote share) / 2, for comparison with the observed rise in pi.
Writes output/tables/closures_screened/format_*.csv. Run from the repo root.
"""
import numpy as np
import pandas as pd
import pyfixest as pf
import scipy.sparse as sp

from closure_data import GLOBALS, INDIR_DERIVED, LoadCovariates, OUTDIR_TABLES
from closure_gst import BASE_YEARS, LoadSample, MAX_TOP_STATE_SHARE, PandemicTerms, PartyGap
from closure_mechanism import FORMAT_STRICT, SlantByYear

N_PLACEBO = 500
MEETING_ONLY = ["unmute", "muted", "mute", "webinar", "webex"]   # words that describe the meeting and rarely remote school
COVID_HEALTH = ["mask", "masks", "vaccine", "vaccines", "quarantine", "distancing", "cdc"]
REMOTE_MIN = 3
SEED = 2361

def Main():
    OUTDIR_TABLES.mkdir(parents=True, exist_ok=True)
    reps, df, vocab, rng = LoadSample(True)
    Placebo(reps, df, vocab)
    WithinDistrict(FORMAT_STRICT, "")
    WithinDistrict(MEETING_ONLY, "_meeting_only")
    return

def RiseWithControl(s, years, values):
    """Rise in the party gap (x 1e4) from 2018-19 to 2021 and to 2023, controlling for standardized `values` by year."""
    gaps = {}
    for y in years:
        slant, meta = s[y]
        x = values[y]
        z = (x - x.mean()) / x.std() if x.std() > 0 else x * 0
        b = np.linalg.lstsq(np.column_stack([meta.is_r.to_numpy(float), np.ones(len(meta)), z]), slant, rcond=None)[0][0]
        gaps[y] = b / 2 * 1e4
    base = np.mean([gaps[y] for y in BASE_YEARS])
    return gaps[2021] - base, gaps[2023] - base

def Rates(mat, df, years, idx):
    m = np.asarray(mat.sum(1)).ravel()
    r = np.asarray(mat[:, idx].sum(1)).ravel() / m * 1e4
    return {y: r[np.flatnonzero(df.label == y)] for y in years}

def Placebo(reps, df, vocab):
    years = sorted(df.label.unique())
    s = SlantByYear(reps, df, years, df.leaid.unique())
    pos = {t: i for i, t in enumerate(vocab)}
    risers = [t for t in PandemicTerms(reps, df, vocab) if t not in FORMAT_STRICT]
    mat = reps[0]
    m = np.asarray(mat.sum(1)).ravel()
    sel = np.flatnonzero(df.label == 2021)
    is_r = df.iloc[sel].is_r.to_numpy(bool)
    def Gap(t):
        r = np.asarray(mat[sel, pos[t]].todense()).ravel() / m[sel] * 1e4
        return r[~is_r].mean() - r[is_r].mean()
    gaps = pd.Series({t: Gap(t) for t in risers})
    real_gap = sum(Gap(t) for t in FORMAT_STRICT if t in pos)
    # Words leaning the same way as the format list: only these could absorb the gap in the same direction.
    pool = gaps[gaps > 0].index.to_numpy()
    print(f"Riser pool: {len(risers)}; D-leaning in 2021: {len(pool)}. FORMAT_STRICT D-R gap {real_gap:.2f} per 10k; pool mean sum of 7: {gaps[pool].mean() * 7:.2f}")
    named = {"none": None, "format strict": FORMAT_STRICT, "meeting only": MEETING_ONLY, "covid health": COVID_HEALTH}
    rows = []
    for name, terms in named.items():
        if terms is None:
            d21, d23 = RiseWithControl(s, years, {y: np.zeros(s[y][0].shape) for y in years})
        else:
            d21, d23 = RiseWithControl(s, years, Rates(mat, df, years, [pos[t] for t in terms if t in pos]))
        rows.append({"set": name, "d2021": d21, "d2023": d23})
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_PLACEBO):
        terms = rng.choice(pool, 7, replace=False)
        d21, d23 = RiseWithControl(s, years, Rates(mat, df, years, [pos[t] for t in terms]))
        draws.append({"gap": gaps[terms].sum(), "d2021": d21, "d2023": d23})
    draws = pd.DataFrame(draws)
    draws.to_csv(OUTDIR_TABLES / "format_placebo_draws.csv", index=False)
    out = pd.DataFrame(rows)
    real = out.set_index("set").loc["format strict"]
    for c in ("d2021", "d2023"):
        out[f"share_placebo_le_{c}"] = [(draws[c] <= v).mean() for v in out[c]]
    close = draws[(draws.gap > real_gap * 0.5) & (draws.gap < real_gap * 2)]
    print(out.round(3).to_string(index=False))
    print("Placebo d2021 pctiles 1/5/50/95:", draws.d2021.quantile([.01, .05, .5, .95]).round(2).to_dict())
    print("Placebo d2023 pctiles 1/5/50/95:", draws.d2023.quantile([.01, .05, .5, .95]).round(2).to_dict())
    print(f"Placebos with similar total gap (n={len(close)}): median d2021 {close.d2021.median():.2f}, share <= real {(close.d2021 <= real.d2021).mean():.3f}; share d2023 <= real {(close.d2023 <= real.d2023).mean():.3f}")
    out.to_csv(OUTDIR_TABLES / "format_placebo_named.csv", index=False)

def WithinDistrict(terms, suffix):
    bow = INDIR_DERIVED / "bow"
    dtm = sp.load_npz(bow / "dtm.npz").tocsr()
    vocab = pd.read_parquet(bow / "vocab.parquet").term.to_numpy()
    docs = pd.read_parquet(bow / "docs.parquet")
    dq_docs = pd.read_parquet(INDIR_DERIVED / "bow_dq" / "docs.parquet", columns=["file", "quarter"])
    docs = docs.merge(dq_docs, on="file", how="left", suffixes=("", "_dq"))
    docs["leaid"] = docs.leaid.astype(str).str.zfill(7)
    docs["year"] = docs.quarter.str[:4].astype(int)
    cov = LoadCovariates()[["leaid", "gop", "state", "closure"]]
    docs = docs.merge(cov, on="leaid", how="left")
    keep = docs.gop.notna() & (docs.gop != 0.5)
    docs, dtm = docs[keep].reset_index(drop=True), dtm[np.flatnonzero(keep)]
    docs["is_r"] = docs.gop > 0.5
    dq_vocab = pd.read_parquet(INDIR_DERIVED / "bow_dq" / "vocab.parquet").term.to_numpy()
    conc = pd.read_parquet(f"output/tables/{GLOBALS['BOW_OUTPUT_TAG']}/term_state_concentration.parquet")
    national = set(dq_vocab[(conc.top_state_share <= MAX_TOP_STATE_SHARE).to_numpy()])
    format_idx = [np.flatnonzero(vocab == t)[0] for t in terms if t in set(vocab)]
    docs["n_format"] = np.asarray(dtm[:, format_idx].sum(1)).ravel()
    # Slant vocabulary excludes the format words so `remote` is not mechanically in the outcome.
    use = np.flatnonzero(np.isin(vocab, list(national - set(FORMAT_STRICT))))
    mat = dtm[:, use].tocsr()
    docs["m"] = np.asarray(mat.sum(1)).ravel()
    docs["slant"] = np.nan
    for y, g in docs.groupby("year"):
        docs.loc[g.index, "slant"] = YearSlant(mat[g.index.to_numpy()], g)
    docs["remote"] = (docs.n_format >= REMOTE_MIN).astype(float)
    docs["remote_any"] = (docs.n_format >= 1).astype(float)
    docs["remote_hi"] = (docs.n_format >= 10).astype(float)
    docs["s"] = docs.slant * 1e4
    print("\nShare of meetings flagged remote by year and party\n", docs.groupby(["year", "is_r"]).remote.mean().unstack().round(3).to_string())
    print("Same-district-year contrast available in", docs.groupby(["leaid", "year"]).remote.agg(lambda x: 0 < x.mean() < 1).groupby(level=1).mean().round(3).to_dict())
    rows = []
    obs = {}
    for y in sorted(docs.year.unique()):
        g = docs[docs.year == y]
        obs[y] = (g[g.is_r].groupby("leaid").s.mean().mean() - g[~g.is_r].groupby("leaid").s.mean().mean()) / 2
        for flag in ["remote", "remote_hi"]:
            if g[flag].sum() < 50:
                continue
            m = pf.feols(f"s ~ {flag} | leaid", g, vcov={"CRV1": "leaid"})
            t = m.tidy().loc[flag]
            share = g.groupby(["leaid", "is_r"])[flag].mean().reset_index()
            d_share = share[share.is_r][flag].mean() - share[~share.is_r][flag].mean()
            rows.append({"year": y, "flag": flag, "delta": t["Estimate"], "se": t["Std. Error"], "n": m._N, "share_R": share[share.is_r][flag].mean(), "share_D": share[~share.is_r][flag].mean(),
                         "implied_gap_contribution": t["Estimate"] * d_share / 2, "observed_half_gap": obs[y]})
            m2 = pf.feols(f"s ~ {flag} + {flag}:is_r_num | leaid", g.assign(is_r_num=g.is_r.astype(float)), vcov={"CRV1": "leaid"})
            ti = m2.tidy()
            rows[-1].update({"delta_D": ti.loc[flag, "Estimate"], "delta_R_minus_D": ti.iloc[1]["Estimate"], "se_R_minus_D": ti.iloc[1]["Std. Error"]})
    out = pd.DataFrame(rows)
    out.to_csv(OUTDIR_TABLES / f"format_within_district{suffix}.csv", index=False)
    print(out.round(3).to_string(index=False))
    if not suffix:
        TopWords(dtm, vocab, docs, national)
    else:
        print("\nFixed-term version on the same flag: share with any flagged meeting by party-year above")

def YearSlant(mat, g):
    """Leave-district-out slant per transcript: word weights from all other districts that year, with equal weight per party's tokens."""
    is_r = g.is_r.to_numpy(bool)
    sR, sD = np.asarray(mat[is_r].sum(0)).ravel().astype(float), np.asarray(mat[~is_r].sum(0)).ravel().astype(float)
    mR, mD = sR.sum(), sD.sum()
    codes, _ = pd.factorize(g.leaid)
    group = sp.csr_matrix((np.ones(len(g)), (codes, np.arange(len(g)))))
    district = (group @ mat).tocsr()
    m_d = np.asarray(district.sum(1)).ravel()
    coo = mat.tocoo()
    i, j, c = coo.row, coo.col, coo.data.astype(float)
    own = np.asarray(district[codes[i], j]).ravel().astype(float)
    r_i = is_r[i]
    qR = np.where(r_i, (sR[j] - own) / (mR - m_d[codes[i]]), sR[j] / mR)
    qD = np.where(r_i, sD[j] / mD, (sD[j] - own) / (mD - m_d[codes[i]]))
    denom = qR + qD
    rho = np.divide(qR, denom, out=np.full_like(denom, 0.5), where=denom > 0)
    m = np.asarray(mat.sum(1)).ravel()
    return np.bincount(i, weights=c / m[i] * (rho - 0.5), minlength=mat.shape[0])

def TopWords(dtm, vocab, docs, national):
    """Words most over- and under-represented in remote meetings compared with in-person meetings of the same district and year (2021)."""
    g = docs[docs.year == 2021]
    both = g.groupby("leaid").remote.transform(lambda x: 0 < x.mean() < 1).astype(bool)
    g = g[both]
    mat = dtm[g.index.to_numpy()].tocsr()
    use = np.flatnonzero(np.isin(vocab, list(national)))
    freq = sp.diags(1 / np.maximum(np.asarray(mat.sum(1)).ravel(), 1)) @ mat
    codes, _ = pd.factorize(g.leaid)
    diffs, weights = [], []
    rem = g.remote.to_numpy(bool)
    n = codes.max() + 1
    Sum = lambda mask: sp.csr_matrix((np.ones(mask.sum()), (codes[mask], np.flatnonzero(mask))), shape=(n, len(g))) @ freq
    a, b = Sum(rem), Sum(~rem)
    na, nb = np.bincount(codes[rem], minlength=n), np.bincount(codes[~rem], minlength=n)
    da = sp.diags(1 / np.maximum(na, 1)) @ a
    db = sp.diags(1 / np.maximum(nb, 1)) @ b
    d = np.asarray((da - db).mean(0)).ravel()[use] * 1e4
    base = np.asarray(db.mean(0)).ravel()[use] * 1e4
    t = pd.DataFrame({"term": vocab[use], "diff_per_10k": d, "in_person_rate": base})
    t = t[t.in_person_rate + t.diff_per_10k > 2]
    t["ratio"] = (t.in_person_rate + t.diff_per_10k + .5) / (t.in_person_rate + .5)
    t.sort_values("diff_per_10k").to_csv(OUTDIR_TABLES / "format_top_words.csv", index=False)
    print("\n2021: more in remote meetings:", ", ".join(t.sort_values("diff_per_10k").term.tail(50)[::-1]))
    print("2021: more in in-person meetings:", ", ".join(t.sort_values("diff_per_10k").term.head(50)))
    print("districts with both types:", n)

if __name__ == "__main__":
    Main()
