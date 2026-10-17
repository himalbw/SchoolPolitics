"""Gentzkow-Shapiro-Taddy (2019) leave-out partisanship of language, by year and quarter.

Speaker = district, party = R if the district's county voted > 50% GOP (two-party) in 2020, else D. For each period:
    q_i       = word frequencies of district i
    rho_j^-i  = q^R_j / (q^R_j + q^D_j), with district i's own counts left out of its party's frequencies
    pi        = 1/2 * mean_{i in R} q_i . rho^-i  +  1/2 * mean_{i in D} q_i . (1 - rho^-i)
pi is the expected posterior that a neutral listener assigns to the right party after hearing ONE word:
0.5 = no partisanship in word choice. The leave-out removes the finite-sample bias of the plug-in estimate.
Placebos (same counts, shuffled labels): `random` (labels shuffled across districts; should be ~0.5, checks the estimator)
and `within_state` (labels shuffled inside each state; keeps state-specific vocabulary, so it measures how much of pi is
just geography). pi minus the within-state placebo is the part of polarization that is not explained by which state you are in.
Writes output/tables/gst_{year,quarter}.csv, gst_year_balanced_thinned.csv and gst_term_excess_year*.parquet. Run from the repo root.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from bow_data import AddStrata, CollapseTo, LoadDQ, OUTDIR_TABLES, TermStateConcentration, WithAnchor
from lexicon import LEXICON

MIN_TOKENS = {"year": 5000, "quarter": 2000}   # a district-period needs this many (post-cut) words to enter
N_PERM = {"year": 10, "quarter": 5}
MIN_PER_PARTY = 5
THIN_TO = 10000   # words per district-year in the balanced/thinned spec
MAX_TOP_STATE_SHARE = 0.3   # 'national vocabulary' = terms with <= 30% of usage in their biggest state

def Main():
    OUTDIR_TABLES.mkdir(parents=True, exist_ok=True)
    dtm, vocab, units = LoadDQ()
    dtm, units = WithAnchor(dtm, units)
    units = AddStrata(units)
    # Spec 1: every district-period with enough words. Composition changes a lot in 2020 (district count x2-3).
    # Spec 2: balanced panel (same districts every year) with each district-year thinned to the same number of words,
    #         so pre/post differences are not driven by who is in the sample or how many meetings they have.
    # Spec 3: spec 2 with state-specific vocabulary removed (terms whose biggest state supplies > MAX_TOP_STATE_SHARE of their
    #         usage: place names, state institutions), so what is left is vocabulary shared across the country.
    df_conc = TermStateConcentration(dtm, units, vocab)
    df_conc.to_parquet(OUTDIR_TABLES / "term_state_concentration.parquet")
    national = np.flatnonzero((df_conc.top_state_share <= MAX_TOP_STATE_SHARE).to_numpy())
    specs = [("all", "year", dtm, vocab, {}), ("all", "quarter", dtm, vocab, {}),
             ("balanced_thinned", "year", dtm, vocab, {"balanced": True, "thin_to": THIN_TO}),
             ("balanced_thinned_national", "year", dtm[:, national], vocab[national], {"balanced": True, "thin_to": THIN_TO}),
             # Spec 4: spec 3 but R and D districts are only compared within the same locale x size stratum
             ("balanced_thinned_national_stratified", "year", dtm[:, national], vocab[national], {"balanced": True, "thin_to": THIN_TO, "stratified": True})]
    for name, level, mat, terms, kwargs in specs:
        df_series, df_excess = PolarizationSeries(mat, units, terms, level, **kwargs)
        suffix = "" if name == "all" else f"_{name}"
        df_series.to_csv(OUTDIR_TABLES / f"gst_{level}{suffix}.csv", index=False)
        print(f"\n== GST pi, {name}, by {level} ==")
        print(df_series.round(4).to_string(index=False))
        if level == "year":
            df_excess.to_parquet(OUTDIR_TABLES / f"gst_term_excess_year{suffix}.parquet")
    return

def LeaveOutPi(counts, is_r):
    """GST leave-out pi for one period. counts: csr (districts x terms). Returns (pi, excess contribution per term)."""
    m = np.asarray(counts.sum(1)).ravel()
    n_r, n_d = is_r.sum(), (~is_r).sum()
    sR = np.asarray(counts[is_r].sum(0)).ravel()
    sD = np.asarray(counts[~is_r].sum(0)).ravel()
    mR, mD = m[is_r].sum(), m[~is_r].sum()
    coo = counts.tocoo()
    i, j, c = coo.row, coo.col, coo.data.astype(float)
    r_i = is_r[i]
    qR = np.where(r_i, (sR[j] - c) / (mR - m[i]), sR[j] / mR)      # R frequency of word j, leaving out i if i is R
    qD = np.where(r_i, sD[j] / mD, (sD[j] - c) / (mD - m[i]))
    denom = qR + qD
    rho = np.divide(qR, denom, out=np.full_like(denom, 0.5), where=denom > 0)   # word used by i alone -> neutral
    correct = np.where(r_i, rho, 1 - rho)
    w = np.where(r_i, 0.5 / n_r, 0.5 / n_d) * c / m[i]                # party weight x q_ij
    V = counts.shape[1]
    contrib = np.bincount(j, weights=w * correct, minlength=V)
    neutral = np.bincount(j, weights=w * 0.5, minlength=V)
    return contrib.sum(), contrib - neutral

def Thin(counts, rng, target):
    """Binomial-thin each row to about `target` words (rows must have >= target)."""
    m = np.asarray(counts.sum(1)).ravel()
    coo = counts.tocoo()
    kept = rng.binomial(coo.data.astype(np.int64), (target / m)[coo.row])
    return sp.csr_matrix((kept.astype(float), (coo.row, coo.col)), shape=counts.shape)

def PolarizationSeries(dtm, units, vocab, level, balanced=False, thin_to=None, n_thin=3, stratified=False, seed=0):
    counts, df, _ = CollapseTo(dtm, units, level)
    rng = np.random.default_rng(seed)
    ok = df.n_tokens >= (thin_to or MIN_TOKENS[level])
    if stratified:
        ok &= df.stratum.notna()
    if balanced:
        n_labels = df.label.nunique()
        in_all = df[ok].groupby("leaid").label.nunique() == n_labels
        ok &= df.leaid.isin(in_all[in_all].index)
        print(f"Balanced panel: {in_all.sum()} districts present in all {n_labels} periods")
    rows_ok = np.flatnonzero(ok)
    counts, df = counts[rows_ok], df.iloc[rows_ok].reset_index(drop=True)
    n_rep = n_thin if thin_to else 1
    reps = [Thin(counts, rng, thin_to) if thin_to else counts for _ in range(n_rep)]
    rows, excess = [], {}
    for label in sorted(df.label.unique()):
        sel = np.flatnonzero(df.label == label)
        meta = df.iloc[sel]
        is_r = meta.is_r.to_numpy(bool)
        if min(is_r.sum(), (~is_r).sum()) < MIN_PER_PARTY:
            continue
        subs = [r[sel] for r in reps]
        stratum = meta.stratum.to_numpy() if stratified else np.zeros(len(meta), dtype=object)
        groups = [np.flatnonzero(stratum == k) for k in np.unique(stratum)]
        res = [PiOverGroups(sub, is_r, groups) for sub in subs]
        pi = np.mean([r[0] for r in res])
        null_random = [PiOverGroups(subs[k % n_rep], PermuteWithin(is_r, stratum, rng), groups)[0] for k in range(N_PERM[level])]
        null_state = [PiOverGroups(subs[k % n_rep], PermuteWithin(is_r, meta.state.to_numpy(), rng), groups)[0] for k in range(N_PERM[level])]
        rows.append({"label": label, "n_districts": len(meta), "n_R": int(is_r.sum()), "n_D": int((~is_r).sum()),
                     "n_tokens": int(meta.n_tokens.sum()), "n_districts_compared": res[0][2], "pi": pi, "pi_thin_sd": np.std([r[0] for r in res]),
                     "random_null_mean": np.mean(null_random), "random_null_sd": np.std(null_random),
                     "within_state_mean": np.mean(null_state), "within_state_sd": np.std(null_state)})
        excess[label] = np.mean([r[1] for r in res], axis=0)
    df_series = pd.DataFrame(rows)
    df_series["pi_minus_within_state"] = df_series.pi - df_series.within_state_mean
    df_series["pi_minus_random_null"] = df_series.pi - df_series.random_null_mean   # use this to compare specs: strata shift the null below 0.5
    df_excess = pd.DataFrame(excess, index=vocab)
    return df_series, df_excess

def PiOverGroups(counts, is_r, groups):
    """Leave-out pi computed separately inside each group (stratum) and averaged with weights = districts compared.
    Groups without >= MIN_PER_PARTY districts of each party are skipped. Returns (pi, excess per term, districts used)."""
    pi_total, excess, used = 0.0, 0.0, 0
    for g in groups:
        if min(is_r[g].sum(), (~is_r[g]).sum()) < MIN_PER_PARTY:
            continue
        p, e = LeaveOutPi(counts[g], is_r[g])
        pi_total, excess, used = pi_total + len(g) * p, excess + len(g) * e, used + len(g)
    return pi_total / used, excess / used, used

def PermuteWithin(is_r, labels, rng):
    out = is_r.copy()
    for s in np.unique(labels):
        idx = np.flatnonzero(labels == s)
        out[idx] = rng.permutation(is_r[idx])
    return out

def PermuteWithinState(is_r, state, rng):
    out = is_r.copy()
    for s in np.unique(state):
        idx = np.flatnonzero(state == s)
        out[idx] = rng.permutation(is_r[idx])
    return out

def CategoryExcess(df_excess):
    """Excess contribution to pi (pi - 0.5 summed) by lexicon category (single terms only), per period."""
    out = {}
    for cat, spec in LEXICON.items():
        terms = [t for t in spec["terms"] if t in df_excess.index]
        out[cat] = df_excess.loc[terms].sum()
    out["ALL_TERMS (pi - 0.5)"] = df_excess.sum()
    return pd.DataFrame(out).T

if __name__ == "__main__":
    Main()
