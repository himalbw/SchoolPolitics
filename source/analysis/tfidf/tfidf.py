"""TF-IDF of school-board language by district x period: top terms, R vs D contrasts, pre-to-post change, pairwise similarity.

Reads  <indir>/{dtm.npz, vocab.parquet, units.parquet}  rows = leaid x quarter, columns = already vocabulary-cut terms
       Datastore/District Covariates/processed/district_static.parquet  partisan anchor, locale and district names
Writes <indir parent>/tfidf/tfidf_district_period.npz, rows.parquet, terms.parquet        sparse matrix + row/column labels
       output/tables/tfidf_<tag>/top_terms_period.csv            (a) top terms per period, by mean district TF-IDF
                                 top_terms_party_period.csv      (b) top terms per party x period
                                 top_terms_party_overlap.csv     how many of the top R and top D terms are the same
                                 distinctive_R_vs_D.csv          (c) Welch t of the R-D gap, by period
                                 changed_pre_post.csv            (c) Welch t of post-pre (+ paired t in the balanced panel)
                                 partisan_movers_pre_post.csv    terms whose R-D standardised gap moved most
                                 example_districts.csv           (d) top terms of a few districts
                                 contamination.csv               how much of each table is place or state-specific vocabulary
                                 unstable_terms.csv              terms dropped from the national_stable pairwise vocabulary
                                 pairwise_cosine.csv             within- vs across-party cosine by period, with placebos
                                 pairwise_cosine_changes.csv     change in that gap between periods, with placebos
       output/figures/tfidf_<tag>/pairwise_gap.png, partisan_terms.png

Every comparison uses one value per district-period, never pooled tokens. Each table reports two samples: all_districts (every
district-period with >= MIN_WORDS_DISTRICT_PERIOD words) and balanced_thinned (districts present in all three periods, randomly
thinned to THIN_TOKENS tokens per period). L2-normalised TF-IDF depends on document length, so only the second is comparable over
time. See README.md for the choices and what they imply.

Run from the repo root:  python3 source/analysis/tfidf/tfidf.py [--indir Datastore/DistrictView/derived_v2/bow_dq] [--tag v2]
"""
import argparse
import re
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

INDIR_DEFAULT = Path("Datastore/DistrictView/derived_v2/bow_dq")
TAG_DEFAULT = "v2"
INDIR_STATIC = Path("Datastore/District Covariates/processed/district_static.parquet")
OUTDIR_TABLES = Path("output/tables")
OUTDIR_FIGURES = Path("output/figures")

PERIODS = {2018: "pre", 2019: "pre", 2020: "covid", 2021: "covid", 2022: "post", 2023: "post"}
PERIOD_ORDER = ["pre", "covid", "post"]
ANCHOR_COL = "gop_2p_2020"
MIN_WORDS_DISTRICT_PERIOD = 20000   # same rule as partisan_lexicon.py (words before vocabulary cuts)
MIN_USING = 50                      # a term must appear in this many district-periods of the period to be ranked
MIN_STATES = 10                     # "national" vocabulary, same thresholds as partisan_lexicon.py
MAX_TOP_STATE_SHARE = 0.3
THIN_TOKENS = 20000                 # tokens per district-period in the balanced panel
N_TOP = 50
N_TOP_DISTRICT = 15
N_PLACEBO = 500
N_BOOT = 300
SEED = 236
MIN_STRATUM_PARTY = 3               # a state/locale enters the stratified t only with this many districts of each party
EXAMPLE_DISTRICTS = ["BREVARD", "Chino Valley Unified", "HOUSTON ISD"]     # plus one random rural R and one random rural D

# Words that appear in district names but are ordinary vocabulary, so not evidence of a place name.
GENERIC_NAME_TOKENS = set("""school schools district districts county city unified independent public central community consolidated
union high elementary area township regional joint town village valley lake lakes park river rivers hill hills mountain mountains creek
grove spring springs beach green grand new north south east west northern southern eastern western saint fort port heights the and for
middle upper lower big little great long lone rock pine oak island bay harbor view mount point falls ridge wood woods center centre
parish borough charter special education board cooperative dept department isd usd cusd unit local free common ind sch dist""".split())
STATE_NAMES = """alabama alaska arizona arkansas california colorado connecticut delaware florida georgia hawaii idaho illinois indiana
iowa kansas kentucky louisiana maine maryland massachusetts michigan minnesota mississippi missouri montana nebraska nevada hampshire
jersey mexico york carolina dakota ohio oklahoma oregon pennsylvania rhode tennessee texas utah vermont virginia washington wisconsin
wyoming""".split()

RED, BLUE, INK, MUTED, GRID, BAND = "#C0392B", "#2C6FAE", "#222222", "#6B6B6B", "#E4E4E4", "#DADADA"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True})

def Main():
    start = time.time()
    args = ParseArgs()
    rng = np.random.default_rng(args.seed)
    indir, outdir_tables, outdir_figures, outdir_matrix = OutputPaths(args)
    counts, vocab, df = LoadDistrictPeriods(indir)
    df_terms = TermInfo(counts, df, vocab)
    X = Tfidf(counts)
    SaveMatrix(X, df, df_terms, outdir_matrix)
    print(f"TF-IDF matrix: {X.shape[0]:,} district-periods x {X.shape[1]:,} terms, {X.nnz:,} non-zeros ({time.time() - start:.0f}s)")
    counts_bt, df_bt = BalancedThinned(counts, df, rng)
    X_bt = Tfidf(counts_bt)
    samples = {"all_districts": (X, df), "balanced_thinned": (X_bt, df_bt)}
    ReportLengthEffect(samples)

    df_top = TopTermsPeriod(samples, df_terms)
    df_top.to_csv(outdir_tables / "top_terms_period.csv", index=False)
    df_party = TopTermsPartyPeriod(samples, df_terms)
    df_party.to_csv(outdir_tables / "top_terms_party_period.csv", index=False)
    PartyOverlap(df_party).to_csv(outdir_tables / "top_terms_party_overlap.csv", index=False)
    gaps = {name: {p: PartyGap(Xs, ds, p, df_terms) for p in PERIOD_ORDER} for name, (Xs, ds) in samples.items()}
    df_dist = DistinctiveTable(gaps)
    df_dist.to_csv(outdir_tables / "distinctive_R_vs_D.csv", index=False)
    df_change = ChangedPrePost(samples, df_terms)
    df_change.to_csv(outdir_tables / "changed_pre_post.csv", index=False)
    PartisanMovers(gaps).to_csv(outdir_tables / "partisan_movers_pre_post.csv", index=False)
    df_examples = ExampleDistricts(samples["all_districts"], df_terms, rng)
    df_examples.to_csv(outdir_tables / "example_districts.csv", index=False)
    Contamination(df_top, df_party, df_dist, df_change, samples["all_districts"], df_terms) \
        .to_csv(outdir_tables / "contamination.csv", index=False)
    print(f"Term tables done ({time.time() - start:.0f}s)")

    unstable = UnstableTerms(samples["balanced_thinned"], df_terms)
    df_terms[unstable][["term", "n_states", "top_state_share", "place_token"]].to_csv(outdir_tables / "unstable_terms.csv", index=False)
    df_pair, df_pair_change = Pairwise(samples, df_terms, unstable, rng)
    df_pair.to_csv(outdir_tables / "pairwise_cosine.csv", index=False)
    df_pair_change.to_csv(outdir_tables / "pairwise_cosine_changes.csv", index=False)
    PlotPairwise(df_pair, outdir_figures)
    PlotPartisanTerms(df_dist, outdir_figures)
    print(f"Done in {time.time() - start:.0f}s. Tables: {outdir_tables}  Figures: {outdir_figures}  Matrix: {outdir_matrix}")
    return

def ParseArgs():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--indir", type=Path, default=INDIR_DEFAULT, help="folder with dtm.npz, vocab.parquet, units.parquet")
    parser.add_argument("--tag", default=TAG_DEFAULT, help="outputs go to output/{tables,figures}/tfidf_<tag>/")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()

def OutputPaths(args):
    outdirs = OUTDIR_TABLES / f"tfidf_{args.tag}", OUTDIR_FIGURES / f"tfidf_{args.tag}", args.indir.parent / "tfidf"
    for path in outdirs:
        path.mkdir(parents=True, exist_ok=True)
    return (args.indir, *outdirs)

def LoadDistrictPeriods(indir):
    """District x period counts (csr) with vocab and one row of covariates per district-period, after the anchor and word filters."""
    dtm = sp.load_npz(indir / "dtm.npz").tocsr()
    vocab = pd.read_parquet(indir / "vocab.parquet").term.to_numpy()
    units = pd.read_parquet(indir / "units.parquet")
    assert dtm.shape == (len(units), len(vocab)), "dtm, units and vocab do not line up"
    units["leaid"] = units.leaid.astype(str).str.zfill(7)
    units["period"] = units.quarter.str[:4].astype(int).map(PERIODS)
    in_window = units.period.notna().to_numpy()
    print(f"Units: {len(units):,} district-quarters, {units.leaid.nunique():,} districts; {(~in_window).sum():,} outside 2018-23 dropped")
    dtm, units = dtm[np.flatnonzero(in_window)], units[in_window].reset_index(drop=True)
    key = pd.MultiIndex.from_arrays([units.leaid, units.period])
    codes, uniq = pd.factorize(key)
    group = sp.csr_matrix((np.ones(len(units), dtype="int32"), (codes, np.arange(len(units)))), shape=(len(uniq), len(units)))
    counts = (group @ dtm).tocsr()
    df = pd.DataFrame(list(uniq), columns=["leaid", "period"])
    df["n_words"] = group @ units.n_words.to_numpy()
    df["n_tokens"] = np.asarray(counts.sum(1)).ravel()
    df = df.merge(LoadCovariates(), on="leaid", how="left")
    keep = (df.gop.notna() & (df.n_words >= MIN_WORDS_DISTRICT_PERIOD)).to_numpy()
    print(f"District-periods: {len(df):,}; no anchor {df.gop.isna().sum():,}; under {MIN_WORDS_DISTRICT_PERIOD:,} words "
          f"{(df.gop.notna() & (df.n_words < MIN_WORDS_DISTRICT_PERIOD)).sum():,}; kept {keep.sum():,}")
    counts, df = counts[np.flatnonzero(keep)], df[keep].reset_index(drop=True)
    df["is_r"] = df.is_r.astype(bool)
    print(df.groupby(["period", "is_r"]).leaid.nunique().unstack().rename(columns={True: "R", False: "D"}).reindex(PERIOD_ORDER))
    return counts, vocab, df

def LoadCovariates():
    """Anchor (R if county 2020 two-party GOP share > 0.5, ties dropped), state, 4-group locale, name tokens. County != district: a proxy."""
    df = pd.read_parquet(INDIR_STATIC, columns=["leaid", ANCHOR_COL, "urban_centric_locale", "lea_name", "place_name", "county_name"])
    df["leaid"] = df.leaid.astype(str).str.zfill(7)
    df = df.rename(columns={ANCHOR_COL: "gop"})
    df = df[df.gop.notna() & (df.gop != 0.5)].copy()
    df["is_r"] = df.gop > 0.5
    df["state"] = df.leaid.str[:2]
    df["locale"] = df.urban_centric_locale.map(lambda c: {1: "city", 2: "suburb", 3: "town", 4: "rural"}.get(int(c) // 10, "unknown") if pd.notna(c) else "unknown")
    df["name_tokens"] = [NameTokens(a, b, c) for a, b, c in zip(df.lea_name, df.place_name, df.county_name)]
    return df.drop(columns="urban_centric_locale")

def NameTokens(*names):
    tokens = set()
    for name in names:
        if isinstance(name, str):
            tokens |= set(re.findall(r"[a-z]{3,}", name.lower()))
    return tokens - GENERIC_NAME_TOKENS

def TermInfo(counts, df, vocab):
    """Per term: states using it, share of district-normalised usage in its biggest state (as in bow_data.TermStateConcentration),
    whether it is 'national', and whether it matches a token of any district/place/county name or a state name."""
    codes, _ = pd.factorize(df.leaid)
    group = sp.csr_matrix((np.ones(len(df)), (codes, np.arange(len(df)))))
    by_district = (group @ counts).tocsr()
    freq = sp.diags(1 / np.asarray(by_district.sum(1)).ravel()) @ by_district
    state_of_district = df.drop_duplicates("leaid").state.to_numpy()
    scode = pd.factorize(state_of_district)[0]
    by_state = (sp.csr_matrix((np.ones(len(scode)), (scode, np.arange(len(scode))))) @ freq).toarray()
    total = by_state.sum(0)
    df_terms = pd.DataFrame({"term": vocab, "n_states": (by_state > 0).sum(0),
                             "top_state_share": np.divide(by_state.max(0), total, out=np.zeros_like(total), where=total > 0)})
    df_terms["national"] = (df_terms.n_states >= MIN_STATES) & (df_terms.top_state_share <= MAX_TOP_STATE_SHARE)
    place = set(STATE_NAMES).union(*df.drop_duplicates("leaid").name_tokens)
    df_terms["place_token"] = df_terms.term.isin(place)
    print(f"Vocabulary: {len(df_terms):,} terms, {df_terms.national.sum():,} national (>= {MIN_STATES} states, top state <= "
          f"{MAX_TOP_STATE_SHARE:.0%}), {df_terms.place_token.sum():,} match a place/district name token")
    return df_terms

def Tfidf(counts):
    """Sublinear tf (1 + log tf), idf = ln((1 + n) / (1 + df)) + 1 with df counted across the rows given, rows L2-normalised."""
    counts = counts.astype("float64").tocsr()
    counts.eliminate_zeros()                                                        # sklearn counts stored zeros in df
    return TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True).fit_transform(counts).astype("float32").tocsr()

def SaveMatrix(X, df, df_terms, outdir_matrix):
    sp.save_npz(outdir_matrix / "tfidf_district_period.npz", X)
    df[["leaid", "period", "is_r", "gop", "state", "locale", "n_words", "n_tokens"]].to_parquet(outdir_matrix / "rows.parquet", index=False)
    df_terms.to_parquet(outdir_matrix / "terms.parquet", index=False)
    return

def BalancedThinned(counts, df, rng):
    """Districts with >= THIN_TOKENS tokens in all three periods, each district-period binomially thinned to about THIN_TOKENS tokens."""
    enough = df[df.n_tokens >= THIN_TOKENS].groupby("leaid").period.nunique()
    keep = df.leaid.isin(enough[enough == len(PERIOD_ORDER)].index).to_numpy()
    rows = np.flatnonzero(keep)
    thin = counts[rows].tocsr().copy()
    probs = THIN_TOKENS / df.n_tokens.to_numpy()[rows]
    thin.data = rng.binomial(thin.data, np.repeat(probs, np.diff(thin.indptr))).astype("int32")
    thin.eliminate_zeros()
    df_bt = df[keep].reset_index(drop=True).copy()
    df_bt["n_tokens"] = np.asarray(thin.sum(1)).ravel()
    print(f"Balanced panel: {df_bt.leaid.nunique():,} districts ({df_bt.drop_duplicates('leaid').is_r.sum():,} R), "
          f"mean tokens after thinning {df_bt.n_tokens.mean():,.0f}")
    return thin, df_bt

def ReportLengthEffect(samples):
    """L2 normalisation spreads weight over more terms in longer documents; print how much document size differs by period."""
    for name, (X, df) in samples.items():
        nnz = np.diff(X.indptr)
        line = ", ".join(f"{p}: {np.median(nnz[(df.period == p).to_numpy()]):,.0f}" for p in PERIOD_ORDER)
        print(f"Median distinct terms per document, {name}: {line}")
    return

def Moments(X):
    """Column means and variances (ddof=1) of a sparse matrix, as float64."""
    X = X.astype("float64")
    n = X.shape[0]
    mean = np.asarray(X.mean(0)).ravel()
    var = (np.asarray(X.multiply(X).sum(0)).ravel() - n * mean ** 2) / (n - 1)
    return mean, np.maximum(var, 0)

def Welch(Xa, Xb):
    """Column-wise Welch test of a - b. Returns a DataFrame with means, t, df, two-sided p and standardised difference."""
    (ma, va), (mb, vb) = Moments(Xa), Moments(Xb)
    na, nb = Xa.shape[0], Xb.shape[0]
    sa, sb = va / na, vb / nb
    se = np.sqrt(sa + sb)
    t = np.divide(ma - mb, se, out=np.zeros_like(se), where=se > 0)
    denom = sa ** 2 / (na - 1) + sb ** 2 / (nb - 1)
    dof = np.divide((sa + sb) ** 2, denom, out=np.ones_like(se), where=denom > 0)
    sd = np.sqrt((va + vb) / 2)
    return pd.DataFrame({"mean_a": ma, "mean_b": mb, "t": t, "dof": dof, "p": 2 * stats.t.sf(np.abs(t), dof),
                         "std_diff": np.divide(ma - mb, sd, out=np.zeros_like(sd), where=sd > 0)})

def StratifiedT(X, is_r, strata):
    """t of the R-D gap comparing districts only within strata: stratum gaps weighted by nR*nD/(nR+nD). Strata too small are skipped."""
    num, var = np.zeros(X.shape[1]), np.zeros(X.shape[1])
    for s in pd.unique(strata):
        r, d = np.flatnonzero((strata == s) & is_r), np.flatnonzero((strata == s) & ~is_r)
        if min(len(r), len(d)) < MIN_STRATUM_PARTY:
            continue
        (mr, vr), (md, vd) = Moments(X[r]), Moments(X[d])
        weight = len(r) * len(d) / (len(r) + len(d))
        num += weight * (mr - md)
        var += weight ** 2 * (vr / len(r) + vd / len(d))
    se = np.sqrt(var)
    return np.divide(num, se, out=np.zeros_like(se), where=se > 0)

def BH(p):
    """Benjamini-Hochberg q-values."""
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out

def NUsing(X):
    return np.bincount(X.indices, minlength=X.shape[1])

def WithTermInfo(df, df_terms):
    return pd.concat([df_terms.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

def VocabularySets(df):
    """(name, subset): every table is reported for all vocabulary and for national vocabulary only."""
    return [("all", df), ("national", df[df.national])]

def TopTermsPeriod(samples, df_terms):
    rows = []
    for name, (X, df) in samples.items():
        for period in PERIOD_ORDER:
            sel = X[np.flatnonzero(df.period == period)]
            d = WithTermInfo(pd.DataFrame({"mean_tfidf": Moments(sel)[0], "share_using": NUsing(sel) / sel.shape[0]}), df_terms)
            for vocabulary, sub in VocabularySets(d):
                top = sub.nlargest(N_TOP, "mean_tfidf").assign(sample=name, period=period, vocabulary=vocabulary)
                top["rank"] = np.arange(1, len(top) + 1)
                rows.append(top)
    return Columns(pd.concat(rows), ["sample", "period", "vocabulary", "rank", "term", "mean_tfidf", "share_using"])

def Columns(df, first):
    """Identifying columns first, then the rest, dropping the raw term-info ones that only filtered."""
    tail = ["n_states", "top_state_share", "place_token"]
    middle = [c for c in df.columns if c not in first + tail + ["national"]]
    return df[first + middle + tail]

def TopTermsPartyPeriod(samples, df_terms):
    rows = []
    for name, (X, df) in samples.items():
        for period in PERIOD_ORDER:
            in_period = (df.period == period).to_numpy()
            r, d = X[np.flatnonzero(in_period & df.is_r)], X[np.flatnonzero(in_period & ~df.is_r)]
            mean = {"R": Moments(r)[0], "D": Moments(d)[0]}
            for party, other in (("R", "D"), ("D", "R")):
                t = WithTermInfo(pd.DataFrame({"mean_tfidf": mean[party], "mean_tfidf_other_party": mean[other]}), df_terms)
                for vocabulary, sub in VocabularySets(t):
                    top = sub.nlargest(N_TOP, "mean_tfidf").assign(sample=name, period=period, party=party, vocabulary=vocabulary)
                    top["rank"] = np.arange(1, len(top) + 1)
                    rows.append(top)
    return Columns(pd.concat(rows), ["sample", "period", "party", "vocabulary", "rank", "term", "mean_tfidf", "mean_tfidf_other_party"])

def PartyOverlap(df_party):
    """Number of terms shared by the top-N R and top-N D lists, for N = 10, 25, 50."""
    rows = []
    for (sample, period, vocabulary), g in df_party.groupby(["sample", "period", "vocabulary"]):
        row = {"sample": sample, "period": period, "vocabulary": vocabulary}
        for n in (10, 25, N_TOP):
            r, d = set(g[(g.party == "R") & (g["rank"] <= n)].term), set(g[(g.party == "D") & (g["rank"] <= n)].term)
            row[f"shared_top{n}"] = len(r & d)
        rows.append(row)
    return pd.DataFrame(rows)

def PartyGap(X, df, period, df_terms):
    """All terms, one period: Welch of R vs D district TF-IDF, q-values over rankable terms, and t comparing only within locale or state."""
    in_period = (df.period == period).to_numpy()
    sel, d_sel = X[np.flatnonzero(in_period)], df[in_period]
    is_r = d_sel.is_r.to_numpy()
    w = Welch(sel[np.flatnonzero(is_r)], sel[np.flatnonzero(~is_r)]).rename(columns={"mean_a": "mean_R", "mean_b": "mean_D"})
    w["n_using"] = NUsing(sel)
    w["t_within_locale"] = StratifiedT(sel, is_r, d_sel.locale.to_numpy())
    w["t_within_state"] = StratifiedT(sel, is_r, d_sel.state.to_numpy())
    out = WithTermInfo(w, df_terms)
    out["rankable"] = out.n_using >= MIN_USING
    out["q"] = np.nan
    out.loc[out.rankable, "q"] = BH(out.loc[out.rankable, "p"].to_numpy())
    return out

def DistinctiveTable(gaps):
    rows = []
    for name, by_period in gaps.items():
        for period, d in by_period.items():
            for vocabulary, sub in VocabularySets(d[d.rankable]):
                ranked = sub.sort_values("t")
                rows.append(ranked.tail(N_TOP).iloc[::-1].assign(side="R", rank=np.arange(1, N_TOP + 1), sample=name, period=period, vocabulary=vocabulary))
                rows.append(ranked.head(N_TOP).assign(side="D", rank=np.arange(1, N_TOP + 1), sample=name, period=period, vocabulary=vocabulary))
            print(f"R vs D, {name}, {period}: {(d.rankable & (d.q < 0.05)).sum():,} of {d.rankable.sum():,} rankable terms have q < .05; "
                  f"among national terms {(d.rankable & d.national & (d.q < 0.05)).sum():,} of {(d.rankable & d.national).sum():,}")
    keep = ["sample", "period", "vocabulary", "side", "rank", "term", "t", "q", "mean_R", "mean_D", "std_diff", "t_within_locale", "t_within_state", "n_using"]
    return Columns(pd.concat(rows), keep).drop(columns=["dof", "p", "rankable"])

def ChangedPrePost(samples, df_terms):
    """Post vs pre. all_districts: Welch (the district mix changes). balanced_thinned: Welch plus a paired t on the same districts."""
    rows = []
    for name, (X, df) in samples.items():
        pre, post = np.flatnonzero(df.period == "pre"), np.flatnonzero(df.period == "post")
        w = Welch(X[post], X[pre]).rename(columns={"mean_a": "mean_post", "mean_b": "mean_pre"})
        w["n_using_pre"], w["n_using_post"] = NUsing(X[pre]), NUsing(X[post])
        if name == "balanced_thinned":
            w["t_paired"] = PairedT(X, df, pre, post)
        d = WithTermInfo(w, df_terms)
        d["rankable"] = (d.n_using_pre + d.n_using_post) >= MIN_USING
        d["q"] = np.nan
        d.loc[d.rankable, "q"] = BH(d.loc[d.rankable, "p"].to_numpy())
        ranked_all = d[d.rankable]
        print(f"Post vs pre, {name}: {(ranked_all.t < 0).mean():.1%} of rankable terms fall (near 50% expected if document size does not drive it); "
              f"{(ranked_all.q < 0.05).sum():,} of {len(ranked_all):,} have q < .05")
        for vocabulary, sub in VocabularySets(ranked_all):
            ranked = sub.sort_values("t")
            rows.append(ranked.tail(N_TOP).iloc[::-1].assign(direction="up", rank=np.arange(1, N_TOP + 1), sample=name, vocabulary=vocabulary))
            rows.append(ranked.head(N_TOP).assign(direction="down", rank=np.arange(1, N_TOP + 1), sample=name, vocabulary=vocabulary))
    keep = ["sample", "vocabulary", "direction", "rank", "term", "t", "q", "mean_pre", "mean_post", "std_diff"]
    return Columns(pd.concat(rows), keep).drop(columns=["dof", "p", "rankable"])

def PairedT(X, df, pre, post):
    """Column-wise paired t of post - pre; rows are matched on leaid."""
    order_pre = pre[np.argsort(df.leaid.to_numpy()[pre])]
    order_post = post[np.argsort(df.leaid.to_numpy()[post])]
    assert (df.leaid.to_numpy()[order_pre] == df.leaid.to_numpy()[order_post]).all()
    mean, var = Moments((X[order_post] - X[order_pre]).tocsr())
    se = np.sqrt(var / len(pre))
    return np.divide(mean, se, out=np.zeros_like(se), where=se > 0)

def PartisanMovers(gaps):
    """Terms whose R-D gap moved most from pre to post. Uses the standardised gap, because t also grows with the district count."""
    rows = []
    for name, by_period in gaps.items():
        pre, post = by_period["pre"], by_period["post"]
        d = pre[["term", "n_states", "top_state_share", "national", "place_token", "t", "std_diff", "rankable"]].rename(
            columns={"t": "t_pre", "std_diff": "std_diff_pre", "rankable": "rankable_pre"})
        d["t_post"], d["std_diff_post"], d["rankable_post"] = post.t.to_numpy(), post.std_diff.to_numpy(), post.rankable.to_numpy()
        d["std_diff_change"] = d.std_diff_post - d.std_diff_pre
        for vocabulary, sub in VocabularySets(d[d.rankable_pre & d.rankable_post]):
            ranked = sub.sort_values("std_diff_change")
            rows.append(ranked.tail(N_TOP).iloc[::-1].assign(direction="more R after", rank=np.arange(1, N_TOP + 1), sample=name, vocabulary=vocabulary))
            rows.append(ranked.head(N_TOP).assign(direction="more D after", rank=np.arange(1, N_TOP + 1), sample=name, vocabulary=vocabulary))
    keep = ["sample", "vocabulary", "direction", "rank", "term", "std_diff_change", "std_diff_pre", "std_diff_post", "t_pre", "t_post"]
    return Columns(pd.concat(rows), keep).drop(columns=["rankable_pre", "rankable_post"])

def TopTermsOfRow(X, i):
    start, end = X.indptr[i], X.indptr[i + 1]
    order = np.argsort(-X.data[start:end])[:N_TOP_DISTRICT]
    return X.indices[start:end][order], X.data[start:end][order]

def ExampleDistricts(sample, df_terms, rng):
    """Top terms of two known flashpoints, one large urban district, and one random rural district of each party."""
    X, df = sample
    three = df.groupby("leaid").period.nunique()
    full = df[df.leaid.isin(three[three == len(PERIOD_ORDER)].index)].drop_duplicates("leaid")
    chosen = list(df[df.lea_name.isin(EXAMPLE_DISTRICTS)].leaid.unique())
    for is_r in (True, False):
        pool = full[(full.locale == "rural") & (full.is_r == is_r)].leaid.to_numpy()
        if len(pool):
            chosen.append(rng.choice(pool))
    national_idx = np.flatnonzero(df_terms.national.to_numpy())
    rows = []
    for vocabulary, Xv, info in (("all", X, df_terms), ("national", normalize(X[:, national_idx]).tocsr(), df_terms.iloc[national_idx])):
        terms, states, shares = info.term.to_numpy(), info.n_states.to_numpy(), info.top_state_share.to_numpy()
        for i in np.flatnonzero(df.leaid.isin(chosen)):
            row = df.iloc[i]
            idx, val = TopTermsOfRow(Xv, i)
            for rank, (j, v) in enumerate(zip(idx, val), start=1):
                rows.append({"vocabulary": vocabulary, "leaid": row.leaid, "lea_name": row.lea_name, "party": "R" if row.is_r else "D", "locale": row.locale,
                             "period": row.period, "rank": rank, "term": terms[j], "tfidf": v, "own_name_token": terms[j] in row.name_tokens,
                             "n_states": states[j], "top_state_share": shares[j]})
    return pd.DataFrame(rows).sort_values(["vocabulary", "lea_name", "period", "rank"])

def ContaminationRow(table, terms, df_terms_indexed, **labels):
    info = df_terms_indexed.loc[list(terms)]
    return {"table": table, **labels, "n_terms": len(info), "share_place_token": info.place_token.mean(),
            "share_top_state_over_30pct": (info.top_state_share > MAX_TOP_STATE_SHARE).mean(), "share_under_10_states": (info.n_states < MIN_STATES).mean()}

def Contamination(df_top, df_party, df_dist, df_change, sample, df_terms):
    """Share of each table's terms that are place-name tokens or state-concentrated, plus the same for every district's own top terms."""
    info = df_terms.set_index("term")
    rows = []
    for (s, p, v), g in df_top[df_top["rank"] <= N_TOP].groupby(["sample", "period", "vocabulary"]):
        rows.append(ContaminationRow("top_terms_period", g.term, info, sample=s, period=p, vocabulary=v))
    for (s, p, v, party), g in df_party.groupby(["sample", "period", "vocabulary", "party"]):
        rows.append(ContaminationRow("top_terms_party_period", g.term, info, sample=s, period=p, vocabulary=v, group=party))
    for (s, p, v, side), g in df_dist.groupby(["sample", "period", "vocabulary", "side"]):
        rows.append(ContaminationRow("distinctive_R_vs_D", g.term, info, sample=s, period=p, vocabulary=v, group=side))
    for (s, v, direction), g in df_change.groupby(["sample", "vocabulary", "direction"]):
        rows.append(ContaminationRow("changed_pre_post", g.term, info, sample=s, period="post-pre", vocabulary=v, group=direction))
    X, df = sample
    terms = df_terms.term.to_numpy()
    national_idx = np.flatnonzero(df_terms.national.to_numpy())
    for vocabulary, Xv, tv in (("all", X, terms), ("national", normalize(X[:, national_idx]).tocsr(), terms[national_idx])):
        by_period = {}
        for i in range(X.shape[0]):
            idx, _ = TopTermsOfRow(Xv, i)
            top = tv[idx]
            by_period.setdefault(df.period.iat[i], []).append((top, np.mean([t in df.name_tokens.iat[i] for t in top])))
        for period, items in by_period.items():
            row = ContaminationRow("each_district_top15", np.concatenate([a for a, _ in items]), info, sample="all_districts", period=period, vocabulary=vocabulary)
            row["share_own_name_token"] = np.mean([b for _, b in items])
            row["n_terms"] = len(items[0][0])
            rows.append(row)
    out = pd.DataFrame(rows)
    print(out[out.table.isin(["each_district_top15", "distinctive_R_vs_D"]) & (out["sample"] == "all_districts")]
          .drop(columns=["table", "sample", "n_terms"]).round(3).to_string(index=False))
    return out

def CosineMatrix(X):
    """Dense cosine similarity with a zero diagonal. Rows must already be L2-normalised."""
    dense = X.toarray().astype("float32")
    S = dense @ dense.T
    np.fill_diagonal(S, 0)
    return S

def WeightedGap(SM, M, s, w=None):
    """Mean cosine of same-label pairs minus different-label pairs. SM = S * M, M = pair mask, s = +1/-1 labels, w = district weights."""
    v, sv = (np.ones_like(s), s) if w is None else (w, s * w)
    total_S, diff_S = v @ (SM @ v), sv @ (SM @ sv)
    total_M, diff_M = v @ (M @ v), sv @ (M @ sv)
    return (total_S + diff_S) / (total_M + diff_M) - (total_S - diff_S) / (total_M - diff_M)

def ShuffleLabels(s, groups, rng):
    if groups is None:
        return rng.permutation(s)
    out = s.copy()
    for idx in groups:
        out[idx] = rng.permutation(s[idx])
    return out

def GroupIndex(values):
    codes, uniq = pd.factorize(values)
    return [np.flatnonzero(codes == k) for k in range(len(uniq))]

def UnstableTerms(sample, df_terms):
    """Boolean mask: terms whose use changed from pre to covid or pre to post in the balanced panel (paired t, q < .05). These are COVID
    vocabulary and caption-format drift; dropping them asks whether the similarity gap moved in the vocabulary that did not itself shift."""
    X, df = sample
    pre = np.flatnonzero(df.period == "pre")
    unstable = np.zeros(X.shape[1], dtype=bool)
    for period in ("covid", "post"):
        other = np.flatnonzero(df.period == period)
        t = PairedT(X, df, pre, other)
        ok = np.flatnonzero((NUsing(X[pre]) + NUsing(X[other])) >= MIN_USING)
        p = 2 * stats.t.sf(np.abs(t[ok]), len(pre) - 1)
        unstable[ok[BH(p) < 0.05]] = True
    print(f"Unstable terms (pre to covid or post, balanced panel, q < .05): {unstable.sum():,}")
    return unstable

def Pairwise(samples, df_terms, unstable, rng):
    """Within- vs across-party TF-IDF cosine by period, for every sample x vocabulary x pair set, against label shuffles (the same shuffle
    in every period, so the change between periods gets a null too) and a district bootstrap."""
    rows, change_rows = [], []
    national = df_terms.national.to_numpy()
    for name, (X, df) in samples.items():
        for vocabulary, Xv in (("all", X), ("national", normalize(X[:, np.flatnonzero(national)]).tocsr()),
                               ("national_stable", normalize(X[:, np.flatnonzero(national & ~unstable)]).tocsr())):
            union = df.drop_duplicates("leaid").reset_index(drop=True)
            pos = pd.Series(np.arange(len(union)), index=union.leaid)
            s_all = np.where(union.is_r, 1.0, -1.0).astype("float32")
            null_groups = {"random": None, "within_state": GroupIndex(union.state.to_numpy()), "within_locale": GroupIndex(union.locale.to_numpy())}
            matrices = {}
            for period in PERIOD_ORDER:
                sel = np.flatnonzero(df.period == period)
                ids = pos[df.leaid.to_numpy()[sel]].to_numpy()
                S = CosineMatrix(Xv[sel])
                same_state = union.state.to_numpy()[ids][:, None] == union.state.to_numpy()[ids][None, :]
                off_diag = 1 - np.eye(len(ids), dtype="float32")
                matrices[period] = (ids, S, {"all_pairs": off_diag, "different_state": off_diag * (~same_state).astype("float32")})
            for pairs in ("all_pairs", "different_state"):
                observed, nulls, boots, cohesion = PairwiseDraws(matrices, pairs, s_all, null_groups, rng)
                for period in PERIOD_ORDER:
                    rows.append(PairwiseRow(name, vocabulary, pairs, period, matrices[period], s_all, observed, nulls, boots, cohesion))
                for a, b in (("pre", "covid"), ("pre", "post"), ("covid", "post")):
                    change_rows.append(PairwiseChangeRow(name, vocabulary, pairs, a, b, matrices, s_all, observed, nulls, boots, cohesion))
                print(f"  {name:17s} {vocabulary:16s}{pairs:16s}", "  ".join(f"{p} {observed[p]:+.4f}" for p in PERIOD_ORDER), flush=True)
    return pd.DataFrame(rows), pd.DataFrame(change_rows)

def PairwiseDraws(matrices, pairs, s_all, null_groups, rng):
    SM = {p: S * masks[pairs] for p, (_, S, masks) in matrices.items()}
    M = {p: masks[pairs] for p, (_, _, masks) in matrices.items()}
    ids = {p: m[0] for p, m in matrices.items()}
    observed = {p: WeightedGap(SM[p], M[p], s_all[ids[p]]) for p in PERIOD_ORDER}
    nulls = {}
    for null_name, groups in null_groups.items():
        draws = {p: np.empty(N_PLACEBO) for p in PERIOD_ORDER}
        for b in range(N_PLACEBO):
            shuffled = ShuffleLabels(s_all, groups, rng)
            for p in PERIOD_ORDER:
                draws[p][b] = WeightedGap(SM[p], M[p], shuffled[ids[p]])
        nulls[null_name] = draws
    boots = {p: np.empty(N_BOOT) for p in PERIOD_ORDER}
    cohesion = {p: np.empty((N_BOOT, 2)) for p in PERIOD_ORDER}
    n = len(s_all)
    for b in range(N_BOOT):
        weights = rng.multinomial(n, np.full(n, 1 / n)).astype("float32")            # resample districts, same draw in every period
        for p in PERIOD_ORDER:
            boots[p][b] = WeightedGap(SM[p], M[p], s_all[ids[p]], weights[ids[p]])
            cohesion[p][b] = Cohesion(SM[p], M[p], s_all[ids[p]], weights[ids[p]])
    return observed, nulls, boots, cohesion

def Cohesion(SM, M, s, w=None):
    """(RR - RD, DD - RD): is each party's mean within-party cosine above the cross-party mean? The overall gap can come from one side only."""
    w = np.ones_like(s) if w is None else w
    r, d = w * (s > 0), w * (s < 0)
    rd = (r @ (SM @ d)) / (r @ (M @ d))
    return (r @ (SM @ r)) / (r @ (M @ r)) - rd, (d @ (SM @ d)) / (d @ (M @ d)) - rd

def TwoSidedP(obs, draws):
    centre = draws.mean()
    return (1 + (np.abs(draws - centre) >= abs(obs - centre)).sum()) / (len(draws) + 1)

def NullColumns(obs, nulls_for_stat):
    row = {}
    for null_name, draws in nulls_for_stat.items():
        row[f"{null_name}_mean"], row[f"{null_name}_sd"] = draws.mean(), draws.std(ddof=1)
        row[f"{null_name}_lo"], row[f"{null_name}_hi"] = np.quantile(draws, [0.025, 0.975])
        row[f"{null_name}_p"] = TwoSidedP(obs, draws)
    return row

def PairwiseRow(name, vocabulary, pairs, period, matrix, s_all, observed, nulls, boots, cohesion):
    ids, S, masks = matrix
    s, M = s_all[ids], masks[pairs]
    r, d = s > 0, s < 0
    mean_of = lambda a, b: (S[np.ix_(a, b)] * M[np.ix_(a, b)]).sum() / M[np.ix_(a, b)].sum()
    row = {"sample": name, "vocabulary": vocabulary, "pairs": pairs, "period": period, "n_R": int(r.sum()), "n_D": int(d.sum()),
           "mean_RR": mean_of(r, r), "mean_DD": mean_of(d, d), "mean_RD": mean_of(r, d), "gap_within_minus_across": observed[period],
           "boot_lo": np.quantile(boots[period], 0.025), "boot_hi": np.quantile(boots[period], 0.975)}
    row["RR_minus_RD"], row["DD_minus_RD"] = row["mean_RR"] - row["mean_RD"], row["mean_DD"] - row["mean_RD"]
    row["RR_minus_RD_lo"], row["RR_minus_RD_hi"] = np.quantile(cohesion[period][:, 0], [0.025, 0.975])
    row["DD_minus_RD_lo"], row["DD_minus_RD_hi"] = np.quantile(cohesion[period][:, 1], [0.025, 0.975])
    row.update(NullColumns(observed[period], {k: v[period] for k, v in nulls.items()}))
    return row

def PairwiseChangeRow(name, vocabulary, pairs, a, b, matrices, s_all, observed, nulls, boots, cohesion):
    change = observed[b] - observed[a]
    row = {"sample": name, "vocabulary": vocabulary, "pairs": pairs, "from": a, "to": b, "gap_change": change,
           "boot_lo": np.quantile(boots[b] - boots[a], 0.025), "boot_hi": np.quantile(boots[b] - boots[a], 0.975)}
    point = {p: Cohesion(matrices[p][1] * matrices[p][2][pairs], matrices[p][2][pairs], s_all[matrices[p][0]]) for p in (a, b)}
    for k, side in enumerate(("RR_minus_RD", "DD_minus_RD")):
        row[f"{side}_change"] = point[b][k] - point[a][k]
        row[f"{side}_change_lo"], row[f"{side}_change_hi"] = np.quantile(cohesion[b][:, k] - cohesion[a][:, k], [0.025, 0.975])
    row.update(NullColumns(change, {k: v[b] - v[a] for k, v in nulls.items()}))
    return row

def PlotPairwise(df_pair, outdir_figures):
    specs = [("balanced_thinned", "Same districts, equal length (comparable over time)"), ("all_districts", "All districts (mix and length change)")]
    d = df_pair[(df_pair.vocabulary == "national") & (df_pair.pairs == "all_pairs")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    x = np.arange(len(PERIOD_ORDER))
    labels = ["pre\n2018-19", "covid\n2020-21", "post\n2022-23"]
    for ax, (sample, title) in zip(axes, specs):
        g = d[d["sample"] == sample].set_index("period").reindex(PERIOD_ORDER)
        ax.fill_between(x, g.random_lo, g.random_hi, color=BAND, label="labels shuffled at random (95% band)")
        ax.plot(x, g.within_state_mean, color=MUTED, lw=2, ls=":", label="labels shuffled within state (mean)")
        ax.plot(x, g.within_locale_mean, color=MUTED, lw=2, ls="--", label="labels shuffled within locale (mean)")
        ax.errorbar(x, g.gap_within_minus_across, yerr=[g.gap_within_minus_across - g.boot_lo, g.boot_hi - g.gap_within_minus_across],
                    color=INK, lw=2, marker="o", ms=8, capsize=3, label="data (bootstrap 95% CI)")
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].set_ylabel("cosine: within-party minus across-party")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Do same-party districts talk more alike? TF-IDF cosine gap, national vocabulary", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir_figures / "pairwise_gap.png", dpi=160)
    plt.close(fig)
    return

def PlotPartisanTerms(df_dist, outdir_figures):
    d = df_dist[(df_dist["sample"] == "all_districts") & (df_dist.vocabulary == "national") & (df_dist.period == "post") & (df_dist["rank"] <= 20)]
    d = d.sort_values("t")
    fig, ax = plt.subplots(figsize=(7.5, 8))
    ax.barh(d.term, d.t, color=[RED if t > 0 else BLUE for t in d.t], height=0.7)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_xlabel("Welch t of R - D district TF-IDF, 2022-23   (right = more in Republican-county districts)")
    ax.set_title("Most partisan terms by TF-IDF, national vocabulary, 2022-23", loc="left", fontsize=11)
    ax.tick_params(axis="y", labelcolor=INK)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(outdir_figures / "partisan_terms.png", dpi=160)
    plt.close(fig)
    return

if __name__ == "__main__":
    Main()
