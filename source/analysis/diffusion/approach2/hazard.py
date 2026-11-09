"""Step 4b: discrete-time hazard of first adoption, with neighbour / same-state / co-partisan exposure. DESCRIPTIVE.

Risk set: panel districts (meetings posted in both PRE and POST years; see adoption.py), phrase p, month t in the POST
years, where the district posted >= 1 kept meeting in month t (adoption is only observable then), had not mentioned p
before t, and did not mention p in the PRE years. y = 1 if the district mentions p in month t (its first mention).
Exposure (all lagged: adopted by t-1), computed over ALL sample districts (not only the panel) other than the district:
  geo_share    share of other districts within RADIUS_KM (district-office lat/lon) that have adopted p
  state_share  share of other same-state districts that have adopted p
  copart_share share of other same-partisanship (county GOP 2020 > / < 50%) districts nationally that have adopted p
Controls: own county GOP share (z), log words posted in month t, number of meetings in month t, no-neighbour dummy.
Models: (1) linear probability model with phrase x month fixed effects (absorbs each phrase's national trend),
        SE clustered by district; (2) logit with phrase and month fixed effects (as in the brief).
Permutation null: district locations (lat/lon) are shuffled across districts N_PERM times (state, partisanship and
adoption dates kept), geo_share recomputed and model (1) refitted; also a stricter within-state shuffle.
Common regional shocks (state mandates, local news markets) also produce a positive geo coefficient: this is NOT a
causal diffusion estimate.
Writes output/tables/diffusion/approach2/{hazard_models.csv, hazard_permutation.csv, hazard_by_phrase.csv}
Run from the repo root (after count_mentions.py):  python3 source/analysis/diffusion/approach2/hazard.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from phrases import PHRASES  # noqa: E402

RADIUS_KM = 100
N_PERM = 200
SEED = cm.GLOBALS["GABRIEL_SEED"]
X_COLS = ["geo_share", "state_share", "copart_share", "gop_z", "log_words", "n_meet", "no_neighbor"]


def Main():
    docs = pd.read_parquet(cm.OUTDIR_DATA / "docs.parquet")
    docs = docs.loc[~docs.is_empty].copy()
    months = pd.period_range(f"{cm.SAMPLE_START}-01", f"{cm.SAMPLE_END}-12", freq="M")
    docs["t"] = (docs.date.dt.to_period("M") - months[0]).apply(lambda x: x.n)
    dist = DistrictFrame(docs)
    n, n_t, phrases = len(dist), len(months), list(PHRASES)
    adopt = AdoptIndex(dist, phrases)  # n x P, month index of first mention (inf = never)
    t_post = (pd.Period(f"{cm.POST_YEARS[0]}-01", "M") - months[0]).n
    # adopted-by-(t-1) indicator, shape n x P x T: A[i,p,t] = adopt[i,p] <= t-1
    tt = np.arange(n_t)
    A = (adopt[:, :, None] <= (tt[None, None, :] - 1)).astype(np.float32)
    state_codes = pd.factorize(dist.state)[0]
    rep_codes = pd.factorize(dist.rep.fillna("NA"))[0]
    S = Peers(state_codes)
    C = Peers(rep_codes)
    W, no_nb = GeoPeers(dist.latitude.values, dist.longitude.values)
    rs = RiskSet(docs, dist, adopt, t_post, n_t, phrases)
    rs["state_share"] = Exposure(S, A, rs)
    rs["copart_share"] = Exposure(C, A, rs)
    rs["geo_share"] = Exposure(W, A, rs)
    rs["no_neighbor"] = no_nb[rs.i.values].astype(float)
    print(f"risk set: {len(rs):,} district-phrase-months, {rs.y.sum():,} adoptions, {rs.i.nunique()} districts")
    models = FitLPM(rs, "LPM, phrase x month FE") + FitLogit(rs)
    df_models = pd.DataFrame(models)
    perm = Permutations(rs, dist, A, state_codes)
    by_phrase = ByPhrase(rs, phrases)
    cm.OUTDIR_TABLES.mkdir(parents=True, exist_ok=True)
    df_models.to_csv(cm.OUTDIR_TABLES / "hazard_models.csv", index=False, float_format="%.5f")
    perm.to_csv(cm.OUTDIR_TABLES / "hazard_permutation.csv", index=False, float_format="%.5f")
    by_phrase.to_csv(cm.OUTDIR_TABLES / "hazard_by_phrase.csv", index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    print(df_models.to_string())
    b_obs = df_models.loc[(df_models.model.str.startswith("LPM")) & (df_models.term == "geo_share"), "coef"].iloc[0]
    for kind, g in perm.groupby("shuffle"):
        print(f"{kind}: observed geo coef {b_obs:.4f}; null mean {g.coef.mean():.4f}, 95th pct {g.coef.quantile(.95):.4f},"
              f" share of draws >= observed {(g.coef >= b_obs).mean():.3f}")
    print(by_phrase.to_string())
    return


def DistrictFrame(docs):
    st = cm.LoadStatic()
    dist = pd.DataFrame({"leaid": sorted(docs.leaid.unique())}).merge(st, on="leaid", how="left", validate="1:1")
    years = docs.date.dt.year
    pre = set(docs.loc[years.isin(cm.PRE_YEARS), "leaid"])
    post = set(docs.loc[years.isin(cm.POST_YEARS), "leaid"])
    dist["panel"] = dist.leaid.isin(pre & post)
    g = dist.gop_2p_2020
    dist["gop_z"] = ((g - g.mean()) / g.std()).fillna(0.0)
    return dist


def AdoptIndex(dist, phrases):
    dm = pd.read_parquet(cm.OUTDIR_DATA / "mentions.parquet")
    dm = dm.loc[dm["count"] > 0]
    t0 = pd.Period(f"{cm.SAMPLE_START}-01", "M")
    first = dm.groupby(["leaid", "phrase"]).date.min().dt.to_period("M").apply(lambda m: (m - t0).n)
    out = np.full((len(dist), len(phrases)), np.inf)
    pos = {k: i for i, k in enumerate(dist.leaid)}
    ppos = {k: i for i, k in enumerate(phrases)}
    for (le, ph), t in first.items():
        out[pos[le], ppos[ph]] = t
    return out


def Peers(codes):
    """Row-normalised 'same group, not self' matrix."""
    M = (codes[:, None] == codes[None, :]).astype(np.float32)
    np.fill_diagonal(M, 0)
    rs = M.sum(1, keepdims=True)
    return np.divide(M, rs, out=np.zeros_like(M), where=rs > 0)


def GeoPeers(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    d = 2 * 6371 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    M = (d <= RADIUS_KM).astype(np.float32)
    np.fill_diagonal(M, 0)
    rs = M.sum(1, keepdims=True)
    return np.divide(M, rs, out=np.zeros_like(M), where=rs > 0), (rs[:, 0] == 0)


def RiskSet(docs, dist, adopt, t_post, n_t, phrases):
    pos = {k: i for i, k in enumerate(dist.leaid)}
    dm = docs.groupby(["leaid", "t"]).agg(n_meet=("video_id", "size"), words=("n_words", "sum")).reset_index()
    dm["i"] = dm.leaid.map(pos)
    dm = dm.loc[(dm.t >= t_post) & dist.panel.values[dm.i.values]]
    frames = []
    for p, ph in enumerate(phrases):
        a = adopt[dm.i.values, p]
        keep = (a >= dm.t.values) & ~(a < t_post)
        f = dm.loc[keep, ["i", "t", "n_meet", "words"]].copy()
        f["p"] = p
        f["y"] = (a[keep] == f.t.values).astype(float)
        frames.append(f)
    rs = pd.concat(frames, ignore_index=True)
    rs["log_words"] = np.log(rs.words)
    rs["gop_z"] = dist.gop_z.values[rs.i.values]
    rs["cell"] = rs.p * n_t + rs.t
    return rs


def Exposure(M, A, rs):
    """share of peers adopted by t-1 for each risk-set row: (M @ A[:, p, t])[i]."""
    n, P, T = A.shape
    E = (M @ A.reshape(n, P * T)).reshape(n, P, T)
    return E[rs.i.values, rs.p.values, rs.t.values]


def Demean(rs, cols):
    g = rs.groupby("cell")[cols].transform("mean")
    return rs[cols] - g


def FitLPM(rs, label):
    cols = ["y"] + X_COLS
    dd = Demean(rs, cols)
    res = sm.OLS(dd.y.values, dd[X_COLS].values).fit(cov_type="cluster", cov_kwds={"groups": rs.i.values})
    sd = rs[X_COLS].std()
    return [{"model": label, "term": c, "coef": res.params[k], "se": res.bse[k], "p": res.pvalues[k],
             "sd_x": sd[c], "n": int(res.nobs), "mean_y": rs.y.mean()} for k, c in enumerate(X_COLS)]


def FitLogit(rs):
    X = pd.concat([rs[X_COLS], pd.get_dummies(rs.p, prefix="p", drop_first=True, dtype=float),
                   pd.get_dummies(rs.t, prefix="t", drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X)
    res = sm.Logit(rs.y.values, X.values).fit(disp=0, maxiter=200, cov_type="cluster",
                                               cov_kwds={"groups": rs.i.values})
    sd = rs[X_COLS].std()
    return [{"model": "logit, phrase + month FE", "term": c, "coef": res.params[k + 1], "se": res.bse[k + 1],
             "p": res.pvalues[k + 1], "sd_x": sd[c], "n": int(res.nobs), "mean_y": rs.y.mean()}
            for k, c in enumerate(X_COLS)]


def Permutations(rs, dist, A, state_codes):
    rng = np.random.default_rng(SEED)
    dd = Demean(rs, ["y"] + X_COLS)
    X0, y = dd[X_COLS].values.copy(), dd.y.values
    j_geo, j_nb = X_COLS.index("geo_share"), X_COLS.index("no_neighbor")
    lat, lon = dist.latitude.values, dist.longitude.values
    cells = rs.cell.values
    out = []
    for kind in ("all districts", "within state"):
        for b in range(N_PERM):
            if kind == "all districts":
                perm = rng.permutation(len(dist))
            else:
                perm = np.arange(len(dist))
                for s in np.unique(state_codes):
                    idx = np.where(state_codes == s)[0]
                    perm[idx] = rng.permutation(idx)
            W, no_nb = GeoPeers(lat[perm], lon[perm])
            geo = Exposure(W, A, rs)
            nb = no_nb[rs.i.values].astype(float)
            X = X0.copy()
            X[:, j_geo] = geo - pd.Series(geo).groupby(cells).transform("mean").values
            X[:, j_nb] = nb - pd.Series(nb).groupby(cells).transform("mean").values
            coef = np.linalg.lstsq(X, y, rcond=None)[0]
            out.append({"shuffle": kind, "draw": b, "coef": coef[j_geo]})
        print(f"permutations done: {kind}", flush=True)
    return pd.DataFrame(out)


def ByPhrase(rs, phrases):
    rows = []
    for p, ph in enumerate(phrases):
        g = rs.loc[rs.p == p]
        if g.y.sum() < 30:
            continue
        dd = Demean(g, ["y"] + X_COLS)
        res = sm.OLS(dd.y.values, dd[X_COLS].values).fit(cov_type="cluster", cov_kwds={"groups": g.i.values})
        j, k = X_COLS.index("geo_share"), X_COLS.index("state_share")
        rows.append({"phrase": ph, "n": len(g), "adoptions": int(g.y.sum()), "geo_coef": res.params[j],
                     "geo_se": res.bse[j], "state_coef": res.params[k], "state_se": res.bse[k],
                     "gop_coef": res.params[X_COLS.index("gop_z")], "gop_se": res.bse[X_COLS.index("gop_z")]})
    return pd.DataFrame(rows).sort_values("adoptions", ascending=False)


if __name__ == "__main__":
    Main()
