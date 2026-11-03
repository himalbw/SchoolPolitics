"""Spectral clustering of districts on LDA topic weights, with partisanship / geography checks and IF-PCA on words.

Functions behind notebooks/run_lda_cluster.ipynb (one year and K per run; Summarize() gives one comparable row per run).
Inputs are the run_lda.py outputs for a year tag (y<year>):
    output/lda_cleaned/<tag>/theta_meetings_k<K>.parquet, theta_chunks_k<K>.parquet, topics_k<K>.csv
    Datastore/DistrictView/derived/lda_cleaned/<tag>/chunks/, vectorizer.joblib   (chunk text; IF-PCA and excerpts only)
Method (Stat 236 lecture 7): standardize each column of the district x topic matrix (X*), keep the top K-1 left singular
vectors, k-means with K groups. K defaults to (# singular values above a column-shuffled null) + 1.
Topic and cluster numbers are local to each fit: compare years by topic content, not by index.
"""
import functools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse, stats
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

REPO = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO / "output/lda_cleaned"
WORK_ROOT = REPO / "Datastore/DistrictView/derived/lda_cleaned"
STATIC = REPO / "Datastore/District Covariates/processed/district_static.parquet"
META = REPO / "Datastore/DistrictView/districtview_share_draft/meta_districtview.parquet"
SEED = 0

LOCALE = {1: "City", 2: "Suburb", 3: "Town", 4: "Rural"}
REGION = {**dict.fromkeys("CT ME MA NH RI VT NJ NY PA".split(), "Northeast"),
          **dict.fromkeys("IL IN MI OH WI IA KS MN MO NE ND SD".split(), "Midwest"),
          **dict.fromkeys("DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX".split(), "South"),
          **dict.fromkeys("AZ CO ID MT NV NM UT WY AK CA HI OR WA".split(), "West")}
HOT_TERMS = ["mask", "masks", "covid", "vaccine", "quarantine", "book", "books", "critical race", "crt",
             "gender", "transgender", "parental rights", "public comment", "curriculum"]


# ---------------------------------------------------------------- inputs

def Tag(year):
    return f"y{year}"


def AvailableRuns():
    """(year, K) pairs with meeting-level topic weights in output/lda_cleaned/."""
    rows = [dict(year=int(m.group(1)), k=int(m.group(2)))
            for f in RESULTS_ROOT.glob("y*/theta_meetings_k*.parquet")
            if (m := re.fullmatch(r"y(\d{4})/theta_meetings_k(\d+)\.parquet", f.relative_to(RESULTS_ROOT).as_posix()))]
    return pd.DataFrame(rows, columns=["year", "k"]).sort_values(["year", "k"]).reset_index(drop=True)


def LoadTopics(year, k):
    """Top 15 words per topic, indexed t1..tK."""
    t = pd.read_csv(RESULTS_ROOT / Tag(year) / f"topics_k{k}.csv")
    return pd.Series(t["top_terms"].to_numpy(), index=[f"t{i}" for i in t["topic"]], name="top_terms")


def Short(topics, t, n=3):
    """'t7 (students, children, parents)' style label."""
    return f"{t} ({', '.join(topics[t].split(', ')[:n])})"


def LoadDistricts(year, k, min_meetings=1):
    """Meeting topic weights and district profiles (each meeting weighted equally). Returns meetings, district, T."""
    T = [f"t{i}" for i in range(1, k + 1)]
    meetings = pd.read_parquet(RESULTS_ROOT / Tag(year) / f"theta_meetings_k{k}.parquet")
    meetings["leaid"] = meetings["leaid"].astype(str).str.zfill(7)
    district = meetings.groupby("leaid")[T].mean()
    district["n_meetings"] = meetings.groupby("leaid").size()
    district = district[district["n_meetings"] >= min_meetings].reset_index()
    return meetings, district, T


@functools.lru_cache(maxsize=2)
def ChunkText(year):
    """id, chunk, leaid, text for every retained chunk of a year (large: ~100 MB per year)."""
    files = sorted((WORK_ROOT / Tag(year) / "chunks").glob("*.parquet"))
    out = pd.concat([pd.read_parquet(f, columns=["id", "chunk", "leaid", "text"]) for f in files], ignore_index=True)
    out["leaid"] = out["leaid"].astype(str).str.zfill(7)
    return out


# ---------------------------------------------------------------- SVD and clustering

def Decompose(district, T, standardize=True, min_topic_mean=.02):
    """SVD of the district x topic matrix.

    standardize=True: columns centered and scaled to SD 1 (the lecture's X*). False: raw weights, first singular vector
    made positive for SCORE. Topics with mean weight < min_topic_mean are dropped first (standardizing inflates their noise).
    """
    topics = [t for t in T if district[t].mean() >= min_topic_mean]
    X = district[topics].to_numpy()
    if standardize:
        X = (X - X.mean(0)) / X.std(0)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    if not standardize:
        U[:, 0] *= np.sign(U[:, 0].sum())
    return dict(X=X, U=U, s=s, Vt=Vt, topics=topics, standardize=standardize, leaid=district["leaid"].to_numpy())


def Embed(sp, r):
    """r-dimensional spectral embedding: top r left singular vectors of X*, or r SCORE ratios U[:, j] / U[:, 0]."""
    if sp["standardize"]:
        return sp["U"][:, :r]
    return sp["U"][:, 1:r + 1] / sp["U"][:, [0]]


def ParallelAnalysis(sp, n_perm=500, q=.95, seed=SEED):
    """q-quantile of each singular value when every column is shuffled independently (kills shared structure).

    Returns (quantiles, n_signal) where n_signal = length of the leading run of observed values above the null.
    """
    rng = np.random.default_rng(seed)
    X = sp["X"]
    null = np.array([np.linalg.svd(np.column_stack([rng.permutation(c) for c in X.T]), compute_uv=False)
                     for _ in range(n_perm)])
    qs = np.quantile(null, q, axis=0)
    return qs, int(np.cumprod(sp["s"] > qs).sum())


def SilhouetteGrid(sp, r_max=5, k_max=8):
    """Silhouette of k-means for each embedding dimension r (rows) and number of clusters k (columns)."""
    rows = {}
    for r in range(1, r_max + 1):
        Z = Embed(sp, r)
        rows[r] = {k: silhouette_score(Z, KMeans(k, n_init=20, random_state=SEED).fit_predict(Z)) for k in range(2, k_max + 1)}
    return pd.DataFrame(rows).T.rename_axis(index="r", columns="k")


def Cluster(sp, nclust, r=None):
    """k-means with nclust groups on the (nclust - 1)-dim embedding (lecture rule) unless r is given."""
    Z = Embed(sp, nclust - 1 if r is None else r)
    return KMeans(nclust, n_init=50, random_state=SEED).fit_predict(Z), Z


def Loadings(sp, r, topics=None, n=4):
    """Topics at the low / high end of each right singular vector."""
    rows = []
    for j in range(r):
        v = pd.Series(sp["Vt"][j], index=sp["topics"]).sort_values()
        name = (lambda t: Short(topics, t)) if topics is not None else str
        rows.append(dict(vector=f"v{j + 1}", low=", ".join(map(name, v.index[:n])), high=", ".join(map(name, v.index[-n:]))))
    return pd.DataFrame(rows).set_index("vector")


def DistinctiveTopics(d, T, cluster, n=2):
    """Topics whose cluster mean most exceeds the overall mean."""
    return (d.loc[d["cluster"] == cluster, T].mean() - d[T].mean()).nlargest(n).index.tolist()


# ---------------------------------------------------------------- plots (each returns the figure)

def PlotScree(sp, null_q=None, title=None):
    fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
    idx = np.arange(1, len(sp["s"]) + 1)
    ax.plot(idx, sp["s"], "o-", label="observed")
    if null_q is not None:
        ax.plot(idx, null_q, "--", color="gray", label="shuffled null, 95%")
        ax.legend(fontsize=8)
    ax.set(xlabel="Index", ylabel="Singular value",
           title=title or f"Singular values of {'standardized X*' if sp['standardize'] else 'raw district × topic matrix'}")
    return fig


def PlotProfile(d, T, title="Mean topic weight by cluster"):
    profile = d.groupby("cluster")[T].mean()
    fig, ax = plt.subplots(figsize=(.55 * len(T) + 2, 2 + .4 * len(profile)), layout="constrained")
    im = ax.imshow(profile, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(T)), T)
    ax.set_yticks(range(len(profile)), [f"cluster {c} (n={(d['cluster'] == c).sum()})" for c in profile.index])
    for i in range(profile.shape[0]):
        for j in range(profile.shape[1]):
            ax.text(j, i, f"{profile.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Mean topic weight")
    ax.set_title(title)
    return fig


def PlotEmbedding(Z, clusters, title="Spectral embedding"):
    fig, ax = plt.subplots(figsize=(6, 4 if Z.shape[1] == 1 else 5), layout="constrained")
    if Z.shape[1] == 1:   # 1-D embedding: histogram by cluster
        for c in sorted(np.unique(clusters)):
            ax.hist(Z[clusters == c, 0], bins=40, alpha=.6, label=f"cluster {c}")
        ax.set(xlabel="Embedding dim 1", ylabel="Districts")
        ax.legend(fontsize=8)
    else:
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=clusters, cmap="tab10", s=12, alpha=.7)
        ax.set(xlabel="Embedding dim 1", ylabel="Embedding dim 2")
        ax.legend(*sc.legend_elements(), title="Cluster", fontsize=8)
    ax.set_title(title)
    return fig


def PlotPartisanship(d):
    p = d.dropna(subset=["gop_2p_2020"])
    groups = sorted(p["cluster"].unique())
    fig, ax = plt.subplots(figsize=(1.2 * len(groups) + 2, 3.5), layout="constrained")
    ax.boxplot([p.loc[p["cluster"] == c, "gop_2p_2020"] for c in groups], tick_labels=[f"cluster {c}" for c in groups])
    ax.axhline(.5, color="gray", lw=.8, ls="--")
    ax.set(ylabel="County GOP two-party share, 2020", title="Partisanship by cluster")
    return fig


def PlotMap(d, title="District clusters, lower 48 (district office location)"):
    m = d[d["longitude"].between(-125, -66) & d["latitude"].between(24, 50)]
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    for c, g in m.groupby("cluster"):
        ax.scatter(g["longitude"], g["latitude"], s=14, alpha=.75, label=f"cluster {c} (n={len(g)})")
    ax.set(xlabel="Longitude", ylabel="Latitude", title=title)
    ax.set_aspect(1.25)
    ax.legend(fontsize=8)
    return fig


# ---------------------------------------------------------------- partisanship and geography

def AddCovariates(d):
    """Merge name, state, coordinates, locale (City/Suburb/Town/Rural), Census region and 2020 county GOP share."""
    static = pd.read_parquet(STATIC)
    cov = static[["leaid", "lea_name", "state_location", "latitude", "longitude", "urban_centric_locale", "gop_2p_2020"]].copy()
    cov["locale"] = (pd.to_numeric(cov["urban_centric_locale"], errors="coerce") // 10).map(LOCALE)
    cov["region"] = cov["state_location"].map(REGION)
    d = d.drop(columns=cov.columns.drop("leaid"), errors="ignore").merge(cov, on="leaid", how="left")
    print(f"merge: {d['lea_name'].notna().mean():.1%} matched; "
          f"gop_2p_2020 {d['gop_2p_2020'].notna().mean():.1%}, locale {d['locale'].notna().mean():.1%}")
    return d


def Eta2(y, g):
    """Share of variance in y explained by the group means of g."""
    y = np.asarray(y, float)
    codes = pd.factorize(np.asarray(g))[0]
    means = np.bincount(codes, y) / np.bincount(codes)
    return means[codes].var() / y.var()


def WithinStrataPerm(y, g, strata, n_perm=5000, seed=SEED):
    """Permutation p-value for Eta2(y, g), shuffling g only within strata (e.g. locale)."""
    rng = np.random.default_rng(seed)
    y, g, strata = np.asarray(y, float), np.asarray(g), np.asarray(strata)
    obs = Eta2(y, g)
    idx = [np.flatnonzero(strata == s) for s in pd.unique(strata)]
    null = np.empty(n_perm)
    for b in range(n_perm):
        gp = g.copy()
        for ix in idx:
            gp[ix] = rng.permutation(g[ix])
        null[b] = Eta2(y, gp)
    return obs, (null >= obs).mean()


def Partisanship(d, n_perm=5000):
    """GOP share by cluster, Kruskal-Wallis, eta^2 and the within-locale permutation p-value."""
    p = d.dropna(subset=["gop_2p_2020"])
    table = p.groupby("cluster")["gop_2p_2020"].describe()[["count", "mean", "25%", "50%", "75%"]]
    kw = stats.kruskal(*[g["gop_2p_2020"] for _, g in p.groupby("cluster")])
    pl = p.dropna(subset=["locale"])
    eta2, perm_p = WithinStrataPerm(pl["gop_2p_2020"], pl["cluster"], pl["locale"], n_perm)
    return dict(table=table, kruskal_stat=kw.statistic, kruskal_p=kw.pvalue, eta2=eta2, perm_p_within_locale=perm_p)


def CramersV(ct):
    chi2, pval, *_ = stats.chi2_contingency(ct)
    return np.sqrt(chi2 / (ct.to_numpy().sum() * (min(ct.shape) - 1))), pval


def Geography(d, variables=("locale", "region")):
    """Cramer's V, chi-square p and row-% crosstab of cluster against each variable."""
    out = {}
    for var in variables:
        ct = pd.crosstab(d["cluster"], d[var])
        v, pval = CramersV(ct)
        out[var] = dict(cramers_v=v, p=pval, row_pct=pd.crosstab(d["cluster"], d[var], normalize="index") * 100)
    return out


def PCRegression(d, sp, r):
    """OLS of GOP share on the r standardized embedding coordinates plus locale dummies (plain OLS SEs)."""
    pcs = pd.DataFrame(Embed(sp, r), columns=[f"PC{j + 1}" for j in range(r)])
    pcs = (pcs - pcs.mean()) / pcs.std()
    pcs["leaid"] = sp["leaid"]
    df = pcs.merge(d[["leaid", "gop_2p_2020", "locale"]], on="leaid").dropna()
    names = [f"PC{j + 1}" for j in range(r)]
    A = np.column_stack([np.ones(len(df)), df[names], pd.get_dummies(df["locale"], drop_first=True).astype(float)])
    y = df["gop_2p_2020"].to_numpy()
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ b
    se = np.sqrt(np.diag(res @ res / (len(y) - A.shape[1]) * np.linalg.inv(A.T @ A)))
    rho = [stats.spearmanr(df[c], y)[0] for c in names]
    return pd.DataFrame(dict(spearman_rho=rho, coef_per_sd=b[1:r + 1], se=se[1:r + 1], t=b[1:r + 1] / se[1:r + 1]), index=names)


# ---------------------------------------------------------------- inspecting a cluster

def TermRates(year, meetings, d, terms=HOT_TERMS):
    """Mentions per 10,000 words of each term, averaged over meetings, by cluster."""
    ct = ChunkText(year)
    text = ct.groupby("id")["text"].agg(" ".join)
    nwords = text.str.count(" ") + 1
    rates = pd.DataFrame({t: text.str.count(rf"\b{re.escape(t)}\b") / nwords * 10_000 for t in terms})
    cl = meetings[["id", "leaid"]].merge(d[["leaid", "cluster"]], on="leaid").set_index("id")["cluster"]
    rates = rates.join(cl).dropna(subset=["cluster"])
    return rates.groupby("cluster")[list(terms)].mean().T


def TopMeetings(meetings, d, cluster, topics, n=15):
    """Meetings in a cluster with the most combined weight on topics, with titles and dates."""
    meta = pd.read_parquet(META, columns=["id", "title", "meeting_date"])
    cols = ["leaid", "cluster", "lea_name", "state_location", "gop_2p_2020"]
    mt = meetings.merge(meta, on="id", how="left").merge(d[[c for c in cols if c in d]], on="leaid", how="left")
    mt["focus"] = mt[topics].sum(axis=1)
    keep = ["id", "lea_name", "state_location", "gop_2p_2020", "meeting_date", "title", *topics, "focus"]
    return mt[mt["cluster"] == cluster].nlargest(n, "focus")[[c for c in keep if c in mt]]


def Excerpts(year, k, top, topics, n=5, chars=800):
    """Text of the chunk with the most weight on topics in each of the first n meetings of top."""
    cw = pd.read_parquet(RESULTS_ROOT / Tag(year) / f"theta_chunks_k{k}.parquet", columns=["id", "chunk", *topics])
    cw["focus"] = cw[topics].sum(axis=1)
    best = cw[cw["id"].isin(top["id"])].sort_values("focus", ascending=False).drop_duplicates("id")
    text = ChunkText(year).merge(best[["id", "chunk"]], on=["id", "chunk"]).set_index("id")["text"]
    out = []
    for _, r in top.head(n).iterrows():
        body = text.get(r["id"])
        out.append((f"{r.get('lea_name', r['id'])} ({r.get('state_location', '')}), {r.get('title', '')}  https://youtu.be/{r['id']}",
                    body[:chars] + " ..." if isinstance(body, str) else "(chunk text not found)"))
    return out


# ---------------------------------------------------------------- IF-PCA on words

def WordCounts(year):
    """District x word count matrix over the run's vocabulary. Returns (D, words, leaids)."""
    import joblib
    sys.path.insert(0, str(REPO / "source/analysis/lda_2018_2023"))
    import workflow  # noqa: F401  (the pickled vectorizer references it)
    vec = joblib.load(WORK_ROOT / Tag(year) / "vectorizer.joblib")
    ct = ChunkText(year)
    C = vec.transform(ct["text"])
    codes, leaids = pd.factorize(ct["leaid"])
    G = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(leaids), len(codes)))
    return (G @ C).toarray(), vec.get_feature_names_out(), np.asarray(leaids)


def DispersionScreen(D, words, min_doc_share=.2):
    """Per-word over-dispersion vs 'same rate in every district' (multinomial null), Efron empirical null -> p-values.

    Only words used by >= min_doc_share of districts. Returns (p-values, kept words, kept column mask).
    """
    keep = (D > 0).mean(0) >= min_doc_share
    Dk = D[:, keep]
    n = Dk.shape[0]
    E = D.sum(1, keepdims=True) * (Dk.sum(0) / D.sum())
    lz = np.log(((Dk - E) ** 2 / E).sum(0) / (n - 1))
    zs = (lz - np.median(lz)) / (1.4826 * np.median(np.abs(lz - np.median(lz))))
    return stats.norm.sf(zs), words[keep], keep


def HCThreshold(pv, n):
    """Number of features to keep: argmax of the IF-PCA Higher Criticism functional (lecture slide 21)."""
    p = len(pv)
    ps, k = np.sort(pv), np.arange(1, p + 1)
    hc = np.sqrt(p) * (k / p - ps) / np.sqrt(k / p + np.maximum(np.sqrt(n) * (k / p - ps), 0))
    ok = (k <= p / 2) & (ps > np.log(p) / p)
    return int(k[ok][np.argmax(hc[ok])])


def IFPCA(year, nclust=2, min_doc_share=.2):
    """Dispersion screen + HC threshold + spectral clustering on standardized sqrt word frequencies."""
    D, words, leaids = WordCounts(year)
    pv, W, keep = DispersionScreen(D, words, min_doc_share)
    khat = HCThreshold(pv, D.shape[0])
    sel = np.argsort(pv)[:khat]
    X = np.sqrt(D[:, keep] / D.sum(1, keepdims=True))
    Xs = ((X - X.mean(0)) / X.std(0))[:, sel]
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    labels = KMeans(nclust, n_init=50, random_state=SEED).fit_predict(U[:, :nclust - 1])
    vectors = [dict(vector=f"v{j + 1}", low=", ".join(W[sel][np.argsort(Vt[j])[:12]]),
                    high=", ".join(W[sel][np.argsort(Vt[j])[-12:]])) for j in range(2)]
    return dict(n_words=len(W), khat=khat, top_words=list(W[sel[:40]]), vectors=pd.DataFrame(vectors).set_index("vector"),
                clusters=pd.DataFrame({"leaid": leaids, "cluster_ifpca": labels, "u1": U[:, 0]}))


def CompareIFPCA(d, ifpca):
    """ARI between topic and word clusters, GOP share and locale mix of the word clusters."""
    cmp = d.merge(ifpca["clusters"], on="leaid")
    return dict(ari=adjusted_rand_score(cmp["cluster"], cmp["cluster_ifpca"]),
                gop=cmp.groupby("cluster_ifpca")["gop_2p_2020"].agg(["count", "mean"]),
                locale=pd.crosstab(cmp["cluster_ifpca"], cmp["locale"], normalize="index") * 100)


# ---------------------------------------------------------------- one run, no plots

def Summarize(year, k, nclust=None, standardize=True, min_topic_mean=.02, min_meetings=1, n_perm=2000):
    """Headline numbers for one (year, K) run, for comparing runs side by side (no plots, no IF-PCA)."""
    meetings, district, T = LoadDistricts(year, k, min_meetings)
    sp = Decompose(district, T, standardize, min_topic_mean)
    null_q, n_signal = ParallelAnalysis(sp)
    nclust = nclust or max(n_signal + 1, 2)
    labels, Z = Cluster(sp, nclust)
    d = district.assign(cluster=labels)
    d = AddCovariates(d)
    part = Partisanship(d, n_perm)
    geo = Geography(d)
    reg = PCRegression(d, sp, nclust - 1)
    gop = part["table"]["mean"]
    return dict(year=year, k=k, meetings=len(meetings), districts=len(d), topics_used=len(sp["topics"]),
                signal_dims=n_signal, nclust=nclust, silhouette=silhouette_score(Z, labels),
                gop_min=gop.min(), gop_max=gop.max(), eta2_gop=part["eta2"], perm_p_within_locale=part["perm_p_within_locale"],
                locale_v=geo["locale"]["cramers_v"], region_v=geo["region"]["cramers_v"],
                max_abs_pc_t=reg["t"].abs().max(), pcs_t_over_2=", ".join(reg.index[reg["t"].abs() > 2]))
