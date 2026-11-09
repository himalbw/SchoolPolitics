"""Step 3: placebo false-link rates, talking-point clusters and sensitivity, per keyword.

  python3 .../03_clusters.py            # all keywords
Link rule (PRIMARY): lex >= 80 OR sh5f >= 4, cross-district pairs only (chosen on the indoctrination trial; the
same placebo rate per pair is applied to every keyword, so expected false links scale with each keyword's pairs).
A talking point = connected component spanning >= 2 districts in >= 2 counties.
Outputs: output/tables/diffusion/approach1/tau_placebo.csv, sensitivity.csv,
         output/tables/diffusion/approach1/keywords/{kw}/talking_points.csv,
         Datastore/.../keywords/{kw}/clusters.parquet (hit_id -> cluster, degree; primary rule).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONTROL, DATA_KW, KEYWORDS, TAB_KW, TAB_OUT, scrub_names

PRIMARY = {"lex": 80, "sh5f": 4}
RULES = [{"lex": 75, "sh5f": 4}, {"lex": 80, "sh5f": 4}, {"lex": 80, "sh5f": 5}, {"lex": 85, "sh5f": 6},
         {"lex": 90, "sh5f": 8}, {"sh5f": 4}, {"lex": 80}, "ex8"]


def rule_name(r):
    return "shared exact 8-gram" if r == "ex8" else " or ".join(f"{k}>={v}" for k, v in r.items())


def passes(E, r):
    if r == "ex8":
        return E.ex8.to_numpy()
    m = np.zeros(len(E), dtype=bool)
    for col, v in r.items():
        m |= (E[col] >= v).to_numpy()
    return m


def n_cross_pairs(sub):
    vc = sub.leaid.value_counts()
    return int((vc.sum() ** 2 - (vc ** 2).sum()) // 2)


def components(hits, E, r):
    e = E[passes(E, r)]
    pos = pd.Series(np.arange(len(hits)), index=hits.hit_id.to_numpy())
    e = e[e.hit_i.isin(pos.index) & e.hit_j.isin(pos.index)]
    n = len(hits)
    A = coo_matrix((np.ones(len(e)), (pos[e.hit_i].to_numpy(), pos[e.hit_j].to_numpy())), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    deg = np.bincount(np.r_[pos[e.hit_i].to_numpy(), pos[e.hit_j].to_numpy()], minlength=n)
    out = hits.assign(cluster=lab, degree=deg)
    g = out.groupby("cluster").agg(n_districts=("leaid", "nunique"), n_counties=("county_fips", "nunique"))
    keep = g[(g.n_districts >= 2) & (g.n_counties >= 2)].index
    return out[out.cluster.isin(keep)], len(e)


def summarise(cl):
    rows = []
    for c, g in cl.groupby("cluster"):
        d = g.drop_duplicates("leaid")
        ex = g.sort_values("degree", ascending=False).drop_duplicates("leaid").head(3)
        r = dict(cluster=c, n_hits=len(g), n_districts=g.leaid.nunique(), n_states=g.state_location.nunique(),
                 n_counties=g.county_fips.nunique(), first_date=g.date.min().date(), last_date=g.date.max().date(),
                 mean_gop_2p_2020=round(d.gop_2p_2020.mean(), 3),
                 share_pc_rule_strict=round(g.flag_public_comment.mean(), 2),
                 share_pc_rule_loose=round(g.flag_public_comment_loose.mean(), 2),
                 share_policy_rule=round(g.flag_policy.mean(), 2), share_definition_rule=round(g.flag_definition.mean(), 2),
                 states=",".join(sorted(g.state_location.dropna().unique())))
        for k, s in enumerate(ex.span.tolist(), 1):
            r[f"example_{k}"] = scrub_names(s)
        rows.append(r)
    cols = ["cluster", "n_hits", "n_districts", "n_states", "n_counties"]
    S = pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)
    return S.sort_values(["n_districts", "n_states", "n_hits"], ascending=False).reset_index(drop=True)


def main():
    Ec = pd.read_parquet(DATA_KW(CONTROL) / "edges.parquet")
    hc = json.loads((DATA_KW(CONTROL) / "hist.json").read_text())
    ctrl = pd.read_parquet(DATA_KW(CONTROL) / "hits.parquet")
    cs = ctrl[ctrl.hit_id.isin(pd.read_parquet(DATA_KW(CONTROL) / "sample_ids.parquet").hit_id)]
    nc = hc["n_pairs"]
    rate = {rule_name(r): passes(Ec, r).sum() / nc for r in RULES}
    pcl = {rule_name(r): components(cs, Ec, r)[0].cluster.nunique() for r in RULES}
    tau, sens = [], []
    for kw in KEYWORDS:
        h = pd.read_parquet(DATA_KW(kw) / "hits.parquet")
        E = pd.read_parquet(DATA_KW(kw) / "edges.parquet")
        nk = n_cross_pairs(h)
        for r in RULES:
            name = rule_name(r)
            tau.append(dict(kw=kw, rule=name, placebo_pairs=nc, placebo_links=int(round(rate[name] * nc)),
                            placebo_rate=f"{rate[name]:.2e}", keyword_pairs=nk, keyword_links=int(passes(E, r).sum()),
                            expected_false_links=round(rate[name] * nk, 1),
                            expected_false_clusters=round(pcl[name] * nk / nc, 1)))
        subsets = {"all hits": h, "public comment (rule, loose)": h[h.flag_public_comment_loose],
                   "not policy (rule)": h[~h.flag_policy]}
        for sname, sub in subsets.items():
            for r in RULES:
                c, ne = components(sub, E, r)
                gs = c.groupby("cluster").leaid.nunique() if len(c) else pd.Series(dtype=int)
                sens.append(dict(kw=kw, subset=sname, n_hits=len(sub), rule=rule_name(r), n_links=ne,
                                 expected_false_links=round(rate[rule_name(r)] * n_cross_pairs(sub), 1),
                                 n_talking_points=len(gs), n_tp_ge3_districts=int((gs >= 3).sum()),
                                 n_tp_ge5_districts=int((gs >= 5).sum()), largest_tp_districts=int(gs.max()) if len(gs) else 0))
        cl, ne = components(h, E, PRIMARY)
        cl[["hit_id", "cluster", "degree"]].to_parquet(DATA_KW(kw) / "clusters.parquet", index=False)
        S = summarise(cl)
        TAB_KW(kw).mkdir(parents=True, exist_ok=True)
        S.to_csv(TAB_KW(kw) / "talking_points.csv", index=False)
        print(f"{kw}: {len(h):,} hits, {nk:,} pairs, {ne} links (expected false {rate[rule_name(PRIMARY)] * nk:.1f}), "
              f"{len(S)} talking points, {len(cl)} hits in them; largest {S.n_districts.max() if len(S) else 0} districts")
    pd.DataFrame(tau).to_csv(TAB_OUT / "tau_placebo.csv", index=False)
    pd.DataFrame(sens).to_csv(TAB_OUT / "sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
