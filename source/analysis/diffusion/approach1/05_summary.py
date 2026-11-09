"""Step 5: combine clusters, LLM labels and pair judgements into per-keyword summaries.

Outputs: output/tables/diffusion/approach1/keyword_summary.csv / .tex, rule_vs_llm.csv, pair_adjudication.csv,
         keywords/{kw}/talking_points.csv (adds LLM-based cluster type), top_public_comment_talking_points.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONTROL, DATA_KW, DATA_OUT, KEYWORDS, TAB_KW, TAB_OUT, scrub_names

MODEL = "gpt-6-luna"
QUOTE = {"dictionary", "news_media", "advocacy_group", "other"}
LABEL = {"indoctrination": "indoctrinat*", "mask": "mask(s)", "crt": "CRT / critical race theory",
         "parental_rights": "parental / parents' rights", "groom": "groom*", "pornography": "pornograph* / porn",
         "equity": "equity", "vaccine": "vaccin* / vax", "transgender": "transgender*"}


def cluster_type(g):
    n = len(g)
    quote = (g.is_quoting & g.quote_source.isin(QUOTE)).sum() / n
    official = (g.speaker_role.isin(["reading_official_text", "board_member", "staff_administrator"])
                | (g.is_quoting & g.quote_source.eq("statute_policy"))).sum() / n
    pc = g.speaker_role.eq("public_commenter").sum() / n
    if quote >= 0.5:
        return "definition/quotation"
    if official >= 0.5:
        return "official/boilerplate"
    if pc >= 0.5:
        return "public comment"
    return "mixed"


def main():
    counts = pd.read_csv(DATA_OUT / "counts.csv").set_index("kw")
    tau = pd.read_csv(TAB_OUT / "tau_placebo.csv")
    rows, agree, pairs, top_pc = [], [], [], []
    for kw in KEYWORDS:
        h = pd.read_parquet(DATA_KW(kw) / "hits.parquet")
        cl = pd.read_parquet(DATA_KW(kw) / "clusters.parquet")
        lab = pd.read_parquet(DATA_KW(kw) / f"llm_hits_{MODEL}.parquet")
        if "error" in lab:
            lab = lab[lab.error.isna()]
        lab = lab.drop_duplicates("hit_id").merge(h, on="hit_id")
        rnd = lab[lab.hit_id.isin(h.sample(min(150, len(h)), random_state=236).hit_id)]
        llm_pc = rnd.speaker_role.eq("public_commenter")
        for fl in ("flag_public_comment", "flag_public_comment_loose"):
            f = rnd[fl]
            agree.append(dict(kw=kw, rule=fl, n=len(rnd), agreement=round((f == llm_pc).mean(), 2),
                              precision=round((f & llm_pc).sum() / max(f.sum(), 1), 2),
                              recall=round((f & llm_pc).sum() / max(llm_pc.sum(), 1), 2),
                              llm_pc_share=round(llm_pc.mean(), 2), rule_share=round(f.mean(), 2)))
        # cluster types from LLM labels of their hits
        S = pd.read_csv(TAB_KW(kw) / "talking_points.csv")
        c = cl.merge(lab[["hit_id", "speaker_role", "is_quoting", "quote_source", "stance",
                          "in_public_comment_period"]], on="hit_id", how="left")
        typ = c.dropna(subset=["speaker_role"]).groupby("cluster").apply(cluster_type, include_groups=False)
        stance = c.groupby("cluster").stance.agg(lambda s: s.value_counts(normalize=True).round(2).to_dict())
        pcs = c.groupby("cluster").speaker_role.apply(lambda s: round(s.eq("public_commenter").mean(), 2))
        if len(S):
            S["type_llm"] = S.cluster.map(typ)
            S["share_public_commenter_llm"] = S.cluster.map(pcs)
            S["stance_llm"] = S.cluster.map(stance).astype(str)
            S.to_csv(TAB_KW(kw) / "talking_points.csv", index=False)
        pcS = S[S.type_llm.eq("public comment")] if len(S) else S
        best = pcS.iloc[0] if len(pcS) else None
        if len(pcS):
            top_pc.append(pcS.head(5).assign(kw=kw))
        prim = tau[(tau.kw == kw) & (tau.rule == "lex>=80 or sh5f>=4")].iloc[0]
        pj = DATA_KW(kw) / f"llm_pairs_{MODEL}.parquet"
        if pj.exists():
            P = pd.read_parquet(pj)
            P = P[P.get("error", pd.Series(index=P.index, dtype=object)).isna()] if "error" in P else P
            E = pd.read_parquet(DATA_KW(kw) / "edges.parquet")
            nb = int((~((E.lex >= 80) | (E.sh5f >= 4)) & (E.sh5f.between(2, 3) | E.lex.between(65, 79))).sum())
            cmap = cl.set_index("hit_id").cluster
            same = P[P.same_talking_point.astype(bool)]
            ci, cj = same.hit_i.map(cmap), same.hit_j.map(cmap)
            pairs.append(dict(kw=kw, borderline_pairs=nb, judged=len(P),
                              share_same_not_already_clustered=round((~(ci.notna() & (ci == cj))).sum() / len(P), 3),
                              share_same_talking_point=round(P.same_talking_point.mean(), 3),
                              kinds=P.kind.value_counts().to_dict(),
                              implied_missed_links=int(round(P.same_talking_point.mean() * nb)),
                              linked_pairs=int(prim.keyword_links)))
        rows.append(dict(
            keyword=LABEL[kw], kw=kw, n_hits_all=int(counts.loc[kw, "n_hits_all"]),
            n_districts_all=int(counts.loc[kw, "n_districts_all"]), n_hits_used=len(h),
            sample_share=round(float(counts.loc[kw, "keep_p"]), 3),
            share_public_comment_llm=round(llm_pc.mean(), 2),
            share_public_comment_rule_loose=round(h.flag_public_comment_loose.mean(), 2),
            links=int(prim.keyword_links), expected_false_links=prim.expected_false_links,
            n_tp_all=len(S), n_tp_public_comment=len(pcS),
            share_tp_official=round(S.type_llm.eq("official/boilerplate").mean(), 2) if len(S) else np.nan,
            share_tp_quotation=round(S.type_llm.eq("definition/quotation").mean(), 2) if len(S) else np.nan,
            largest_tp_districts=int(S.n_districts.max()) if len(S) else 0,
            largest_pc_tp_districts=int(best.n_districts) if best is not None else 0,
            largest_pc_tp_states=int(best.n_states) if best is not None else 0,
            largest_pc_tp_dates=f"{best.first_date} to {best.last_date}" if best is not None else "",
            largest_pc_tp_example=scrub_names(best.example_1) if best is not None else ""))
    K = pd.DataFrame(rows)
    K.to_csv(TAB_OUT / "keyword_summary.csv", index=False)
    pd.DataFrame(agree).to_csv(TAB_OUT / "rule_vs_llm.csv", index=False)
    pp = DATA_KW(CONTROL) / f"llm_pairs_{MODEL}.parquet"
    if pp.exists():   # placebo: the same judge on borderline 'students' pairs = LLM false-positive baseline
        P = pd.read_parquet(pp)
        pairs.append(dict(kw="placebo (students)", judged=len(P),
                          share_same_talking_point=round(P.same_talking_point.mean(), 3), kinds=P.kind.value_counts().to_dict()))
    pd.DataFrame(pairs).to_csv(TAB_OUT / "pair_adjudication.csv", index=False)
    if top_pc:
        pd.concat(top_pc).to_csv(TAB_OUT / "top_public_comment_talking_points.csv", index=False)
    write_tex(K)
    pd.set_option("display.width", 250)
    print(K.drop(columns=["largest_pc_tp_example"]).to_string(index=False))
    print(pd.DataFrame(agree).to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))


def tex_escape(s):
    return str(s).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def write_tex(K):
    lines = [r"\begin{tabular}{lrrrrrrrl}", r"\toprule",
             r"Keyword & Hits & Districts & PC share & Links (exp.\ false) & TPs & PC TPs & Official & Largest PC TP \\",
             r" & & & (LLM) & & & & share & (districts/states) \\", r"\midrule"]
    for r in K.itertuples():
        hits = f"{r.n_hits_all:,}" + (r"$^{*}$" if r.sample_share < 1 else "")
        big = f"{r.largest_pc_tp_districts}/{r.largest_pc_tp_states}" if r.largest_pc_tp_districts else "--"
        off = f"{r.share_tp_official:.2f}" if pd.notna(r.share_tp_official) else "--"
        lines.append(f"{tex_escape(r.keyword)} & {hits} & {r.n_districts_all:,} & {r.share_public_comment_llm:.2f} & "
                     f"{r.links} ({r.expected_false_links:.0f}) & {r.n_tp_all} & {r.n_tp_public_comment} & {off} & {big} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", "",
              r"% $^{*}$ random sample of hits used for the pairwise step (cap 8,000). PC share: gpt-6-luna "
              r"speaker role = public commenter, 150 random hits. Links: cross-district pairs with lex $\geq$ 80 or "
              r"$\geq$ 4 shared non-generic 5-grams; expected false links from the `students' placebo. TPs: clusters "
              r"spanning $\geq$ 2 districts in $\geq$ 2 counties; PC TPs: clusters whose LLM-labelled hits are mostly "
              r"public commenters; Official share: share of TPs mostly official text / board / staff."]
    (TAB_OUT / "keyword_summary.tex").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
