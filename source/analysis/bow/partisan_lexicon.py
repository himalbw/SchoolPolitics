"""Which education-politics vocabulary is partisan, and did it grow? District-level comparison of R vs D districts.

Uses the district x quarter matrix (bow_dq) and the 2020 county GOP anchor (R = GOP two-party share > 0.5).
Everything is computed on DISTRICT-level rates (one observation per district-period), not pooled tokens, because words are
heavily clustered within districts and token-level z-scores are wildly overconfident.
  1. Lexicon categories (lexicon.py): mentions per 10,000 words, R vs D districts, by period (pre 2018-19, covid 2020-21,
     post 2022-23). Welch t of the R-D gap; label "R"/"D" if |t| > T_PARTISAN, else "not partisan"; compared with our prior.
  2. Quarterly trend of each category's rate, by party.
  3. Every single term: Welch t of the R-D difference in relative frequency, by period; top R and D terms (national
     vocabulary only) and the terms whose partisanship moved most from pre to post.
Writes output/tables/lexicon_*.csv and partisan_terms_*.csv. Run from the repo root (after gst_polarization.py).
"""
import numpy as np
import pandas as pd

from bow_data import CollapseTo, LoadDQ, LoadPhraseCounts, OUTDIR_TABLES, TermStateConcentration, WithAnchor
from lexicon import LEXICON

T_PARTISAN = 3.0
MIN_WORDS_DISTRICT_PERIOD = 20000   # a district-period needs this many words (before vocabulary cuts) to count
MIN_DISTRICTS_USING = 50            # a term must appear in this many district-periods to be ranked
MIN_STATES = 10
MAX_TOP_STATE_SHARE = 0.3
N_TOP = 50

def Main():
    OUTDIR_TABLES.mkdir(parents=True, exist_ok=True)
    dtm, vocab, units = LoadDQ()
    dtm, units = WithAnchor(dtm, units)
    df_phrases = LoadPhraseCounts(units)
    df_conc = TermStateConcentration(dtm, units, vocab)
    category_term_idx = {c: [np.flatnonzero(vocab == t)[0] for t in spec["terms"] if t in set(vocab)] for c, spec in LEXICON.items()}
    report = [(c, len(idx), len(LEXICON[c]["terms"])) for c, idx in category_term_idx.items()]
    print("Lexicon terms found in vocabulary:", ", ".join(f"{c} {a}/{b}" for c, a, b in report))
    df_rates = CategoryRates(dtm, units, df_phrases, category_term_idx, "period")
    CategoryGap(df_rates).to_csv(OUTDIR_TABLES / "lexicon_category_gap.csv", index=False)
    CategoryRates(dtm, units, df_phrases, category_term_idx, "quarter").groupby(["label", "category", "is_r"]).rate_total.mean() \
        .unstack("is_r").rename(columns={True: "R", False: "D"}).reset_index() \
        .to_csv(OUTDIR_TABLES / "lexicon_category_trend_quarter.csv", index=False)
    TermPartisanship(dtm, units, vocab, df_conc)
    return

def CategoryRates(dtm, units, df_phrases, category_term_idx, level):
    """One row per district x period x category: mentions per 10k words (single terms from the matrix, phrases from the text)."""
    counts, df, group = CollapseTo(dtm, units, level)
    term_counts = {c: np.asarray(counts[:, idx].sum(1)).ravel() if idx else np.zeros(counts.shape[0])
                   for c, idx in category_term_idx.items()}
    phrase_counts = {c: group @ df_phrases[spec["phrases"]].sum(1).to_numpy() for c, spec in LEXICON.items()}
    min_words = MIN_WORDS_DISTRICT_PERIOD if level == "period" else MIN_WORDS_DISTRICT_PERIOD / 4
    ok = df.n_words >= min_words
    rows = []
    for c in LEXICON:
        d = df[ok][["leaid", "label", "is_r", "n_words"]].copy()
        d["category"] = c
        d["rate_terms"] = term_counts[c][ok] / d.n_words * 1e4
        d["rate_phrases"] = phrase_counts[c][ok] / d.n_words * 1e4
        d["rate_total"] = d.rate_terms + d.rate_phrases      # a phrase's words are also single terms, so this slightly double-counts
        rows.append(d)
    return pd.concat(rows, ignore_index=True)

def Welch(a, b):
    na, nb = len(a), len(b)
    se = np.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
    return (a.mean() - b.mean()) / se if se > 0 else np.nan

def CategoryGap(df_rates):
    out = []
    for (cat, label), g in df_rates.groupby(["category", "label"]):
        r, d = g[g.is_r], g[~g.is_r]
        row = {"category": cat, "period": label, "expected": LEXICON[cat]["expected"], "n_R": len(r), "n_D": len(d)}
        for col in ("rate_terms", "rate_phrases", "rate_total"):
            row[f"{col}_R"], row[f"{col}_D"] = r[col].mean(), d[col].mean()
            row[f"{col}_t"] = Welch(r[col], d[col])
        row["observed"] = "R" if row["rate_total_t"] > T_PARTISAN else "D" if row["rate_total_t"] < -T_PARTISAN else "not partisan"
        out.append(row)
    df = pd.DataFrame(out)
    df["period"] = pd.Categorical(df.period, ["pre", "covid", "post"], ordered=True)
    return df.sort_values(["category", "period"])

def TermPartisanship(dtm, units, vocab, df_conc):
    counts, df, _ = CollapseTo(dtm, units, "period")
    n = np.asarray(counts.sum(1)).ravel()
    ok = (df.n_words >= MIN_WORDS_DISTRICT_PERIOD).to_numpy()
    q_all = counts.multiply(1 / n[:, None]).tocsr()
    national = ((df_conc.n_states >= MIN_STATES) & (df_conc.top_state_share <= MAX_TOP_STATE_SHARE)).to_numpy()
    lex_cat = {t: c for c, spec in LEXICON.items() for t in spec["terms"]}
    tables = {}
    for period in ["pre", "covid", "post"]:
        sel = np.flatnonzero(ok & (df.label == period).to_numpy())
        q, is_r = q_all[sel], df.is_r.to_numpy()[sel].astype(bool)
        R, D = q[is_r], q[~is_r]
        mR, mD = np.asarray(R.mean(0)).ravel(), np.asarray(D.mean(0)).ravel()
        vR = (np.asarray(R.multiply(R).sum(0)).ravel() - R.shape[0] * mR ** 2) / (R.shape[0] - 1)
        vD = (np.asarray(D.multiply(D).sum(0)).ravel() - D.shape[0] * mD ** 2) / (D.shape[0] - 1)
        se = np.sqrt(vR / R.shape[0] + vD / D.shape[0])
        t = np.divide(mR - mD, se, out=np.zeros_like(se), where=se > 0)
        tables[period] = pd.DataFrame({
            "term": vocab, "t": t, "freq_R_per_10k": mR * 1e4, "freq_D_per_10k": mD * 1e4,
            "log2_R_over_D": np.log2((mR + 1e-9) / (mD + 1e-9)),
            "n_districts_using": np.asarray((q > 0).sum(0)).ravel(), "n_states": df_conc.n_states.to_numpy(),
            "top_state_share": df_conc.top_state_share.to_numpy(), "national": national,
            "category": [lex_cat.get(x, "") for x in vocab]})
    rankable = lambda d: d[(d.n_districts_using >= MIN_DISTRICTS_USING) & d.national]
    for period, d in tables.items():
        r = rankable(d).sort_values("t")
        pd.concat([r.tail(N_TOP).iloc[::-1].assign(side="R"), r.head(N_TOP).assign(side="D")]) \
            .to_csv(OUTDIR_TABLES / f"partisan_terms_{period}.csv", index=False)
    lex = pd.concat([d[d.category != ""].assign(period=p) for p, d in tables.items()])
    lex.to_csv(OUTDIR_TABLES / "lexicon_term_t.csv", index=False)
    move = tables["pre"][["term", "t", "category"]].merge(tables["post"][["term", "t", "n_districts_using", "n_states", "national"]],
                                                          on="term", suffixes=("_pre", "_post"))
    move["t_change"] = move.t_post - move.t_pre
    move = rankable(move).sort_values("t_change")
    pd.concat([move.tail(N_TOP).iloc[::-1].assign(direction="more R after"), move.head(N_TOP).assign(direction="more D after")]) \
        .to_csv(OUTDIR_TABLES / "partisan_term_movers_pre_to_post.csv", index=False)
    return

if __name__ == "__main__":
    Main()
