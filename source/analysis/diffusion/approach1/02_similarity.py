"""Step 2: cross-district pairwise similarity of keyword spans, and of the shared placebo ('students') spans.

  python3 .../02_similarity.py placebo          # once: builds the placebo sample and its pair statistics
  python3 .../02_similarity.py <kw> [<kw> ...]  # per keyword (or 'all')

Measures for every pair of hits from DIFFERENT districts (keyword tokens masked):
  lex  = rapidfuzz fuzz.ratio (0-100) on the 15-word span w[i-7 : i+L+7];
  sh5  = number of word 5-grams shared within +-20 words of the keyword (offset-invariant);
  sh5f = sh5 after dropping generic 5-grams (in >= 2% of 600 reference transcripts, one per district);
  ex8  = identical 8-gram containing the keyword.
(The indoctrination trial also had TF-IDF cosine and best-8-gram-half ratios; they added no links beyond
lex + sh5f and were dropped.)
Outputs per keyword in Datastore/.../approach1/keywords/{kw}/: edges.parquet (pairs with lex >= 60, sh5f >= 2 or
ex8; the rest are summarised in hist.json); placebo: keywords/students/{edges.parquet, hist.json, sample_ids}.
"""
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONTROL, DATA_KW, DATA_OUT, KEYWORDS, MANIFEST, MASK, SEED, TRANSCRIPTS

LEX_FLOOR, SH_FLOOR, SH_HALF, CHUNK = 60, 2, 20, 1500
N_PLACEBO = 8000


def generic_5grams(n_docs=600, min_share=0.02):
    path = DATA_OUT / "generic_5grams.parquet"
    if path.exists():
        return set(pd.read_parquet(path).ngram)
    man = pd.read_parquet(MANIFEST, columns=["file", "leaid", "status"])
    man = man[man.status == "kept"].groupby("leaid").sample(1, random_state=SEED).sample(n_docs, random_state=SEED)
    cnt = Counter()
    for f in man.file:
        w = gzip.open(TRANSCRIPTS / f, "rt").read().split()
        cnt.update({" ".join(w[k:k + 5]) for k in range(len(w) - 4)})
    g = pd.DataFrame([(k, v) for k, v in cnt.items() if v >= min_share * n_docs], columns=["ngram", "n_docs"])
    g.to_parquet(path, index=False)
    return set(g.ngram)


def masked(text, kpos, L):
    w = text.split()
    for k in range(kpos, min(len(w), kpos + L)):
        w[k] = MASK
    return w


def ex8_pairs(spans, kpos, lea):
    idx = {}
    for n, (w, k) in enumerate(zip(spans, kpos)):
        for s in range(max(0, k - 7), k + 1):
            if s + 8 <= len(w):
                idx.setdefault(tuple(w[s:s + 8]), []).append(n)
    out = set()
    for m in idx.values():
        if 2 <= len(m) <= 2000:
            for a in range(len(m)):
                for b in range(a + 1, len(m)):
                    i, j = sorted((m[a], m[b]))
                    if lea[i] != lea[j]:
                        out.add((i, j))
    return out


def run(df, outdir, generic):
    df = df.reset_index(drop=True)
    n = len(df)
    lea = df.leaid.to_numpy()
    spans = [masked(s, k, L) for s, k, L in zip(df.span, df.span_kpos, df.kw_len)]
    span_txt = [" ".join(w) for w in spans]
    ctx = [masked(s, k, L) for s, k, L in zip(df.ctx30, df.ctx30_kpos, df.kw_len)]
    win = [" ".join(w[max(0, k - SH_HALF): k + L + SH_HALF]) for w, k, L in zip(ctx, df.ctx30_kpos, df.kw_len)]
    cv = CountVectorizer(ngram_range=(5, 5), binary=True, token_pattern=r"(?u)\b[\w']+\b", dtype=np.int32)
    B = cv.fit_transform(win).tocsr()
    gen = np.array([t in generic for t in cv.get_feature_names_out()])
    Bf = B.multiply((~gen).astype(np.int32)[None, :]).tocsr()
    Bf.eliminate_zeros()
    sh = None
    for col, M in (("sh5", B), ("sh5f", Bf)):
        S = (M @ M.T).tocoo()
        k = (S.row < S.col) & (lea[S.row] != lea[S.col])
        d = pd.DataFrame({"i": S.row[k], "j": S.col[k], col: S.data[k].astype(int)})
        sh = d if sh is None else sh.merge(d, on=["i", "j"], how="outer")
    sh = sh.fillna(0).astype(int)
    ex8 = ex8_pairs(spans, df.span_kpos.tolist(), lea)
    lex_hist = np.zeros(101, dtype=np.int64)
    edges, n_pairs = [], 0
    for s in range(0, n, CHUNK):
        e = min(n, s + CHUNK)
        L = process.cdist(span_txt[s:e], span_txt, scorer=fuzz.ratio, dtype=np.uint8, workers=-1)
        m = (np.arange(n)[None, :] > np.arange(s, e)[:, None]) & (lea[s:e, None] != lea[None, :])
        n_pairs += int(m.sum())
        lex_hist += np.bincount(L[m], minlength=101)
        ii, jj = np.nonzero(m & (L >= LEX_FLOOR))
        edges.append(pd.DataFrame(dict(i=ii + s, j=jj, lex=L[ii, jj].astype(int))))
    E = pd.concat(edges, ignore_index=True)
    E = E.merge(sh[sh.sh5f >= SH_FLOOR], on=["i", "j"], how="outer")
    E = E.merge(pd.DataFrame(sorted(ex8), columns=["i", "j"]).assign(ex8=True), on=["i", "j"], how="outer")
    miss = E.lex.isna()
    E.loc[miss, "lex"] = [fuzz.ratio(span_txt[i], span_txt[j]) for i, j in zip(E.i[miss], E.j[miss])]
    E = E.drop(columns=["sh5", "sh5f"]).merge(sh, on=["i", "j"], how="left")
    E[["sh5", "sh5f"]] = E[["sh5", "sh5f"]].fillna(0).astype(int)
    E["ex8"] = E.ex8.fillna(False).astype(bool)
    E["lex"] = E.lex.astype(int)
    E["hit_i"] = df.hit_id.to_numpy()[E.i]
    E["hit_j"] = df.hit_id.to_numpy()[E.j]
    E[["hit_i", "hit_j", "lex", "sh5", "sh5f", "ex8"]].to_parquet(outdir / "edges.parquet", index=False)
    hist = dict(n=n, n_pairs=n_pairs, n_districts=int(df.leaid.nunique()), lex_hist=lex_hist.tolist(),
                sh5f_hist=np.bincount(np.clip(sh.sh5f, 0, 40), minlength=41).tolist(), n_ex8=len(ex8),
                n_generic_5grams_in_windows=int(gen.sum()))
    (outdir / "hist.json").write_text(json.dumps(hist))
    print(f"  {outdir.name}: {n:,} hits, {n_pairs:,} cross-district pairs, {len(E):,} edges saved, "
          f"sh5f>=4: {int((E.sh5f >= 4).sum())}, lex>=80: {int((E.lex >= 80).sum())}", flush=True)


def main():
    args = sys.argv[1:]
    generic = generic_5grams()
    if args == ["placebo"]:
        ctrl = pd.read_parquet(DATA_KW(CONTROL) / "hits.parquet")
        pooled = pd.concat([pd.read_parquet(DATA_KW(k) / "hits.parquet", columns=["year"]) for k in KEYWORDS])
        target = (pooled.year.value_counts(normalize=True) * N_PLACEBO).round().astype(int).to_dict()
        parts = [g.sample(min(len(g), target.get(y, 0)), random_state=SEED) for y, g in ctrl.groupby("year")]
        cs = pd.concat(parts)
        print(f"placebo: {len(cs):,} '{CONTROL}' hits matched to the pooled keyword year mix "
              f"{cs.year.value_counts().sort_index().to_dict()}")
        cs[["hit_id"]].to_parquet(DATA_KW(CONTROL) / "sample_ids.parquet", index=False)
        run(cs, DATA_KW(CONTROL), generic)
        return
    kws = list(KEYWORDS) if args == ["all"] else args
    for kw in kws:
        run(pd.read_parquet(DATA_KW(kw) / "hits.parquet"), DATA_KW(kw), generic)


if __name__ == "__main__":
    main()
