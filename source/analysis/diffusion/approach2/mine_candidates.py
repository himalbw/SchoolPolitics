"""Step 1b: data-driven candidate talking-point phrases (free, local; no API calls).

Pass 1 (all kept transcripts): every 3-5-gram that contains at least one ANCHOR word (contested-politics vocabulary) is
hashed to 64 bits. Per district we keep, for PRE and POST years separately, the number of transcripts containing each
hash. Aggregated: districts and states using it in POST, transcript share in PRE vs POST.
Candidates = n-grams in >= MIN_POST_DISTRICTS districts and >= MIN_POST_STATES states in POST whose transcript share
rises at least MIN_RATIO-fold from PRE to POST (with +1 smoothing). Overlapping n-grams are not merged here: we pick by
hand from the ranked list (phrases.py).
Pass 2 (a sample of POST transcripts): recovers the strings behind the candidate hashes.
Also: all 3-5-grams of the GABRIEL illustrative quotes (output/tables/quotes_*.csv) are scored with the same counts.

Writes output/tables/diffusion/approach2/candidates_ngram.csv (top MAX_OUT by rise) and candidates_quotes.csv.
Run from the repo root:  python3 source/analysis/diffusion/approach2/mine_candidates.py
"""
import re
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402

ANCHORS = set("""parent parents parental rights freedom freedoms liberty mask masks masking unmask unmasked mandate mandates
mandated choice indoctrination indoctrinate indoctrinating ideology ideologies gender crt woke groom groomer groomers
grooming porn pornography pornographic explicit obscene curriculum transparency constitution constitutional tyranny
vaccine vaccines vaccinated vaccination breathe medical experimental equity inclusion diversity racism racist
lgbtq transgender pronouns marxist marxism agenda divisive sexual sexually biological bathroom bathrooms books library
abuse child children kids""".split())
NS = (3, 4, 5)
MIN_POST_DISTRICTS, MIN_POST_STATES, MIN_RATIO = 20, 5, 5.0
MAX_OUT = 4000
# caption-format artefacts (fillers absent from pre-2020 captions) and procedural / reopening boilerplate
BOILERPLATE = (r"\b(um|uh|i i|and and|the the|agenda|indivisible|pledge|kids back|children back|child care)\b"
               r"|back (in|to|into) school|in person")
COVID_LOGISTICS = r"mask|mass|vaccin|covid|quarantin|distanc|virtual|remote|zoom|hybrid|wear|pfizer"
N_PASS2_DOCS = 8000


def Main():
    df = cm.LoadSample()
    df["post"] = df.year.isin(cm.POST_YEARS)
    if len(sys.argv) > 1:  # quick test on the first N districts
        df = df.loc[df.leaid.isin(df.leaid.drop_duplicates().head(int(sys.argv[1])))]
    groups = [(g.file.tolist(), g.post.tolist()) for _, g in df.groupby("leaid")]
    leaids = [k for k, _ in df.groupby("leaid")]
    state_of = df.drop_duplicates("leaid").set_index("leaid").state
    n_pre_docs, n_post_docs = int((~df.post).sum()), int(df.post.sum())
    with Pool(cm.N_WORKERS) as pool:
        res = pool.map(DistrictHashes, groups, chunksize=4)
    print(f"pass 1 done: {len(res)} districts", flush=True)
    stats = Aggregate(res, [state_of[k] for k in leaids])
    stats["pre_share"] = stats.pre_docs / n_pre_docs
    stats["post_share"] = stats.post_docs / n_post_docs
    stats["ratio"] = ((stats.post_docs + 1) / n_post_docs) / ((stats.pre_docs + 1) / n_pre_docs)
    keep = stats.loc[(stats.post_districts >= MIN_POST_DISTRICTS) & (stats.post_states >= MIN_POST_STATES)
                     & (stats.ratio >= MIN_RATIO)].copy()
    print(f"{len(keep):,} hashes pass the spread/rise screen", flush=True)
    sample = df.loc[df.post].sample(min(N_PASS2_DOCS, int(df.post.sum())), random_state=cm.GLOBALS["GABRIEL_SEED"])
    wanted = set(keep.index.tolist())
    with Pool(cm.N_WORKERS) as pool:
        maps = pool.starmap(RecoverStrings, [(f, wanted) for f in np.array_split(sample.file.values, 40)])
    strings = {}
    for mp in maps:
        strings.update(mp)
    keep["ngram"] = [strings.get(int(h), "") for h in keep.index]
    keep = keep.loc[keep.ngram != ""].sort_values("ratio", ascending=False)
    keep = keep.loc[~keep.ngram.str.contains(BOILERPLATE)]
    keep["covid_logistics"] = keep.ngram.str.contains(COVID_LOGISTICS)
    keep = keep.sort_values(["post_districts"], ascending=False).head(MAX_OUT)
    cm.OUTDIR_TABLES.mkdir(parents=True, exist_ok=True)
    cols = ["ngram", "covid_logistics", "post_districts", "post_states", "pre_districts", "pre_docs", "post_docs", "pre_share", "post_share", "ratio"]
    keep[cols].to_csv(cm.OUTDIR_TABLES / "candidates_ngram.csv", index=False, float_format="%.5g")
    print(keep[cols].head(80).to_string())
    ScoreQuotes(stats, n_pre_docs, n_post_docs)
    return


def Tokens(text):
    return text.split()


def AnchoredNgrams(toks):
    """Yield every 3-5-gram (as a string) that contains at least one anchor word."""
    n_tok = len(toks)
    seen_start = set()
    for i, t in enumerate(toks):
        if t not in ANCHORS:
            continue
        for n in NS:
            for s in range(max(0, i - n + 1), min(i, n_tok - n) + 1):
                if (s, n) in seen_start:
                    continue
                seen_start.add((s, n))
                yield " ".join(toks[s:s + n])


def Hash(s):
    b = s.encode()
    return (zlib.crc32(b) << 32) | zlib.adler32(b)


def DistrictHashes(args):
    files, posts = args
    out = {False: [], True: []}
    for f, p in zip(files, posts):
        toks = Tokens(cm.ReadText(f))
        hs = np.fromiter({Hash(g) for g in AnchoredNgrams(toks)}, dtype=np.uint64)
        out[p].append(hs)
    res = {}
    for p in (False, True):
        if out[p]:
            u, c = np.unique(np.concatenate(out[p]), return_counts=True)
        else:
            u, c = np.array([], dtype=np.uint64), np.array([], dtype=np.int64)
        res[p] = (u, c.astype(np.int32))
    return res


def Aggregate(res, states):
    """Per hash: districts and transcripts in PRE and POST, and number of states in POST."""
    out = {}
    for p, name in ((False, "pre"), (True, "post")):
        h = np.concatenate([r[p][0] for r in res])
        c = np.concatenate([r[p][1] for r in res])
        u, inv = np.unique(h, return_inverse=True)
        out[f"{name}_districts"] = pd.Series(np.bincount(inv), index=u)
        out[f"{name}_docs"] = pd.Series(np.bincount(inv, weights=c).astype(np.int64), index=u)
        if p:
            st_codes = pd.factorize(pd.Series(states))[0]
            per_state = [np.unique(np.concatenate([r[p][0] for r, s in zip(res, st_codes) if s == k]))
                         for k in range(st_codes.max() + 1)]
            u2, n_st = np.unique(np.concatenate(per_state), return_counts=True)
            out["post_states"] = pd.Series(n_st, index=u2)
    stats = pd.DataFrame(out).fillna(0).astype(np.int64)
    return stats


def RecoverStrings(files, wanted):
    mp = {}
    for f in files:
        for g in set(AnchoredNgrams(Tokens(cm.ReadText(f)))):
            h = Hash(g)
            if h in wanted:
                mp[int(h)] = g
    return mp


def ScoreQuotes(stats, n_pre_docs, n_post_docs):
    rows = []
    for path in cm.INDIR_QUOTES:
        dq = pd.read_csv(path)
        for q in dq.quote.dropna():
            toks = re.sub(r"[^a-z' ]", " ", q.lower().replace("[name]", " ")).split()
            for n in NS:
                for s in range(len(toks) - n + 1):
                    rows.append({"ngram": " ".join(toks[s:s + n]), "quote_file": path.name})
    dq = pd.DataFrame(rows).groupby("ngram").agg(n_quotes=("quote_file", "size")).reset_index()
    dq["h"] = [np.uint64(Hash(g)) for g in dq.ngram]
    dq = dq.merge(stats, left_on="h", right_index=True, how="left").fillna(0)
    dq["ratio"] = ((dq.post_docs + 1) / n_post_docs) / ((dq.pre_docs + 1) / n_pre_docs)
    dq = dq.loc[(dq.post_districts >= MIN_POST_DISTRICTS) & (dq.post_states >= MIN_POST_STATES)]
    dq = dq.sort_values("ratio", ascending=False).drop(columns="h")
    dq.to_csv(cm.OUTDIR_TABLES / "candidates_quotes.csv", index=False, float_format="%.5g")
    print(f"\nquote n-grams (anchored only have counts) passing spread screen: {len(dq)}")
    print(dq.head(40).to_string())


if __name__ == "__main__":
    Main()
