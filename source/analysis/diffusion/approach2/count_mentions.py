"""Step 2: count every tracked phrase (phrases.py) in every kept transcript of the screened sample.

Writes (Datastore/DistrictView/derived/diffusion/approach2/)
  mentions.parquet   long table, nonzero rows only: video_id, leaid, date, phrase, count, count_excluded
                     (count = matches without procedural context; count_excluded = matches whose window hit `exclude`)
  docs.parquet       one row per kept transcript: video_id, leaid, date, month, n_words, is_empty (file read as empty)
  kwic.parquet       up to KWIC_PER_PHRASE random keyword-in-context snippets per phrase, for validation
Run from the repo root:  python3 source/analysis/diffusion/approach2/count_mentions.py
"""
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as cm  # noqa: E402
from phrases import EXCL_WINDOW, Compiled  # noqa: E402

PATTERNS = Compiled()
KWIC_PER_PHRASE = 30
KWIC_CHARS = 120


def Main():
    df = cm.LoadSample()
    with Pool(cm.N_WORKERS) as pool:
        res = pool.map(CountOne, df.file.tolist(), chunksize=100)
    rows, kwic = [], []
    df["is_empty"] = [r[0] for r in res]
    for i, (_, counts, ctx) in enumerate(res):
        for ph, (n, n_ex) in counts.items():
            rows.append((df.id[i], df.leaid[i], df.date[i], ph, n, n_ex))
        kwic += [(df.id[i], df.leaid[i], df.date[i], ph, c) for ph, c in ctx]
    dm = pd.DataFrame(rows, columns=["video_id", "leaid", "date", "phrase", "count", "count_excluded"])
    dm[["count", "count_excluded"]] = dm[["count", "count_excluded"]].astype("int32")
    dk = pd.DataFrame(kwic, columns=["video_id", "leaid", "date", "phrase", "context"])
    dk = dk.sample(frac=1, random_state=cm.GLOBALS["GABRIEL_SEED"]).groupby("phrase").head(KWIC_PER_PHRASE)
    cm.OUTDIR_DATA.mkdir(parents=True, exist_ok=True)
    dm.to_parquet(cm.OUTDIR_DATA / "mentions.parquet", index=False)
    dk.to_parquet(cm.OUTDIR_DATA / "kwic.parquet", index=False)
    docs = df[["id", "leaid", "date", "n_words", "is_empty"]].rename(columns={"id": "video_id"})
    docs.to_parquet(cm.OUTDIR_DATA / "docs.parquet", index=False)
    print(f"{len(df):,} transcripts, {int(df.is_empty.sum())} read as empty; {len(dm):,} nonzero (transcript, phrase) rows")
    summ = dm.groupby("phrase").agg(mentions=("count", "sum"), excluded=("count_excluded", "sum"),
                                    docs=("count", lambda s: (s > 0).sum()),
                                    districts=("leaid", lambda s: s[dm.loc[s.index, "count"] > 0].nunique()))
    print(summ.sort_values("districts", ascending=False).to_string())
    return


def CountOne(fname):
    text = cm.ReadText(fname)
    if not text.strip():
        return True, {}, []
    counts, ctx = {}, []
    rng = random.Random(fname)
    for ph, (pat, exc) in PATTERNS.items():
        n = n_ex = 0
        hits = []
        for m in pat.finditer(text):
            if exc is not None and exc.search(text, max(0, m.start() - EXCL_WINDOW), m.end() + EXCL_WINDOW):
                n_ex += 1
            else:
                n += 1
            hits.append(m)
        if hits:
            counts[ph] = (n, n_ex)
            m = rng.choice(hits)
            ctx.append((ph, text[max(0, m.start() - KWIC_CHARS):m.end() + KWIC_CHARS]))
    return False, counts, ctx


if __name__ == "__main__":
    Main()
