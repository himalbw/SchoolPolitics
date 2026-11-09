"""Step 1: extract keyword hits (all keywords in one pass) and sampled control-word hits, with rule-based flags.

  python3 .../01_extract_hits.py            # count pass, then extraction with per-keyword sampling caps
Writes Datastore/.../approach1/keywords/{kw}/hits.parquet (sampled to KEYWORDS[kw]['cap']),
       .../keywords/students/hits.parquet (placebo), counts.csv (all hits, before the cap) and read_log.parquet.
"""
import gzip
import hashlib
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CONTROL, CONTROL_KEEP_P, CTX_HALF, DATA_KW, DATA_OUT, KEYWORDS, MANIFEST, META, READ_HALF,
                    SEED, SPAN_HALF, STATIC, TRANSCRIPTS, flags, matcher)

MATCH = {k: matcher(v) for k, v in KEYWORDS.items()}
KEEP_P = {}   # filled in the extraction pass (per-keyword sampling probability)


def scan(fname, keep_p=None):
    try:
        w = gzip.open(TRANSCRIPTS / fname, "rt", encoding="utf-8").read().split()
    except FileNotFoundError:
        return fname, "missing", {}, []
    except Exception as e:  # noqa: BLE001
        return fname, f"error:{type(e).__name__}", {}, []
    if not w:
        return fname, "empty", {}, []
    rng = random.Random(int(hashlib.md5(fname.encode()).hexdigest()[:8], 16) + SEED)
    counts, rows = {}, []
    for i, tok in enumerate(w):
        for kw, f in MATCH.items():
            L = f(w, i)
            if L:
                counts[kw] = counts.get(kw, 0) + 1
                if keep_p is not None and rng.random() < keep_p[kw]:
                    rows.append(record(w, i, L, kw, fname))
        if keep_p is not None and tok == CONTROL and rng.random() < CONTROL_KEEP_P:
            rows.append(record(w, i, 1, CONTROL, fname))
    return fname, "ok", counts, rows


def record(w, i, L, kw, fname):
    lo = max(0, i - SPAN_HALF)
    c0 = max(0, i - CTX_HALF)
    return dict(kw=kw, file=fname, pos=i, kw_len=L, token=" ".join(w[i:i + L]), n_words=len(w),
                span=" ".join(w[lo: i + L + SPAN_HALF]), span_kpos=i - lo,
                ctx30=" ".join(w[c0: i + L + CTX_HALF]), ctx30_kpos=i - c0,
                ctx60=" ".join(w[max(0, i - READ_HALF): i + L + READ_HALF]), **flags(w, i))


def scan_count(f):
    return scan(f)


def scan_keep(f):
    return scan(f, KEEP_P)


def init(p):
    KEEP_P.update(p)


def main():
    man = pd.read_parquet(MANIFEST, columns=["id", "leaid", "year", "file", "status"])
    man = man[man.status == "kept"]
    files = man.file.tolist()
    print(f"screened sample: {len(files):,} transcripts, {man.leaid.nunique():,} districts")
    t0 = time.time()
    tot, per_file, log = {}, [], []
    with Pool(10) as pool:
        for fname, st, cnt, _ in pool.imap_unordered(scan_count, files, chunksize=64):
            log.append((fname, st))
            for k, v in cnt.items():
                tot[k] = tot.get(k, 0) + v
                per_file.append((fname, k, v))
    log = pd.DataFrame(log, columns=["file", "read_status"])
    log.to_parquet(DATA_OUT / "read_log.parquet", index=False)
    print("read status:", log.read_status.value_counts().to_dict(), f"({time.time() - t0:.0f}s)")
    pf = pd.DataFrame(per_file, columns=["file", "kw", "n"]).merge(man[["file", "leaid"]], on="file")
    counts = pd.DataFrame(dict(kw=list(KEYWORDS))).assign(
        n_hits_all=lambda d: d.kw.map(tot).fillna(0).astype(int),
        n_transcripts=lambda d: d.kw.map(pf.groupby("kw").file.nunique()).fillna(0).astype(int),
        n_districts_all=lambda d: d.kw.map(pf.groupby("kw").leaid.nunique()).fillna(0).astype(int))
    keep_p = {k: (1.0 if KEYWORDS[k]["cap"] is None or tot.get(k, 0) <= KEYWORDS[k]["cap"]
                  else KEYWORDS[k]["cap"] / tot[k]) for k in KEYWORDS}
    counts["keep_p"] = counts.kw.map(keep_p)
    counts.to_csv(DATA_OUT / "counts.csv", index=False)
    print(counts.to_string(index=False))

    rows = []
    with Pool(10, initializer=init, initargs=(keep_p,)) as pool:
        for _, _, _, r in pool.imap_unordered(scan_keep, files, chunksize=64):
            rows.extend(r)
    hits = pd.DataFrame(rows).merge(man[["file", "id", "leaid", "year"]], on="file", how="left")
    meta = pd.read_parquet(META, columns=["id", "meeting_date", "published_at"])
    hits = hits.merge(meta, on="id", how="left")
    d = pd.to_datetime(hits.meeting_date, errors="coerce", utc=True).dt.tz_localize(None)
    hits["date"] = d.fillna(pd.to_datetime(hits.published_at, errors="coerce", utc=True).dt.tz_localize(None))
    hits["leaid"] = hits.leaid.astype(str).str.zfill(7)
    st = pd.read_parquet(STATIC, columns=["leaid", "state_location", "county_fips", "gop_2p_2020"])
    st["leaid"] = st.leaid.astype(str).str.zfill(7)
    hits = hits.merge(st, on="leaid", how="left").drop(columns=["meeting_date", "published_at"])
    print(f"county merge {hits.county_fips.notna().mean():.1%}, gop_2p_2020 {hits.gop_2p_2020.notna().mean():.1%}")
    for kw, g in hits.groupby("kw"):
        g = g.reset_index(drop=True)
        g.insert(0, "hit_id", range(len(g)))
        DATA_KW(kw).mkdir(parents=True, exist_ok=True)
        g.drop(columns="kw").to_parquet(DATA_KW(kw) / "hits.parquet", index=False)
        print(f"{kw}: kept {len(g):,} hits, {g.leaid.nunique():,} districts, PC strict "
              f"{g.flag_public_comment.mean():.1%} loose {g.flag_public_comment_loose.mean():.1%} "
              f"policy {g.flag_policy.mean():.1%}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
