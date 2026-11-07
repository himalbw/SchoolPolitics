"""Count the closure_lexicon phrases in every kept cleaned transcript (same order as bow/docs.parquet).

Writes {BOW_DERIVED_DIR}/bow_phrases/closure_phrase_counts.parquet. Run from the repo root (about 5 min).
"""
import gzip
import json
import re
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from closure_lexicon import AllClosurePhrases

GLOBALS = json.loads(Path("globals.json").read_text())
INDIR_DERIVED = Path(GLOBALS["BOW_DERIVED_DIR"])
OUTDIR = INDIR_DERIVED / "bow_phrases"
PHRASES = AllClosurePhrases()
PATTERNS = [re.compile(r"\b" + re.escape(p) + r"\b") for p in PHRASES]

def Main():
    df_manifest = pd.read_parquet(INDIR_DERIVED / "manifest.parquet", columns=["file", "status"])
    files = df_manifest.loc[df_manifest.status == "kept", "file"].tolist()
    with Pool(6) as pool:
        rows = pool.map(CountOne, files, chunksize=200)
    df = pd.DataFrame(rows, columns=PHRASES, dtype="int32")
    df.insert(0, "file", files)
    df.to_parquet(OUTDIR / "closure_phrase_counts.parquet")
    print(f"{len(df):,} transcripts x {len(PHRASES)} phrases")
    print(df[PHRASES].sum().sort_values(ascending=False).to_string())
    return

def CountOne(fname):
    with gzip.open(INDIR_DERIVED / "transcripts" / fname, "rt", encoding="utf-8") as f:
        text = f.read()
    return [len(pat.findall(text)) for pat in PATTERNS]

if __name__ == "__main__":
    Main()
