"""Build cleaned ("derived") transcripts from the raw ones, restricted to the analysis sample.

Reads  Datastore/DistrictView/raw/transcripts_districtview_part*/{channel_id}__{id}.txt.gz   (never modified)
Writes Datastore/DistrictView/derived/transcripts/{channel_id}__{id}.txt.gz                   (cleaned text)
       Datastore/DistrictView/derived/manifest.parquet                                       (one row per in-window video)

Sample + filters come from globals.json (repo root): SAMPLE_START, SAMPLE_END (meeting year, inclusive;
falls back to the YouTube upload year when meeting_date is missing) and MIN_WORDS (minimum cleaned words).
Cleaning: drop [H:MM:SS] timestamps, drop all [bracketed] caption tags ([music], [applause], ...), drop '>>'
speaker-turn markers (their count is kept in the manifest), lowercase, strip punctuation (keeping apostrophes
inside words, e.g. "don't"), collapse whitespace to a single paragraph.
Run from the repo root:  python3 source/derived/sb_transcripts/build_derived_transcripts.py
"""
import gzip
import json
import re
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

GLOBALS = json.loads(Path("globals.json").read_text())
INDIR_META = Path("Datastore/DistrictView/districtview_share_draft/meta_districtview.parquet")
INDIR_RAW = Path("Datastore/DistrictView/raw/")
OUTDIR_DERIVED = Path("Datastore/DistrictView/derived/")
OUTDIR_TRANSCRIPTS = OUTDIR_DERIVED / "transcripts"

RE_TIMESTAMP = re.compile(r"^\[\d+:\d{2}:\d{2}\]\s*", re.M)
RE_TAG = re.compile(r"\[[^\]]*\]")
RE_NONWORD = re.compile(r"[^\w']|_")
RE_EDGE_APOSTROPHE = re.compile(r"(?<!\w)'|'(?!\w)")
PLACEHOLDER = "<No transcript available>"

def Main():
    df_sample = SelectSample()
    OUTDIR_TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    part_dirs = sorted(INDIR_RAW.glob("transcripts_districtview_part*"))
    raw_index = {p.name: p for d in part_dirs for p in d.iterdir()}
    jobs = [(f"{r.channel_id}__{r.id}.txt.gz", raw_index.get(f"{r.channel_id}__{r.id}.txt.gz")) for r in df_sample.itertuples()]
    with Pool(6) as pool:
        results = pool.map(ProcessOne, jobs, chunksize=500)
    df_manifest = pd.concat([df_sample.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    RemoveStale(set(df_manifest.loc[df_manifest.status == "kept", "file"]))
    df_manifest.to_parquet(OUTDIR_DERIVED / "manifest.parquet")
    print(f"Sample {GLOBALS['SAMPLE_START']}-{GLOBALS['SAMPLE_END']}, MIN_WORDS={GLOBALS['MIN_WORDS']}: "
          f"{len(df_manifest):,} videos in window")
    print(df_manifest.status.value_counts().to_string())
    print(f"Derived transcripts saved to {OUTDIR_TRANSCRIPTS}; manifest at {OUTDIR_DERIVED / 'manifest.parquet'}")
    return

def SelectSample():
    df = pd.read_parquet(INDIR_META, columns=["id", "channel_id", "leaid", "meeting_date", "published_at"])
    df["leaid"] = df.leaid.astype(str).str.zfill(7)
    df["year"] = df.meeting_date.dt.year.fillna(df.published_at.dt.year).astype(int)
    df = df[df.year.between(GLOBALS["SAMPLE_START"], GLOBALS["SAMPLE_END"])]
    return df.drop(columns=["meeting_date", "published_at"])

def Clean(text):
    text = RE_TIMESTAMP.sub("", text)
    n_turns = text.count(">>")
    text, n_tags = RE_TAG.subn(" ", text)
    text = text.lower().replace("’", "'")
    text = RE_EDGE_APOSTROPHE.sub(" ", RE_NONWORD.sub(" ", text))
    return " ".join(text.split()), n_turns, n_tags

def ProcessOne(job):
    fname, path = job
    out = {"file": fname, "status": "kept", "n_words": 0, "n_turn_markers": 0, "n_tags": 0}
    if path is None:
        out["status"] = "missing_file"
        return out
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    if raw.strip().startswith(PLACEHOLDER):
        out["status"] = "placeholder"
        return out
    clean, out["n_turn_markers"], out["n_tags"] = Clean(raw)
    out["n_words"] = len(clean.split())
    if out["n_words"] < GLOBALS["MIN_WORDS"]:
        out["status"] = "too_short"
        return out
    with gzip.open(OUTDIR_TRANSCRIPTS / fname, "wt", encoding="utf-8") as f:
        f.write(clean + "\n")
    return out

def RemoveStale(keep):
    # So changing globals.json (e.g. a narrower window) never leaves old files behind
    stale = [p for p in OUTDIR_TRANSCRIPTS.iterdir() if p.name not in keep]
    for p in stale:
        p.unlink()
    if stale:
        print(f"Removed {len(stale):,} derived files outside the current sample/filters.")
    return

if __name__ == "__main__":
    Main()
