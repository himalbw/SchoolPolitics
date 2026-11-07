"""Build the bag-of-words input from Stefan's cleaned transcripts (Drive: cleaned_transcripts_2026_10_28).

Reads  Datastore/Cleaned Data/metadata.zip                                  (one metadata/{id}.json.gz per video, from Drive)
       Datastore/DistrictView/raw/transcripts_districtview_part*/*.txt.gz   (never modified)
       Datastore/DistrictView/derived/transcripts/*.txt.gz                  (only to report what changed against the old cleaning)
Writes Datastore/DistrictView/derived_v2/transcripts/{channel_id}__{id}.txt.gz              (tokenised text, same format as derived/)
       Datastore/DistrictView/derived_v2/manifest.parquet                                   (one row per in-window video)

Stefan's cleaning only removes timestamps and normalises quotes, apostrophes, zero-width characters and whitespace, and his metadata
stores the SHA-256 of each cleaned text. So the text is rebuilt from raw and checked against that hash (byte-identical to the Drive
copy, without ~176k single-file Drive downloads); a mismatch is dropped with status "sha_mismatch". The text keeps case, punctuation
and caption tags, so it then goes through the same Clean() as build_derived_transcripts.py before counting words.
Sample + filters come from globals.json, as in build_derived_transcripts.py.
Run from the repo root:  python3 source/derived/sb_transcripts/build_cleaned_transcripts.py
"""
import gzip
import hashlib
import json
import re
import zipfile
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from build_derived_transcripts import Clean, GLOBALS, INDIR_RAW, SelectSample

INDIR_METADATA = Path("Datastore/Cleaned Data/metadata.zip")
INDIR_OLD_TRANSCRIPTS = Path("Datastore/DistrictView/derived/transcripts/")
OUTDIR_DERIVED = Path("Datastore/DistrictView/derived_v2/")
OUTDIR_TRANSCRIPTS = OUTDIR_DERIVED / "transcripts"

RE_TIMESTAMP = re.compile(r"^\[\d+:\d{2}:\d{2}\]\s*", re.M)
RE_WHITESPACE = re.compile(r"\s+")
TYPOGRAPHIC = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "﻿": None, "​": None})

def Main():
    df_sample = SelectSample().merge(ReadCleanedMetadata(), on="id", how="left", validate="1:1")
    assert df_sample.stefan_status.notna().all(), "some in-window videos are missing from metadata.zip"
    OUTDIR_TRANSCRIPTS.mkdir(parents=True, exist_ok=True)
    jobs = [(f"{r.channel_id}__{r.id}.txt.gz", r.stefan_status, r.source_archive, r.cleaned_sha) for r in df_sample.itertuples()]
    with Pool(6) as pool:
        results = pool.map(ProcessOne, jobs, chunksize=500)
    df_manifest = pd.concat([df_sample.drop(columns=["source_archive", "cleaned_sha"]).reset_index(drop=True), pd.DataFrame(results)], axis=1)
    df_manifest.to_parquet(OUTDIR_DERIVED / "manifest.parquet")
    Report(df_manifest)
    return

def ReadCleanedMetadata():
    rows = []
    with zipfile.ZipFile(INDIR_METADATA) as z:
        for name in z.namelist():
            if name.endswith(".json.gz"):
                d = json.loads(gzip.decompress(z.read(name)))
                rows.append({"id": d["id"], "stefan_status": d["status"], "source_archive": d.get("source_archive"),
                             "cleaned_sha": d.get("cleaned_text_sha256"), "quality_flags": "|".join(d.get("quality_flags") or [])})
    return pd.DataFrame(rows)

def StefanClean(raw):
    text = RE_TIMESTAMP.sub("", raw).translate(TYPOGRAPHIC)
    return RE_WHITESPACE.sub(" ", text).strip()

def ProcessOne(job):
    fname, stefan_status, source_archive, cleaned_sha = job
    out = {"file": fname, "status": "kept", "n_words": 0, "n_turn_markers": 0, "n_tags": 0, "same_as_old": pd.NA}
    if stefan_status != "available":
        out["status"] = {"placeholder": "placeholder", "missing_archive_transcript": "missing_file"}[stefan_status]
        return out
    with gzip.open(INDIR_RAW / source_archive.removesuffix(".tar") / fname, "rt", encoding="utf-8", errors="replace", newline="") as f:
        stefan_text = StefanClean(f.read())
    if hashlib.sha256(stefan_text.encode("utf-8")).hexdigest() != cleaned_sha:
        out["status"] = "sha_mismatch"
        return out
    clean, out["n_turn_markers"], out["n_tags"] = Clean(stefan_text)
    out["n_words"] = len(clean.split())
    if out["n_words"] < GLOBALS["MIN_WORDS"]:
        out["status"] = "too_short"
        return out
    with gzip.open(OUTDIR_TRANSCRIPTS / fname, "wt", encoding="utf-8") as f:
        f.write(clean + "\n")
    old_path = INDIR_OLD_TRANSCRIPTS / fname
    if old_path.exists():
        with gzip.open(old_path, "rt", encoding="utf-8") as f:
            out["same_as_old"] = f.read() == clean + "\n"
    return out

def Report(df_manifest):
    print(f"Sample {GLOBALS['SAMPLE_START']}-{GLOBALS['SAMPLE_END']}, MIN_WORDS={GLOBALS['MIN_WORDS']}: {len(df_manifest):,} videos in window")
    print(df_manifest.status.value_counts().to_string())
    kept = df_manifest[df_manifest.status == "kept"]
    print(f"Kept: {len(kept):,} transcripts, {kept.leaid.nunique():,} districts, {kept.n_words.sum():,} words")
    print(f"Against derived/: {kept.same_as_old.eq(True).sum():,} tokenised texts identical, {kept.same_as_old.eq(False).sum():,} differ, "
          f"{kept.same_as_old.isna().sum():,} not in the old sample")
    print(f"Saved to {OUTDIR_TRANSCRIPTS}; manifest at {OUTDIR_DERIVED / 'manifest.parquet'}")
    return

if __name__ == "__main__":
    Main()
