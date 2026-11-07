"""Draw the transcript sample for GABRIEL (LLM) coding: up to N meetings per district-year, optionally limited to some states.

Reads  Datastore/DistrictView/derived/manifest.parquet + derived/transcripts/   (built by build_derived_transcripts.py)
       Datastore/District Covariates/processed/district_static.parquet          (state abbreviation, county GOP share)
Writes Datastore/DistrictView/derived/gabriel_sample/sample.parquet             (one row per sampled transcript, with text)

Parameters come from globals.json: GABRIEL_PER_DISTRICT_YEAR (N), GABRIEL_STATES (postal codes; [] = all states),
GABRIEL_SEED. Each kept transcript gets a fixed pseudo-random key from hash(seed, id), and the N lowest keys in each
district-year are drawn. The draw is therefore nested: a state's rows are identical whether you sample it alone or with
other states, and raising N only adds rows. District-years with fewer than N kept transcripts contribute all of them.
Run from the repo root:  python3 source/derived/sb_transcripts/build_gabriel_sample.py
"""
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

GLOBALS = json.loads(Path("globals.json").read_text())
INDIR_DERIVED = Path("Datastore/DistrictView/derived/")
INDIR_STATIC = Path("Datastore/District Covariates/processed/district_static.parquet")
OUTDIR_SAMPLE = INDIR_DERIVED / "gabriel_sample"
TOKENS_PER_WORD = 1.3  # rough, for the cost printout only

def Main():
    df_sample = SelectSample()
    df_sample["text"] = [ReadText(f) for f in df_sample.file]
    OUTDIR_SAMPLE.mkdir(parents=True, exist_ok=True)
    df_sample.to_parquet(OUTDIR_SAMPLE / "sample.parquet", index=False)
    PrintSummary(df_sample)
    print(f"Saved {len(df_sample):,} transcripts to {OUTDIR_SAMPLE / 'sample.parquet'}")
    return

def SelectSample():
    df = pd.read_parquet(INDIR_DERIVED / "manifest.parquet", columns=["id", "channel_id", "leaid", "year", "file", "status", "n_words"])
    df = df[df.status == "kept"].drop(columns="status")
    df["leaid"] = df.leaid.astype(str).str.zfill(7)
    df_static = pd.read_parquet(INDIR_STATIC, columns=["leaid", "state_location", "lea_name", "gop_2p_2020"])
    df = df.merge(df_static.rename(columns={"state_location": "state"}), on="leaid", how="left", validate="m:1")
    assert df.state.notna().all(), "kept transcripts without a state in district_static"
    if GLOBALS["GABRIEL_STATES"]:
        df = df[df.state.isin(GLOBALS["GABRIEL_STATES"])]
    df["draw_key"] = [DrawKey(i) for i in df.id]
    df = df.sort_values(["leaid", "year", "draw_key"])
    df = df.groupby(["leaid", "year"]).head(GLOBALS["GABRIEL_PER_DISTRICT_YEAR"])
    return df.drop(columns="draw_key").reset_index(drop=True)

def DrawKey(video_id):
    return hashlib.sha256(f"{GLOBALS['GABRIEL_SEED']}:{video_id}".encode()).hexdigest()

def ReadText(fname):
    with gzip.open(INDIR_DERIVED / "transcripts" / fname, "rt", encoding="utf-8") as f:
        return f.read()

def PrintSummary(df):
    states = ", ".join(GLOBALS["GABRIEL_STATES"]) or "all"
    print(f"GABRIEL sample: {GLOBALS['GABRIEL_PER_DISTRICT_YEAR']} per district-year, states = {states}")
    print(f"{len(df):,} transcripts, {df.leaid.nunique():,} districts, "
          f"{df.n_words.sum() * TOKENS_PER_WORD / 1e6:,.0f}M input tokens (approx.)")
    df_counts = df.groupby(["state", "year"]).size().unstack(fill_value=0)
    df_counts["districts"] = df.groupby("state").leaid.nunique()
    print(df_counts.to_string())
    return

if __name__ == "__main__":
    Main()
