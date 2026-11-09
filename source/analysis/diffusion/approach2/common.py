"""Shared paths and loaders for Approach 2 of the topic-diffusion extension (tracking known phrases).

Sample = the screened BoW sample: rows of {BOW_DERIVED_DIR}/manifest.parquet with status == "kept" (80,063 transcripts).
Date = meeting_date from the DistrictView metadata (the screen drops undated meetings), falling back to published_at.
Pre / post windows are derived from globals.json: PRE = the first two sample years, POST = the rest.
"""
import gzip
import json
from pathlib import Path

import pandas as pd

GLOBALS = json.loads(Path("globals.json").read_text())
SAMPLE_START, SAMPLE_END = GLOBALS["SAMPLE_START"], GLOBALS["SAMPLE_END"]
PRE_YEARS = list(range(SAMPLE_START, SAMPLE_START + 2))
POST_YEARS = list(range(SAMPLE_START + 2, SAMPLE_END + 1))

INDIR_DERIVED = Path(GLOBALS["BOW_DERIVED_DIR"])
INDIR_TRANSCRIPTS = INDIR_DERIVED / "transcripts"
INDIR_META = Path("Datastore/DistrictView/districtview_share_draft/meta_districtview.parquet")
INDIR_STATIC = Path("Datastore/District Covariates/processed/district_static.parquet")
INDIR_QUOTES = sorted(Path("output/tables").glob("quotes_*.csv"))

OUTDIR_DATA = Path("Datastore/DistrictView/derived/diffusion/approach2")
OUTDIR_TABLES = Path("output/tables/diffusion/approach2")
OUTDIR_FIGURES = Path("output/figures/diffusion/approach2")
N_WORKERS = 10


def LoadSample():
    """One row per kept transcript: file, id, leaid, date, year, month (period), n_words, state, gop_2p_2020, locale."""
    df = pd.read_parquet(INDIR_DERIVED / "manifest.parquet", columns=["id", "leaid", "file", "status", "n_words"])
    df = df.loc[df.status == "kept"].drop(columns="status")
    df_meta = pd.read_parquet(INDIR_META, columns=["id", "meeting_date", "published_at"])
    df = df.merge(df_meta, on="id", how="left", validate="1:1")
    pub = pd.to_datetime(df.published_at, utc=True).dt.tz_localize(None)
    df["date"] = pd.to_datetime(df.meeting_date).fillna(pub).dt.normalize()
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.to_period("M")
    df["leaid"] = df.leaid.str.zfill(7)
    df_static = LoadStatic()
    df = df.merge(df_static, on="leaid", how="left", validate="m:1")
    return df.drop(columns=["meeting_date", "published_at"]).reset_index(drop=True)


def LoadStatic():
    cols = ["leaid", "state_location", "latitude", "longitude", "urban_centric_locale", "county_fips", "gop_2p_2020"]
    df = pd.read_parquet(INDIR_STATIC, columns=cols).rename(columns={"state_location": "state"})
    df["leaid"] = df.leaid.str.zfill(7)
    df["locale4"] = LocaleGroup(df.urban_centric_locale)
    df["rep"] = (df.gop_2p_2020 > 0.5).map({True: "R", False: "D"}).where(df.gop_2p_2020.notna())
    return df


def LocaleGroup(code):
    """NCES urban-centric locale codes 11-43 -> city / suburb / town / rural (first digit)."""
    first = pd.to_numeric(code, errors="coerce") // 10
    return first.map({1: "city", 2: "suburb", 3: "town", 4: "rural"})


def ReadText(fname):
    with gzip.open(INDIR_TRANSCRIPTS / fname, "rt", encoding="utf-8") as f:
        return f.read()
