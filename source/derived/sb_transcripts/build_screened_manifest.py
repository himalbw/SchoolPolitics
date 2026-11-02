"""Apply Stefan's dedupe + eligibility screen to the bag-of-words sample.

Reads  Datastore/DistrictView/derived_v2/manifest.parquet                  (the unscreened bag-of-words sample)
       Datastore/DistrictView/districtview_share_draft/*                   (metadata; tar order decides which exact duplicate is kept)
       Datastore/DistrictView/raw/transcripts_districtview_part*/*.txt.gz  (never modified)
Writes {BOW_DERIVED_DIR}/manifest.parquet       status = "kept", or "screen_<reason>" for transcripts the screen removes
       {BOW_DERIVED_DIR}/transcripts            symlink to derived_v2/transcripts (the text is unchanged, so it is not copied)
       {BOW_DERIVED_DIR}/screen/                cleaning_manifest.csv + chunks/ (inputs to his audit), eligibility_audit.csv (his output)

The retained set follows prepare() in source/analysis/lda_2018_2023/workflow.py: non-null meeting_date in 2018-2023, no placeholder,
at least 200 words, first copy of each exact token sequence. Only the first 600 tokens are stored, which is all audit() reads, in
place of his ~1,000-word chunks of every transcript. audit() in source/analysis/eligibility_hc/workflow.py then runs unchanged, and
"eligible" is its primary_eligible (audit_status starts with include_). The counts should match his README: 97,926 and 83,369.
MIN_WORDS in globals.json still applies on top of his 200-word floor, because the team sample is defined that way.
Run from the repo root:  python3 source/derived/sb_transcripts/build_screened_manifest.py
"""
import gzip
import hashlib
import importlib.util
import json
import subprocess
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

GLOBALS = json.loads(Path("globals.json").read_text())
INDIR_BASE = Path("Datastore/DistrictView/derived_v2/")
INDIR_RAW = Path("Datastore/DistrictView/raw/")
INDIR_TARS = Path("Datastore/DistrictView/districtview_share_draft/")
OUTDIR = Path(GLOBALS["BOW_DERIVED_DIR"])
OUTDIR_SCREEN = OUTDIR / "screen"
PREFIX_WORDS = 600

def LoadModule(name, path):
    # Both of Stefan's modules are called workflow.py, so they are loaded by path rather than by name.
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

LDA = LoadModule("lda_workflow", "source/analysis/lda_2018_2023/workflow.py")
HC = LoadModule("hc_workflow", "source/analysis/eligibility_hc/workflow.py")

def Main():
    (OUTDIR_SCREEN / "chunks").mkdir(parents=True, exist_ok=True)
    df_clean = CleaningManifest()
    HC.LDA, HC.OUTDIR = OUTDIR_SCREEN, OUTDIR_SCREEN
    HC.audit()
    df_audit = pd.read_csv(OUTDIR_SCREEN / "eligibility_audit.csv", usecols=["id", "audit_status", "primary_eligible"], dtype={"id": str})
    print(f"Retained {(df_clean.status == 'retained').sum():,} (Stefan: 97,926); eligible {df_audit.primary_eligible.sum():,} (Stefan: 83,369)")
    df_manifest = ScreenManifest(df_clean, df_audit)
    df_manifest.to_parquet(OUTDIR / "manifest.parquet")
    LinkTranscripts()
    Report(df_manifest)
    return

def CleaningManifest():
    df_meta = pd.read_parquet(INDIR_TARS / "meta_districtview.parquet", columns=["id", "leaid", "meeting_date", "title"])
    df_meta = df_meta[df_meta.meeting_date.dt.year.between(2018, 2023)].copy()
    df_meta["year"] = df_meta.meeting_date.dt.year
    meta = df_meta.set_index("id").to_dict("index")
    jobs = [job for job in TarOrder() if job[0] in meta]
    with Pool(6) as pool:
        results = pool.imap(ProcessOne, jobs, chunksize=200)
        rows, prefixes, seen = [], [], {}
        for vid, status, n_words, digest, prefix in results:
            if status == "retained":
                if digest in seen:
                    status = "exact_duplicate"
                else:
                    seen[digest] = vid
                    prefixes.append({"id": vid, "chunk": 0, "text": prefix})
            rows.append({"id": vid, "status": status, "words": n_words, **meta[vid]})
    found = {row["id"] for row in rows}
    rows += [{"id": vid, "status": "missing_archive", "words": 0, **m} for vid, m in meta.items() if vid not in found]
    df_clean = pd.DataFrame(rows)
    df_clean.to_csv(OUTDIR_SCREEN / "cleaning_manifest.csv", index=False)
    pd.DataFrame(prefixes).to_parquet(OUTDIR_SCREEN / "chunks" / "shard_00.parquet")
    print(df_clean.status.value_counts().to_string())
    return df_clean

def TarOrder():
    """(video id, raw path) in the order prepare() streams them: sorted archives, member order inside each."""
    jobs = []
    for tar in sorted(INDIR_TARS.glob("*.tar")):
        names = subprocess.run(["tar", "-tf", str(tar)], capture_output=True, text=True, check=True).stdout.split("\n")
        for name in names:
            if name.endswith(".txt.gz"):
                fname = Path(name).name
                jobs.append((fname.removesuffix(".txt.gz").partition("__")[2], str(INDIR_RAW / tar.stem / fname)))
    return jobs

def ProcessOne(job):
    vid, path = job
    try:
        raw = gzip.decompress(Path(path).read_bytes()).decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return vid, "decode_error", 0, None, None
    if raw.strip() == "<No transcript available>":
        return vid, "placeholder", 0, None, None
    tokens = LDA.WORD.findall(LDA.clean_text(raw))
    if len(tokens) < LDA.MIN_WORDS:
        return vid, "under_200_words", len(tokens), None, None
    digest = hashlib.sha256(" ".join(tokens).encode()).hexdigest()
    return vid, "retained", len(tokens), digest, " ".join(tokens[:PREFIX_WORDS])

def ScreenManifest(df_clean, df_audit):
    df = pd.read_parquet(INDIR_BASE / "manifest.parquet")
    df = df.merge(df_clean[["id", "status"]].rename(columns={"status": "lda_status"}), on="id", how="left", validate="1:1")
    df = df.merge(df_audit, on="id", how="left", validate="1:1")
    reason = df.audit_status.where(df.lda_status == "retained", df.lda_status).fillna("undated")
    screened_out = (df.status == "kept") & (df.primary_eligible.ne(True))
    df.loc[screened_out, "status"] = "screen_" + reason[screened_out]
    return df

def LinkTranscripts():
    link = OUTDIR / "transcripts"
    if not link.exists():
        link.symlink_to(Path("../derived_v2/transcripts"))
    return

def Report(df_manifest):
    print(df_manifest.status.value_counts().to_string())
    kept = df_manifest[df_manifest.status == "kept"]
    print(f"Kept {len(kept):,} transcripts, {kept.leaid.nunique():,} districts, {kept.n_words.sum():,} words")
    before = df_manifest[df_manifest.status.eq("kept") | df_manifest.status.str.startswith("screen_")]
    print(pd.DataFrame({"before": before.groupby("year").size(), "after": kept.groupby("year").size()}).assign(share=lambda d: (d.after / d.before).round(3)).to_string())
    return

if __name__ == "__main__":
    Main()
