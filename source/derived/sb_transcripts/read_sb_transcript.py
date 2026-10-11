import gzip
import re
import pandas as pd
from pathlib import Path
import tarfile

INDIR_SB = Path("Datastore/DistrictView/raw/")   # raw = original, untouched transcripts (derived/ is built by build_derived_transcripts.py)
INDIR_RAW_TRANSCRIPTS = Path("Datastore/DistrictView/districtview_share_draft/")
OUTDIR_SAMPLES = Path("Datastore/DistrictView/sample_transcripts/")

def Main():
    df_metadata = pd.read_parquet(INDIR_RAW_TRANSCRIPTS / "meta_districtview.parquet")
    ReadTranscripts()
    SampleTranscripts(df_metadata)
    return

def ReadTranscripts():
    tar_paths = sorted(INDIR_RAW_TRANSCRIPTS.glob("*.tar*"))
    for tar_path in tar_paths:
        extract_dir = INDIR_SB / tar_path.name.split('.')[0]
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"{extract_dir.name} already extracted, skipping.")
            continue
        extract_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {tar_path.name} to {extract_dir}...")
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(path=extract_dir, filter="data")
    print(f"Transcripts are saved to {INDIR_SB}!")
    return

def Slug(text):
    return re.sub(r"[^A-Za-z0-9]+", "-", str(text)).strip("-").lower()

def SampleTranscripts(df_metadata, n=15, n_lines=15, seed=236):
    # Files are named {channel_id}__{id}.txt.gz; part dirs come from ReadTranscripts
    part_dirs = sorted(INDIR_SB.glob("transcripts_districtview_part*"))
    OUTDIR_SAMPLES.mkdir(exist_ok=True)

    found = 0
    # Shuffle all videos, walk until n real transcripts are found
    # (some files are just the placeholder "<No transcript available>")
    for row in df_metadata.sample(frac=1, random_state=seed).itertuples():
        fname = f"{row.channel_id}__{row.id}.txt.gz"
        path = next((d / fname for d in part_dirs if (d / fname).exists()), None)
        if path is None:
            continue
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            text = f.read()
        if text.strip().startswith("<No transcript available>"):
            continue

        found += 1
        header = f"{row.place_names} | {row.title} | {row.meeting_date:%Y-%m-%d}"
        print(f"\n=== Sample {found}: {header} ===")
        print("\n".join(text.splitlines()[:n_lines]))
        base = f"{row.leaid}_{Slug(row.place_names)}_{row.meeting_date:%Y-%m-%d}"
        out_name, k = f"{base}.txt", 2
        while (OUTDIR_SAMPLES / out_name).exists():  # same district + same day
            out_name, k = f"{base}_{k}.txt", k + 1
        (OUTDIR_SAMPLES / out_name).write_text(f"{header}\n\n{text}")
        if found == n:
            break

    print(f"\nSample transcripts for your viewing are saved to {OUTDIR_SAMPLES}!")
    return

if __name__ == '__main__':
    Main()
