"""Pull NCES CCD (directory, enrollment by race, finance) and Census SAIPE for the districts in DistrictView,
via the Urban Institute Education Data Portal API. Output: Datastore/District Covariates/ccd/*.parquet
Run: python3 -I 02_download_ccd_saipe.py [dataset ...]   (no args = all; datasets run well in parallel)"""
import json, time, sys, urllib.request
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "Datastore"
OUT = ROOT / "District Covariates" / "ccd"
OUT.mkdir(parents=True, exist_ok=True)
B = "https://educationdata.urban.org/api/v1/school-districts/"
meta = pd.read_parquet(ROOT / "DistrictView/districtview_share_draft/meta_districtview.parquet", columns=["leaid"])
ARGS = sys.argv[1:]
LEAIDS = set(meta.leaid.astype(str).str.zfill(7))

def fetch_all(path):
    url, rows = B + path + "/", []
    while url:
        for attempt in range(5):
            try:
                d = json.load(urllib.request.urlopen(url, timeout=300)); break
            except Exception as e:
                print("  retry", url, e, file=sys.stderr); time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(url)
        rows += [r for r in d["results"] if str(r["leaid"]).zfill(7) in LEAIDS]
        url = d.get("next")
    return pd.DataFrame(rows)

def run(name, path_fmt, years):
    if ARGS and name not in ARGS: return
    f = OUT / f"{name}.parquet"
    if f.exists(): print("skip", name); return
    parts = []
    for y in years:
        df = fetch_all(path_fmt.format(y=y)); print(name, y, len(df), flush=True)
        if len(df): parts.append(df)
    pd.concat(parts, ignore_index=True).to_parquet(f)

run("ccd_directory", "ccd/directory/{y}", range(2008, 2025))
run("ccd_enroll_race", "ccd/enrollment/{y}/grade-99/race", range(2008, 2024))
run("saipe", "saipe/{y}", range(2008, 2024))
run("ccd_finance", "ccd/finance/{y}", range(2008, 2022))
