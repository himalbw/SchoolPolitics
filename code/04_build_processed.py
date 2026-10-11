"""Harmonize raw downloads into compact parquet files keyed on leaid (7-char NCES LEA ID) or county fips (5-char).
Output: Datastore/District Covariates/processed/ and Datastore/Covid School Data Hub/processed/"""
import glob, re
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "Datastore"
DV = ROOT / "DistrictView/districtview_share_draft/meta_districtview.parquet"
COV = ROOT / "District Covariates"; HUB = ROOT / "Covid School Data Hub"
(COV / "processed").mkdir(exist_ok=True); (HUB / "processed").mkdir(exist_ok=True)
meta = pd.read_parquet(DV, columns=["leaid", "place_names", "meeting_date", "published_at"])
meta["leaid"] = meta.leaid.astype(str).str.zfill(7)
LEAIDS = set(meta.leaid)

# ---------- 1. DistrictView district roster + video panel ----------
yr = meta.meeting_date.dt.year.fillna(meta.published_at.dt.year).astype(int)
panel = meta.assign(year=yr).groupby(["leaid", "year"]).size().rename("n_videos").reset_index()
panel.to_parquet(COV / "processed/dv_videos_leaid_year.parquet")
roster = meta.groupby("leaid").agg(place_name=("place_names", "first"), n_videos=("leaid", "size"),
                                   first_date=("meeting_date", "min"), last_date=("meeting_date", "max")).reset_index()
roster["state_fips"] = roster.leaid.str[:2]
roster.to_parquet(COV / "processed/dv_roster.parquet")

# ---------- 2. SEDA 6.0 ----------
S = COV / "seda"
def seda_keep(df, id_col):
    df = df[df[id_col].notna()].copy(); df["leaid"] = df[id_col].astype(int).astype(str).str.zfill(7); return df
pool = seda_keep(pd.read_csv(S / "seda_admindist_pool_cs_6.0.csv"), "sedaadmin")
pool = pool[pool.leaid.isin(LEAIDS) & pool.subgroup.isin(["all", "ecd", "nec", "wht", "blk", "hsp"])]
pool.to_parquet(COV / "processed/seda_admindist_pool_cs.parquet")
ann = seda_keep(pd.read_csv(S / "seda_admindist_annual_cs_6.0.csv"), "sedaadmin")
ann = ann[ann.leaid.isin(LEAIDS) & (ann.subgroup == "all")]
ann.to_parquet(COV / "processed/seda_admindist_annual_cs.parquet")
for k in ["pool", "annual"]:
    c = seda_keep(pd.read_csv(S / f"seda_cov_admindist_{k}_6.0.csv"), "sedaadmin")
    c[["gslo", "gshi"]] = c[["gslo", "gshi"]].astype(str)   # mixed str/num grade labels
    c[c.leaid.isin(LEAIDS)].to_parquet(COV / f"processed/seda_cov_admindist_{k}.parquet")
# county-level SEDA (keyed on 5-digit county fips; joined via CCD county_code later)
cp = pd.read_csv(S / "seda_county_pool_cs_6.0.csv"); cp = cp[(cp.subgroup == "all") & cp.sedacounty.notna()]
cp["county_fips"] = cp.sedacounty.astype(int).astype(str).str.zfill(5); cp.to_parquet(COV / "processed/seda_county_pool_cs.parquet")
cc = pd.read_csv(S / "seda_cov_county_pool_6.0.csv"); cc = cc[cc.sedacounty.notna()]; cc[["gslo", "gshi"]] = cc[["gslo", "gshi"]].astype(str); cc["county_fips"] = cc.sedacounty.astype(int).astype(str).str.zfill(5)
cc.to_parquet(COV / "processed/seda_cov_county_pool.parquet")

# ---------- 3. County presidential vote (2008-2024; GitHub tonmcg compilation of Guardian/Townhall/Fox/Politico/NYT) ----------
E = COV / "elections"
a = pd.read_csv(E / "US_County_Level_Presidential_Results_08-16.csv", dtype={"fips_code": str})
a["county_fips"] = a.fips_code.str.zfill(5)
out = a[["county_fips"]].copy()
for y in (2008, 2012, 2016):
    out[f"dem_{y}"], out[f"gop_{y}"], out[f"total_{y}"] = a[f"dem_{y}"], a[f"gop_{y}"], a[f"total_{y}"]
for y in (2020, 2024):
    b = pd.read_csv(E / f"{y}_US_County_Level_Presidential_Results.csv", dtype={"county_fips": str})
    b["county_fips"] = b.county_fips.str.zfill(5)
    out = out.merge(b[["county_fips", "votes_dem", "votes_gop", "total_votes"]].rename(
        columns={"votes_dem": f"dem_{y}", "votes_gop": f"gop_{y}", "total_votes": f"total_{y}"}), on="county_fips", how="outer")
for y in (2008, 2012, 2016, 2020, 2024):
    out[f"gop_share_{y}"] = out[f"gop_{y}"] / out[f"total_{y}"]          # share of all votes
    out[f"gop_2p_{y}"] = out[f"gop_{y}"] / (out[f"gop_{y}"] + out[f"dem_{y}"])  # two-party share
out.to_parquet(COV / "processed/county_presidential.parquet")

# ---------- 4. COVID School Data Hub (district-level files) ----------
R = HUB / "raw"
def read(f):
    return pd.read_csv(f, dtype=str, encoding="utf-8-sig", low_memory=False)
def leaid_col(df, col):
    return df[col].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(7)
shares, mask, wk = [], [], []
for f in sorted(glob.glob(str(R / "DistrictsByShare_*.csv"))):
    d = read(f); d["leaid"] = leaid_col(d, "NCESDistrictID"); d["src_file"] = Path(f).name; shares.append(d)
for f in sorted(glob.glob(str(R / "*MaskingData*.csv"))):
    d = read(f).drop(columns=[""], errors="ignore"); d["leaid"] = leaid_col(d, "NCESDistrictID"); d["src_file"] = Path(f).name; mask.append(d)
for f in sorted(glob.glob(str(R / "*_Districts_LearningModelData_Final.csv"))):
    d = read(f); d = d[d.NCESDistrictID.notna()]; d["leaid"] = leaid_col(d, "NCESDistrictID")
    d = d[d.leaid.isin(LEAIDS)][["StateAbbrev", "leaid", "DistrictName", "EnrollmentTotal", "TimePeriodStart", "TimePeriodEnd",
                                  "LearningModel", "LearningModelStateCat"]]
    wk.append(d)
shares = pd.concat(shares, ignore_index=True); mask = pd.concat(mask, ignore_index=True); wk = pd.concat(wk, ignore_index=True)
for c in ["share_inperson", "share_virtual", "share_hybrid"]: shares[c] = pd.to_numeric(shares[c], errors="coerce")
shares.to_parquet(HUB / "processed/hub_districts_by_share_all.parquet")
mask.to_parquet(HUB / "processed/hub_masking_all.parquet")
wk.to_parquet(HUB / "processed/hub_learning_model_weekly_dv.parquet")
print("shares", shares.shape, shares.leaid.isin(LEAIDS).sum(), "| mask", mask.shape, mask.leaid.isin(LEAIDS).sum(), "| weekly (DV districts)", wk.shape, wk.leaid.nunique())
print("processed files written")
