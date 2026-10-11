"""Merge everything onto the DistrictView district roster.
Outputs (Datastore/District Covariates/processed/):
  district_static.parquet  one row per leaid  (roster + CCD + SEDA pooled + SEDA covariates + county votes + COVID hub)
  district_year.parquet    leaid x year       (video counts + CCD enrollment/race + SAIPE + finance + SEDA annual score)"""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent / "Datastore"
C = ROOT / "District Covariates"; P = C / "processed"; H = ROOT / "Covid School Data Hub/processed"
rd = lambda f: pd.read_parquet(f)
roster = rd(P / "dv_roster.parquet")

# ---- CCD directory: static attributes (latest obs) + county (mode) ----
d = rd(C / "ccd/ccd_directory.parquet"); d["leaid"] = d.leaid.astype(str).str.zfill(7)
d = d.sort_values(["leaid", "year"])
cty = d.dropna(subset=["county_code"]).groupby("leaid").county_code.agg(lambda s: s.mode().iat[0]).rename("county_fips")
last = d.groupby("leaid").tail(1).set_index("leaid")[["lea_name", "state_location", "city_location", "latitude", "longitude",
        "urban_centric_locale", "cbsa", "csa", "county_name", "agency_type", "year"]].rename(columns={"year": "ccd_last_year"})
static = roster.merge(last, on="leaid", how="left").merge(cty, on="leaid", how="left")
static["county_fips"] = static.county_fips.astype("string").str.zfill(5)
# county FIPS reassignment guard: CT planning regions etc. may not match election files -> flagged below
# ---- SEDA pooled (all students) + pooled covariates ----
sp = rd(P / "seda_admindist_pool_cs.parquet"); a = sp[sp.subgroup == "all"].set_index("leaid")
static = static.merge(a[["cs_mn_avg_eb", "cs_mn_lrn_eb", "cs_mn_tav_eb", "tot_asmts"]].add_prefix("seda_"), on="leaid", how="left")
gap = sp.pivot_table(index="leaid", columns="subgroup", values="cs_mn_avg_eb").add_prefix("seda_avg_").reset_index()
static = static.merge(gap, on="leaid", how="left")
cv = rd(P / "seda_cov_admindist_pool.parquet")
keep = ["leaid", "urban", "suburb", "town", "rural", "totenrl", "perwht", "perblk", "perhsp", "perasn", "perfrl", "perecd", "perell", "perspeced",
        "sesavgall", "lninc50avgall", "baplusavgall", "unempavgall", "snapavgall", "povertyavgall", "single_momavgall", "hswhtblk", "rswhtblk"]
static = static.merge(cv[keep].add_prefix("sedacov_").rename(columns={"sedacov_leaid": "leaid"}), on="leaid", how="left")
# ---- County: presidential vote + SEDA county covariates ----
el = rd(P / "county_presidential.parquet")
static = static.merge(el[["county_fips"] + [c for c in el if c.startswith(("gop_2p_", "total_"))]], on="county_fips", how="left")
cc = rd(P / "seda_cov_county_pool.parquet")[["county_fips", "totenrl", "perfrl", "sesavgall", "baplusavgall", "povertyavgall", "lninc50avgall"]]
static = static.merge(cc.add_prefix("cty_").rename(columns={"cty_county_fips": "county_fips"}), on="county_fips", how="left")
# ---- COVID School Data Hub ----
sh = rd(H / "hub_districts_by_share_all.parquet").drop_duplicates("leaid")[["leaid", "share_inperson", "share_hybrid", "share_virtual"]]
mk = rd(H / "hub_masking_all.parquet").drop_duplicates("leaid")
mk = mk[["leaid", "StudentMaskPolicy", "StaffMaskPolicy"]].rename(columns=lambda c: "hub_" + c if c != "leaid" else c)
static = static.merge(sh.add_prefix("hub_").rename(columns={"hub_leaid": "leaid"}), on="leaid", how="left").merge(mk, on="leaid", how="left")
static.to_parquet(P / "district_static.parquet")

# ---- district x year ----
v = rd(P / "dv_videos_leaid_year.parquet")
e = rd(C / "ccd/ccd_enroll_race.parquet"); e["leaid"] = e.leaid.astype(str).str.zfill(7)
race = {99: "total", 1: "white", 2: "black", 3: "hispanic", 4: "asian", 5: "aian", 6: "nhpi", 7: "twoplus"}
e = e[e.race.isin(race) & (e.sex == 99)].assign(r=lambda x: x.race.map(race)).pivot_table(index=["leaid", "year"], columns="r", values="enrollment").reset_index()
for r in list(race.values())[1:]: e[f"share_{r}"] = e[r] / e["total"]
e = e.rename(columns={"total": "enroll_total"})[["leaid", "year", "enroll_total"] + [c for c in e if c.startswith("share_")]]
sa = rd(C / "ccd/saipe.parquet"); sa["leaid"] = sa.leaid.astype(str).str.zfill(7)
sa = sa[["leaid", "year", "est_population_5_17", "est_population_5_17_poverty", "est_population_5_17_poverty_pct"]]
fin = rd(C / "ccd/ccd_finance.parquet"); fin["leaid"] = fin.leaid.astype(str).str.zfill(7)
fin = fin[["leaid", "year", "rev_total", "rev_fed_total", "rev_state_total", "rev_local_total", "rev_local_prop_tax", "exp_total", "exp_current_instruction_total", "enrollment_fall_responsible"]]
for c in fin.columns[2:]: fin[c] = fin[c].where(fin[c] >= 0)      # Urban API codes missing as negatives
fin["exp_pp"] = fin.exp_total / fin.enrollment_fall_responsible
dd = d[["leaid", "year", "enrollment", "number_of_schools", "teachers_total_fte", "spec_ed_students", "english_language_learners", "urban_centric_locale"]]
sy = rd(P / "seda_admindist_annual_cs.parquet")[["leaid", "year", "cs_mn_avg_eb", "tot_asmts"]].rename(columns={"cs_mn_avg_eb": "seda_cs_avg", "tot_asmts": "seda_asmts"})
panel = (v.merge(dd, on=["leaid", "year"], how="outer").merge(e, on=["leaid", "year"], how="outer").merge(sa, on=["leaid", "year"], how="outer")
          .merge(fin, on=["leaid", "year"], how="outer").merge(sy, on=["leaid", "year"], how="outer"))
panel = panel[panel.leaid.isin(set(roster.leaid)) & panel.year.between(2008, 2025)].sort_values(["leaid", "year"])
panel["n_videos"] = panel.n_videos.fillna(0)
panel.to_parquet(P / "district_year.parquet")
print("static", static.shape, "panel", panel.shape)
print("county match (election):", static.gop_2p_2020.notna().mean().round(3), "| leaids w/o county:", static.county_fips.isna().sum())
