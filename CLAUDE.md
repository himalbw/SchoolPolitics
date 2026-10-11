# CLAUDE.md — School Board Polarization Project (Stat 236)

Team: Calista Wilik, Lillian Yang, Stefan Nikov, Himal Bamzai-Wokhlu.
Proposal: `docs/Stat 236 Final Project Proposal HBW Edits.pdf`

## Research question & roadmap (from proposal)
**Has local education politics become more partisan?** Use school-board meeting transcripts (DistrictView) to study how
language, topics, and political alignment changed before / during / after COVID-19.
1. **Topic prevalence over time** — LDA or structural topic models (Roberts et al. 2014).
2. **Partisanship of language** — Gentzkow-Shapiro-Taddy (2019) congressional-speech approach (needs a partisan anchor: county/district vote shares are the planned proxy).
3. **What drives pre/post-pandemic shift** — linguistic features and topics most associated with the change.
4. **Network analysis** across districts (topic/language links, community detection, Newman 2006): is polarization widespread or concentrated by partisanship/geography?
5. *If time allows:* effect of COVID-era policy (lockdown severity, parents'-rights-group prevalence) on tension in meetings.

Data sources named in the proposal: DistrictView (main), SEDA / Stanford (educational outcomes; CALDER WP 309), Oster's COVID School Data Hub (closures),
SafeGraph cell-phone data near schools (OSF project tpwqf — **not yet obtained**).

## Repository layout
```
SchoolBoards/
  CLAUDE.md                    <- this file
  README.md, requirements.txt
  data cleaning/               <- numbered, re-runnable covariate scripts (see "Pipeline"; quote the path, it has a space)
  source/derived/sb_transcripts/read_sb_transcript.py  <- extracts transcript tars + saves samples (run first, from repo root)
  notebooks/  output/{figures,tables}/  docs/   <- tracked; docs/ holds a copy of the proposal PDF
  Datastore/
    DistrictView/              <- transcripts + meta (5.1 GB, tar files; do NOT commit)
    District Covariates/
      seda/                    <- SEDA 6.0 raw csv + docs/ (codebooks, documentation pdf)
      ccd/                     <- NCES CCD / SAIPE / finance pulled via Urban Institute API (filtered to DistrictView LEAIDs)
      elections/               <- county presidential returns 2008-2024
      processed/               <- ***analysis-ready parquet files*** (start here)
    Covid School Data Hub/
      raw/                     <- Oster hub state files (312 files)
      processed/
    Writeups/
```
`SchoolBoards/.gitignore` excludes `Datastore/` (size + SEDA data-use terms). `SchoolBoards/` is its own git repo (nested inside the home-dir repo, which does not track it); run git from inside it.

## Shared Drive copy (Google Drive: "Stat236 Project Datastore", owner himalbamzaiwokhlu@college.harvard.edu)
- rclone remote `personal_gdrive:` (Himal's). Always address the folder by ID, never by name — a different, unrelated folder named `datastore` (abortion/fertility project) also exists in the same Drive:
  `DEST="personal_gdrive,root_folder_id=1LTr8kZcD_cJmo75EAQ5J5amulIK7uJ0O:"`
- Push: `rclone copy Datastore "$DEST" --exclude ".DS_Store" -v` (use `copy`, not `sync`; add `--ignore-existing` to avoid overwriting teammates' files; `--dry-run` first). Pull: swap the arguments.
- Uploaded 2026-10-11: everything under Datastore/ (tars, SEDA raw + processed, CCD, elections, COVID hub, Literature, Writeups). Extracted per-transcript folders (`DistrictView/transcripts_districtview_part{1,2,3}/`, ~199k files) were NOT uploaded — the tars hold the same data.
- SEDA raw files carry Stanford's data-use terms (no public redistribution): keep the Drive folder restricted to the team, never link-share publicly.

## DistrictView corpus (`Datastore/DistrictView/districtview_share_draft/`)
- `meta_districtview.parquet`: **201,568 videos, 1,605 districts (LEAIDs), 49 states, 1,606 YouTube channels.** Median 93 videos/district (max 1,497).
- Columns: `id, title, description, published_at, thumbnail, view_count, like_count, comment_count, duration (ISO 8601), channel_id, st_fips (empty), place_names, multiple_cities (always 0), predicted_st_fips (empty), leaid, meeting_date, government_type`. Codebook: `codebook.md`.
- **`leaid` is a 7-char string key to NCES** (state FIPS = first 2 chars). Always `zfill(7)`.
- `meeting_date` is null for 9.2% of videos (fallback = `published_at`, as codebook says; code uses `meeting_date` then `published_at` year).
- **Time coverage is 2008–2025, not 2005** as the proposal says; volume is thin before ~2015 (≤ 1.6k videos/yr through 2013), jumps in 2020 (18.5k) and is ~24–26k/yr 2021–2025. Pre-period (2015–2019) is therefore much smaller than post — mind this for any pre/post design (weight / normalise by district-meeting).
- Transcripts: three tar files (`transcripts_districtview_part{1,2,3}.tar`, 78k+77k+44k = ~199k files; ~2.8k fewer than metadata rows). Members are **`{channel_id}__{id}.txt.gz`** (gzipped! codebook says `.txt`). Format: one line per caption segment `[H:MM:SS] text` — **raw YouTube auto-captions, no speaker labels, no punctuation/casing reliability, noisy ("um")**. Speaker/role (board member vs public commenter) is not available; any "who said what" analysis needs inference.
- Read one: `tar -xOf part1.tar <channel>__<id>.txt.gz | gunzip`. Extract needed subsets by filtering metadata first (5 GB total tarballs).

## Pipeline (idempotent; the `data cleaning/` scripts resolve paths from their own location, `read_sb_transcript.py` needs the repo root as cwd)
| script | does |
|---|---|
| `data cleaning/01_download_seda.sh` | SEDA 6.0 (Stanford Digital Repository druid `xh833nn4025`, world-downloadable): admindist pool/poolsub/annual (CS scale), cov_admindist pool/annual, county pool + covariates |
| `data cleaning/02_download_ccd_saipe.py [dataset]` | Urban Institute Education Data API → CCD directory 2008-24, CCD enrollment by race 2008-23, SAIPE poverty 2008-23, CCD finance 2008-20 (2021 returns empty) (filtered to the 1,605 LEAIDs). Slow for old years (API timeouts) — run datasets in parallel |
| `data cleaning/03_download_covid_hub.py` | Scrapes covidschooldatahub.com asset links; keeps district-level files (DistrictsByShare, LearningModelData, MaskingData, etc.) |
| `data cleaning/04_build_processed.py` | Harmonizes raw → `processed/*.parquet` (SEDA subsets, county votes, hub files) |
| `data cleaning/05_merge_panel.py` | Builds `district_static.parquet` (1 row / LEAID) and `district_year.parquet` (LEAID × year) |

## Analysis-ready files (`Datastore/District Covariates/processed/`)
- **`district_static.parquet`** — one row per DistrictView LEAID. Roster (place_name, n_videos, first/last date), CCD (name, lat/lon, locale, CBSA, `county_fips`), SEDA pooled achievement (`seda_cs_mn_avg_eb` = mean achievement in CS SD units, `seda_cs_mn_lrn_eb` learning rate, `seda_cs_mn_tav_eb` cohort trend; `seda_avg_{ecd,nec,wht,blk,hsp}` subgroup means), SEDA covariates (`sedacov_*`: race shares, FRL, SES composite, income, BA+, poverty, unemployment, SNAP, single mothers), county presidential vote (`gop_2p_{2008..2024}` two-party GOP share, `total_{year}`), county SEDA covariates (`cty_*`), COVID hub (`hub_share_inperson/hybrid/virtual`, `hub_StudentMaskPolicy`, `hub_StaffMaskPolicy`).
- **`district_year.parquet`** — LEAID × year (2008–2025): `n_videos`, CCD enrollment/teachers/ELL/SpEd, enrollment race shares, SAIPE child poverty, finance (revenue by source, `exp_pp`; ends 2020), SEDA annual mean score (`seda_cs_avg`, 2009–2019 only).
- Other processed pieces: `seda_admindist_pool_cs`, `seda_admindist_annual_cs`, `seda_cov_admindist_{pool,annual}` (annual covariates, incl. year-varying SES), `seda_county_pool_cs`, `seda_cov_county_pool`, `county_presidential`, `dv_roster`, `dv_videos_leaid_year`; and in `Covid School Data Hub/processed/`: `hub_districts_by_share_all`, `hub_masking_all`, `hub_learning_model_weekly_dv` (weekly in-person/hybrid/remote status, DV districts only).

## Coverage of DistrictView districts (1,605 districts, 201,568 videos; verified 2026-10-11)
| block | districts | videos |
|---|---|---|
| CCD directory / county FIPS / locale | 100% | 100% |
| SEDA pooled achievement (`seda_cs_mn_avg_eb`) | 97.4% | 97.9% |
| SEDA covariates (FRL, SES, race) | 98–99% | 98–99% |
| County presidential vote 2016 / 2020 | 99.7% | 99.7% |
| County presidential vote 2024 | 97.8% | 97.2% |
| COVID hub share in-person/hybrid/virtual | 96.2% | 96.7% |
| COVID hub masking policy | 80.1% | 79.7% |
| COVID hub weekly learning-model panel | 47% (751 districts) | 49% |
Panel (`district_year`): CCD enrollment/race ~99% of districts for 2008-2023 (none for 2024-25), SAIPE ~99% through 2023, finance (`exp_pp`) 2008-2020 only, SEDA annual 2009-2019 (only 1.1k-1.5k districts per year; thin 2014-2019).
Sanity: SEDA score vs. FRL share r = -0.88, vs. SES r = +0.83; county GOP-2p 2020 vs. white share r = +0.41.
Missing votes: the 5 Alaska districts (AK reports by house district, not county) and a few CT / non-standard-county districts in 2024.

## Data caveats / gotchas
- **SEDA 6.0 ends in 2019** (years 2009–2019; many state-years missing, e.g. only ~1.1–1.4k DV districts per year 2014–2019). Therefore SEDA gives **pre-pandemic baseline covariates only**. Pandemic-era achievement losses (Education Recovery Scorecard, educationscorecard.org — has 2019→2022/24 district losses) have **no bulk download link found**; TODO ask team/pull from site or Stanford "recovery" release.
- SEDA unit: use **administrative district** (`sedaadmin` = NCES LEAID, 7 digits after zero-pad) — matches DistrictView `leaid`. Geographic-district files (`geodist`) are Census-boundary based and key on a different ID. Recommended scale per SEDA docs: **CS** (cohort-standardized) for research. Estimates: use `_eb` (empirical Bayes) columns.
- SEDA files are **not for redistribution** (Stanford DUA: non-commercial, cite as Reardon, Ho, Shear, Fahle, Kalogrides, saliba (2026) SEDA v6.0, https://purl.stanford.edu/xh833nn4025). Do not commit/share raw files. SEDA's own site (edopportunity.org) gates the download behind an email form; we used the public SDR record instead.
- **County ≠ district.** `county_fips` = CCD county of the district office (mode over years). Districts can span counties; county vote share is a proxy for district partisanship. Better district-level partisanship (precinct-to-district crosswalk, e.g. Harvard Dataverse school-district presidential returns, or Census/EDGE GRF + precinct results) is a TODO. Alaska election results are by house district, DC by ward (non-county FIPS).
- Presidential returns come from the GitHub compilation `tonmcg/US_County_Level_Election_Results_08-24` (Guardian/Townhall/Fox/Politico/NYT; "not authoritative"). Sanity-checked: Cook 2020 24% GOP-2p, LA 27%, Autauga 73%. MIT Election Lab file (Harvard Dataverse doi:10.7910/DVN/VOQCHQ) requires a guestbook form (name/email) — not used; swap in if the team registers.
- CCD via Urban API: negative values are missing codes (−1, −2, −3); finance fields are masked to NaN in `district_year`. CCD directory for older years has noisy county codes (CT etc.).
- COVID hub coverage is partial (not all states; state files differ in format/date). `DistrictsByShare` = share of 2020-21 school year in each learning mode; weekly file only for states that published weekly data. Hub `Demographics_*.csv` are state-level aggregates (not used). IDs are NCES district IDs with dropped leading zeros — re-padded.
- Not yet pulled: SafeGraph/OSF cell-phone data; Oster hub school-level files; ACS/EDGE district demographics (partly covered by SEDA covariates, which are ACS-derived); Chronic absenteeism; Census Household Pulse; parents'-rights-group (Moms for Liberty chapters) and state mask/lockdown policy trackers.

## Conventions
- Python 3 (miniconda, pandas 3.x + pyarrow). Run scripts from the repo root with plain `python3` (not `-I`: isolated mode ignores `PYTHONPATH`). Keys: `leaid` = 7-char string, `county_fips` = 5-char string.
- Keep scratch/temp in the session scratchpad; commit code (`data cleaning/`, `source/`), not data.
- Verify merge rates after every join (see 05 script printout) — do not assume.

## Status log
- 2026-10-11: Initial setup. Proposal read; DistrictView profiled; SEDA 6.0, NCES CCD (via Urban API), SAIPE, CCD finance, county presidential returns 2008-24 and Oster COVID hub downloaded, harmonized and merged onto all 1,605 DistrictView LEAIDs (`district_static`, `district_year`). Video counts reconcile (201,568 = 201,568).
- Next-step suggestions: (1) get pandemic-era achievement (Education Recovery Scorecard) and district-level partisanship (school-district presidential returns); (2) extract transcripts for a pilot subset (e.g. 2018-2024, ~200 districts) from the tars into one tidy parquet; (3) decide partisan anchor for the GST measure; (4) obtain SafeGraph/OSF data if still wanted.
