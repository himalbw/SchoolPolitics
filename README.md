# School Board Polarization (Stat 236)

Has local education politics become more partisan? We use DistrictView school-board meeting
transcripts to study how language, topics, and partisan alignment changed before / during / after COVID-19.

Team: Calista Wilik, Lillian Yang, Stefan Nikov, Himal Bamzai-Wokhlu.
Project context and data notes for contributors: see [CLAUDE.md](CLAUDE.md). Proposal: `docs/`.

## Layout
```
code/        numbered, re-runnable pipeline scripts (01-05 build the data; later numbers = analysis)
notebooks/   exploratory notebooks
output/      figures/ and tables/ that are small enough to commit
docs/        proposal, notes, writeups
Datastore/   ALL data — gitignored, not in this repo (see below)
```

## Data (not in the repo)
`Datastore/` is excluded from git because of size (~6 GB) and the SEDA data-use terms
(non-commercial, no redistribution). Each teammate rebuilds it locally:

| Source | How |
|---|---|
| DistrictView transcripts + metadata | Obtain the share from the project owner; unpack to `Datastore/DistrictView/districtview_share_draft/` |
| SEDA 6.0 | `bash code/01_download_seda.sh` |
| NCES CCD / SAIPE / finance | `python3 -I code/02_download_ccd_saipe.py` |
| Oster COVID School Data Hub | `python3 -I code/03_download_covid_hub.py` |
| Processed panels | `python3 -I code/04_build_processed.py && python3 -I code/05_merge_panel.py` |

Scripts resolve paths relative to their own location, so run them from anywhere.

## Setup
```bash
pip install -r requirements.txt
```

## Citation
SEDA: Reardon, Ho, Shear, Fahle, Kalogrides, saliba (2026), SEDA v6.0, https://purl.stanford.edu/xh833nn4025
