# School Board Polarization (Stat 236)

Has local education politics become more partisan? We use  school-board meeting
transcripts to study how language, topics, and partisan alignment changed before / during / after COVID-19.

Team: Calista Wilik, Lillian Yang, Stefan Nikov, Himal Bamzai-Wokhlu.
Project context and data notes for contributors: see [CLAUDE.md](CLAUDE.md). 

## Setup
Everything below is run **from the `SchoolBoards/` project root** (scripts use paths relative to it, e.g. `Datastore/...`).

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Get the data.** Download the shared Google Drive folder *Stat236 Project Datastore* (ask Himal for access; keep it private, SEDA has data-use terms) and place it at `SchoolBoards/Datastore/`. Do not commit it.
3. **Extract the transcripts** (required before any text analysis; ~5 GB of tars become ~199k `.txt.gz` files, safe to re-run):
   ```bash
   cd SchoolBoards
   python3 source/derived/sb_transcripts/read_sb_transcript.py
   ```
   This writes `Datastore/DistrictView/transcripts_districtview_part{1,2,3}/` and two sample transcripts to `Datastore/DistrictView/sample_transcripts/`.
4. **(Only if we add shared modules) set `PYTHONPATH` to the project root** so scripts in any folder can import them (e.g. `from source.utils import ...`):
   ```bash
   cd SchoolBoards
   export PYTHONPATH="$PWD"            # this terminal session only
   echo "export PYTHONPATH=\"$PWD\"" >> ~/.zshrc   # optional: persist (use ~/.bashrc for bash)
   ```
   Do **not** run scripts with `python -I` (isolated mode ignores `PYTHONPATH`).
5. *(Optional, to rebuild covariates from scratch instead of using the Drive copy)* run `data cleaning/01..05` in order, e.g. `python3 "data cleaning/04_build_processed.py"` (note the quotes: the folder name has a space).

## Citation
SEDA: Reardon, Ho, Shear, Fahle, Kalogrides, saliba (2026), SEDA v6.0, https://purl.stanford.edu/xh833nn4025
