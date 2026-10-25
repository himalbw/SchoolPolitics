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
3. **Extract the raw transcripts** (~5 GB of tars become ~199k `.txt.gz` files in `Datastore/DistrictView/raw/`, never modified; safe to re-run):
   ```bash
   python3 source/derived/sb_transcripts/read_sb_transcript.py
   ```
   It also saves two sample transcripts to `Datastore/DistrictView/sample_transcripts/`.
4. **Build the cleaned (derived) transcripts for the analysis sample:**
   ```bash
   python3 source/derived/sb_transcripts/build_derived_transcripts.py
   ```
   Writes `Datastore/DistrictView/derived/transcripts/` (lowercase, no punctuation, no `[music]`/`[applause]`/timestamps) and `derived/manifest.parquet` (one row per video in the sample with `status`: kept / too_short / placeholder / missing_file, plus word counts). The sample window and minimum length are set in **`globals.json`** (`SAMPLE_START`, `SAMPLE_END`, `MIN_WORDS`); edit it and re-run, and stale files are removed automatically.
5. **(Only if we add shared modules) set `PYTHONPATH` to the project root** so scripts in any folder can import them (e.g. `from source.utils import ...`):
   ```bash
   cd SchoolBoards
   export PYTHONPATH="$PWD"            # this terminal session only
   echo "export PYTHONPATH=\"$PWD\"" >> ~/.zshrc   # optional: persist (use ~/.bashrc for bash)
   ```
   Do **not** run scripts with `python -I` (isolated mode ignores `PYTHONPATH`).
6. **Analyses**: bag-of-words / GST in `source/analysis/bow/README.md`; topic models + vocabulary-change tests in `source/analysis/lda_2018_2023/README.md`.
7. *(Optional, to rebuild covariates from scratch instead of using the Drive copy)* run `data cleaning/01..05` in order, e.g. `python3 "data cleaning/04_build_processed.py"` (note the quotes: the folder name has a space).

## Citation
SEDA: Reardon, Ho, Shear, Fahle, Kalogrides, saliba (2026), SEDA v6.0, https://purl.stanford.edu/xh833nn4025
