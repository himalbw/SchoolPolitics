# Stefan's STAT 236 analyses

Self-contained contribution: initial DistrictView EDA, full-corpus LDA with
10/20/30 topics, topic-label report code, and eligibility/Higher Criticism analysis
of vocabulary changes between 2018–2019 and 2022–2023. Existing repository files
and the original local analyses are unchanged.

Only code, notebook source, and documentation are committed. Notebook outputs,
raw data, metadata extracts, transcript excerpts, matrices, model caches, plots,
and generated reports are excluded. Numerical narratives describe the original
October 2026 runs; packaging these files does not constitute a new model fit.

## Data placement

Obtain the DistrictView files privately from the team's **Stat236 Project
Datastore** Google Drive folder. This contribution needs no SEDA or election
files. Keep the transcript archives intact; scripts read their gzipped members
without extracting the entire corpus.

Default layout relative to the repository root:

```text
stefan_analysis/
  README.md
  requirements.txt
  data/                            # ignored; create locally
    raw/
      meta_districtview.parquet
      transcripts_districtview_part1.tar
      transcripts_districtview_part2.tar
      transcripts_districtview_part3.tar
  analysis/
    initial_eda/                    # code + ignored generated outputs
      initial_eda.ipynb
    lda_2018_2023/
      LDA_2018_2023_comparison.ipynb
      workflow.py
      chunks/shard_00.parquet ...  # generated, ignored
      vectorizer.joblib            # generated, ignored
      lda_k10.joblib ...           # generated, ignored
    eligibility_hc/
      eligibility_and_vocabulary_change.ipynb
      workflow.py
      district_profiles.npz       # generated, ignored
```

Alternatively set `STAT236_RAW_DIR` to the absolute directory containing all four
raw files. Set it before starting Python/Jupyter. This avoids copying large
archives. For Stefan's existing local download:

```bash
export STAT236_RAW_DIR="$HOME/Desktop/University/Courses/2026 Fall/STAT 236/School Board Project/data/raw"
```

For the team's Datastore layout, from the repository root:

```bash
export STAT236_RAW_DIR="$PWD/Datastore/DistrictView/districtview_share_draft"
```

Adjust that directory if your Drive download has different nesting. The variable
controls raw input only; generated files go alongside each analysis's scripts.
No machine-specific absolute path is embedded in the code.

## Environment

The original analysis used Python 3.12. From the repository root:

```bash
python3.12 -m venv stefan_analysis/.venv
source stefan_analysis/.venv/bin/activate
python -m pip install -r stefan_analysis/requirements.txt
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
```

Core analysis dependencies are pinned to the original environment. Jupyter and
IPython are notebook interface dependencies. Leave the root requirements file
alone. Full regeneration can take hours and needs several GB of disk for raw
archives, cleaned chunks, and model caches.

## Rebuild order

Run these commands from the repository root, with the environment activated.
Data-dependent outputs are intentionally absent on a fresh clone.

### 1. Initial EDA and diagnostic pilot

```bash
python stefan_analysis/analysis/initial_eda/inspect_data.py
python stefan_analysis/analysis/initial_eda/availability.py
python stefan_analysis/analysis/initial_eda/finalize_audit.py
python stefan_analysis/analysis/initial_eda/lda_pilot.py
python stefan_analysis/analysis/initial_eda/build_report.py
```

These create metadata summaries, archive availability counts, a seeded
1,500-entry text sample, and a small LDA pilot. The final script generates local
figures, `report.html`, and the EDA notebook. These reports can include transcript
evidence and must remain local. The pilot is not the full-corpus model.

### 2. Full retained-corpus LDA

```bash
python stefan_analysis/analysis/lda_2018_2023/workflow.py prepare
python stefan_analysis/analysis/lda_2018_2023/verify_outputs.py
python stefan_analysis/analysis/lda_2018_2023/workflow.py vocabulary
python stefan_analysis/analysis/lda_2018_2023/workflow.py fit
python stefan_analysis/analysis/lda_2018_2023/workflow.py evaluate
```

Preparation creates `chunks/`, `cleaning_manifest.csv`, `cleaning_by_year.csv`,
and `preparation.json`. Vocabulary creates `vectorizer.joblib` and summaries.
Fitting creates K=10/20/30 models and a training checkpoint. Evaluation creates
`comparison.csv`, `topics.csv`, `topic_examples.json`, and `evaluation.json`.
All these outputs are ignored. The original fit used about 1.365 million chunks.

Run the LDA notebook to display the results and generate its figures. Then,
after initial EDA and the LDA notebook have run, generate the topic report:

```bash
python stefan_analysis/analysis/lda_2018_2023/build_topic_report.py
```

This writes `topic_report/report.html` and labeling/evidence files locally.
The 30 labels in the generator are historical interpretations, not an automatic
labeling method; revisit them if the fitted models change. The original LDA
models predate the eligibility audit and were not refitted after screening.

### 3. Eligibility audit and vocabulary tests

This stage needs the LDA cleaned chunks, cleaning manifest, and vectorizer, but
not its fitted topic models. LDA `prepare` and `vocabulary` are sufficient if
only vocabulary testing is wanted.

```bash
python stefan_analysis/analysis/eligibility_hc/workflow.py audit
python stefan_analysis/analysis/eligibility_hc/workflow.py aggregate
python stefan_analysis/analysis/eligibility_hc/workflow.py compare
python stefan_analysis/analysis/eligibility_hc/collect_contexts.py
```

Audit produces eligibility/channel CSVs, a summary, and manual-review candidates.
Aggregation produces the matched recording list and district profiles.
Comparison produces global tests, word changes, bootstrap arrays, and a
conditional Gaussian simulation. The final command generates transcript-context
examples. All outputs stay local and are ignored by Git.

The original `manual_review_log.csv` contains transcript evidence and is not
committed. Obtain it privately from Stefan and place it in `analysis/eligibility_hc/`
to reproduce the historical manual-review section, or conduct a new documented
review. **Code generates candidates, not human labels.** The notebook explicitly
skips missing manual-review evidence; automated tests can still run. The context
section similarly skips missing examples until `collect_contexts.py` is run.

### View notebooks

```bash
cd stefan_analysis
python -m jupyterlab
```

Open each notebook in its analysis subfolder. Its kernel must start in that
notebook's directory (normal Jupyter behavior). Use a fresh kernel for each:
both LDA and HC use a module named `workflow.py`. HC defaults to reading caches;
its rebuild flags are optional after running the commands above. Cached stages
are reused. To change the specification, use a fresh copy of this contribution
folder; do not mix caches from different settings or delete teammates' files.

## Historical findings and interpretation

The LDA cleaning retained 97,926 dated 2018–2023 recordings. The subsequent
primary eligibility screen retained 83,369. The matched vocabulary comparison
used 21,878 recordings from 273 districts, testing 11,804 words. HC+, minimum-p,
and sum of squares all reached the 0.0005 Monte Carlo resolution floor, including
sensitivity checks. This does not identify causal effects or establish which
test has the best power. Apparent COVID transcription variants and number-format
changes occur alongside interpretable vocabulary shifts.

## Relationship to the existing team pipeline

This contribution preserves the earlier specification and does not read or
modify the root `globals.json`, root requirements, or existing `source/` code:

- Uses non-null `meeting_date` in 2018–2023; does not add upload-year fallback for
  missing dates. The team's current pipeline uses that fallback.
- Keeps at least 200 alphabetic words, deduplicates exact cleaned text, and splits
  into roughly 1,000-word chunks. The team's current minimum is 1,000 words per
  transcript. Sample counts therefore differ.
- Fits a training-only vocabulary with a fixed video-level content-hash split;
  each LDA model gets one complete online pass over training chunks.
- HC uses equal video weight within district-year, equal year weight within
  period, and equal district weight. It tests paired district vocabulary shares,
  not pooled token counts or partisan alignment.
- Eligibility is automated rules plus sampled text review, not full audio/manual
  validation. Global-test calibration is approximate and does not fully handle
  shared cross-district shocks.

Agree on a common sample/cleaning definition before combining results across
pipelines. Do not treat the preserved historical figures as updated estimates
for a different dataset or configuration.

## Before later commits

The nested `.gitignore` allows source/documentation and ignores generated data,
reports, plots, archives, models and caches. **Executed notebook outputs are
still part of tracked notebook files:** clear all outputs and save before
committing. Never force-add data. Inspect the staged diff before pushing.
Original executed notebooks and reports remain in Stefan's local School Board
Project folder; they have not been moved or replaced by this contribution.
