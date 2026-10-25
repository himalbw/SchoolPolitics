# Topic models + vocabulary-change tests (Stefan; Stat 236 school boards)

Three linked analyses: an initial DistrictView EDA with a small LDA pilot (`source/analysis/initial_eda/`), a full-corpus
LDA comparing 10/20/30 topics (`source/analysis/lda_2018_2023/`), and an eligibility audit + Higher Criticism test of
vocabulary change between 2018–19 and 2022–23 (`source/analysis/eligibility_hc/`). Notebooks are in `notebooks/`
(`initial_eda.ipynb`, `lda_2018_2023.ipynb`, `eligibility_hc.ipynb`).

Run everything from the repo root. Inputs are read straight from the tars in `Datastore/DistrictView/districtview_share_draft/`
(no extraction; override with `export STAT236_RAW_DIR=/path/to/dir` holding `meta_districtview.parquet` + the three tars).
All generated files go to `Datastore/DistrictView/derived/{initial_eda,lda_2018_2023,eligibility_hc}/` (never committed:
they include transcript excerpts, models and caches). Every stage caches its outputs and is reused on re-run; to change a
setting, delete that stage's output folder rather than mixing caches from different settings.

Extra dependencies are in the root `requirements.txt`. The original runs used Python 3.12 with numpy 1.26.4, pandas 3.0.2,
pyarrow 23.0.1, scipy 1.15.3, scikit-learn 1.5.1, joblib 1.5.3, matplotlib 3.8.2 (the saved `.joblib` models need a matching
scikit-learn). Set `OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4` for the fit. Full regeneration takes hours and several GB.

## Steps

| step | script | does |
|---|---|---|
| 1 | `source/analysis/initial_eda/inspect_data.py` | metadata + archive index, missingness, coverage by year/district, seeded 1,500-entry text sample |
| 2 | `source/analysis/initial_eda/availability.py` | full gzip/placeholder census of every tar member |
| 3 | `source/analysis/initial_eda/finalize_audit.py` | annual availability, adds census results to `summary.json` |
| 4 | `source/analysis/initial_eda/lda_pilot.py` | diagnostic K=12 LDA on 600 sampled videos (not the full-corpus model) |
| 5 | `source/analysis/initial_eda/build_report.py` | figures, `report.html`, regenerates `notebooks/initial_eda.ipynb` |
| 6 | `source/analysis/lda_2018_2023/workflow.py prepare` | clean, dedupe, ~1,000-word chunks of every dated 2018–23 transcript -> `chunks/`, `cleaning_manifest.csv` |
| 7 | `source/analysis/lda_2018_2023/verify_outputs.py` | accounting + train/test separation checks |
| 8 | `source/analysis/lda_2018_2023/workflow.py vocabulary` | training-only CountVectorizer -> `vectorizer.joblib` |
| 9 | `source/analysis/lda_2018_2023/workflow.py fit` | online LDA, K = 10/20/30, one pass each -> `lda_k*.joblib` |
| 10 | `source/analysis/lda_2018_2023/workflow.py evaluate` | held-out completion perplexity, NPMI, diversity -> `comparison.csv`, `topics.csv`, `topic_examples.json` |
| 11 | `notebooks/lda_2018_2023.ipynb` | displays results, saves `corpus_eda.png`, `model_comparison.png` (needed by step 12) |
| 12 | `source/analysis/lda_2018_2023/build_topic_report.py` | `topic_report/report.html` with the 30 hand labels + evidence (needs steps 1-5 and 11) |
| 13 | `source/analysis/eligibility_hc/workflow.py audit` | rule-based eligibility screen of the LDA-retained videos + manual-review candidates |
| 14 | `source/analysis/eligibility_hc/workflow.py aggregate` | matched district panel (2018, 2019, 2022, 2023) -> `district_profiles.npz` |
| 15 | `source/analysis/eligibility_hc/workflow.py compare` | HC+, max \|Z\|, sum-of-squares tests, district multiplier bootstrap, Gaussian power sim |
| 16 | `source/analysis/eligibility_hc/collect_contexts.py` | transcript-context examples for selected words |

`source/analysis/lda_2018_2023/make_notebook.py` regenerates `notebooks/lda_2018_2023.ipynb` from source (no data needed).
The HC stage needs LDA steps 6 and 8 only, not the fitted topic models.

Notebooks find the repo root themselves (the folder with `globals.json`) and `chdir` there, so open them from anywhere.
Use a fresh kernel per notebook (LDA and HC both import a module named `workflow`). The HC notebook reads caches by default;
`REBUILD_FROM_TRANSCRIPTS` / `RERUN_INFERENCE` rebuild. **Clear notebook outputs before committing** (they can hold
transcript text).

`manual_review_log.csv` (Stefan's hand review, contains transcript evidence) is not in git: get it from Stefan and put it in
`Datastore/DistrictView/derived/eligibility_hc/`. Without it the notebook skips that section. Code generates review
candidates, not human labels.

## Sample definition (differs from the team pipeline)
This analysis keeps its own, earlier specification and does **not** read `globals.json` or `Datastore/DistrictView/derived/transcripts/`:
- Non-null `meeting_date` in 2018–2023 only; no upload-year fallback (team pipeline falls back to upload year).
- At least 200 alphabetic words (team: 1,000), exact cleaned-text duplicates dropped, split into ~1,000-word chunks
  (200–1,199 words) with timestamp spans kept. Own cleaning: contractions expanded, bracket tags removed, own stopwords.
- Fixed video-level train/test split by content hash (prefix mod 10 = 0 -> test); vocabulary fit on training chunks only.
- HC weights: equal video weight within district-year, equal year weight within period, equal district weight. It tests
  paired district vocabulary shares, not pooled token counts, and has no partisan dimension.
- Eligibility is automated rules plus sampled text review, not full manual validation. Global-test calibration is
  approximate and does not fully handle shared cross-district shocks.

Agree on a common sample/cleaning definition before combining these results with the bag-of-words pipeline.

## Findings (October 2026 runs, Stefan's machine)
- LDA cleaning retained 97,926 dated 2018–2023 recordings (~1.365M chunks). The eligibility screen then kept 83,369.
- The matched vocabulary comparison used 21,878 recordings from 273 districts and 11,804 words. HC+, minimum-p and sum of
  squares all hit the 0.0005 Monte Carlo floor, including in sensitivity checks: vocabulary changed between 2018–19 and
  2022–23. This is not causal and does not say which test is most powerful.
- Some apparent change is COVID-era transcription variants ("covet", "coven") and number formatting, alongside
  interpretable shifts (ESSER, shortages, inflation).
- K=30 has the lowest held-out perplexity (3.65% below K=20, 9.35% below K=10); coherence and diversity do not clearly
  favour it. The 30 labels in `build_topic_report.py` are hand interpretations; revisit them if the models are refit.
  The LDA models predate the eligibility audit and were not refit after screening.
