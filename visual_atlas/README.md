# School-board visual atlas

Open **index.html** for the figure gallery, or **School_board_visual_atlas.ipynb** for the reproducible analysis (Git copy has outputs cleared). The `figures/` directory contains PNG and PDF exports. `tables/` contains plot inputs, sample definitions, date conflicts and geographic linkage audits. This is an additive analysis: no source transcripts or previous model results are edited.

## Reproduce

Use Python 3.12 and install `requirements.txt`. Open the notebook with this folder as its working directory and run all cells, or run:

```bash
python download_sources.py
python prepare.py
python figures.py
python geography.py
python build_delivery.py
```

`build_delivery.py` creates and executes the notebook (which also regenerates the figures), then creates the HTML gallery. The public downloads are cached in `external/`, so normal reruns work offline. Install pyshp from requirements.txt; no vendored dependencies are committed.

## Local inputs (not committed)

Set `SCHOOL_BOARD_PROJECT` to your local **School Board Project** folder before launching Jupyter. For example:

```bash
export SCHOOL_BOARD_PROJECT="/absolute/path/to/School Board Project"
cd visual_atlas
python -m jupyterlab
```

Without this variable, scripts look under `visual_atlas/local_data/School Board Project/` (ignored). The separate v2 corpus and historical analysis outputs must be obtained privately; this folder does not regenerate upstream cleaning or model fitting. It does not plug into or modify the team's existing pipeline.

Set `SCHOOL_BOARD_PROJECT` to that project's location before starting Python/Jupyter. Required folders below it are:

- `data/modeling_v2_2018_2023/model_manifest.parquet` and `core_tokens/`
- `analysis/lda_cleaning_comparison/` with comparison.csv and topic_matches.csv
- `analysis/eligibility_hc_2018_2019_vs_2022_2023/` with word_changes_primary.csv, test_comparison.csv and cleaning_followup definitions

The readable `cleaned_transcripts_2026_10_28/transcripts/` folder alone is not enough to rebuild these v2 counts. Cached plot inputs, when obtained privately or generated locally, let you reproduce figures without rescanning transcripts. They are not included in Git. To rebuild counts after changing dictionaries or tokenization, remove this analysis's `tables/phrase_counts_v2.parquet` and rerun `prepare.py`; never remove source data.

## Figure families

1. District-quarter coverage heatmap and observation counts.
2. Six subject changes with pointwise intervals, all vs outside COVID chunks.
3. Continuous 2018–2023 quarterly trajectories using a fixed district panel.
4. LDA coherence/diversity, within-version perplexity and 30-topic matches.
5. Earlier-pipeline word changes and global-test threshold comparison.
6. County coverage/voting maps, county subject-change maps and state estimates.
7. Subject changes versus baseline county voting, with state-clustered intervals.
8. Quarterly trajectories by fixed 2016 county political groups.

## Interpret carefully

The new phrase analysis uses v2 core tokens (including numbers and stopwords), not LDA-filtered tokens. Its denominator differs from earlier phrase outputs. Six exploratory lexicons count mentions within 1,000-token blocks; themes may overlap. `no_covid` omits whole COVID-flagged blocks, not merely the COVID words. It is not a causal adjustment. ASR errors, sponsor ads, repetitions and quoted/played media remain possible; no audio validation or stance classification is claimed.

38 explicit title-year conflicts are excluded from new substantive analyses, leaving 83,328 recordings. That is a limited consistency check, not full date validation. Rates average recordings within district-year/quarter and districts equally. The period comparison requires four recordings and three distinct quarters in each endpoint year; the 24-quarter trajectory panel requires at least one recording every quarter.

NCES and Census geography are fixed at 2020. County analysis excludes districts overlapping multiple counties and records all exclusions. State summaries retain a district only if its linked counties agree on state. Maps show the contiguous US; non-contiguous states remain in eligible statistical tables. Gray areas mean unavailable or unrepresented, not zero discussion. District names and source attribution have not been independently validated against audio.

Election data are the official MEDSL 2000–2016 GitHub release; only 2016 is used as a pre-period baseline. The newer Dataverse data require a guestbook submission and are not used. Republican share is Republican/(Republican+Democratic) votes. County voting is contextual, not an estimate of speakers' views or district-specific voting. Political regressions are unadjusted, exploratory associations, with state-clustered intervals; trajectory bands use a within-group state bootstrap. No causal or nationally representative claims are made.

Existing HC outputs are from earlier preprocessing; they have not been rerun on v2. Existing paired LDA metrics are reused, not refitted. Perplexity levels are not comparable between token representations. Pointwise intervals across the many exploratory plots are not simultaneous or multiplicity-adjusted.

Public URLs and checksums are saved in `external/sources.json`; software versions in `environment.json`; figure manifest in `figure_catalog.json`; deliverable hashes in `output_checksums.json`.

## Git package

This additive folder includes the notebook source, scripts, all 13 PNG/PDF figure pairs, the HTML gallery, phrase dictionaries and documentation. The notebook's executed outputs are deliberately cleared. Figures contain aggregate results only. `tables/`, `external/`, transcript excerpts, caches, downloaded data and local paths are excluded. The original executed notebook remains in the local analysis folder.

Running `build_delivery.py` executes the notebook and embeds outputs locally. Clear notebook outputs before committing a regenerated copy. `verify.py` checks a full local regeneration; it requires private upstream inputs and generated tables. Existing figures can be inspected without any data or Python setup by opening index.html.
