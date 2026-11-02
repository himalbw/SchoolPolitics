# Bag-of-words analysis (Stat 236 school boards)

Run everything from the repo root, in this order. Data paths are under `BOW_DERIVED_DIR` and figures/tables go to `output/{figures,tables}/<BOW_OUTPUT_TAG>/` (both in `globals.json`; now `Datastore/DistrictView/derived_screened` and `bow_screened`). Set them to `Datastore/DistrictView/derived_v2` and `bow_v2` for the unscreened sample. The older files directly in `output/{figures,tables}/` are the October 17 run.

| step | script | does |
|---|---|---|
| 0 | `source/derived/sb_transcripts/build_cleaned_transcripts.py` | rebuilds Stefan's cleaned text from raw, checks each file against the SHA-256 in `Datastore/Cleaned Data/metadata.zip`, tokenises it -> `transcripts/`, `manifest.parquet` |
| 0b | `source/derived/sb_transcripts/build_screened_manifest.py` | applies Stefan's dedupe + eligibility screen to the step 0 sample -> `manifest.parquet`, `screen/` (`transcripts/` is a symlink to `derived_v2/`) |
| 1 | `source/derived/sb_transcripts/clean_transcripts.py` | transcript x term count matrix -> `bow/` (stopwords from `STOPWORDS`; `MIN_DF=20`, `MAX_DF=1.0`; ~13 min) |
| 2 | `source/derived/sb_transcripts/aggregate_dtm.py` | cuts vocabulary (digits, <3 chars, possessives merged, procedural words, terms in <25 districts) and sums to district x quarter -> `bow_dq/` (`vocab_cuts.parquet` says why each term was cut) |
| 3 | `source/analysis/bow/count_phrases.py` | counts the multi-word phrases in `lexicon.py` per transcript -> `bow_phrases/` (~15 min) |
| 4 | `source/analysis/bow/gst_polarization.py` | Gentzkow-Shapiro-Taddy leave-out partisanship by year/quarter + placebos -> `output/tables/gst_*.csv` |
| 5 | `source/analysis/bow/partisan_lexicon.py` | which lexicon categories / terms differ between R and D districts, and how that moved -> `output/tables/lexicon_*.csv`, `partisan_terms_*.csv` |
| 6 | `source/analysis/bow/plot_bow.py` | `output/figures/*.png` |
| 7 | `source/analysis/bow/conflict_map.py` | in-meeting conflict index per district/state + maps (`conflict_*.csv`, `conflict_map.png`, `conflict_state_by_period.png`) |

`bow_data.py` = shared loaders. `lexicon.py` = hand-built culture-war / control word lists (edit here to change categories).

## Design choices
- Partisan anchor: county 2020 two-party GOP share > 0.5 -> R, else D (fixed across periods; county != district, so this is a proxy).
- Unit = district. GST gives each district equal weight. Lexicon/term tests use district-level rates (Welch t), never pooled tokens.
- Periods: pre 2018-19, covid 2020-21, post 2022-23. Quarter from `meeting_date` (fallback upload date).
- GST was validated on simulated data (leave-out matches the true value; the plug-in estimate is biased up). Placebos: random label shuffle (~0.5000) and within-state shuffle (how much is just geography).
- Comparable-over-time spec: balanced panel (349 districts present every year with >= 10k words) thinned to 10k words per district-year, because the district count jumps in 2020 and more data raises the leave-out estimate. "National vocabulary" additionally drops terms with > 30% of their usage in one state.

## Screened sample: current headline results (2026-11-02)
`build_screened_manifest.py` runs Stefan's own `audit()` unchanged and reproduces his counts exactly (97,926 retained, 83,369 eligible). The team's `MIN_WORDS` = 1,000 still applies, so the sample goes from 103,019 to **80,063 transcripts**, and from 1,532 to 1,485 districts. Removed: 11,098 ambiguous (no clear school-board title or text evidence), 9,074 with no `meeting_date`, 2,184 other public bodies, 433 possible clips or campaign events, 76 joint bodies, 90 exact duplicates, and 1 under his 200-word floor. The share kept is similar by year (76-80%).

**The screen does not change the conclusions**, so the earlier results were not being driven by non-school-board videos, duplicates or undated videos.

| Balanced panel, national vocabulary | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|---|
| pi, unscreened (345 districts) | .5028 | .5029 | .5034 | .5042 | .5039 | .5034 |
| pi, screened (300 districts) | .5030 | .5030 | .5038 | .5045 | .5043 | .5035 |
| net of shuffled null, screened | .0029 | .0029 | .0039 | .0045 | .0043 | .0036 |
| net of null, screened, within locale x size (254 of 299 compared) | .0024 | .0025 | .0040 | .0046 | .0038 | .0034 |

### Reading the figures (`output/figures/bow_screened/`)
- `gst_polarization.png`: same shape as before. Partisanship is small, peaks in 2021, and by 2023 has fallen most of the way back but stays above 2018-19. Holding locale and size fixed does not remove the peak. The balanced panels differ by 45 districts, so differences of a few ten-thousandths between the two rows can reflect the panel as well as the screen. The jump at 2020Q2 in the quarterly panel coincides with the sample doubling, so it should not be read as a change in language.
- `lexicon_category_gap.png`: equity/SEL is still the largest gap in 2022-23 (1.96 R vs 4.67 D per 10k words, t = -17.5; -9.0 in 2018-19). Unions/pay (-6.8), COVID mandates (-6.0), immigration (-5.3) and `gender_lgbtq` (-4.2) lean D; athletics (+9.8), facilities (+6.8) and religion (+3.1) lean R. Book content, CRT, parental rights and school choice remain indistinguishable (|t| < 1.3). The facilities and athletics controls also differ, so some of this reflects district type rather than politics. These t-values have no locale control.
- `lexicon_category_trend.png`: culture-war terms still move together in R and D districts. CRT spikes in 2021Q3 and is back near 0.1 by 2023; book content and parental rights stay above 2018-19. `gender_lgbtq` opens up after 2022. COVID terms flip from R-leaning in 2020-21 (t = +2.2, weaker than +3.1 unscreened) to D-leaning after. Mentions are not stance.
- `conflict_map.png`, `conflict_state_by_period.png`: `meeting_conflict` peaks in 2021Q3 at .98 per 10k words (about .50 in 2018-19) and is about .59 in 2023; it is lower in 2020 (about .38). Top states by index: FL 70, NJ 68, PA 63 (FL moves ahead of NJ). Conflict words remain more common in D-county districts in 2022-23 (`meeting_conflict` t = -5.1, `conflict_legal` t = -3.5). State rankings with few districts are noisy (WV 7, NM 7).

### Limits of the screen
- It is rule-based. `review_ambiguous` is excluded because it is unresolved, not because those videos were shown to be wrong, so real meetings with uninformative titles are lost. Stefan's hand-review log is not in the repo and was not checked here.
- Requiring `meeting_date` drops videos by whether the title or description gave a parsable date, which is not random.
- It does not address caption drift, county-as-proxy, or the absence of sampling CIs on pi.

## Rerun on Stefan's cleaned transcripts, before the screen (2026-11-02)
Nothing substantive changed, because the inputs did not change. Stefan's Drive transcripts (`cleaned_transcripts_2026_10_28`) only remove timestamps and normalise quotes and whitespace, which is lighter than `build_derived_transcripts.py`. After tokenising, all 103,019 kept transcripts are identical to `derived/`; the sample is the same and the transcript-level and district-quarter matrices differ in 0 cells. All GST, partisan-term and conflict tables are byte-identical to the earlier run. The three lexicon tables only add rows for `meeting_conflict`, which was added to `lexicon.py` after those tables were last written; shared rows differ by 0.

So this rerun is a reproduction check, not new evidence. It does not include Stefan's substantive steps (deduplication and the eligibility screen in `source/analysis/lda_2018_2023/`, 97,926 -> 83,369 recordings).

### Reading the unscreened figures (`output/figures/bow_v2/`)
- `gst_polarization.png`: language is slightly partisan and the gap widened during COVID, then partly closed. Balanced panel, national vocabulary: pi = .5028, .5029, .5034, .5042, .5039, .5034 (2018-23), against about .5000 with shuffled labels. 2023 is still above 2018-19. The jump in the quarterly panel at 2020Q2 coincides with the sample doubling, so use the balanced lines for change over time. The within-locale x size version peaks in 2021 too (net of its null: .0025, .0023, .0036, .0043, .0038, .0033). There are no sampling CIs on pi yet.
- `lexicon_category_gap.png`: partisan differences sit in the professional and everyday vocabulary of districts, not the culture-war flashpoints. Equity/SEL is the largest gap (1.92 R vs 4.67 D mentions per 10k words, t = -17.7, up from -9.5 in 2018-19). Book content, CRT, parental rights and school choice are indistinguishable (|t| < 1). `ctrl_athletics` and `ctrl_facilities` lean R, and academics leans D, which suggests district type matters; these t-values have no locale control.
- `lexicon_category_trend.png`: culture-war terms rose in R and D districts together. CRT spikes to about 1.1-1.2 per 10k in 2021Q3 and returns to about 0.1 by 2023; book content and parental rights stay above 2018-19 levels. `gender_lgbtq` is the one that opened up (t = -1.1 pre, -4.1 post, more in D). COVID-mandate terms flip from R-heavy in 2020-21 (t = +3.1) to D-heavy after (t = -6.6). Mentions are not stance.
- `conflict_map.png`, `conflict_state_by_period.png`: national conflict words peak in 2021Q3 (`meeting_conflict` .91 per 10k vs about .50 in 2018-19) and settle a little above baseline (about .57 in 2023). The rate is lower in 2020 (about .37), when many meetings were remote. Conflict words are more common in D than R districts in 2022-23 (`meeting_conflict` t = -5.1, `conflict_legal` t = -3.9), though R and D districts differ in size and locale and that comparison has no control. State cells rest on few districts in places, and caption text carries no tone.

Together with TF-IDF (`source/analysis/tfidf/README.md`): meeting language became slightly more partisan around 2021, and much of that is gone by 2023. The evidence points more to shared national salience plus a professional-vocabulary divide than to two camps fighting over culture-war topics. County vote share is a proxy, and these are descriptive patterns, not causal estimates of the pandemic.

## Findings (first pass, 2018-2023)
1. Language is only slightly partisan per word: pi ~ 0.503-0.506 vs 0.5000 under shuffled labels. Real, but small.
2. It rose, then partly fell. Balanced panel, national vocabulary: 0.5028 (2018), 0.5029, 0.5034, **0.5042 (2021)**, 0.5039, 0.5034 (2023). Same shape with all vocabulary and with all districts. The jump in the quarterly series at 2020Q2 is mostly the sample doubling.
3. Roughly half of pi is geography (within-state shuffle gives ~0.501-0.502).
4. What drives it is mostly not the culture-war lexicon. Top R-leaning terms are informal/local talk (got, christmas, basketball, coach, ffa, football, prices). Top D-leaning are professional-administrative jargon (equity, inclusive, supporting, metrics, engagement, families). This looks like rural/small vs. urban/professionalised districts: county vote share is confounded with locale.
5. Lexicon categories, 2022-23 (district-mean mentions per 10k words, R vs D): equity/SEL strongly D (t = -17.7, up from -9.5 pre); unions D; immigration D; COVID terms flipped from R-heavy during 2020-21 (mask t = +7.4) to D-heavy after (vaccination, mandates); athletics, buses/facilities and religion R (rural). Book bans, CRT, parental rights and school choice are NOT detectably partisan in these meetings, and are rare (0.2-1 mentions per 10k words).
6. Culture-war topics became *salient everywhere at once*, not more partisan: CRT spikes in 2021Q3 and parental-rights/book-content terms rise from 2021 in R and D districts alike (`lexicon_category_trend.png`).
7. Our priors in `lexicon.py` (`expected`) were wrong for covid_mandates and gender_lgbtq: mentions measure who is discussing/implementing a topic, not who opposes it. Do not read the R/D tilt as stance.

## Locale/size control (GST)
`balanced_thinned_national_stratified`: R and D districts are compared only within the same locale (city/suburb/town/rural) x enrollment-tercile stratum (291 of 343 panel districts have both parties in their stratum). Net of the shuffled-label null, pi is .0025, .0023, .0036, .0043, .0038, .0033 (2018-23) vs .0028, .0029, .0034, .0043, .0040, .0034 without the control: the 2021 peak is not a rural/urban artifact, and the rise from 2019 is slightly larger once locale is held fixed.

## Conflict index and map
Index (0-100) = mean percentile of two word-list rates per district (>= 100k words; 1,355 of 1,532 districts): `meeting_conflict` (decorum, "out of order", "point of order", shouting, liar, arrests, ...) and `conflict_legal` (lawsuit, resignation, censure, misconduct, ...). Routine words are excluded on purpose (public comment = ~40% of the raw legal list, recall, transparency, harassment policy, disruption). Top districts include well-known flashpoints (Chino Valley CA, Brevard FL, Stockton CA); top states NJ, FL, PA (mean index 68, 68, 63). Lowest: MS, MN, UT, KY, NC, KS, GA. No boundary files offline, so maps are district points + a state tile grid.
Caveats: captions carry no tone, so this is conflict that is said, not heard; state means with < 5 districts are blanked; word lists were chosen by us and tightened after inspecting the top words.

## Caveats / next steps
- No sampling uncertainty on pi yet (only placebo and thinning noise): add district subsampling.
- Unigrams only; auto-caption quality and channel effects vary by year. Phrase counts cover the hand-picked phrases only.
- No speaker roles: public comment and board talk are mixed (matters for "parents' rights" talk).
