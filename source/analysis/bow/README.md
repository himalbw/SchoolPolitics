# Bag-of-words analysis (Stat 236 school boards)

Run everything from the repo root, in this order (data paths are under `Datastore/DistrictView/derived/`):

| step | script | does |
|---|---|---|
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
