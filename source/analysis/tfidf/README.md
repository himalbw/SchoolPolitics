# TF-IDF analysis (Stat 236 school boards)

Which words characterise each period and party, and do same-party districts talk more alike over time?

```
python3 source/analysis/tfidf/tfidf.py --indir Datastore/DistrictView/derived/bow_dq --tag v1    # about 35 s
python3 source/analysis/tfidf/tfidf.py                                                           # defaults: derived_v2/bow_dq, tag v2
```
Run from the repo root. `--indir` needs `dtm.npz`, `vocab.parquet`, `units.parquet`. Outputs: `output/tables/tfidf_<tag>/`, `output/figures/tfidf_<tag>/`, and the matrix in `<indir parent>/tfidf/` (`tfidf_district_period.npz`, `rows.parquet`, `terms.parquet`; 134 MB on v1). The script does not read `globals.json` or import `bow_data.py`. It reads the inputs named above and `district_static.parquet` directly.

The findings below are for the v1 matrix (3,121 district-periods x 60,445 terms, 28.4M non-zeros). The v2 input (Stefan's cleaned transcripts) has an identical count matrix, so they also describe `tfidf_v2`.

## Screened sample (Stefan's dedupe + eligibility screen; current headline)
`python3 source/analysis/tfidf/tfidf.py --indir Datastore/DistrictView/derived_screened/bow_dq --tag screened`. 3,121 -> 2,949 district-periods (pre 490, covid 1,187, post 1,272); balanced panel 383 districts (145 R, 238 D). The findings below hold in the same direction. National vocabulary, all pairs:

| sample, vocabulary | pre | covid | post | post - pre [95% bootstrap CI] |
|---|---|---|---|---|
| balanced, national | .0030 | .0048 | .0059 | +.0029 [.0017, .0043] |
| balanced, national_stable | .0021 | .0026 | .0037 | +.0016 [.0007, .0027] |
| all districts, national | .0067 | .0086 | .0080 | +.0013 [-.0022, .0048] |
| all districts, national_stable | .0084 | .0088 | .0083 | -.0001 [-.0043, .0036] |

- In the same districts at equal length, the rise is a little larger, and covid to post is now borderline (+.0011 [.0000, .0026]; stable vocabulary +.0011 [.0001, .0021]), rather than clearly flat. With all districts there is still no detectable change, so this is not a robust rise.
- The gap is still D-county cohesion: balanced post DD - RD = +.0088 [.0058, .0119], RR - RD = -.0020 [-.0041, .0005].
- GST peaks in 2021 and falls by 2023, while this gap keeps edging up. They measure different things (word-level party predictability vs overall document similarity), and the panels differ, so the two should not be treated as one series.

- **Document** = one district x period (pre 2018-19, covid 2020-21, post 2022-23), with quarters summed. Each district counts once per period, whatever its volume.
- **Filter** = at least 20,000 words before vocabulary cuts (as in `partisan_lexicon.py`) and a partisan anchor. This drops 179 + 7 of 3,307 district-periods.
- **TF-IDF** = sklearn `TfidfTransformer`: sublinear tf `1 + ln(tf)`, smooth idf `ln((1+n)/(1+df)) + 1`, rows L2-normalised.
  - df is counted across district-periods, so a district that uses a term in all three periods counts three times.
  - One idf is shared across periods so that weights compare over time. It leans towards covid and post, which have more documents.
- **Anchor** = county 2020 two-party GOP share > 0.5 is R. It is fixed over time, and county is a proxy for district.
- **Tests** use one value per district-period: Welch t with Benjamini-Hochberg q over rankable terms (used in >= 50 district-periods of the period), plus a standardised difference. t grows with the number of districts, so `partisan_movers` ranks on the standardised difference.
- **`t_within_locale`, `t_within_state`** = the R-D gap compared only inside a locale (city/suburb/town/rural) or state.

### Two samples, because L2-normalised TF-IDF is not length-neutral
| sample | what | use for |
|---|---|---|
| `all_districts` | every district-period passing the filter (pre 520, covid 1,259, post 1,342) | cross-sections, largest n |
| `balanced_thinned` | 436 districts (166 R, 270 D) with >= 20,000 tokens in all three periods, binomially thinned to about 20,000 tokens each, with their own idf | anything over time |

Cosine levels are not comparable between the samples, because longer documents are less noisy. The balanced panel is also a selected group: 436 districts that were already posting long transcripts in 2018-19.

### Pairwise measure
All pairs per period (no sampling): the mean cosine of same-party pairs minus different-party pairs, from a dense matrix with a zero diagonal. It is reported for:
- **Vocabulary:** `all`, `national`, and `national_stable` (national minus 4,976 terms whose use moved from pre to covid or post in the balanced panel, paired t, q < .05).
- **Pairs:** `all_pairs` and `different_state`.
- **Nulls (500 draws):** labels shuffled at random, within state, and within locale. The same shuffle is used in every period, so changes get a null too. p-values floor at .002.
- **Bootstrap (300 draws):** districts resampled, with the same draw in every period.

`RR_minus_RD` and `DD_minus_RD` show which party's cohesion produces the gap.

In `all_districts`, the shuffle p-value for a *change* ignores the changing district mix and is far too small (shuffle sd about .0004, against a bootstrap 95% width of about .008). Use the bootstrap CI there.

## Place names and state-specific words
Thresholds for `national` match `partisan_lexicon.py`: at least 10 states and no more than 30% of usage in one state. `place_token` means the term matches a token of any district, place or county name, or a state name. `contamination.csv` has every share.

| table | result |
|---|---|
| Top terms by period or party | Not contaminated: 0% state-concentrated, at most 6% place tokens. Ordinary words dominate, and the all and national lists are identical. |
| R vs D, all vocabulary | Mild, and on the D side: 6-12% of the top 50 are state-concentrated, such as `latino`, `latinx`, `los`, `eld`, `bilingual`, `pacific`, `islander`. |
| R vs D, national | 0% by construction. Place tokens are 0-6% and look like false matches (`thousand`, `state`, `ball`). |
| Each district's own top 15 | **Heavily local.** With all vocabulary, 66-72% of terms are state-concentrated, about 12% are place tokens, and about 5% match the district's own name. |

Three limits on these numbers:
- The national filter does not fix the district-level lists. Brevard's national list is still `schiller mcdougall takis mullens gibbs mims melbourne ...`, mostly surnames and local words that span enough states to pass. Treat `example_districts.csv` as a check on what TF-IDF is doing, not as a description of agendas.
- The own-name share understates the problem. Nearby towns such as `batavia` and `viera` are not in the district's name.
- The filter also over-cuts. `latino`, `ethnic` and `bilingual` are demographic vocabulary rather than place names, and they are removed because of where they are used. The `all` rows are kept so this stays visible.

Place vocabulary does not drive the pairwise result: `different_state` gaps are within .0006 of `all_pairs` in every national row.

## Caption drift: read this before using the pre-to-post tables
Much of the change over time is the captioning system, not the speakers. In an unweighted spot check of 360 raw transcripts (60 dated meetings per year, one draw, rates per 10k words):

| term | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|---|---|---|---|---|---|---|
| `alright` | 0.63 | 1.47 | 0.53 | 0.00 | 0.00 | 0.00 |
| `mister` | 0.39 | 0.33 | 0.41 | 0.01 | 0.02 | 0.03 |
| `seventy` | 0.77 | 0.80 | 0.30 | 0.30 | 0.11 | 0.03 |
| `foreign` | 0.14 | 0.14 | 0.16 | 0.50 | 1.75 | 5.53 |

Read in context, `foreign` often sits at pauses or beside `[music]` tags, so it looks like a caption artifact for non-speech rather than a word people said. Some uses are real ("foreign language"). The top risers also include mis-captions of COVID: `covet`, `coven`, `coveted`, `kovid`, `coid`, `cobit`. The top fallers are number words and spellings such as `alright`, `mister`, `nineteen`, `seventy`. These readings are inferred from the patterns and were not checked against any record of a caption-model change.

## Findings (v1)
Gap = within-party minus across-party mean cosine, national vocabulary, all pairs.

| sample, vocabulary | pre | covid | post | post - pre [95% bootstrap CI] |
|---|---|---|---|---|
| balanced, national | .0027 | .0048 | .0053 | +.0026 [.0013, .0040] |
| balanced, national_stable | .0021 | .0025 | .0033 | +.0012 [.0003, .0020] |
| all districts, national | .0072 | .0091 | .0079 | +.0007 [-.0036, .0046] |
| all districts, national_stable | .0096 | .0094 | .0082 | -.0014 [-.0054, .0025] |

1. **Same-party districts are slightly more alike, and the gap is outside placebo noise but small.** Each gap in the table exceeds the 95% random-shuffle band (at most about ±.0013). The largest gap in the balanced panel, .0053, compares with mean cosines near .37. Shuffling within state or within locale still gives about .0014 of the post-period gap of .0053, so geography and locale each account for roughly a quarter of it, and these overlap. Locale here has four groups and no size control.
2. **Whether the gap grew depends on the spec, so this is not a robust rise.**
   - Same districts at equal length: it grew from pre to post, and most of the rise came by covid. The covid-to-post change is within noise (+.0005 [-.0007, .0018], shuffle p = .35).
   - With unstable vocabulary removed, under half of the rise remains. This does not separate caption drift from real COVID talk, because both are in the removed terms.
   - All districts: no detectable change, with wide CIs. The point estimate falls after covid.
   - Three other thinning seeds gave post - pre of +.0025 to +.0027 (national) and +.0011 to +.0014 (stable).
3. **The gap is mostly D-district cohesion, not two camps.** In the balanced panel with national vocabulary, post-period DD - RD = +.0078 [.0047, .0103], while RR - RD = -.0011 [-.0030, .0011]. R districts are no more similar to each other than to D districts in the post period. `national_stable` points the same way (DD - RD = +.0042 [.0020, .0060]; RR - RD = +.0008 [-.0007, .0029]). From pre to post, DD - RD rose by .0048 [.0024, .0073] and RR - RD fell by .0030 [-.0053, -.0010]. This fits a shared professional vocabulary spreading among D-county districts, but the data do not establish that reading. It is not evidence of two opposed vocabularies.
4. **R vs D terms repeat the bag-of-words result.** In 2022-23 (national, all districts):
   - R-leaning: `christmas` (t = 16.3), `ffa`, `pay`, `paid`, `got`, `bus`, `football`, `coach`.
   - D-leaning: `culturally` (t = -14.7), `asian`, `colleague`, `inclusive`, `equity` (-12.9), `lens`, `diversity`.
   - The gaps stay large within state and within locale (median |t| across the top 100 terms is 9.2 and 7.1). The contrast is still informal/local against professional/equity talk, which may reflect district type as well as politics. Mentions are not stance.
5. **Terms with detectable R-D gaps became more common, then flattened.** Share of national terms with q < .05 in the balanced panel: 7.7% pre, 12.8% covid, 11.2% post. In `all_districts` these counts are not comparable, because n more than doubles.
6. **Top-term tables say little.** School, year, students, board, want and time lead every period and party, and the top-50 R and D lists share 44-47 terms. With smooth idf, mean TF-IDF rewards words every district uses. The contrasts in `distinctive_R_vs_D` and `changed_pre_post` carry the information.
7. **`partisan_movers_pre_post.csv` is exploratory only.** It ranks the extremes of about 8,000 noisy terms and has no null. The balanced-panel list mixes suggestive terms (`conservative`, `america`, `systemic`, `equity`) with noise (`pounds`, `leaves`, `opera`). Do not cite single terms from it.

## What was verified
- TF-IDF matches a hand calculation of the formula on one row (max difference 2e-9). Row norms are 1, tokens are conserved, and the saved matrix equals the recomputed one with labels aligned.
- Welch matches `scipy.stats.ttest_ind` on `equity`; BH matches `scipy.stats.false_discovery_control`; the stratified t reduces to Welch with one stratum on a toy example.
- The fast gap formula matches a brute-force pair average, weighted and unweighted, on 200 districts.
- Top terms and the R/D contrast match the existing bag-of-words findings.

## Not done
- No simulation with a planted partisan signal was run for the pairwise measure.
- Caption model version is not observed, so drift is only partly handled. If R and D districts moved to newer captions at different times, the over-time gap could still be affected in ways `national_stable` does not remove.
- No enrollment or joint state x locale control in the nulls. County is a proxy for district partisanship, and there are no speaker roles.
- No multiple-comparison adjustment across the pairwise specs, and the bootstrap drops self-pairs of resampled districts, which is a small approximation.
