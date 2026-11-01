# GABRIEL results (descriptive): stance and tone in school-board meetings, 2018–2024

Updated 2026-10-28 (sample extended to 2020 and 2024, 2021 topped up, public-comment share added). All results are
descriptive associations. Event-study coefficients are per 10 pp of county GOP share × year, 2019 = 0, district + year
FE, locale × year, 95% CI clustered by district, random excerpts (`output/tables/gabriel_event_study*.csv`).

## Headline

**1. The R–D gap in conservative positions and parental-rights framing opens in 2021 (mostly 2021H2), is not
distinguishable from zero in 2022, is present again in 2023, and is back to zero in 2024. 2020 is flat.**

| | 2018 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| Conservative positions argued | +0.10 [−0.14, 0.34] | −0.05 [−0.36, 0.25] | **+1.14** [0.64, 1.63] | +0.24 [−0.24, 0.72] | **+0.54** [0.11, 0.96] | +0.06 [−0.21, 0.33] |
| Parental-rights framing | +0.28 [−0.01, 0.58] | +0.25 [−0.16, 0.66] | **+1.04** [0.60, 1.49] | +0.38 [−0.09, 0.86] | **+0.65** [0.27, 1.03] | +0.26 [−0.04, 0.56] |
| Hostility | +0.03 [−0.34, 0.39] | +0.24 [−0.16, 0.63] | +0.38 [−0.12, 0.88] | +0.40 [−0.12, 0.92] | **+0.54** [0.10, 0.99] | +0.27 [−0.11, 0.66] |
| COVID contention | +0.01 [−0.02, 0.03] | −0.50 [−1.01, 0.02] | **+1.06** [0.36, 1.76] | −0.35 [−0.86, 0.17] | +0.03 [−0.04, 0.10] | −0.03 [−0.09, 0.03] |

Half-year event study for 2020–21 (calendar years otherwise, 2019 = 0; `gabriel_event_study_halfyear.png`):

| | 2020H1 | 2020H2 | 2021H1 | 2021H2 |
|---|---|---|---|---|
| Conservative positions argued | −0.12 [−0.42, 0.18] | +0.16 [−0.10, 0.43] | +0.41 [0.01, 0.82] | **+1.88** [1.05, 2.71] |
| Parental-rights framing | +0.23 [−0.20, 0.65] | +0.41 [−0.02, 0.84] | +0.49 [0.13, 0.85] | **+1.63** [0.94, 2.33] |
| COVID contention | +0.13 [−0.17, 0.43] | −1.08 [−2.07, −0.10] | +0.07 [−0.60, 0.74] | **+2.13** [1.06, 3.21] |
| Hostility | +0.29 [−0.16, 0.74] | +0.26 [−0.18, 0.70] | +0.13 [−0.35, 0.61] | +0.65 [−0.01, 1.31] |

The divergence is small in 2021H1 and concentrated in 2021H2 (the fall 2021 mask-mandate and curriculum fights;
COVID contention peaks in the same half-year). In levels (`gabriel_headline.png`), conservative positions rise in both
R and D counties in 2021 (D 0.24 → 4.30, R 0.32 → 6.82 from 2019) and fall back to ~0.9 in both by 2024. The R − D gap
in the headline figure (bootstrap over districts) is +2.52 [1.11, 4.03] in 2021, +0.22 [−1.20, 1.76] in 2022,
+1.42 [0.24, 2.60] in 2023 and +0.05 [−0.54, 0.74] in 2024; for parental-rights framing the 2023 gap is
+0.80 [−0.20, 1.84] (the event study, which adds district FE and locale × year, gives a significant +0.65).
"Persists through 2023" is therefore too strong: 2022 is indistinguishable from zero for both attributes.

**2. The 2021 rise is concentrated in public-comment excerpts, not board or staff discussion.**
Yearly event study split by `public_comment_share` (≥ 50 = mostly members of the public speaking;
`gabriel_event_study_by_speaker.csv/.png`):

| conservative positions | 2018 | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|---|
| Mostly public comment | −1.55 [−6.80, 3.70] | −1.15 [−8.19, 5.89] | **+12.05** [5.28, 18.82] | +1.28 [−7.94, 10.50] | +2.31 [−3.72, 8.34] | +2.09 [−3.35, 7.52] |
| Mostly board / staff | +0.05 [−0.12, 0.21] | +0.04 [−0.08, 0.15] | **+0.42** [0.07, 0.78] | +0.29 [−0.08, 0.65] | +0.31 [−0.09, 0.72] | +0.03 [−0.20, 0.26] |

**Small-cell caveat.** Only ~6–10% of random excerpts are mostly public comment: 55, 68, 118, 199, 66, 67, 99
excerpts in 2018–2024, from 49–149 districts per year (vs ~970–1,900 board/staff excerpts in all 348). Every
public-comment coefficient except 2021 has a CI including 0 and ~±6–9 wide; the 2021 point estimate is large but
imprecise. Hostility shows no partisan divergence in either group (public 2021 +0.66 [−8.09, 9.41]; board 2023
+0.49 [0.03, 0.96] is the only CI excluding 0). Public comment itself was more common in 2021 (9.7% of random
excerpts ≥ 50, vs 5–6.5% in other years). Excerpts are 1,500-word windows, so "mostly public comment" is a
classification of the window, not of individual speakers. Extra random excerpts (2021, 2023) have no public-comment
score and are excluded from this split.

**3. Hostility rose everywhere and does not replicate in split B.** Random-excerpt hostility rose in both R and D
counties in 2021 (D 3.69 → 5.16, R 2.54 → 5.17 from 2019). R-county meetings were less hostile before COVID (2019 R − D
gap −1.15 [−2.12, −0.21]) and level afterwards (2021–24 gaps −0.54 to +0.28, all CIs include 0), so the positive
GOP × post coefficient (+0.39, p = .007) reflects catch-up rather than R counties overtaking D. It is significant on
split A (+0.47, p = .04) but **not on split B (+0.25, p = .16)**, the clean replication split, and in the event study
only 2023 is individually significant.

## Summary regression (single post period; random excerpts)

score ~ GOP share (per 10 pp) × post + district FE + year FE + locale × post, clustered by district, 2020 excluded
(neither pre nor post). `gabriel_regressions.csv` (both post definitions), `gabriel_regressions_split.csv` (post =
2021–24). Wording was developed on split A, so **split B is the clean replication**.

| attribute | post = 2021–24 | 95% CI | post = 2021–23 | split A | split B |
|---|---|---|---|---|---|
| **Conservative positions argued** | **+0.45** (p .002) | 0.17, 0.74 | +0.59 (p .001) | +0.29 (p .10) | **+0.55 (p .014)** |
| **Parental-rights framing** | **+0.45** (p .001) | 0.18, 0.71 | +0.55 (p .001) | +0.33 (p .09) | **+0.51 (p .004)** |
| Hostility | +0.39 (p .007) | 0.11, 0.67 | +0.43 (p .010) | +0.47 (p .04) | +0.25 (p .16) |
| Share on culture-war issues | +0.34 (p .14) | −0.11, 0.79 | +0.38 (p .17) | +0.15 | +0.43 |
| Contention over COVID policy | +0.18 (p .14) | −0.06, 0.41 | +0.24 (p .12) | +0.01 | +0.32 (p .08) |
| Progressive positions argued | −0.03 (p .90) | −0.42, 0.37 | −0.07 | +0.01 | −0.14 |
| Stance (secondary) | +0.48 (p .014) | 0.10, 0.86 | +0.66 (p .003) | +0.27 | +0.68 (p .006) |

Adding 2024 shrinks every coefficient (the 2024 gap is ~0). Compared with the 2018–23 run (conservative +0.68,
parental rights +0.67, hostility +0.59), the 2021–23 coefficients are also somewhat smaller, because 2021 now has 6
meetings per district (3 per half-year) instead of 3 and 2021 and 2023 have two extra random windows per meeting.
Split A no longer reaches p < .05 for conservative positions or parental rights; split B does.

## Sample and design changes (2026-10-28)

- **Same 348 districts and A/B split.** Eligibility rules unchanged (`build_excerpts.py --extend`); channel stats are
  now computed over 2018–2024, under which 4 of the 5,196 original meetings would no longer qualify (kept).
- **2020 added**: 6 meetings per district-year, 3 per half-year where available (a short half is filled from the
  other half); 2,036 meetings in 347 districts (mean 5.9 per district).
- **2021 topped up** from 3 to 6 meetings per district (original 3 kept; 3 per half-year where available); 2,060
  meetings in 348 districts.
- **2024 added**: 1,916 meetings in 332 districts (16 districts have no eligible 2024 meeting).
- Random + flagged excerpts built exactly as before, name-stripped with the current rules: 11,295 new excerpts
  (4,971 random, 6,324 flagged), scored on gpt-6-luna with attributes v3. `scores_full.parquet` has `batch`
  (original / extension / extra_random) and `half_year` for every row; the pre-extension file is kept as
  `scores_full_original.parquet`.
- **Public-comment share** (`public_comment_share.parquet`, keyed by excerpt_id): one-attribute GABRIEL pass over all
  22,174 original + extension excerpts (logged in `attribute_log.md`). Median 0, mean 12.3; share ≥ 50 is 5–10% of
  random and 9–20% of flagged excerpts by year.
- **Extra random excerpts (step 3).** Variance of random-excerpt conservative_positions before adding them:
  **72.7% within district-year**, 21.6% between years within a district, 5.7% between districts
  (`gabriel_variance_split.csv` now shows the split including the extra excerpts: 79.6 / 13.5 / 6.9). Because within
  > 50%, 2 more random windows per meeting were built for every sampled meeting, but **only 2021 and 2023 were scored**
  (budget): 6,200 excerpts. Random excerpts per district-year are therefore 3 (2018, 2019, 2022), 6 (2020, 2024),
  18 (2021) and 9 (2023); the district-year mean is the estimand either way. Width of the R − D gap CI in the headline
  figure (`gabriel_extra_random_ci.csv`):

  | | 2021 before → after | 2023 before → after |
  |---|---|---|
  | Conservative positions argued | 3.55 → 2.94 | 2.98 → 2.31 |
  | Parental-rights framing | 2.97 → 2.43 | 2.82 → 1.94 |

### Caveats specific to the new years

- **2020**: many meetings were virtual or shortened, and public comment was often restricted, suspended or read into
  the record; transcripts that pass the ≥ 3,000-word filter may over-represent longer, in-person or hybrid meetings.
  Low 2020 scores partly reflect what was possible to say at the meeting, not only what people wanted to say.
- **2024 is outside the team's `globals.json` window** (`SAMPLE_END = 2023`). The GABRIEL scripts set `END_YEAR = 2024`
  locally; BoW / LDA results are not directly comparable for 2024. 16 of 348 districts have no 2024 data.

## The 2022 dip

Still not a composition change: 2022 and 2023 have the same 348 districts, ~3 meetings per district and ~1,490 words
per excerpt (`gabriel_dip_composition.csv`); 2022 has fewer July–September meetings (19.8% vs 24.7% in 2021 and 23.6%
in 2024). The half-year results suggest 2021 was a fall-2021 spike rather than a level shift, so the "dip" is better
read as 2021H2 being unusual. Individual-year gaps remain imprecise (`gabriel_gap_conservative_by_split.csv`).

## Illustrative districts and quotes (gpt-6.1-terra)

Picked from the top-3 lists in `gabriel_change_district.csv` (random excerpts, 2018–19 → 2021–24, ≥ 10 sampled
meetings; all candidates have 30):

| leaid | district | state | GOP share 2020 | meetings | attribute | pre | post |
|---|---|---|---|---|---|---|---|
| 1200930 | Indian River | FL | 0.61 | 30 | conservative_positions | 0.0 | 27.5 |
| 4219920 | Radnor Township SD | PA | 0.36 | 30 | progressive_positions | 0.0 | 11.7 |

Other candidates: Mentor Exempted Village OH (R, +15.2), Connetquot Central NY (R, GOP 0.50, +13.6); Virginia Beach VA
(D, +23.0), Old Bridge Township NJ (D, +12.3).

Method (`quotes.py`): for each district, the 25 highest-scoring excerpts on its position attribute and the 10
highest on hostility; Terra received the RAW (not name-stripped) window and returned the single sentence (or a 10–60
word span when captions have no punctuation) that most strongly argues the position / is most hostile. A quote is kept
only if it is an exact substring of the raw window text (timestamps and `[tags]` removed, as in
`build_excerpts.CleanRaw`; a sentence spanning caption lines cannot match the timestamped file itself). Names of
private individuals were then replaced with `[NAME]` (Terra's list + "my name is …" + spaCy PERSON, so some officials'
names are also redacted); district and place names kept. Caption errors are kept verbatim (e.g. "coveted" for
"covid", "rner"/"ratner" for Radnor).

- **Indian River** (`output/tables/quotes_1200930.csv`): 35 excerpts → 35 verified → 34 quotes (25 position, 9
  hostility). Mask-optional / parental-choice arguments (2020–21), then book removal (2022–24), CRT and the
  "safe, respectful and inclusive education" policy; most come from public comment (public_comment_share 72–100).
  Example: "i just want to say that we need to put the choice of masking children back in the hands of the parents …"
  (2021-01-26, score 82).
- **Radnor** (`output/tables/quotes_4219920.csv`): 35 excerpts → 28 verified → 26 quotes (19 position, 7 hostility);
  the 7 failures are all excerpts where Terra found no fitting sentence (empty quote), none were mismatches. Radnor's
  top progressive excerpts score far lower than Indian River's conservative ones (only 8 of 25 above 30), so its
  low-scoring quotes are weak illustrations (e.g. a charter-funding remark, score 4). Strong ones: DEI as board policy
  (2022), mandatory masking (2020), staff diversity (2024). Its most hostile quotes come from the conservative side
  (mask-optional, the "Raider" mascot name change, 2021–22).
- Quotes with a missing public_comment_share come from extra random excerpts (not scored for public comment).

## Within a topic (flagged excerpts)

`gabriel_topic_period.png/.csv` (pre = 2018–19, post = 2021–24; 2020 shown in the CSV as `covid`). Hostility when a
contested topic comes up rose after 2020 in every category except equity/SEL, most for CRT/race (R 6.5 → 24.0,
D 12.2 → 21.2) and parental rights (R 3.2 → 12.9, D 7.8 → 10.0). Stance moved in the conservative direction in both R
and D counties for books (R −1.3 → +2.3, D −8.2 → −4.2), COVID mandates, parental rights and CRT/race (R +2.3 → +27.1,
D −15.4 → +6.9; pre-period thin). Within-category GOP × post regressions show no detectable partisan divergence (all
CIs include 0; CRT/race has 14 districts observed both pre and post and is `thin`).

## Where (maps)

`gabriel_change_map.png` (state tiles) and `gabriel_change_map_appendix.png`; per-district values (now also pre/post
means of conservative and progressive positions, n_meetings, place name) in `gabriel_change_district.csv`. Change
2018–19 → 2021–24: conservative positions rose in all 22 states with ≥ 5 districts (largest FL +5.8, NC +5.0, WI +4.1,
KS +4.0, SC +3.9); hostility rose in 17 of 22 (largest FL +3.4, NC +2.7, SC +2.5) and fell in MI −3.4, IN −2.4, MA −0.7,
CT −0.6, MD −0.3. State means rest on 5–32 districts each.

## Stance (secondary) and model dependence

Unchanged from the 2018–23 run (Terra check uses the same 1,000 original excerpts): progressive_positions correlates
only 0.70 between Luna and Terra in random excerpts, so stance levels are model-dependent; lead with
conservative_positions (r = 0.97). `gabriel_terra_check.csv`, `gabriel_model_check.csv`.

## Checks and limitations

- **Leaked names** (`gabriel_leak_check.csv`): the stance–GOP gradient holds in excerpts without a detected leftover
  place name (flagged +2.73, random +1.16 per 10 pp, both p < .001); weaker where one is detected (flagged +0.82,
  p = .14; random +0.59, p = .05). Bare surnames still leak (`PIPELINE_NOTES.md`).
- County vote share proxies district partisanship; public-comment share is inferred from caption text (no speaker
  labels); requiring ≥ 2 eligible meetings in every year 2018, 2019, 2021–23 leaves 348 districts that over-represent
  suburbs and D-leaning counties.
- Many attributes and wording revisions were tried (`attribute_log.md`); cite split-B estimates as confirmatory.

## Figures and tables

Figures (`output/figures/`): `gabriel_headline.png` (slide: R vs D trends and R − D gap, conservative positions and
parental-rights framing, 2018–2024), `gabriel_trends_random.png`, `gabriel_gap_by_year.png`,
`gabriel_event_study.png` (yearly, 2018–2024), `gabriel_event_study_halfyear.png` (2020H1–2021H2),
`gabriel_event_study_by_speaker.png` (public comment vs board/staff), `gabriel_topic_period.png`,
`gabriel_change_map.png`, `gabriel_change_map_appendix.png`.

Tables (`output/tables/`): `gabriel_event_study.csv`, `gabriel_event_study_halfyear.csv`,
`gabriel_event_study_by_speaker.csv`, `gabriel_regressions.csv`, `gabriel_regressions_split.csv`,
`gabriel_gap_by_year.csv`, `gabriel_trends_random.csv`, `gabriel_variance_split.csv`, `gabriel_extra_random_ci.csv`,
`gabriel_change_district.csv`, `gabriel_topic_period.csv`, `gabriel_dip_composition.csv`,
`gabriel_gap_conservative_by_split.csv`, `gabriel_leak_check.csv`, `gabriel_terra_check.csv`,
`gabriel_model_check.csv`, `gabriel_district_year.csv`, `quotes_1200930.csv`, `quotes_4219920.csv`.

## Costs (actual, from API usage fields)

| session | key | step | calls | cost |
|---|---|---|---|---|
| 2026-10-27 | (earlier key) | pilots | 600 | $0.17 |
| 2026-10-27 | | full run, original 10,879 excerpts (Luna) | 10,879 | $3.18 |
| 2026-10-27 | | Terra check, 1,000 excerpts | 1,000 | $2.92 |
| 2026-10-28 | nwaHuy (by mistake) | model check | 4 | ≈ $0.00 |
| 2026-10-28 | nwaHuy | extension scoring (2020, 2021 top-up, 2024) | 11,295 | $3.31 |
| 2026-10-28 | nwaHuy | partial extra-random + public-comment runs before credits ran out (saved) | 3,000 | $0.73 |
| 2026-10-28 | nwaHuy | in-flight calls lost when those runs were stopped (estimate, upper bound) | ~1,333 | ~$0.35 |
| 2026-10-28 | 5eeu | key check | 1 | ≈ $0.00 |
| 2026-10-28 | 5eeu | public-comment share, original + extension | 20,174 | $4.44 |
| 2026-10-28 | 5eeu | extra random excerpts, 2021 + 2023 | 5,753 | $1.68 |
| 2026-10-28 | 5eeu | Terra quotes, 2 districts | 70 | $0.14 |
| | | **2026-10-28 session** | | **≈ $10.66** (nwaHuy ≈ $4.40, 5eeu $6.26) |
| | | **all sessions** | | **≈ $16.93** |

Ledgers: `Datastore/DistrictView/derived/gabriel/session_costs_{nwaHuy,5eeu}.csv`. Reused responses (2,000
public-comment and 447 extra-random rows first billed to nwaHuy) are counted once, under nwaHuy.
