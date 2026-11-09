# Topic diffusion, Approach 2: tracking known talking-point phrases (prototype)

This is a trial run of Approach 2 in `docs/topic_diffusion_extension.tex`. We take a fixed list of known talking points,
record when each district first says each one, and ask whether adoption follows geography and partisanship.
Everything runs locally and calls no API. All results are **descriptive**.

## How to run (from the repo root, plain `python3`)
| step | script | time | writes |
|---|---|---|---|
| 1b | `mine_candidates.py` | ~1.5 min | `output/tables/diffusion/approach2/candidates_{ngram,quotes}.csv` |
| 1 | `phrases.py` | (imported) | the phrase list (regex variants, source, exclusions) |
| 2 | `count_mentions.py` | ~4 min | `Datastore/DistrictView/derived/diffusion/approach2/{mentions,docs,kwic}.parquet` |
| 3, 4a, 5 | `adoption.py` | ~20 s | `phrase_table.csv`, `adoption_monthly.csv`, `meeting_share_quarterly.csv`, `timing_regressions.csv`, `output/figures/diffusion/approach2/adoption_curves{,_talking_points}.png` |
| 4b | `hazard.py` | ~35 s | `hazard_models.csv`, `hazard_permutation.csv`, `hazard_by_phrase.csv` |

`common.py` holds the paths and loaders. The sample is the screened BoW sample: `status == "kept"` in
`{BOW_DERIVED_DIR}/manifest.parquet`, 80,063 transcripts and 1,485 districts. Years come from `globals.json`:
PRE = the first two sample years (2018-19) and POST = the rest (2020-23). Dates are `meeting_date`, falling back to
`published_at`. Every kept transcript has a meeting date, and **0 files read as empty** (no iCloud eviction in this run).

## Step 1: phrase list (34 phrases; regexes and exclusions in `phrases.py`)
Source codes: S = hand seed of national talking points. N = mined n-gram. Q = n-gram that recurs in the GABRIEL quotes.
- **Parents' rights:** parents bill of rights (S,N,Q: parent/parents/parent's/parental bill of rights) · parental rights (S,N) · moms for liberty (N) · let parents decide (N) · parent's choice (N,Q) · co-parent with government (S,N) · not what to think (S)
- **Masks and vaccines:** mask mandate (S,N,Q: mask/mass/math mandate[s], which are caption variants) · masks optional (N,Q) · unmask our children (S) · let them breathe (S) · right to breathe / breathe freely (N) · medical freedom (S) · medical decisions for my child (N) · my child my choice (S) · my body my choice (N) · vaccine mandate (S) · experimental vaccine (S) · god-given rights (N)
- **Curriculum, race and gender:** critical race theory (S,N) · crt (acronym) (S,N) · 1619 project (S) · indoctrination (S,N) · gender ideology (S) · groomer / grooming kids (S) · sexually explicit (S,N) · pornographic (S) · book bans (N,Q)
- **Left-coded counterparts:** my pronouns are · lgbtq plus · diversity equity inclusion · anti-racist · systemic racism · grading for equity (all N, some also Q)

**How the n-grams were mined (`mine_candidates.py`):**
- Every 3-5-gram in all 80k transcripts that contains an "anchor" word (about 70 contested-politics words) is hashed. For each district we count the PRE and POST transcripts that contain it.
- A candidate must appear in at least 20 districts and at least 5 states in POST, and its transcript share must rise at least 5-fold from PRE (with +1 smoothing). 18,750 hashes pass. Their strings are then recovered from 8k POST transcripts.
- Filler words ("um", "uh") are nearly absent from pre-2020 captions, and so is reopening and procedural boilerplate ("kids back in school", "agenda"). Both are filtered out. COVID-logistics n-grams are flagged.
- We picked the final phrases by hand from the top of `candidates_ngram.csv`.

The GABRIEL quotes are only about 60 sentences. Their n-grams (`candidates_quotes.csv`) mainly confirm the mask, DEI, book and "parent's bill of rights" items.

**Validation (`kwic.parquet`, 30 random contexts per phrase).** Hits in a procedural context are counted separately (`count_excluded`) and never define adoption. They are matches within 150 characters of an exclusion term:
- parental rights: FERPA, IEP, 504, special education, annual notice (3% of hits)
- crt: crisis response team or CRT training (1%)
- parent's choice: choosing a learning model during COVID (7%)
- grooming: hygiene or dress code
- pornographic: "child pornography"
- book bans: "banned books week"

## Steps 2-3: counts and adoption
- **Long table:** `mentions.parquet` has 35,877 nonzero (transcript, phrase) rows.
- **Adoption** is a district's first month with at least one non-excluded mention.
- **Panel districts:** the curves and regressions use only the 529 districts with meetings in both PRE and POST. Without this, a district that starts posting in 2021 would look like a 2021 adopter, and PRE mentions could not be seen.
- **District-weighted meeting shares:** `meeting_share_quarterly.csv` averages each district's own share of meetings that mention the phrase, so heavy posters do not dominate. The pooled share is also kept.

**Top phrases by spread.** All columns use all 1,485 districts, except median adoption, which uses panel adopters. The R share of adopters should be compared with the 47% of sample districts in counties where Trump got more than 50% of the 2020 two-party vote ("R counties").

| phrase | districts | states | first month | median adoption (panel) | R share of adopters |
|---|---|---|---|---|---|
| mask mandate | 990 | 48 | 2018-08* | 2021-06 | 44% |
| diversity equity inclusion | 726 | 44 | 2018-04 | 2021-03 | 32% |
| masks optional | 677 | 45 | 2018-08* | 2021-08 | 43% |
| critical race theory | 670 | 47 | 2018-07 | 2021-06 | 45% |
| parental rights | 583 | 46 | 2018-01 | 2021-08 | 44% |
| anti-racist | 517 | 47 | 2018-01 | 2020-07 | 29% |
| book bans | 292 | 40 | 2018-02 | 2022-05 | 40% |
| parents bill of rights | 181 | 25 | 2019-06 | 2022-06 | 50% |
| moms for liberty | 128 | 30 | 2018-07 | 2022-09 | 46% |
| co-parent with government | 118 | 28 | 2020-06 | 2021-09 | 36% |

\*Isolated false positives before 2020 ("mass mandate", "math mandate"). The full table is in `phrase_table.csv`.

**Timing.** The race and equity vocabulary (anti-racist, systemic racism) arrives first, in summer 2020. Then come masks (2021, with a cliff in August 2021) and CRT (May-July 2021). Books, grooming and gender ideology follow in 2022, and parents bill of rights and Moms for Liberty in 2022-23.

## Step 4a: partisanship and locale (`timing_regressions.csv`)
Sample: panel districts, excluding PRE adopters. Outcome: the chance of adopting in POST. Controls: locale and log meetings posted.

**Ever adopting, per 1 SD of county GOP share:**
- *Right-coded phrases are more likely in R counties:* pornographic +7.0 pp, parental rights +6.9, parents bill of rights +6.5, crt +6.2, god-given rights +5.5, moms for liberty +5.0, groomer +4.5, medical freedom +4.4.
- *Left-coded phrases are much more likely in D counties:* anti-racist -10.9 pp, DEI -9.1, lgbtq plus -8.2, grading for equity -8.2, systemic racism -6.2.
- *The pooled effect is about zero.*

**Masks: same reach, earlier in R counties.** Masks reach both sides equally, but R counties adopt "mask mandate" and "masks optional" about 2 months sooner per SD. Rural and town districts are about 10 pp less likely to adopt any phrase.

**Raw R shares understate the R gradient.** R-county districts post fewer meetings, so the raw R share of adopters is below 47% even for right-coded phrases.

## Step 4b: hazard and neighbour test (`hazard.py`)
- **Risk set:** panel district × phrase × POST month in which the district posted a meeting and had not yet adopted. That gives 523,255 rows and 4,946 adoptions, a base hazard of 0.95%.
- **Exposure:** the share of other districts that had adopted by t-1, measured three ways: (i) within 100 km, (ii) in the same state, (iii) with the same partisanship nationally.
- **Main model:** a linear probability model with phrase × month fixed effects, so each phrase's national trend is absorbed, and SEs clustered by district.

| exposure | LPM coef (se) | effect of +1 SD on the hazard |
|---|---|---|
| neighbours ≤ 100 km | 0.0123 (0.0040) | +0.16 pp (about +17% of base) |
| same state | 0.0331 (0.0070) | +0.38 pp (about +40%) |
| own GOP share (z) | 0.0018 (0.0003) | +0.18 pp |

The logit with phrase and month fixed effects gives the same signs for the neighbour and state terms (0.59 and 1.26). The co-partisan share flips sign between the two models, because it is nearly collinear with the fixed effects. Do not interpret it.

**Permutation null (200 draws each, `hazard_permutation.csv`).**
- *Shuffling locations across all districts:* null mean 0.000, 95th percentile 0.0043. No draw reaches the observed 0.0123 (p < 0.005).
- *Shuffling only within state (stricter):* null mean 0.0036, 95th percentile 0.0090, p = 0.01.

So adoption clusters locally beyond what the state level explains. This is consistent with diffusion, but also with shared regional shocks such as county health orders, local media markets and coordinated campaigns. Per phrase (`hazard_by_phrase.csv`), the neighbour effect is clearest for indoctrination, crt, lgbtq plus and masks optional. The state effect is strongest for parents bill of rights (state bills such as Florida HB 1557) and mask mandate.

## Data problems and caveats
- **Caption spelling:** "mask" is often captioned "mass" or "math" ("mass mandate" appears in about 500 districts). Hyphens and "+" become spaces ("co parent", "lgbtq plus"), and "parents'" loses its apostrophe. The regexes include these variants. The rarer variants we found by mining, so others probably remain.
- **The caption format changes in 2020:** fillers ("um", "uh") almost never appear before 2020. Any "new n-gram" mining has to drop them, or the format change looks like new vocabulary.
- **A mention is not an endorsement:** opponents quote phrases too ("calling teachers groomers", "manufactured talking points around book banning crt"). Neutral or procedural uses also remain, for example FERPA "parental rights", parent's choice of learning model, and grooming in child-abuse training before 2020. The exclusion windows catch only some of these.
- **Sample composition:** only 529 of 1,485 districts post in both periods. PRE has far fewer meetings than POST, and districts differ in posting frequency. Adoption is only observable in months a district posts, so the hazard controls for words and meetings in the month.
- **District location:** distance uses the district-office latitude and longitude. Partisanship is the county vote, not the district's.

## Next steps
1. Hand-label about 50 KWIC contexts per phrase for precision, and separate stance (pro vs quoting or opposing), for example with GABRIEL on a small sample.
2. Add advocacy-group phrase lists (Moms for Liberty, Parents Defending Education) with dated first use, and test lead-lag from national events.
3. Replace county partisanship with school-district returns, and add state law and mandate dates as controls to separate regional shocks from peer diffusion.
4. Check robustness to the radius (50 and 200 km), to defining adoption as 2+ mentions, and to measuring exposure among panel districts only.
5. Connect to Approach 1 by seeding phrase discovery with its similarity clusters.
