# School closures and educational outcomes (Stat 236 school boards)

Do closures and educational outcomes account for the partisan-language results in `source/analysis/bow/`? Uses the screened BoW sample (`BOW_DERIVED_DIR` in `globals.json`). Outputs go to `output/tables/closures_screened/`. Needs `pyfixest`.

Run from the repo root with `PYTHONPATH=source/analysis/closures:source/analysis/bow`, in this order:

| step | script | does |
|---|---|---|
| 1 | `count_closure_phrases.py` | counts `closure_lexicon.py` phrases per transcript -> `bow_phrases/closure_phrase_counts.parquet` (about 5 min) |
| 2 | `closure_profile.py` | what predicts closures, enrollment first stage, state-level test-score bins |
| 3 | `closure_language.py` | levels, long differences, closure x GOP, event study, weekly timing (`language_*.csv`) |
| 4 | `closure_gst.py` | covariate-adjusted pi, pi without pandemic vocabulary, pi with other labels (`gst_*.csv`; about 10 min; `--unbalanced` for all district-years) |
| 5 | `closure_mechanism.py` | paired reductions, closure slope, fixed 2018-19 word weights, remote-meeting words (`mech_*.csv`; about 10 min) |
| 6 | `closure_format_tests.py` | placebo word sets and same-district-year remote vs other meetings (`format_*.csv`; over 10 min) |
| 7 | `closure_exit_voice.py` | enrollment loss vs later contested talk, with pre-trend placebo (`exit_voice.csv`) |

`closure_data.py` = shared covariates and rates. Closure index = virtual share + 0.5 x hybrid share, school year 2020-21. Achievement = SEDA pooled 2009-19, so pre-pandemic. The hub's `AssmtOutcomesByInPersonAccess` files are state x access-bin cells, not district data.

## Findings (2026-11-04)
One point = 0.0001 of pi. Balanced panel, thinned, national vocabulary, 286 districts; delete-20% jackknife SEs. Descriptive, not causal.

- pi - 0.5 equals half the R-D gap in leave-out slant, so regressing slant on party with controls gives an adjusted pi. The rise from 2018-19 to 2021 is 14.0 (SE 3.5); to 2023 it is 6.0 (SE 3.7).
- Closure removes 4.3 points of the rise (SE 1.2, 31%), or 1.9 within state (SE 1.2, 17%). It already removes 7.0 points of the gap in 2018-19, so much of the link is selection. Achievement + SES remove 0.1 (SE 0.4).
- With word weights fixed at 2018-19, the gap does not rise: -0.2 (SE 2.1) to 2021 and +1.1 (SE 2.1) to 2023. Dropping 507 pandemic terms still leaves 11.7 of the 14 points.
- Remote-meeting words absorb 52% of the rise as a district control and beat about 95% of placebo word sets. Mute and webinar words alone absorb 38% and beat about 86%. Same-district-year contrasts imply only 7-14%. The flag mixes format with content, so this is suggestive only.
- Closures do not predict conflict words or culture-war talk once state and GOP share are held fixed. GOP share does predict a larger rise in parental-rights talk (+0.12 per 10k per SD, SE 0.03, survives BH).
- One SD more closure predicts about 1% lower enrollment by fall 2021 (SE 0.28), with a clean 2017-19 placebo. The enrollment-loss link to parental-rights talk fails its pre-trend placebo.

## Caveats
Closures are chosen by boards and sorted on politics, with no instrument here. County vote is a proxy for district partisanship. No district-level post-pandemic test scores, so the learning-loss channel is untested. The meeting-format hypothesis came from inspecting the data.
