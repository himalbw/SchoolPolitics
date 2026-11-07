# GABRIEL attribute log

Every attribute name and wording ever tried, in order. Wording lives in `attributes.py` (`VERSION` must match the
latest entry here). Develop on split A only; report on split B.

## v1 — 2026-10-27 (draft, not yet run)
Initial draft from the lane brief. Five attributes, one `rate()` call, each 0–100:
`hostility`, `culture_war_share`, `parental_rights_framing`, `covid_contention`, `stance` (bipolar: 0 = progressive,
100 = conservative, 50 = no position). Exact wording: `attributes.py` at v1.

Open questions before the pilot:
- `stance` is bipolar, while GABRIEL's `rate` template asks how strongly an attribute is *present*. Check in the pilot
  that the model honours "50 = no position" (and doesn't score routine business low). Fallback (would be v2): two unipolar
  attributes, `progressive_positions` and `conservative_positions`, then use their difference.
- `culture_war_share` includes masks/vaccines, so it overlaps `covid_contention` by design in 2020–21; report both.
- Team sanity check on the `stance` wording still pending.

## v2 — 2026-10-27 (draft, not yet run; v1 was never run)
Replaced the bipolar `stance` with two one-sided attributes, `progressive_positions` and `conservative_positions`
(0 = no such positions argued, 100 = strongly progressive / conservative positions argued throughout; rate positions
argued, not topics discussed). Reason: GABRIEL's `rate` asks how strongly an attribute is *present*, so a bipolar
scale with "50 = no position" risks routine business being scored low (i.e. read as "progressive"). Stance is computed
later as `conservative_positions - progressive_positions`; meetings with both sides argued now show up as high on both
rather than averaging to 50. The other four attributes are unchanged from v1.
Pilot (2026-10-27, gpt-6-luna, reasoning none, 200 split-A excerpts x 2 runs): run-to-run r = 0.957–0.990.

## v3 — 2026-10-27 (FINAL; wording frozen from here on)
Approved with two added examples: `conservative_positions` gains "asserting parents' authority over curriculum,
opposing CRT or 'woke' content"; `progressive_positions` gains "defending inclusive race or history curriculum against
those criticisms". Exact wording: `attributes.py` at v3.

Draft notes:
`progressive_positions` / `conservative_positions` now require arguing one side of a *contested dispute*; routine
descriptions of district programs/plans/services count as 0. Reason: in the v2 pilot, random excerpts scored
progressive far more than conservative (mean 4.4 vs 0.3) partly because staff presenting e.g. an equity plan was scored
as a progressive position (h2uNRx7jglg_0: progressive 57), mirroring the BoW finding that professional/equity jargon is
D-coded. The other four attributes are unchanged from v2. Wording frozen once approved (no changes after v3).

## public_comment_share — 2026-10-28 (separate one-attribute pass; v3 attributes unchanged)
Not part of the v3 rate() call and not an outcome: a classifier used to split excerpts by who is speaking. Wording
(`run_gabriel_stance.py`, `PUBLIC_COMMENT`): share of the excerpt that is public comment by members of the public speaking to
the board, as opposed to board member or staff discussion; cues "state your name and address", "three minutes",
"next speaker", "public comment", a speaker introducing themselves as a parent or resident; 0-100. Run once on all
22,174 original + extension excerpts (gpt-6-luna, reasoning none); no pilot, no revisions. Cutoff 50 fixed in advance.
