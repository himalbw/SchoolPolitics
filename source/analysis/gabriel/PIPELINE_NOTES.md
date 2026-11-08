# GABRIEL pipeline notes

## 2026-10-27 — excerpt counts changed between runs (9,389 → 7,919): iCloud evicted the raw transcripts

**Symptom.** Re-running `build_excerpts.py` with only name-stripping changes gave ~30% fewer flagged excerpts in every
lexicon category, with the same districts and meetings.

**Hypothesis ruled out.** Flagging was *not* running on stripped text: windows are chosen on the raw text in
`ExcerptsForMeeting` before `StripNames` runs. Check on 200 random meetings (raw window vs stripped window):
2,277 → 2,260 lexicon hits (0.75% lost; 13 of the 17 lost hits were "DEI" tagged by spaCy as a person/place), far from 30%.

**Cause.** The repo sits on the Desktop, which is synced by iCloud Drive with "Optimize Mac Storage" on, and the disk
was ~96% full. macOS evicted 73,481 of the 199,183 extracted files in `Datastore/DistrictView/raw/` (flag `dataless`
in `ls -lO`; 28.8k / 28.4k / 16.3k in parts 1–3). An evicted file keeps its size but reads as **0 bytes with no error**
(`brctl` reports iCloud sync disabled, so nothing is re-downloaded). The old reader treated these as empty transcripts,
so their flagged windows disappeared; the random window was still emitted (empty). More files were evicted between runs,
which is why counts drifted. The 9,389 run was very likely affected too — neither earlier count should be used.
58 of the 200 checked meetings have an evicted raw copy.

**Fix.**
- Transcripts are now read straight from the original tars (`districtview_share_draft/*.tar`, currently not evicted)
  through a cached byte-offset index (`derived/gabriel/tar_index.parquet`). `Datastore/DistrictView/raw/` is no
  longer read by this pipeline.
- `ReadClean` fails loudly: a short read or a gzip CRC/length error raises instead of returning empty text, and every
  meeting's word count must match the eligibility scan.
- The scan cache was rebuilt from the tars (`scan_tar.parquet`; the old `scan.parquet` was deleted). Eligibility counts
  were identical (84,740 / 67,815 / 63,538), so the scan itself had run before most files were evicted.
- Name-stripping never replaces a spaCy entity that contains a lexicon term. After this, the same 200 meetings lose
  2,277 → 2,276 hits (0.04%); across all 10,879 excerpts 50,483 → 50,419 (0.13%, residual from metadata place-name
  matches).
- Every build now reports lexicon hits before vs after name-stripping (`n_hits_raw`, `n_hits_stripped` columns).

**Name-stripping limits (accepted 2026-10-27; a limitation to report).** Captions before ~2022 are all lowercase, so
spaCy misses many names. Rules now catch a title (mr/mrs/ms/dr/superintendent/trustee/president/chair/director/
principal/board member/…) plus up to two name words, "my name is …", and any phrase ending in High School / Middle
School / Elementary / Academy / ISD / School District / Unified. Still leaking: **bare names with no title or school
word** (first names, surnames, e.g. "brent moore", "pastorick"; a school name used alone, e.g. "Porter"). "Miss X" is
not stripped ("miss" is too often a verb), nor "I'm X" / "this is X" (they precede opinions as often as names).
Individual surnames carry little partisan signal; a bare school or town name could hint at location. The pilot leak
check was inconclusive (its pattern mostly matched generic phrases); the GOP-share regressions should be read with this
caveat.

**Still at risk (needs a person, not code).** The tars are on the same iCloud-managed Desktop and can be evicted too;
the pipeline would then stop with "Unreadable transcript …". To prevent it, either move `Datastore/` off the
Desktop/Documents (e.g. `~/Stat236Data`, with a symlink back) or right-click `Datastore` → "Keep Downloaded", and/or
turn off "Optimize Mac Storage". Other team scripts that read `Datastore/DistrictView/raw/`
(e.g. `build_derived_transcripts.py`) are exposed to the same silent empty reads on this machine.

## 2026-10-28 — paid runs: wrong key, credits, and a GABRIEL crash on resume

- The shell's `OPENAI_API_KEY` was not the intended key; ~$4.40 was billed to it before its credits ran out (the
  "rate-limit" errors GABRIEL reported were `429 You have no credits remaining`). Paid calls now read the key from
  `~/.openai_key_5eeu` and every paid step writes its actual cost to `session_costs_<key>.csv` (`run_gabriel_stance.py`).
- GABRIEL 1.1.9 crashes with `ValueError: task_done() called too many times` when a burst of errors (quota, DNS)
  cancels its workers. Responses are checkpointed every 1,000 calls in `<save_dir>/ratings_raw_responses.csv`, and a
  re-run with `reset_files=False` resumes from them, so wrap long runs in a retry loop; in-flight calls since the last
  checkpoint are billed but lost.
- Do not run two GABRIEL jobs on one key at once: they share the rate limit and GABRIEL throttles to 1 worker.
