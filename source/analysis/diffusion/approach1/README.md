# Topic Diffusion, Approach 1: similarity-based talking points across a sample of keywords

This is a prototype of Approach 1 in `docs/topic_diffusion_extension.tex`. It started as a trial on
"indoctrinat*" and was then extended to 9 keywords. The text-reuse steps run locally. LLM labelling uses the
OpenAI models the project already uses (as in `source/analysis/gabriel/run_gabriel_stance.py`), under a hard
$20 cap. **Actual spend: $0.50** (gpt-6-luna, plus $0.04 of gpt-6.1-terra for the validation check).

## How to run (from the repo root)
```
python3 source/analysis/diffusion/approach1/01_extract_hits.py            # ~3.5 min, all keywords in one pass
python3 source/analysis/diffusion/approach1/02_similarity.py placebo      # 'students' placebo pairs
python3 source/analysis/diffusion/approach1/02_similarity.py all          # ~1 min per capped keyword
python3 source/analysis/diffusion/approach1/03_clusters.py                # tau table, clusters, sensitivity
python3 source/analysis/diffusion/approach1/04_llm_label.py check         # PAID: 40 hand-coded hits, luna + terra
python3 source/analysis/diffusion/approach1/04_llm_label.py label         # PAID: cluster hits + 150 random hits/kw
python3 source/analysis/diffusion/approach1/04_llm_label.py pairs         # PAID: 200 borderline pairs/kw
python3 source/analysis/diffusion/approach1/05_summary.py                 # keyword_summary.csv/.tex etc.
```
- **Configuration.** Keywords, caps and windows are set in `common.py` (`KEYWORDS`). The sample comes from
  `globals.json` (`BOW_DERIVED_DIR`, rows with `status == "kept"`; 80,063 transcripts, 0 unreadable).
- **Spend control (`llm.py`).** Every call's actual usage is written to
  `Datastore/.../approach1/llm_ledger.csv`. Calls stop once the ledger reaches $18; the hard cap is $20.
  Labelling resumes where it left off and skips hits that are already labelled.
- **Where outputs go.** Per-keyword data are in `Datastore/DistrictView/derived/diffusion/approach1/keywords/{kw}/`
  (`hits`, `edges`, `clusters`, `llm_hits_gpt-6-luna`, `llm_pairs_gpt-6-luna`); this folder is ~45 MB.
  Tables are in `output/tables/diffusion/approach1/`.

## Keywords (stems)
- `indoctrinat*`.
- `mask(s)`: mask, masks, masking, masked, unmask(ed/ing), facemask(s).
- CRT: the token "crt" or the phrase "critical race theor(y/ies)".
- Parental rights: parental / parents / parent / parent's rights, and "parental right".
- `groom*`: groom, grooming, groomer(s), groomed, grooms. The token list keeps out words like "groomsmen", but
  dress-code "grooming" still matches; it was kept on purpose as a boilerplate test.
- `pornograph*`, plus "porn" and "porno".
- Equity: equity, inequity, inequities.
- Vaccine: `vaccin*`, `unvaccin*`, vax, (un)vaxxed.
- `transgender*`.

The keywords span culture-war terms that conservatives use (indoctrination, groom, pornography, parental
rights, CRT), equity language that progressives use (equity), COVID topics (mask, vaccine) and transgender
rights (contested from both sides).

**Cap.** A keyword with more than 8,000 hits is randomly sampled down to 8,000 for the pairwise step. The kept
shares are mask 2.8%, equity 4.3%, vaccine 4.6%, CRT 47% and transgender 67%. The full counts are in
`counts.csv` and in the summary table.

## Design
1. **Spans and flags (`01`).** Each hit gets its span `w[i-7:i+L+7]` (the union of all 8-grams that contain
   the keyword), ±30 and ±60 words of context, and rule-based flags. The flags are public comment
   (strict/loose), policy text and definition; the cues are scored over ±250 words.
2. **Similarity (`02`).** Only pairs from different districts are compared, with the keyword tokens masked.
   - `lex`: rapidfuzz ratio on the span.
   - `sh5f`: the number of shared word 5-grams within ±20 words, after dropping 4,425 "generic" 5-grams found
     in at least 2% of 600 reference transcripts. This handles the pledge, "i just want to say thank" and
     similar stock phrases.
   - `ex8`: the pair shares an exact 8-gram that contains the keyword.

   TF-IDF and best-half measures from the trial added nothing and were dropped.
3. **Threshold and clusters (`03`).**
   - Link rule: **lex ≥ 80 or sh5f ≥ 4** (roughly 8 identical non-generic words in a row anywhere within ±20
     words).
   - Placebo: 7,259 "students" hits matched to the year mix of the keyword hits. 12 of 26.3M placebo pairs
     pass, a rate of 4.6e-7 per pair. Expected false links = rate × the keyword's pairs, which is about 14 per
     capped keyword.
   - The same placebo is reused for every keyword. Both measures mask the keyword and drop generic phrasing,
     so the placebo measures chance verbatim overlap in board-meeting speech. Its weakness is that it does
     not capture *topic-specific* stock phrases (see Limits).
   - A talking point (TP) is a connected component that spans ≥ 2 districts in ≥ 2 counties.
4. **LLM labels (`04`, gpt-6-luna, reasoning "none", structured JSON).**
   - What is sent: each hit's ±250-word window (re-read from disk), with the keyword marked.
   - What comes back: `speaker_role` (public commenter / board member / staff-administrator / student /
     reading official text / unclear), `in_public_comment_period`, `is_quoting` + `quote_source` (statute or
     policy, dictionary, news, advocacy group, other), and stance (agree / disagree / neutral) toward a topic
     statement defined per keyword (`TOPIC`).
   - What is labelled: (a) every hit in a TP (1,962 hits) and (b) 150 random hits per keyword.
   - Cluster type is assigned in this order, each needing ≥ 50% of the cluster's hits:
     1. definition/quotation: quoting a dictionary, news, an advocacy group or another source;
     2. official/boilerplate: reading official text, a board member or staff speaking, or quoting a statute;
     3. public comment: a public commenter speaking;
     4. otherwise mixed.
   - Pairs: 200 borderline pairs per keyword (not linked, but sh5f 2–3 or lex 65–79) are judged "same
     talking point?". The same judge is run on 200 borderline placebo pairs as a false-positive baseline.
5. **Validation.** On the 38 non-ambiguous hand-coded indoctrination hits, a public commenter vs not label
   agrees with my hand reading 92% of the time for luna and 89% for terra. Luna was therefore used throughout.

## Results (`keyword_summary.csv` / `.tex`)
| keyword | hits (all) | districts | PC share (LLM) | links (exp. false) | TPs | PC TPs | official share | largest PC TP |
|---|---|---|---|---|---|---|---|---|
| indoctrinat* | 4,431 | 575 | 0.85 | 503 (4) | 33 | 19 | 0.21 | 2 dist / 2 st |
| mask(s) | 286,327* | 1,355 | 0.45 | 95 (15) | 47 | 12 | 0.51 | 13 / 11 |
| CRT | 17,058* | 763 | 0.81 | 667 (14) | 134 | 50 | 0.33 | 33 / 20 |
| parental rights | 5,413 | 616 | 0.63 | 368 (7) | 72 | 31 | 0.42 | 11 / 9 |
| groom* | 3,872 | 724 | 0.48 | 173 (3) | 31 | 3 | 0.61 | 3 / 2 |
| pornograph* | 6,302 | 584 | 0.72 | 389 (9) | 44 | 8 | 0.43 | 4 / 3 |
| equity | 187,604* | 1,295 | 0.22 | 20 (15) | 15 | 1 | 0.60 | 2 / 2 |
| vaccin* | 174,960* | 1,266 | 0.39 | 51 (14) | 33 | 7 | 0.58 | 4 / 4 |
| transgender* | 11,868* | 708 | 0.64 | 1,197 (14) | 113 | 31 | 0.33 | 9 / 5 |

\* sampled to 8,000 hits for the pairwise step.

**Rule flag vs LLM** (`rule_vs_llm.csv`, 150 random hits per keyword).
- The strict rule flag is precise: 0.82–1.00 precision against the LLM's public-commenter label.
- It misses most public comment: recall is 0.19–0.35.
- The loose flag has precision 0.76–0.93 and recall 0.57–0.79.

**Most convincing public-comment talking points** (`top_public_comment_talking_points.csv`):
- **Parental rights.** "educating and empowering parents to defend their parental rights at all levels of
  government": 11 districts, 9 states, 2021-05 to 2023-07. This is an advocacy group's mission statement
  recited by commenters.
- **CRT.** "[we] have been told ... that critical race theory is not being taught in our schools": 33
  districts, 20 states, 2021-05 to 2023-07. It also includes a state school-boards-association statement.
  Another cluster is "critical race theory is an academic concept / discipline that ...": 18 districts and 9
  districts.
- **Transgender.** "the rights of transgender and gender non conforming students": 9 districts, 5 states.
  Also bullying and suicide statistics; the Trevor-Project-style statistic "more than half of transgender and
  non-binary youth considered attempting suicide" spans 68 districts in 22 states but is typed as a
  quotation.
- **Mask.** "i don't want to wear a mask": 13 districts, 11 states. This is probably a natural common phrase,
  not a talking point.

**Official text and boilerplate dominate many clusters:**
- Florida statute on pornography in instructional material (16 districts).
- The federal definition of child pornography.
- State parental-rights statutes (Arizona's statute A.R.S. 1-602, the Texas Education Code).
- Dress-code "grooming" rules and child-abuse prevention "grooming behaviors" policy.
- Board-policy text on indoctrination.
- Virginia's model policies on transgender students.

**Paraphrase missed by the lexical rule** (`pair_adjudication.csv`). On the placebo, the LLM judges 3.5% of
borderline pairs to be the same talking point. On keyword borderline pairs it judges:
- 12% for equity;
- 18–33% for mask, vaccine, CRT, transgender and groom;
- 50–78% for parental rights, pornography and indoctrination.

Most of these are not already in the same cluster. So the lexical rule is conservative: for the
culture-war keywords, the borderline zone holds roughly as many paraphrased talking-point links again. This
judge has not been validated against hand reading.

## Limits / verdict
- **Text reuse works for rare, contested terms** that the public brings to meetings (indoctrination,
  parental rights, CRT, pornography, transgender). Links there far exceed the placebo (for example, 503 links
  vs 4 expected), and the shared passages are identifiable: advocacy mission statements, definitions,
  statistics, statutes and slogans.
- **It does not work for high-frequency administrative terms** (equity, vaccine, mask). Links barely exceed
  the placebo (equity: 20 vs 15 expected), and the clusters are mostly official guidance. Sampling at 3–5%
  also makes a talking point shared by k districts nearly undetectable, because both hits of a pair must be
  sampled. Full-population matching would need a 5-gram index over all hits and a much stricter τ, since
  expected false links grow with the number of pairs.
- **Topic-specific stock phrases** ("i don't want to wear a mask", "critical race theory is not being taught")
  pass both the placebo test and the generic-phrase filter. Telling an organised talking point from a natural
  phrase needs a source match (advocacy materials, news) or timing evidence (Approach 2).
- Captions have no speaker labels. Public-comment status comes from an LLM label of the ±250-word window, not
  from segmentation.
- Names after introduction cues are scrubbed from printed examples, and the printed tables were checked
  against a rare-word list. One private name found that way is redacted via an md5 list in `common.py`.
  Officials' names can remain in `ctx60`.

The first indoctrination-only tables are archived in `output/tables/diffusion/approach1/keywords/indoctrination/trial_v1/`.
The rerun reproduces them: 503 links and 33 talking points.
