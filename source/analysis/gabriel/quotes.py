"""Illustrative districts and verbatim quotes (step 5 of the 2026-10-28 extension).

Run from the repo root:
  micromamba run -n stat236 python source/analysis/gabriel/quotes.py candidates            # no API calls
  micromamba run -n stat236 python source/analysis/gabriel/quotes.py quotes LEAID [LEAID]  # gpt-6.1-terra (costs money)
candidates: from gabriel_change_district.csv, the 3 R-county districts with the largest pre -> post rise in
conservative_positions and the 3 D-county districts with the largest rise in progressive_positions (>= MIN_MEETINGS
sampled meetings each).
quotes: per district, the TOP_POSITION highest-scoring excerpts on its position attribute and the TOP_HOSTILITY highest on
hostility. Terra gets the RAW (not name-stripped) window and returns the single sentence that most strongly argues
the position (or is most hostile). A quote is kept only if it is an exact substring of the raw window text (timestamps
and [tags] removed, as in build_excerpts.CleanRaw); names of private individuals are then replaced with [NAME].
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_excerpts as be  # noqa: E402
from run_gabriel_stance import OUTDIR, SessionCost, TERRA_MODEL, TERRA_PRICE_IN, TERRA_PRICE_OUT  # noqa: E402

INDIR_TABLES = Path("output/tables/")
MIN_MEETINGS = 10
N_CANDIDATES = 3
TOP_POSITION, TOP_HOSTILITY = 25, 10
EXCERPT_FILES = ["excerpts.parquet", "excerpts_extension.parquet", "excerpts_extra_random.parquet"]
TARGETS = {
    "conservative_positions": "argues most strongly for the conservative side of a contested political or social "
                              "dispute (e.g. removing books, opposing equity/DEI or LGBTQ-inclusive policies, parents' "
                              "authority over curriculum, opposing CRT, opposing masks or vaccine requirements)",
    "progressive_positions": "argues most strongly for the progressive side of a contested political or social dispute "
                             "(e.g. defending books against removal, defending equity/DEI or LGBTQ-inclusive policies, "
                             "defending inclusive race or history curriculum, supporting masks or vaccine requirements)",
    "hostility": "is most hostile: a personal attack or insult, an accusation of lying, corruption or bad faith, a "
                 "threat, or a call for officials to resign or be removed",
}
PROMPT = """Below is an excerpt of an automatic YouTube caption transcript of a US school board meeting.
Find the single sentence that {target}.
Copy it VERBATIM, character for character, exactly as it appears in the excerpt (same spelling, casing, punctuation and
spacing; do not fix caption errors). If the captions have no punctuation, return one contiguous span of 10-60 words.
If no sentence fits, return an empty quote.
Also list every name of a private individual (a parent, student, resident or other member of the public; not an
elected official, superintendent or district staff member speaking in an official role) that appears in your quote.
Return JSON only: {{"quote": "...", "private_names": ["..."]}}

Excerpt:
{text}"""


def Main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "candidates":
        Candidates()
    elif mode == "quotes" and len(sys.argv) > 2:
        Quotes(sys.argv[2:])
    else:
        sys.exit(__doc__)
    return


def Candidates():
    d = pd.read_csv(INDIR_TABLES / "gabriel_change_district.csv", dtype={"leaid": str})
    d = d[d.n_meetings >= MIN_MEETINGS]
    cols = ["leaid", "place_name", "lea_name", "state_location", "gop_2p_2020", "n_meetings"]
    for party, attr in [("R", "conservative_positions"), ("D", "progressive_positions")]:
        top = d[d.party == party].assign(change=d[f"{attr}_post"] - d[f"{attr}_pre"]).nlargest(N_CANDIDATES, "change")
        print(f"\n{party}-county districts, largest rise in {attr} (2018-19 -> 2021-24, random excerpts):")
        print(top[cols + [f"{attr}_pre", f"{attr}_post", "change"]].round(2).to_string(index=False))
    return


def RawWindows(rows):
    # Raw (not name-stripped) text of each excerpt's window, re-read from the tars.
    meta = pd.read_parquet(be.INDIR_META / "meta_districtview.parquet", columns=["id", "channel_id"])
    meta["name"] = meta.channel_id + "__" + meta.id + ".txt.gz"
    loc = meta.merge(be.TarIndex(), on="name").set_index("id")
    out = []
    for r in rows.itertuples():
        words = be.ReadClean((loc.at[r.video_id, "tar"], int(loc.at[r.video_id, "offset"]),
                              int(loc.at[r.video_id, "size"]))).split()
        out.append(" ".join(words[r.window_start:r.window_start + be.WINDOW]))
    return out


def Redact(quote, names):
    # Model-listed private names, then "my name is X" and spaCy PERSON entities (may also catch officials' names).
    for n in sorted({n for n in names if n and len(n) > 1}, key=len, reverse=True):
        quote = re.sub(re.escape(n), "[NAME]", quote, flags=re.I)
    quote = re.sub(r"\b(my name is|my name's)(\s+)[A-Za-z'-]+(?:\s+[A-Z][a-z'-]+)?", r"\1\2[NAME]", quote, flags=re.I)
    doc = NLP(quote)
    for ent in reversed([e for e in doc.ents if e.label_ == "PERSON" and "NAME" not in e.text]):
        quote = quote[:ent.start_char] + "[NAME]" + quote[ent.end_char:]
    return quote


def Quotes(leaids):
    global NLP
    import spacy
    from openai import OpenAI
    NLP = spacy.load("en_core_web_sm")
    SessionCost()
    client = OpenAI()
    scores = pd.read_parquet(OUTDIR / "scores_full.parquet")
    pc = pd.read_parquet(OUTDIR / "public_comment_share.parquet", columns=["excerpt_id", "public_comment_share"])
    excerpts = pd.concat([pd.read_parquet(OUTDIR / f, columns=["excerpt_id", "leaid", "video_id", "meeting_date",
                                                               "window_start"]) for f in EXCERPT_FILES
                          if (OUTDIR / f).exists()])
    df = scores.merge(excerpts, on="excerpt_id", validate="one_to_one").merge(pc, on="excerpt_id", how="left")
    change = pd.read_csv(INDIR_TABLES / "gabriel_change_district.csv", dtype={"leaid": str}).set_index("leaid")
    tok_in = tok_out = 0
    log = []
    for leaid in leaids:
        attr = "conservative_positions" if change.at[leaid, "party"] == "R" else "progressive_positions"
        d = df[df.leaid == leaid]
        jobs = pd.concat([d.nlargest(TOP_POSITION, attr).assign(attribute=attr),
                          d.nlargest(TOP_HOSTILITY, "hostility").assign(attribute="hostility")], ignore_index=True)
        jobs["score"] = [r[r.attribute] for _, r in jobs.iterrows()]
        jobs["raw"] = RawWindows(jobs)
        rows = []
        for r in jobs.itertuples():
            resp = client.responses.create(model=TERRA_MODEL, reasoning={"effort": "none"},
                                           input=PROMPT.format(target=TARGETS[r.attribute], text=r.raw),
                                           text={"format": {"type": "json_object"}})
            tok_in += resp.usage.input_tokens
            tok_out += resp.usage.output_tokens
            try:
                out = json.loads(resp.output_text)
            except json.JSONDecodeError:
                out = {"quote": "", "private_names": []}
            quote = (out.get("quote") or "").strip()
            verified = bool(quote) and quote in r.raw
            log.append({"leaid": leaid, "excerpt_id": r.excerpt_id, "attribute": r.attribute, "quote_raw": quote,
                        "verified": verified})
            if verified:
                rows.append({"video_id": r.video_id, "meeting_date": r.meeting_date.date(), "attribute": r.attribute,
                             "score": r.score, "public_comment_share": r.public_comment_share,
                             "quote": Redact(quote, out.get("private_names") or [])})
        out = pd.DataFrame(rows).drop_duplicates(["video_id", "quote"])
        out.to_csv(INDIR_TABLES / f"quotes_{leaid}.csv", index=False)
        n = sum(l["leaid"] == leaid for l in log)
        print(f"{leaid}: {n} excerpts -> {sum(l['verified'] for l in log if l['leaid'] == leaid)} verified quotes "
              f"({len(out)} after de-duplication) -> {INDIR_TABLES / f'quotes_{leaid}.csv'}")
    pd.DataFrame(log).to_csv(OUTDIR / "quotes_log.csv", index=False)
    # Record cost in the session ledger (SessionCost reads a ratings_raw_responses.csv with token columns).
    usage_dir = OUTDIR / "quotes_terra"
    usage_dir.mkdir(exist_ok=True)
    pd.DataFrame([{"Input Tokens": tok_in, "Output Tokens": tok_out}]).to_csv(usage_dir / "ratings_raw_responses.csv",
                                                                             index=False)
    SessionCost("quotes_terra", usage_dir, TERRA_PRICE_IN, TERRA_PRICE_OUT)
    return


if __name__ == "__main__":
    Main()
