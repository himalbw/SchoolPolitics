"""Shared paths and helpers for Topic Diffusion Approach 1 (similarity-based talking points)."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
GLOBALS = json.loads((ROOT / "globals.json").read_text())
DERIVED = ROOT / GLOBALS["BOW_DERIVED_DIR"]          # screened derived corpus
TRANSCRIPTS = DERIVED / "transcripts"
MANIFEST = DERIVED / "manifest.parquet"
META = ROOT / "Datastore/DistrictView/districtview_share_draft/meta_districtview.parquet"
STATIC = ROOT / "Datastore/District Covariates/processed/district_static.parquet"
DATA_OUT = ROOT / "Datastore/DistrictView/derived/diffusion/approach1"
TAB_OUT = ROOT / "output/tables/diffusion/approach1"

DATA_KW = lambda kw: DATA_OUT / "keywords" / kw   # noqa: E731  per-keyword data folder
TAB_KW = lambda kw: TAB_OUT / "keywords" / kw     # noqa: E731  per-keyword table folder

# Keyword definitions. A hit is anchored at the first token of the match; `tokens` = exact tokens, `prefixes` =
# token prefixes, `phrases` = multi-token sequences (all matched tokens are masked in the similarity step).
# `cap` = max hits kept for the pairwise step (random sample of all hits; None = keep all).
KEYWORDS = {
    "indoctrination": dict(prefixes=["indoctrinat"], cap=None),
    "mask": dict(tokens=["mask", "masks", "masking", "masked", "unmask", "unmasked", "unmasking", "facemask",
                         "facemasks"], cap=8000),
    "crt": dict(tokens=["crt"], phrases=[["critical", "race", "theory"], ["critical", "race", "theories"]],
                cap=8000),
    "parental_rights": dict(phrases=[["parental", "rights"], ["parents", "rights"], ["parent", "rights"],
                                     ["parents'", "rights"], ["parent's", "rights"], ["parental", "right"]],
                            cap=8000),
    "groom": dict(tokens=["groom", "grooming", "groomer", "groomers", "groomed", "grooms"], cap=8000),
    "pornography": dict(prefixes=["pornograph"], tokens=["porn", "porno"], cap=8000),
    "equity": dict(tokens=["equity", "inequity", "inequities"], cap=8000),
    "vaccine": dict(prefixes=["vaccin", "unvaccin"], tokens=["vax", "vaxxed", "unvaxxed"], cap=8000),
    "transgender": dict(prefixes=["transgender"], cap=8000),
}
CONTROL = "students"                  # placebo control word (exact token), shared across keywords
CONTROL_KEEP_P = 0.003                # sampling prob per control hit at extraction
SPAN_HALF = 7                         # span = w[i-7 : i+L+7] = union of all 8-grams containing the keyword
CTX_HALF = 30                         # +-30 words stored (5-gram window uses +-20 of it)
READ_HALF = 60                        # +-60 words stored for reading
WIDE_HALF = 250                       # +-250 words: cue scoring at extraction; re-read from disk for the LLM step
MASK = "kwx"                          # placeholder replacing the keyword / control tokens
SEED = 236


def matcher(cfg):
    """Return f(words, i) -> match length (0 if no hit at i)."""
    toks = set(cfg.get("tokens", []))
    pref = tuple(cfg.get("prefixes", []))
    phrases = [tuple(p) for p in cfg.get("phrases", [])]
    first = {p[0] for p in phrases}

    def f(w, i):
        t = w[i]
        if t in toks or (pref and t.startswith(pref)):
            return 1
        if t in first:
            for p in phrases:
                if tuple(w[i:i + len(p)]) == p:
                    return len(p)
        return 0
    return f


# ---- rule-based context flags (stand-in for LLM segmentation) ------------------------------------------------
PC_CUES = {  # public-comment cues
    "public comment": r"public comments?|public participation|citizen comments?|audience participation|comments from the public",
    "state your name": r"state your (?:name|address)|name and (?:address|city|town)",
    "my name is": r"my name is",
    "i live in": r"i live in|i'm a resident|i am a resident|resident of|i reside",
    "three minutes": r"(?:three|3|two|2|five|5) minutes",
    "time is up": r"(?:your )?time is up|that's time|out of time|times up|time's up",
    "as a parent": r"as a (?:parent|mother|father|mom|dad|grandparent|taxpayer)|i'm a (?:parent|mom|dad|mother|father)|i am a (?:parent|mom|dad|mother|father)",
    "my child": r"my (?:child|children|kids|son|daughter|grandchild|grandchildren|grandkids)",
    "thank you for your time": r"thank you for your time|thank you for listening|thanks for your time",
    "next speaker": r"next speaker|our next (?:speaker|commenter)|next (?:person|name) (?:is|on)|speaker card|signed up to speak",
}
POL_CUES = {  # official / policy text cues
    "policy": r"\bpolic(?:y|ies)\b",
    "section": r"\bsection\b|\bsubsection\b",
    "shall": r"\bshall\b",
    "board policy": r"board policy|administrative regulation",
    "state law": r"state law|statute|education code|ed code|house bill|senate bill|\bhb\b|\bsb\b",
    "pursuant": r"pursuant|in accordance with|hereby",
    "the district shall": r"the district shall|the board shall|teachers shall|employees shall",
    "the board will": r"the board will|the board may|superintendent or (?:his or her )?designee|designee",
    "provided": r"\bprovided (?:that|they|it)\b|\bprovided\b",
    "instructional program": r"instructional (?:program|materials|goals)|course of study|controversial issues?",
    "enumeration": r"(?:^| )[abc] (?:does|is|shall|encourages|the|to)\b",
}
DEF_RE = re.compile(r"definition|defin(?:e|es|ed) (?:as|it)|dictionary|webster|merriam|oxford|the process of teaching|is the process of")
FIRST_PERSON = re.compile(r"\b(?:i|i'm|my|me|i've|i'd|mine)\b")
PC_RE = {k: re.compile(r"\b(?:" + v + r")\b") for k, v in PC_CUES.items()}
POL_RE = {k: re.compile(v) for k, v in POL_CUES.items()}


def flags(w, i):
    """Cue scores for the hit at word i of transcript words w (weights: <=100 words 1, 100-250 words 0.5)."""
    lo = max(0, i - WIDE_HALF)
    wide = " ".join(w[lo: i + WIDE_HALF + 1])
    c = len(" ".join(w[lo:i]))
    pc = 0.0
    for rx in PC_RE.values():
        best = 0.0
        for m in rx.finditer(wide):
            best = max(best, 1.0 if abs(m.start() - c) / 6.0 <= 100 else 0.5)
            if best == 1.0:
                break
        pc += best
    ctx = " ".join(w[max(0, i - CTX_HALF): i + CTX_HALF + 1])
    pol = sum(bool(rx.search(ctx)) for rx in POL_RE.values())
    n_fp = len(FIRST_PERSON.findall(ctx))
    is_pol = pol >= 2 or (pol >= 1 and n_fp == 0)
    return dict(pc_score=pc, pol_score=pol, flag_policy=is_pol, flag_definition=bool(DEF_RE.search(ctx)),
                flag_public_comment=(pc >= 1.5) and not is_pol, flag_public_comment_loose=(pc >= 0.5) and not is_pol)


# md5 prefixes of private individuals' name tokens found by hand in printed examples (names kept out of the code)
REDACT_MD5 = {'dbbc546e3570', '5b9f593b8fc9'}


def _redact(text):
    import hashlib
    return " ".join("[name]" if hashlib.md5(t.encode()).hexdigest()[:12] in REDACT_MD5 else t for t in text.split())


def scrub_names(text: str) -> str:
    """Best-effort removal of private individuals' names from printed examples: replace the 1-2 tokens after an
    introduction cue ('my name is', 'next speaker is', 'this is from' ...) when they are not common words.
    Not exhaustive: printed examples are also checked by hand (see README)."""
    common = {"the", "a", "an", "to", "for", "and", "so", "you", "your", "all", "everyone", "very", "much", "our",
              "here", "board", "members", "i", "is", "my", "not", "going", "that", "this", "it", "we", "a",
              "parent", "mom", "dad", "teacher", "resident", "student", "senior", "member", "concerned",
              "again", "about", "really", "just", "glad", "sorry", "speaking", "in", "of", "on", "at", "with"}

    def rep(m):
        toks = m.group(2).split()
        keep = []
        for k, t in enumerate(toks):
            if t in common:
                keep.extend(toks[k:])
                break
            keep_name = "[name]" if not keep or keep[-1] != "[name]" else None
            if keep_name:
                keep.append(keep_name)
        return m.group(1) + " " + " ".join(keep)

    text = re.sub(r"\b(my name is|name is|speaker is|this is from|next we have|next up is|next is)((?:\s+\w+){1,2})",
                  rep, text)
    return _redact(text)
