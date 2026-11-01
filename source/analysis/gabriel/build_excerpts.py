"""Build name-stripped ~1,500-word excerpts of RAW transcripts for the GABRIEL ratings (no API calls, costs nothing).

Run from the repo root:  micromamba run -n stat236 python source/analysis/gabriel/build_excerpts.py
Steps: (1) title screen ported from Stefan's eligibility audit (source/analysis/eligibility_hc/workflow.py, rules
unchanged, but channel stats / transcript prefixes computed from raw files since his LDA outputs are not needed);
(2) scan candidate transcripts (cached in scan_tar.parquet); (3) keep districts with >= MIN_MEETINGS eligible meetings in
every pre (SAMPLE_START-2019) and post (2021-SAMPLE_END) year; (4) stratify by gop_2p_2020 tercile x locale and sample
N_DISTRICTS; (5) MEETINGS_PER_YEAR random meetings per district-year; (6) one random window + up to 2 lexicon-flagged
windows per meeting, flagged on the raw text; (7) strip names from the chosen windows; (8) count tokens and estimate cost.
Transcripts are read straight from the original tars, not Datastore/DistrictView/raw/: see PIPELINE_NOTES.md (iCloud).
`--extend` (2026-10-28): same districts and splits; adds 2020 and 2024 (EXT_PER_YEAR meetings per district-year) and tops
2021 up to EXT_PER_YEAR keeping the original 3, all balanced EXT_PER_HALF per half-year -> excerpts_extension.parquet.
`--extra-random` adds EXTRA_RANDOM more random windows per sampled meeting (all years) -> excerpts_extra_random.parquet.
"""
import gzip
import hashlib
import json
import os
import re
import sys
import tarfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bow"))
from lexicon import LEXICON  # noqa: E402
from attributes import ATTRIBUTES  # noqa: E402

GLOBALS = json.loads(Path("globals.json").read_text())
SAMPLE_START, SAMPLE_END = GLOBALS["SAMPLE_START"], GLOBALS["SAMPLE_END"]
PRE_YEARS = list(range(SAMPLE_START, 2020))
POST_YEARS = list(range(2021, SAMPLE_END + 1))

INDIR_META = Path("Datastore/DistrictView/districtview_share_draft/")
INDIR_COV = Path("Datastore/District Covariates/processed/")
OUTDIR = Path("Datastore/DistrictView/derived/gabriel/")

SEED = 236
MIN_WORDS = 3000
MIN_MEETINGS = 2          # eligible meetings required in every year of both periods
N_DISTRICTS = 400         # all qualifying districts are taken if fewer
MEETINGS_PER_YEAR = 3     # per district-year where available, else all
WINDOW = 1500             # words per excerpt
MAX_FLAGGED = 2
MIN_FLAG_HITS = 2         # a flagged window needs >= 2 lexicon hits, so one stray "mask" or "gender" doesn't count
# Rarest first: flagged windows prefer these categories in this order (common ones would otherwise crowd them out).
FLAG_CATEGORIES = ["crt_race", "parental_rights", "book_content", "gender_lgbtq", "covid_mandates", "equity_sel"]
PRICE_IN, PRICE_OUT = 0.10, 0.50   # $ per 1M tokens, GPT-6 Luna (check current pricing)
OUT_TOKENS_PER_CALL = 300
PROMPT_TEMPLATE_TOKENS = 250       # rough size of GABRIEL's rate() instructions around our attribute text
LOCALES = {1: "city", 2: "suburb", 3: "town", 4: "rural"}
STATES = ["alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware", "florida",
          "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
          "maryland", "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "montana", "nebraska",
          "nevada", "new hampshire", "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio",
          "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota", "tennessee", "texas",
          "utah", "vermont", "virginia", "washington", "west virginia", "wisconsin", "wyoming"]
GENERIC_NAME_WORDS = {"school", "schools", "district", "districts", "public", "unified", "independent", "community",
                      "consolidated", "central", "city", "county", "parish", "township", "borough", "regional", "union",
                      "free", "high", "elementary", "joint", "area", "local", "exempted", "village", "town",
                      "of", "the", "and", "no", "sd", "isd", "usd", "csd", "cusd", "ufsd", "comm", "co", "twp", "dist",
                      "sch", "pub", "schs", "unif", "ind", "cons", "reg", "boe", "board", "education", "north", "south",
                      "east", "west", "new", "saint", "st", "mount", "mt", "lake", "valley", "park", "river", "point"}
TITLES = ["mr", "mrs", "ms", "dr",   # not "miss": too often a verb ("kids miss school")
          "superintendent", "trustee", "president", "vice president", "chair",
          "chairman", "chairwoman", "chairperson", "director", "principal", "commissioner", "board member", "member",
          "councilman", "councilwoman", "coach", "officer", "mayor", "senator", "representative", "pastor", "reverend",
          "professor"]
NOT_NAME_WORDS = {"said", "says", "mentioned", "asked", "talked", "thank", "thanks", "agreed", "noted", "stated",
                  "presented", "explained", "shared", "moved", "seconded", "motion", "chair", "president",
                  "superintendent", "board", "member", "members", "trustee", "trustees", "yes", "no", "okay", "ok",
                  "yeah", "um", "uh", "report", "comments", "comment", "office", "question", "questions", "aye", "nay",
                  "present", "absent", "here", "pro", "tem", "elect", "emeritus", "brought", "made", "gave", "spoke"}
SCHOOL_GENERIC_WORDS = ["new", "old", "current", "local", "public", "private", "junior", "senior", "district", "county",
                        "area", "alternative", "charter", "magnet", "online", "virtual", "summer", "community",
                        "middle", "high", "elementary", "school", "schools", "other", "every", "each", "both"]
EXT_YEARS = [2020, 2021, 2024]
EXT_PER_YEAR, EXT_PER_HALF = 6, 3   # 3 per half-year where available; a short half is filled from the other half
EXTRA_RANDOM = 2
ED_WORDS = ["school", "schools", "student", "students", "teacher", "teachers", "education", "superintendent", "curriculum"]


def Main():
    if "--restrip" in sys.argv:   # redo name-stripping only, keeping the sample and windows fixed
        return Restrip()
    if "--extend" in sys.argv:
        return Extend()
    if "--extra-random" in sys.argv:
        return ExtraRandom()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    meta = LoadMeta()
    meta = TitleScreen(meta)
    scan = ScanTranscripts(meta[meta.title_candidate])
    meta = ApplyEligibility(meta, scan)
    districts = SelectDistricts(meta)
    meetings = SampleMeetings(meta, districts)
    excerpts = BuildExcerpts(meetings)
    excerpts = StripNames(excerpts, districts)
    Summarise(excerpts, districts)
    return


def Period(year):
    return np.where(year < 2020, "pre", np.where(year == 2020, "covid", "post"))


def LoadMeta(years=PRE_YEARS + POST_YEARS):
    meta = pd.read_parquet(INDIR_META / "meta_districtview.parquet",
                           columns=["id", "title", "channel_id", "leaid", "meeting_date", "published_at", "place_names"])
    meta["leaid"] = meta.leaid.astype(str).str.zfill(7)
    meta["meeting_date"] = meta.meeting_date.fillna(meta.published_at.dt.tz_localize(None).dt.normalize())  # codebook fallback
    meta["year"] = meta.meeting_date.dt.year
    meta = meta[meta.year.isin(years)].copy()
    meta["period"] = Period(meta.year)
    meta["name"] = meta.channel_id + "__" + meta.id + ".txt.gz"
    meta = meta.merge(TarIndex(), on="name", how="left", validate="many_to_one")
    print(f"{len(meta):,} videos in {years}; {meta.tar.isna().sum():,} not in the tars")
    return meta[meta.tar.notna()]


def TarIndex():
    # Byte offset of every member in the three tars (cached), so any transcript can be read with one seek.
    cache = OUTDIR / "tar_index.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    rows = []
    for tar_path in sorted(INDIR_META.glob("transcripts_districtview_part*.tar")):
        print(f"Indexing {tar_path.name}...", flush=True)
        with tarfile.open(tar_path, "r:") as tar:
            rows += [(Path(m.name).name, str(tar_path), m.offset_data, m.size) for m in tar if m.isfile()]
    index = pd.DataFrame(rows, columns=["name", "tar", "offset", "size"]).drop_duplicates("name")
    index.to_parquet(cache, index=False)
    return index


def TitleScreen(meta):
    # Title and channel rules copied from Stefan's audit(); status priority is the same np.select order.
    title = meta.title.str.lower().fillna("")
    school = r"school|education|\bboe\b|\bboard of ed\b|\b[a-z]*isd\b|\b[a-z]+usd\b|\b[a-z]+csd\b|\brsu\s*\d|\bmsad\s*\d|\bsau\s*\d"
    body = r"board|committee|\bboe\b|council|governance"
    other = (r"zoning board|zoning advisory|zoning (?:and|&) planning|planning board|planning commission|city council|"
             r"town council|borough council|selectmen|select board|board of (?:finance|health|elections|assessors|"
             r"supervisors|commissioners|adjustments|parks)|county commission|conservation commission|water board|"
             r"sewer board|library board|housing authority|land use committee|charter review committee")
    event = r"graduation|commencement|concert|football game|basketball game|soccer game|promotional"
    meta["explicit_school_title"] = title.str.contains(school) & title.str.contains(body)
    meta["other_body_title"] = title.str.contains(other)
    meta["event_title"] = title.str.contains(event) & ~title.str.contains(body)
    meta["generic_governance_title"] = title.str.contains(body)
    meta["clip_or_campaign_title"] = (title.str.contains(r"candidate.{0,30}(?:debate|forum)|(?:debate|forum).{0,30}candidate")
                                      | (title.str.contains(r"report|update|presentation")
                                         & ~title.str.contains(r"meeting|session|hearing|committee")))
    channel = meta.groupby("channel_id").agg(videos=("id", "size"), explicit_channel=("explicit_school_title", "sum"))
    channel["explicit_fraction"] = channel.explicit_channel / channel.videos
    meta = meta.join(channel[["explicit_channel", "explicit_fraction"]], on="channel_id")
    blocked = meta.clip_or_campaign_title | meta.other_body_title | meta.event_title
    meta["title_candidate"] = ~blocked & (meta.explicit_school_title | meta.generic_governance_title)
    return meta


def CleanRaw(text):
    # Raw text kept as spoken: drop only timestamps, [bracket tags] and >> turn markers; keep casing and punctuation.
    text = re.sub(r"\[\d+:\d{2}:\d{2}\]", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = text.replace(">>", " ")
    return " ".join(text.split())


def ReadClean(loc):
    # Fail loudly: an iCloud-evicted file reads as zero bytes without any error (see PIPELINE_NOTES.md).
    # gzip's CRC/length check also rejects any partially read member.
    tar, offset, size = loc
    with open(tar, "rb") as f:
        f.seek(offset)
        data = f.read(size)
    try:
        if len(data) != size:
            raise EOFError(f"read {len(data)} of {size} bytes")
        text = gzip.decompress(data).decode("utf-8", errors="replace")
    except (EOFError, OSError) as e:
        raise RuntimeError(f"Unreadable transcript at {tar}:{offset} ({e}); is the tar evicted by iCloud?") from e
    return CleanRaw(text)


def Loc(row):
    return row["tar"], int(row["offset"]), int(row["size"])


def ScanOne(args):
    vid, loc = args
    words = ReadClean(loc).split()
    prefix = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in words[:600]]
    counts = [prefix.count(w) for w in ED_WORDS]
    direct = bool(re.search(r"school board|board of education|school committee|school district.{0,50}board|"
                            r"public schools.{0,40}board|board.{0,50}school district", " ".join(prefix)[:3500]))
    return vid, len(words), sum(counts), sum(c > 0 for c in counts), direct


def ScanTranscripts(meta):
    cache = OUTDIR / "scan_tar.parquet"   # scan.parquet (raw-folder reads) is superseded: some reads hit evicted files
    done = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["id"])
    todo = meta[~meta.id.isin(done.id)]
    if len(todo):
        print(f"Scanning {len(todo):,} transcripts ({len(done):,} cached)...", flush=True)
        with Pool(max(1, os.cpu_count() - 1)) as pool:
            rows = pool.map(ScanOne, zip(todo.id, [Loc(r) for r in todo.to_dict("records")]), chunksize=200)
        new = pd.DataFrame(rows, columns=["id", "n_words", "education_mentions", "education_distinct_terms",
                                          "direct_school_body_prefix"])
        done = pd.concat([done, new], ignore_index=True)
        done.to_parquet(cache, index=False)
    return done


def ApplyEligibility(meta, scan):
    meta = meta.merge(scan, on="id", how="left")
    corroborated = (meta.generic_governance_title & (meta.explicit_channel >= 10) & (meta.explicit_fraction >= .5)
                    & (meta.education_mentions >= 3) & (meta.education_distinct_terms >= 2))
    corroborated = corroborated | (meta.generic_governance_title & meta.direct_school_body_prefix.fillna(False).astype(bool))
    meta["eligible_title"] = meta.title_candidate & (meta.explicit_school_title | corroborated)
    meta["eligible"] = meta.eligible_title & (meta.n_words >= MIN_WORDS)
    # One meeting per district-date: duplicate uploads of the same meeting keep the longest transcript.
    elig = (meta[meta.eligible].sort_values("n_words", ascending=False)
            .drop_duplicates(["leaid", "meeting_date"]))
    print(f"Eligible: {meta.eligible_title.sum():,} meeting-titled videos, {meta.eligible.sum():,} with >= {MIN_WORDS:,} "
          f"words, {len(elig):,} distinct district-meetings")
    return elig


def Stratum(df):
    return df.gop_tercile + " x " + df.locale


def SelectDistricts(meta):
    static = pd.read_parquet(INDIR_COV / "district_static.parquet",
                             columns=["leaid", "place_name", "lea_name", "city_location", "county_name",
                                      "gop_2p_2020", "urban_centric_locale"])
    static["leaid"] = static.leaid.astype(str).str.zfill(7)
    # Terciles over all DistrictView districts (fixed cutoffs, independent of which districts survive the filters).
    static["gop_tercile"] = pd.qcut(static.gop_2p_2020, 3, labels=["D", "mid", "R"]).astype(str)
    static["locale"] = (static.urban_centric_locale // 10).map(LOCALES)
    counts = meta.groupby(["leaid", "year"]).size().unstack(fill_value=0).reindex(columns=PRE_YEARS + POST_YEARS, fill_value=0)
    ok = counts.index[(counts >= MIN_MEETINGS).all(axis=1)]
    pool = static[static.leaid.isin(ok) & static.gop_2p_2020.notna() & static.locale.notna()].copy()
    pool["stratum"] = Stratum(pool)
    print(f"Districts with >= {MIN_MEETINGS} eligible meetings in every year {PRE_YEARS + POST_YEARS}: {len(ok):,} "
          f"({len(pool):,} with vote share and locale)")

    # Spread N_DISTRICTS as evenly as possible over the 12 strata (water-filling: small strata are taken whole;
    # if the pool is smaller than N_DISTRICTS every qualifying district is taken).
    sizes = pool.stratum.value_counts()
    quota, left = {}, N_DISTRICTS
    for i, (s, n) in enumerate(sizes.sort_values().items()):
        quota[s] = min(n, left // (len(sizes) - i))
        left -= quota[s]
    districts = pd.concat([g.sample(quota[s], random_state=SEED) for s, g in pool.groupby("stratum")])
    # Seeded coin flip per LEAID (hash-based, so a district's split never changes if the sample changes).
    districts["split"] = districts.leaid.map(lambda x: "A" if int(hashlib.md5(f"{SEED}-{x}".encode()).hexdigest(), 16) % 2 == 0 else "B")
    districts.to_parquet(OUTDIR / "districts.parquet", index=False)
    return districts


def SampleMeetings(meta, districts):
    meta = meta[meta.leaid.isin(districts.leaid)]
    # Seeded shuffle, then the first MEETINGS_PER_YEAR per district-year (all of them where fewer exist).
    meetings = meta.sample(frac=1, random_state=SEED).groupby(["leaid", "year"]).head(MEETINGS_PER_YEAR)
    meetings = meetings.merge(districts[["leaid", "split", "gop_2p_2020", "locale", "gop_tercile"]], on="leaid")
    print(f"Sampled {len(meetings):,} meetings from {meetings.leaid.nunique()} districts")
    return meetings


def LexiconIndex():
    terms, phrases = {}, {}
    for cat in FLAG_CATEGORIES:
        for t in LEXICON[cat]["terms"]:
            terms[t] = cat
        for p in LEXICON[cat]["phrases"]:
            toks = tuple(p.split())
            phrases.setdefault(toks[0], []).append((toks, cat))
    return terms, phrases


def LexiconHits(norm, terms, phrases):
    hits = []
    for i, w in enumerate(norm):
        if w in terms:
            hits.append((i, terms[w]))
        for toks, cat in phrases.get(w, []):
            if tuple(norm[i:i + len(toks)]) == toks:
                hits.append((i, cat))
    return hits


def FlaggedWindows(n, hits):
    # A candidate window is centred on each hit and needs >= MIN_FLAG_HITS hits. Its category is the rarest one with
    # >= MIN_FLAG_HITS hits inside (else its most frequent). Greedy pick: rarest category first, then most hits;
    # overlapping candidates are skipped.
    if not hits:
        return []
    pos = np.array([h[0] for h in hits])
    cats = np.array([h[1] for h in hits])
    candidates = []
    for s in np.unique(np.clip(pos - WINDOW // 2, 0, max(0, n - WINDOW))):
        inside = cats[(pos >= s) & (pos < s + WINDOW)]
        if len(inside) < MIN_FLAG_HITS:
            continue
        names, counts = np.unique(inside, return_counts=True)
        common = dict(zip(names, counts))
        cat = next((c for c in FLAG_CATEGORIES if common.get(c, 0) >= MIN_FLAG_HITS), names[counts.argmax()])
        candidates.append((FLAG_CATEGORIES.index(cat), -len(inside), int(s), str(cat)))
    chosen = []
    for _, neg_hits, s, cat in sorted(candidates):
        if len(chosen) == MAX_FLAGGED:
            break
        if not any(abs(s - c) < WINDOW for c, _, _ in chosen):
            chosen.append((s, cat, -neg_hits))
    return chosen


def CountHits(text, terms, phrases):
    return len(LexiconHits([re.sub(r"[^a-z0-9]", "", w.lower()) for w in text.split()], terms, phrases))


def ExcerptsForMeeting(args):
    row, seed = args
    words = ReadClean(Loc(row)).split()
    n = len(words)
    if n != row["n_words"]:
        raise RuntimeError(f"{row['id']}: read {n} words, scan found {row['n_words']}")
    rng = np.random.default_rng(seed)
    terms, phrases = LexiconIndex()
    norm = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in words]
    windows = [("random", int(rng.integers(0, max(1, n - WINDOW + 1))), None, None)]
    windows += [("flagged", int(s), cat, k) for s, cat, k in FlaggedWindows(n, LexiconHits(norm, terms, phrases))]
    out = []
    for j, (kind, s, cat, k) in enumerate(windows):
        text = " ".join(words[s:s + WINDOW])
        out.append({"excerpt_id": f"{row['id']}_{j}", "leaid": row["leaid"], "video_id": row["id"],
                    "meeting_date": row["meeting_date"], "year": row["year"], "period": row["period"],
                    "split": row["split"], "excerpt_type": kind, "lexicon_category": cat, "n_lexicon_hits": k,
                    "window_start": s, "gop_2p_2020": row["gop_2p_2020"], "locale": row["locale"],
                    "gop_tercile": row["gop_tercile"], "n_hits_raw": CountHits(text, terms, phrases), "text": text})
    return out


def BuildExcerpts(meetings):
    rows = meetings.to_dict("records")
    seeds = [int(hashlib.md5(f"{SEED}-{r['id']}".encode()).hexdigest(), 16) % 2**32 for r in rows]
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        out = pool.map(ExcerptsForMeeting, zip(rows, seeds), chunksize=20)
    excerpts = pd.DataFrame([e for ex in out for e in ex])
    excerpts["n_lexicon_hits"] = excerpts.n_lexicon_hits.astype("Int64")
    return excerpts


def NamePatterns(row):
    # Full names from metadata/CCD, plus their distinctive words ("Tacoma School District" -> "Tacoma").
    names = {str(row[c]).lower() for c in ["place_name", "lea_name", "city_location"] if pd.notna(row[c])}
    if pd.notna(row["county_name"]):
        county = re.sub(r"\s+(county|parish|borough|census area|municipality)$", "", str(row["county_name"]).lower())
        names |= {county, county + " county"}
    names |= {w for name in list(names) for w in re.findall(r"[a-z][a-z'.-]+", name)
              if w not in GENERIC_NAME_WORDS and len(w) > 2}
    return sorted(names, key=len, reverse=True)


def StripNames(excerpts, districts):
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
    state_re = re.compile(r"\b(" + "|".join(sorted(STATES, key=len, reverse=True)) + r")\b", re.I)
    patterns = {r["leaid"]: re.compile(r"\b(" + "|".join(map(re.escape, NamePatterns(r))) + r")\b", re.I)
                for r in districts.to_dict("records")}
    from spacy.lang.en.stop_words import STOP_WORDS
    # A "name token" is any word that is not a stop word, a common meeting word or a lexicon term, so rules don't eat
    # ordinary speech.
    lexicon_terms = {t for c in LEXICON.values() for t in c["terms"]}
    not_name = sorted(STOP_WORDS | NOT_NAME_WORDS | lexicon_terms, key=len, reverse=True)
    tok = r"(?!(?:" + "|".join(map(re.escape, not_name)) + r")\b)[a-z][a-z'-]*[a-z]"
    school_tok = r"(?:\[PLACE\]|(?!(?:" + "|".join(SCHOOL_GENERIC_WORDS) + r")\b)" + tok + ")"
    # Lowercase captions hide names from spaCy, so: school names; a title + up to two name tokens; self-introductions.
    school_re = re.compile(r"(?<![\w\]])(?:" + school_tok + r"\s+){1,2}(?:high school|middle school|junior high(?: school)?|"
                           r"elementary(?: school)?|academy|isd|school district|unified(?: school district)?)\b", re.I)
    title_re = re.compile(r"\b(" + "|".join(TITLES) + r")(\.?\s+)" + tok + r"(?:\s+" + tok + r")?\b", re.I)
    # Only "my name is": "i'm" / "this is" are followed by opinions ("i'm disgusted") as often as by names.
    intro_re = re.compile(r"\b(my name is|my name's)(\s+)" + tok + r"(?:\s+" + tok + r")?\b", re.I)
    texts = [state_re.sub("[PLACE]", patterns[l].sub("[PLACE]", t)) for l, t in zip(excerpts.leaid, excerpts.text)]
    texts = [intro_re.sub(r"\1\2[NAME]", title_re.sub(r"\1\2[NAME]", school_re.sub("[PLACE]", t))) for t in texts]
    # spaCy sometimes tags lexicon words ("DEI", "Gender") as PERSON/GPE; never strip an entity containing one.
    protected = {t for c in LEXICON.values() for t in c["terms"]}
    keep = lambda e: (not any(re.sub(r"[^a-z0-9]", "", w.lower()) in protected for w in e.text.split())
                      and "NAME" not in e.text and "PLACE" not in e.text)   # don't re-tag our own [NAME]/[PLACE]
    stripped = []
    for doc in nlp.pipe(texts, batch_size=64, n_process=4):
        t = doc.text
        for ent in reversed([e for e in doc.ents if e.label_ in ("PERSON", "GPE") and keep(e)]):
            t = t[:ent.start_char] + ("[NAME]" if ent.label_ == "PERSON" else "[PLACE]") + t[ent.end_char:]
        stripped.append(t)
    excerpts["text"] = stripped
    excerpts["n_words"] = excerpts.text.str.split().str.len()
    # Check: name-stripping should remove almost no lexicon hits (flagging itself ran on the raw text).
    terms, phrases = LexiconIndex()
    excerpts["n_hits_stripped"] = [CountHits(t, terms, phrases) for t in excerpts.text]
    return excerpts


def WindowsOne(args):
    loc, starts = args
    words = ReadClean(loc).split()
    return [" ".join(words[s:s + WINDOW]) for s in starts]


def Restrip():
    # Re-read each excerpt's raw window from the tars (same video, same window_start) and strip names again.
    excerpts = pd.read_parquet(OUTDIR / "excerpts.parquet")
    districts = pd.read_parquet(OUTDIR / "districts.parquet")
    meta = pd.read_parquet(INDIR_META / "meta_districtview.parquet", columns=["id", "channel_id"])
    meta["name"] = meta.channel_id + "__" + meta.id + ".txt.gz"
    loc = meta.merge(TarIndex(), on="name").set_index("id")
    groups = excerpts.groupby("video_id", sort=False).window_start.apply(list)
    jobs = [((loc.at[v, "tar"], int(loc.at[v, "offset"]), int(loc.at[v, "size"])), starts) for v, starts in groups.items()]
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        windows = pool.map(WindowsOne, jobs, chunksize=20)
    raw = {(v, s): w for (v, starts), ws in zip(groups.items(), windows) for s, w in zip(starts, ws)}
    excerpts["text"] = [raw[(v, s)] for v, s in zip(excerpts.video_id, excerpts.window_start)]
    terms, phrases = LexiconIndex()
    if [CountHits(t, terms, phrases) for t in excerpts.text] != excerpts.n_hits_raw.tolist():
        raise RuntimeError("Re-read raw windows do not match the original build")
    excerpts = StripNames(excerpts, districts)
    Summarise(excerpts, districts)
    return


def Summarise(excerpts, districts, out_name="excerpts.parquet"):
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    excerpts["n_tokens"] = [len(enc.encode(t)) for t in excerpts.text]
    excerpts.to_parquet(OUTDIR / out_name, index=False)

    attr_tokens = len(enc.encode(json.dumps(ATTRIBUTES)))
    calls = len(excerpts)
    tok_in = int(excerpts.n_tokens.sum()) + calls * (attr_tokens + PROMPT_TEMPLATE_TOKENS)
    tok_out = calls * OUT_TOKENS_PER_CALL
    cost = tok_in / 1e6 * PRICE_IN + tok_out / 1e6 * PRICE_OUT
    print("\nDistricts per stratum (GOP tercile x locale):")
    print(pd.crosstab(districts.gop_tercile, districts.locale).reindex(["D", "mid", "R"]).to_string())
    print(f"\nSplit: {districts.split.value_counts().to_dict()}")
    print("\nExcerpts by type and period:")
    print(pd.crosstab(excerpts.excerpt_type, excerpts.period, margins=True).to_string())
    print("\nFlagged excerpts by lexicon category and period:")
    print(pd.crosstab(excerpts.lexicon_category, excerpts.period, margins=True).to_string())
    print(f"\nLexicon hits in all excerpts: {excerpts.n_hits_raw.sum():,} raw -> {excerpts.n_hits_stripped.sum():,} "
          f"after name-stripping")
    print(f"\nExcerpts: {calls:,}; median {excerpts.n_words.median():.0f} words, {excerpts.n_tokens.median():.0f} tokens")
    print(f"Tokens: {excerpts.n_tokens.sum():,} excerpt + {calls * (attr_tokens + PROMPT_TEMPLATE_TOKENS):,} prompt "
          f"= {tok_in:,} input; {tok_out:,} output ({OUT_TOKENS_PER_CALL}/call)")
    print(f"Estimated cost per full run at ${PRICE_IN}/M in, ${PRICE_OUT}/M out: ${cost:,.2f}")

    sample = excerpts.sample(20, random_state=SEED)
    with open(OUTDIR / f"inspect_name_stripping{out_name.removeprefix('excerpts').removesuffix('.parquet')}.txt", "w") as f:
        for r in sample.itertuples():
            f.write(f"===== {r.excerpt_id} | leaid {r.leaid} | {r.meeting_date:%Y-%m-%d} | {r.excerpt_type} "
                    f"{r.lexicon_category or ''}\n{r.text}\n\n")
    print(f"\nWrote {OUTDIR / out_name} and 20 excerpts to {f.name}")
    return


def HalfYear(dates):
    return dates.dt.year.astype(str) + np.where(dates.dt.month <= 6, "H1", "H2")


def Extend():
    # Same 348 districts and A/B split, same eligibility rules (channel stats now computed over 2018-2024).
    districts = pd.read_parquet(OUTDIR / "districts.parquet")
    old = pd.read_parquet(OUTDIR / "excerpts.parquet", columns=["video_id", "leaid", "year", "meeting_date"])
    meta = TitleScreen(LoadMeta(list(range(SAMPLE_START, 2025))))
    meta = meta[meta.leaid.isin(districts.leaid)]
    meta = ApplyEligibility(meta, ScanTranscripts(meta[meta.title_candidate]))
    lost = set(old.video_id) - set(meta.id)
    print(f"Original sampled meetings no longer eligible under the wider channel window: {len(lost)}")
    meetings = SampleExtension(meta, districts, old.drop_duplicates("video_id"))
    excerpts = BuildExcerpts(meetings)
    excerpts = StripNames(excerpts, districts)
    excerpts["half_year"] = HalfYear(excerpts.meeting_date)
    excerpts["batch"] = "extension"
    Summarise(excerpts, districts, "excerpts_extension.parquet")
    return


def SampleExtension(meta, districts, old):
    # Seeded shuffle; per district-year keep the already-sampled meetings, take up to EXT_PER_HALF per half-year,
    # then fill to EXT_PER_YEAR from either half (all meetings where fewer exist). Never two meetings on one date.
    meta = meta[meta.year.isin(EXT_YEARS)].sample(frac=1, random_state=SEED)
    meta["half"] = np.where(meta.meeting_date.dt.month <= 6, 1, 2)
    old = old[old.year.isin(EXT_YEARS)].assign(half=lambda d: np.where(d.meeting_date.dt.month <= 6, 1, 2))
    picks = []
    for (leaid, year), g in meta.groupby(["leaid", "year"], sort=False):
        have = old[(old.leaid == leaid) & (old.year == year)]
        g = g[~g.id.isin(have.video_id) & ~g.meeting_date.isin(have.meeting_date)]
        take = []
        for h in (1, 2):
            take += list(g.index[g.half == h][:max(0, EXT_PER_HALF - int((have.half == h).sum()))])
        rest = [i for i in g.index if i not in take]
        take += rest[:max(0, EXT_PER_YEAR - len(have) - len(take))]
        picks += take
    meetings = meta.loc[picks].drop(columns="half")
    meetings = meetings.merge(districts[["leaid", "split", "gop_2p_2020", "locale", "gop_tercile"]], on="leaid")
    per = pd.concat([meetings[["leaid", "year"]], old[["leaid", "year"]]]).groupby(["leaid", "year"]).size()
    print(f"Sampled {len(meetings):,} new meetings; meetings per district-year incl. kept originals:\n"
          + per.groupby("year").describe()[["count", "mean", "min", "max"]].to_string())
    return meetings


def ExtraWindowsOne(args):
    # EXTRA_RANDOM more random windows for one meeting (own seed stream, independent of the first random window).
    row, seed = args
    words = ReadClean(Loc(row)).split()
    if len(words) != row["n_words"]:
        raise RuntimeError(f"{row['video_id']}: read {len(words)} words, scan found {row['n_words']}")
    rng = np.random.default_rng(seed)
    terms, phrases = LexiconIndex()
    out = []
    for k in range(1, EXTRA_RANDOM + 1):
        s = int(rng.integers(0, max(1, len(words) - WINDOW + 1)))
        text = " ".join(words[s:s + WINDOW])
        out.append({**{c: row[c] for c in ["leaid", "video_id", "meeting_date", "year", "period", "split",
                                           "gop_2p_2020", "locale", "gop_tercile"]},
                    "excerpt_id": f"{row['video_id']}_r{k}", "excerpt_type": "random", "lexicon_category": None,
                    "n_lexicon_hits": None, "window_start": s, "n_hits_raw": CountHits(text, terms, phrases),
                    "text": text})
    return out


def ExtraRandom():
    # Every meeting sampled so far (original + extension), all years.
    districts = pd.read_parquet(OUTDIR / "districts.parquet")
    cols = ["video_id", "leaid", "meeting_date", "year", "period", "split", "gop_2p_2020", "locale", "gop_tercile"]
    meetings = pd.concat([pd.read_parquet(OUTDIR / f, columns=cols) for f in
                          ["excerpts.parquet", "excerpts_extension.parquet"]]).drop_duplicates("video_id")
    meta = pd.read_parquet(INDIR_META / "meta_districtview.parquet", columns=["id", "channel_id"])
    meta["name"] = meta.channel_id + "__" + meta.id + ".txt.gz"
    meta = meta.merge(TarIndex(), on="name").merge(pd.read_parquet(OUTDIR / "scan_tar.parquet", columns=["id", "n_words"]),
                                                   on="id")
    meetings = meetings.merge(meta.rename(columns={"id": "video_id"})[["video_id", "tar", "offset", "size", "n_words"]],
                              on="video_id", how="left", validate="one_to_one")
    if meetings.tar.isna().any():
        raise RuntimeError(f"{meetings.tar.isna().sum()} sampled meetings not found in the tar index")
    rows = meetings.to_dict("records")
    seeds = [int(hashlib.md5(f"{SEED}-extra-{r['video_id']}".encode()).hexdigest(), 16) % 2**32 for r in rows]
    with Pool(max(1, os.cpu_count() - 1)) as pool:
        out = pool.map(ExtraWindowsOne, zip(rows, seeds), chunksize=20)
    excerpts = pd.DataFrame([e for ex in out for e in ex])
    excerpts["n_lexicon_hits"] = excerpts.n_lexicon_hits.astype("Int64")
    excerpts = StripNames(excerpts, districts)
    excerpts["half_year"] = HalfYear(excerpts.meeting_date)
    excerpts["batch"] = "extra_random"
    print(f"{len(meetings):,} meetings -> {len(excerpts):,} extra random excerpts")
    Summarise(excerpts, districts, "excerpts_extra_random.parquet")
    return


if __name__ == "__main__":
    Main()
