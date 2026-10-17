import gzip
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

INDIR_DERIVED = Path("Datastore/DistrictView/derived/")
INDIR_TRANSCRIPTS = INDIR_DERIVED / "transcripts"

# sklearn's list has no contractions, but our text keeps in-word apostrophes ("don't"), so list the common ones.
CONTRACTIONS = {
    "i'm", "i've", "i'll", "i'd", "you're", "you've", "you'll", "you'd", "he's", "he'll", "he'd", "she's", "she'll",
    "she'd", "it's", "it'll", "it'd", "we're", "we've", "we'll", "we'd", "they're", "they've", "they'll", "they'd",
    "that's", "that'll", "there's", "there'll", "here's", "what's", "who's", "where's", "when's", "how's", "let's",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "won't",
    "wouldn't", "can't", "couldn't", "shouldn't", "mustn't", "needn't", "ain't",
}
# Auto-caption / spoken-language noise. "applause" and "music" are already stripped as [tags] in derived/; kept as a safety net.
CAPTION_NOISE = {
    "applause", "music", "laughter", "inaudible", "um", "uh", "umm", "uhh", "hmm", "mm", "ah", "oh", "yeah", "yep",
    "okay", "ok", "gonna", "wanna", "gotta", "kinda", "sorta", "like", "just", "really", "going", "know", "think",
}

PROCEDURAL = set()

STOPWORDS = frozenset(ENGLISH_STOP_WORDS | CONTRACTIONS | CAPTION_NOISE | PROCEDURAL)

def Main(n_docs=200):
    docs = []
    for i, (fname, text) in enumerate(GetCorpus()):
        if i >= n_docs:
            break
        docs.append((fname, text))
    n_before = sum(len(t.split()) for _, t in docs)
    n_after = sum(len(RemoveStopwords(t)) for _, t in docs)
    print(f"{len(docs)} docs: {n_before:,} words -> {n_after:,} after stopwords ({1 - n_after / n_before:.1%} removed)")
    return

def GetCorpus():
    df_manifest = pd.read_parquet(INDIR_DERIVED / "manifest.parquet", columns=["file", "status"])
    for fname in df_manifest.loc[df_manifest.status == "kept", "file"]:
        with gzip.open(INDIR_TRANSCRIPTS / fname, "rt", encoding="utf-8") as f:
            yield fname, f.read()

def RemoveStopwords(text, stopwords=STOPWORDS):
    return [w for w in text.split() if w not in stopwords]

def TopTerms(docs, n=40):
    counts = Counter()
    for _, text in docs:
        counts.update(RemoveStopwords(text))
    return pd.Series(dict(counts.most_common(n)), name="count")

if __name__ == "__main__":
    Main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
