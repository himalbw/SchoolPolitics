import gzip
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

INDIR_DERIVED = Path("Datastore/DistrictView/derived/")
INDIR_TRANSCRIPTS = INDIR_DERIVED / "transcripts"
OUTDIR_BOW = INDIR_DERIVED / "bow"   # a run with n_docs set writes to bow_sample/ instead, so tests never overwrite the full matrix
MIN_DF = 20      # drop words in fewer than 20 transcripts (caption typos, rare names)
MAX_DF = 0.5     # drop words in more than half of all transcripts

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

def Main(n_docs=None):
    """n_docs=None builds the full corpus; an int uses only the first n_docs kept transcripts (for testing)."""
    df_docs = KeptDocs(n_docs)
    dtm, vocab = BuildDocTermMatrix(df_docs)
    SaveDTM(dtm, vocab, df_docs, OUTDIR_BOW if n_docs is None else OUTDIR_BOW.with_name("bow_sample"))
    return

def KeptDocs(n_docs=None):
    """One row per kept transcript (manifest order = row order of the matrix)."""
    df = pd.read_parquet(INDIR_DERIVED / "manifest.parquet", columns=["file", "leaid", "year", "n_words", "status"])
    df = df[df.status == "kept"].drop(columns="status")
    return df.head(n_docs).reset_index(drop=True)

def GetCorpus(n_docs=None):
    """Yield (file, text) for every kept transcript, one at a time."""
    for fname in KeptDocs(n_docs).file:
        with gzip.open(INDIR_TRANSCRIPTS / fname, "rt", encoding="utf-8") as f:
            yield fname, f.read()

def RemoveStopwords(text, stopwords=STOPWORDS):
    return [w for w in text.split() if w not in stopwords]

def BuildDocTermMatrix(df_docs, min_df=MIN_DF, max_df=MAX_DF):
    """Return (sparse documents x terms count matrix, vocabulary array). Rows follow df_docs order."""
    # Text is already cleaned, so split on whitespace only; the custom tokenizer keeps "don't" as one token.
    vectorizer = CountVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False,
                                 stop_words=sorted(STOPWORDS), min_df=min_df, max_df=max_df, dtype="int32")
    dtm = vectorizer.fit_transform(text for _, text in GetCorpus(len(df_docs)))
    vocab = vectorizer.get_feature_names_out()
    print(f"Document-term matrix: {dtm.shape[0]:,} docs x {dtm.shape[1]:,} terms, {dtm.nnz:,} nonzeros, {dtm.sum():,} tokens")
    return dtm, vocab

def SaveDTM(dtm, vocab, df_docs, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    sp.save_npz(outdir / "dtm.npz", dtm)
    pd.DataFrame({"term": vocab}).to_parquet(outdir / "vocab.parquet")
    df_docs.to_parquet(outdir / "docs.parquet")
    print(f"Saved dtm.npz, vocab.parquet (column j of dtm), docs.parquet (row i of dtm) to {outdir}")
    return

def TopTerms(docs, n=40):
    counts = Counter()
    for _, text in docs:
        counts.update(RemoveStopwords(text))
    return pd.Series(dict(counts.most_common(n)), name="count")

if __name__ == "__main__":
    Main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
