"""Collapse the transcript-level document-term matrix to district x quarter, with a tightened vocabulary.

Reads  Datastore/DistrictView/derived/bow/{dtm.npz, vocab.parquet, docs.parquet}   (from clean_transcripts.py)
       Datastore/DistrictView/districtview_share_draft/meta_districtview.parquet   (for the meeting date -> quarter)
Writes Datastore/DistrictView/derived/bow_dq/dtm.npz        districts-quarters x terms (counts)
                                         vocab.parquet      term for each column
                                         units.parquet      row i = (leaid, quarter, n_docs, n_tokens, n_words)
                                         docs.parquet       bow/docs.parquet plus quarter (row order = phrase_counts.parquet)
                                         vocab_cuts.parquet every original term with the reason it was cut (or "kept")

Vocabulary cuts, in order: contains a digit; fewer than 3 characters; leftover apostrophes; in STOPWORDS (clean_transcripts.py);
used in fewer than MIN_DISTRICTS districts (place names, local acronyms, names of individuals identify a district, not a topic).
Possessives are merged into the base word ("student's" -> "student").
Quarter = meeting_date, falling back to upload date (same rule as build_derived_transcripts.py).
Run from the repo root:  python3 source/derived/sb_transcripts/aggregate_dtm.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from clean_transcripts import STOPWORDS

INDIR_BOW = Path("Datastore/DistrictView/derived/bow/")
INDIR_META = Path("Datastore/DistrictView/districtview_share_draft/meta_districtview.parquet")
OUTDIR_DQ = Path("Datastore/DistrictView/derived/bow_dq/")
MIN_DISTRICTS = 25

def Main():
    dtm = sp.load_npz(INDIR_BOW / "dtm.npz").tocsr()
    vocab = pd.read_parquet(INDIR_BOW / "vocab.parquet").term.to_numpy()
    df_docs = pd.read_parquet(INDIR_BOW / "docs.parquet")
    df_docs["quarter"] = AssignQuarter(df_docs.file)
    unit_codes, df_units = DistrictQuarterIndex(df_docs)
    dtm, vocab, df_cuts = CutVocabulary(dtm, vocab, df_docs.leaid.to_numpy())
    dtm_dq = sp.csr_matrix((np.ones(len(df_docs), dtype="int32"), (unit_codes, np.arange(len(df_docs)))),
                           shape=(len(df_units), len(df_docs))) @ dtm
    df_units["n_docs"] = np.bincount(unit_codes, minlength=len(df_units))
    df_units["n_tokens"] = np.asarray(dtm_dq.sum(1)).ravel()                       # words left after vocabulary cuts
    df_units["n_words"] = np.bincount(unit_codes, weights=df_docs.n_words, minlength=len(df_units))   # words before cuts
    Save(dtm_dq, vocab, df_units, df_cuts, df_docs)
    return

def AssignQuarter(files):
    df_meta = pd.read_parquet(INDIR_META, columns=["id", "channel_id", "meeting_date", "published_at"])
    df_meta["file"] = df_meta.channel_id + "__" + df_meta.id + ".txt.gz"
    when = df_meta.meeting_date.fillna(df_meta.published_at.dt.tz_convert(None))
    df_meta["quarter"] = when.dt.to_period("Q").astype(str)
    quarter = pd.Series(files).map(df_meta.drop_duplicates("file").set_index("file").quarter)
    assert quarter.notna().all(), "some derived transcripts are missing from the metadata"
    return quarter.to_numpy()

def DistrictQuarterIndex(df_docs):
    keys = pd.MultiIndex.from_frame(df_docs[["leaid", "quarter"]])
    codes, uniques = pd.factorize(keys, sort=True)
    return codes, pd.DataFrame(list(uniques), columns=["leaid", "quarter"])

def CanonicalTerm(term):
    """Base form of a term, or the reason it is cut (prefixed with '!')."""
    base = re.sub(r"'s$", "", term).rstrip("'")
    if re.search(r"\d", base):
        return "!digit"
    if len(base) < 3:
        return "!short"
    if "'" in base:
        return "!apostrophe"
    if base in STOPWORDS or term in STOPWORDS:
        return "!stopword"
    return base

def CutVocabulary(dtm, vocab, leaids):
    canon = np.array([CanonicalTerm(t) for t in vocab])
    keep = ~np.char.startswith(canon, "!")
    new_terms, new_idx = np.unique(canon[keep], return_inverse=True)
    mapping = sp.csr_matrix((np.ones(keep.sum(), dtype="int32"), (np.flatnonzero(keep), new_idx)),
                            shape=(len(vocab), len(new_terms)))
    dtm = dtm @ mapping    # merges possessives into base words and drops the cut columns
    dist_codes = pd.factorize(leaids)[0]
    by_district = sp.csr_matrix((np.ones(len(dist_codes), dtype="float32"), (dist_codes, np.arange(len(dist_codes)))))
    n_districts = np.asarray(((by_district @ (dtm > 0).astype("float32")) > 0).sum(0)).ravel()
    enough = n_districts >= MIN_DISTRICTS
    df_cuts = pd.DataFrame({"term": vocab, "canonical": canon, "status": np.where(keep, "kept", canon)})
    few = df_cuts.canonical.isin(new_terms[~enough])
    df_cuts.loc[few & keep, "status"] = "!few_districts"
    print(f"Vocabulary {len(vocab):,} -> {enough.sum():,} terms. Cuts:\n{df_cuts.status.value_counts().to_string()}")
    return dtm[:, np.flatnonzero(enough)], new_terms[enough], df_cuts

def Save(dtm_dq, vocab, df_units, df_cuts, df_docs):
    OUTDIR_DQ.mkdir(parents=True, exist_ok=True)
    sp.save_npz(OUTDIR_DQ / "dtm.npz", dtm_dq)
    pd.DataFrame({"term": vocab}).to_parquet(OUTDIR_DQ / "vocab.parquet")
    df_units.to_parquet(OUTDIR_DQ / "units.parquet")
    df_cuts.to_parquet(OUTDIR_DQ / "vocab_cuts.parquet")
    df_docs.to_parquet(OUTDIR_DQ / "docs.parquet")
    print(f"Saved district-quarter matrix {dtm_dq.shape} ({df_units.leaid.nunique()} districts, "
          f"{df_units.quarter.nunique()} quarters) to {OUTDIR_DQ}")
    return

if __name__ == "__main__":
    Main()
