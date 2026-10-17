"""Shared loaders for the bag-of-words analysis (district x quarter matrix + partisan anchor).

Rows of the district-quarter matrix are described by units.parquet; the partisan anchor is the county presidential
two-party GOP share from district_static.parquet (county != district: a proxy, see CLAUDE.md).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

INDIR_DQ = Path("Datastore/DistrictView/derived/bow_dq/")
INDIR_PHRASES = Path("Datastore/DistrictView/derived/bow_phrases/phrase_counts.parquet")
INDIR_DOCS = Path("Datastore/DistrictView/derived/bow/docs.parquet")
INDIR_STATIC = Path("Datastore/District Covariates/processed/district_static.parquet")
OUTDIR_TABLES = Path("output/tables/")
OUTDIR_FIGURES = Path("output/figures/")

PERIODS = {2018: "pre", 2019: "pre", 2020: "covid", 2021: "covid", 2022: "post", 2023: "post"}

def LoadDQ():
    """(dtm csr [units x terms], vocab array, units DataFrame) with year, period and state added."""
    dtm = sp.load_npz(INDIR_DQ / "dtm.npz").tocsr()
    vocab = pd.read_parquet(INDIR_DQ / "vocab.parquet").term.to_numpy()
    units = pd.read_parquet(INDIR_DQ / "units.parquet")
    units["year"] = units.quarter.str[:4].astype(int)
    units["period"] = units.year.map(PERIODS)
    units["state"] = units.leaid.str[:2]
    return dtm, vocab, units

def LoadAnchor(anchor_year=2020):
    """leaid -> gop two-party share in `anchor_year` and a binary party label (R if share > 0.5); missing/0.5 dropped.
    The anchor is held fixed across periods so changes in measured polarization are not changes in the labels."""
    df = pd.read_parquet(INDIR_STATIC, columns=["leaid", f"gop_2p_{anchor_year}"])
    df = df.rename(columns={f"gop_2p_{anchor_year}": "gop"})
    df["leaid"] = df.leaid.astype(str).str.zfill(7)
    df = df[df.gop.notna() & (df.gop != 0.5)].copy()
    df["is_r"] = df.gop > 0.5
    return df

def WithAnchor(dtm, units, anchor_year=2020):
    """Attach the anchor to units and drop units with no anchor (rows of dtm are dropped consistently)."""
    units = units.merge(LoadAnchor(anchor_year), on="leaid", how="left")
    keep = units.gop.notna().to_numpy()
    print(f"Partisan anchor gop_2p_{anchor_year}: {units.loc[keep, 'leaid'].nunique()} of {units.leaid.nunique()} districts "
          f"({keep.sum():,} of {len(units):,} district-quarters)")
    return dtm[np.flatnonzero(keep)], units[keep].reset_index(drop=True)

def CollapseTo(dtm, units, level):
    """Sum district-quarter rows to district x `level` ('year', 'quarter' or 'period'). Returns (csr, DataFrame label/leaid/is_r/state/n_tokens/n_words, unit->group sparse matrix)."""
    key = pd.MultiIndex.from_arrays([units.leaid, units[level]])
    codes, uniq = pd.factorize(key)
    group = sp.csr_matrix((np.ones(len(units), dtype="float64"), (codes, np.arange(len(units)))), shape=(len(uniq), len(units)))
    counts = (group @ dtm).tocsr()
    df = pd.DataFrame(list(uniq), columns=["leaid", "label"])
    df = df.merge(units[["leaid", "is_r", "gop", "state"]].drop_duplicates("leaid"), on="leaid", how="left")
    df["is_r"] = df.is_r.astype(bool)
    df["n_tokens"] = np.asarray(counts.sum(1)).ravel()
    df["n_words"] = group @ units.n_words.to_numpy()
    return counts, df, group

def LoadPhraseCounts(units):
    """Phrase counts (units x phrases DataFrame, rows aligned with `units`) summed from transcripts to district-quarter."""
    docs = pd.read_parquet(INDIR_DQ / "docs.parquet", columns=["leaid", "quarter"])
    counts = pd.read_parquet(INDIR_PHRASES).drop(columns="file")
    assert len(docs) == len(counts), "phrase counts and docs.parquet are out of sync; re-run count_phrases.py"
    by_unit = counts.groupby([docs.leaid.to_numpy(), docs.quarter.to_numpy()]).sum()
    return by_unit.reindex(pd.MultiIndex.from_frame(units[["leaid", "quarter"]])).fillna(0).reset_index(drop=True)

def TermStateConcentration(dtm, units, vocab):
    """Per term: share of its (district-normalised) usage that comes from its single biggest state, and number of states using it.
    High share = state-specific vocabulary (place names, state institutions) that proxies geography, not agenda."""
    codes = pd.factorize(units.leaid)[0]
    group = sp.csr_matrix((np.ones(len(units)), (codes, np.arange(len(units)))))
    counts = (group @ dtm).tocsr()
    freq = sp.diags(1 / np.asarray(counts.sum(1)).ravel()) @ counts                 # district relative frequencies
    state_of_district = units.drop_duplicates("leaid").set_index("leaid").state.reindex(pd.unique(units.leaid)).to_numpy()
    scode = pd.factorize(state_of_district)[0]
    by_state = (sp.csr_matrix((np.ones(len(scode)), (scode, np.arange(len(scode))))) @ freq).toarray()
    return pd.DataFrame({"term": vocab, "top_state_share": by_state.max(0) / by_state.sum(0), "n_states": (by_state > 0).sum(0)})
