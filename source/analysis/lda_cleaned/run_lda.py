"""LDA on the team-cleaned transcripts (cleaned_transcripts_*), with per-document topic weights.

Reuses Stefan's text normalisation, stopwords, chunking rule, vocabulary builder and priors from
source/analysis/lda_2018_2023/workflow.py; only the input (new cleaned folder instead of the raw tars), the number of
training passes, and the topic-weight extraction are new.

Run from the repo root (stat236 env):
    python source/analysis/lda_cleaned/run_lda.py all --years 2022 --k 30 --passes 10
Stages (each cached in its output folder; delete the folder to rebuild): prepare, vocabulary, fit, infer, all.
Working files (chunks with transcript text, vectorizer, fitted model) -> Datastore/DistrictView/derived/lda_cleaned/<tag>/;
results (topic weights, topic terms, manifest, run summaries) -> output/lda_cleaned/<tag>/ (tag = years, e.g. y2022).
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.decomposition import LatentDirichletAllocation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lda_2018_2023"))
import workflow as wf  # noqa: E402  (Stefan's pipeline: clean_text, WORD, CHUNK, STOP, priors, vocabulary())

CLEAN_DIR = Path(os.environ.get("STAT236_CLEAN_DIR", "Datastore/cleaned_transcripts_2026_10_28"))
OUTROOT = Path("Datastore/DistrictView/derived/lda_cleaned")
RESULTS_ROOT = Path("output/lda_cleaned")
RESULTS = None   # set in Main(): output/lda_cleaned/<tag>/
GLOBALS = json.loads(Path("globals.json").read_text())
MIN_DOC_WORDS = GLOBALS["MIN_WORDS"]   # transcript-level cut (team definition); wf.MIN_WORDS is only the chunk-tail rule


def Main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prepare", "vocabulary", "fit", "infer", "all"])
    p.add_argument("--years", type=int, nargs="+", default=list(range(GLOBALS["SAMPLE_START"], GLOBALS["SAMPLE_END"] + 1)))
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--passes", type=int, default=10, help="online passes over the training chunks (Stefan used 1 on the full corpus)")
    args = p.parse_args()

    tag = f"y{args.years[0]}" if len(args.years) == 1 else f"y{min(args.years)}_{max(args.years)}"
    wf.OUTDIR = OUTROOT / tag   # wf.vocabulary() and wf.iter_frames() read/write here
    wf.OUTDIR.mkdir(parents=True, exist_ok=True)
    global RESULTS
    RESULTS = RESULTS_ROOT / tag
    RESULTS.mkdir(parents=True, exist_ok=True)
    stages = ["prepare", "vocabulary", "fit", "infer"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"== {stage} ({wf.OUTDIR})", flush=True)
        if stage == "prepare":
            print(json.dumps(Prepare(args.years)["statuses"], indent=2))
        elif stage == "vocabulary":
            print(len(wf.vocabulary().vocabulary_), "terms")
        elif stage == "fit":
            Fit(args.k, args.passes)
        elif stage == "infer":
            Infer(args.k)
    return


def Prepare(years):
    """Select videos by year from the cleaned metadata, normalise with Stefan's clean_text, write ~1,000-word chunks."""
    if (RESULTS / "preparation.json").exists():
        return json.loads((RESULTS / "preparation.json").read_text())
    (wf.OUTDIR / "chunks").mkdir(exist_ok=True)
    writers, buffers, audit, seen, counts = {}, collections.defaultdict(list), [], {}, collections.Counter()

    def Flush(shard):
        if not buffers[shard]:
            return
        if shard not in writers:
            writers[shard] = pq.ParquetWriter(wf.OUTDIR / "chunks" / f"shard_{shard:02}.parquet", wf.schema, compression="zstd")
        writers[shard].write_table(pa.Table.from_pylist(buffers[shard], schema=wf.schema))
        buffers[shard].clear()

    for f in sorted((CLEAN_DIR / "metadata").glob("*.json.gz")):
        d = json.loads(gzip.decompress(f.read_bytes()))
        m = d.get("raw_metadata", {})
        date = m.get("meeting_date") or m.get("published_at") or ""
        year = int(date[:4]) if date[:4].isdigit() else None
        if year not in years:
            continue
        rec = dict(id=d["id"], leaid=str(m.get("leaid", "")).zfill(7), year=year,
                   year_source="meeting_date" if m.get("meeting_date") else "published_at", chunks=0)
        path = CLEAN_DIR / (d.get("transcript_path") or f"transcripts/{d['id']}.txt.gz")
        if d.get("status") != "available":
            rec["status"] = d.get("status")
        elif not path.exists():
            rec["status"] = "not_downloaded"
        else:
            tokens = [x.group() for x in wf.WORD.finditer(wf.clean_text(gzip.decompress(path.read_bytes()).decode("utf-8")))]
            rec["words"] = len(tokens)
            digest = hashlib.sha256(" ".join(tokens).encode()).hexdigest()
            if len(tokens) < MIN_DOC_WORDS:
                rec["status"] = f"under_{MIN_DOC_WORDS}_words"
            elif digest in seen:
                rec.update(status="exact_duplicate", duplicate_of=seen[digest])
            else:
                seen[digest] = d["id"]
                split = "test" if int(digest[:8], 16) % 10 == 0 else "train"   # same rule as Stefan
                shard = int(hashlib.sha256((str(wf.SEED) + d["id"]).encode()).hexdigest()[:8], 16) % wf.SHARDS
                starts = list(range(0, len(tokens), wf.CHUNK))
                if len(starts) > 1 and len(tokens) - starts[-1] < wf.MIN_WORDS:
                    starts.pop()   # merge short tail into previous chunk
                for i, start in enumerate(starts):
                    end = starts[i + 1] if i + 1 < len(starts) else len(tokens)
                    # Timestamps were stripped by the new cleaning, so chunk start/end seconds are unknown.
                    buffers[shard].append(dict(id=d["id"], leaid=rec["leaid"], year=year, chunk=i, start_seconds=None,
                                               end_seconds=None, words=end - start, split=split, text=" ".join(tokens[start:end])))
                    if len(buffers[shard]) >= 128:
                        Flush(shard)
                counts[split + "_chunks"] += len(starts)
                counts[split + "_videos"] += 1
                rec.update(status="retained", split=split, chunks=len(starts))
        audit.append(rec)
    for shard in list(buffers):
        Flush(shard)
    for w in writers.values():
        w.close()
    a = pd.DataFrame(audit)
    a.to_csv(RESULTS / "cleaning_manifest.csv", index=False)
    result = dict(clean_dir=str(CLEAN_DIR), years=years, min_doc_words=MIN_DOC_WORDS, statuses=a.status.value_counts().to_dict(),
                  counts=dict(counts), districts_retained=int(a.loc[a.status == "retained", "leaid"].nunique()))
    (RESULTS / "preparation.json").write_text(json.dumps(result, indent=2))
    return result


def Frames(split=None):
    """Yield chunk batches from every shard that exists (small samples may leave some shards empty)."""
    for path in sorted((wf.OUTDIR / "chunks").glob("*.parquet")):
        for b in pq.ParquetFile(path).iter_batches(batch_size=wf.BATCH):
            frame = b.to_pandas()
            yield frame if split is None else frame.loc[frame.split == split]


def Fit(k, passes):
    """Online LDA (Stefan's priors and learning schedule), several passes over the training chunks."""
    out = wf.OUTDIR / f"lda_k{k}.joblib"
    if out.exists():
        return joblib.load(out)
    prep = json.loads((RESULTS / "preparation.json").read_text())
    vec = wf.vocabulary()
    model = LatentDirichletAllocation(n_components=k, doc_topic_prior=wf.ALPHA, topic_word_prior=wf.ETA, learning_method="online",
                                      learning_decay=.7, learning_offset=64, total_samples=prep["counts"]["train_chunks"],
                                      batch_size=wf.BATCH, max_doc_update_iter=20, mean_change_tol=.001, random_state=wf.SEED, n_jobs=4)
    started = time.time()
    for p in range(passes):
        for frame in Frames("train"):
            frame = frame.sample(frac=1, random_state=wf.SEED + p)
            X = vec.transform(frame.text)
            X = X[X.getnnz(axis=1) > 0]
            if X.shape[0]:
                model.partial_fit(X)
        print(f"pass {p + 1}/{passes} done, {time.time() - started:.0f}s", flush=True)
    joblib.dump(model, out)
    words = vec.get_feature_names_out()
    beta = model.components_ / model.components_.sum(axis=1, keepdims=True)
    pd.DataFrame([dict(topic=t + 1, top_terms=", ".join(words[np.argsort(beta[t])[::-1][:15]])) for t in range(k)]).to_csv(
        RESULTS / f"topics_k{k}.csv", index=False)
    (RESULTS / f"training_k{k}.json").write_text(json.dumps(dict(k=k, passes=passes, seconds=time.time() - started,
                                                                    train_chunks=prep["counts"]["train_chunks"]), indent=2))
    return model


def Infer(k):
    """Topic weights for every retained chunk (train + test), then token-weighted meeting-level shares."""
    vec = joblib.load(wf.OUTDIR / "vectorizer.joblib")
    model = joblib.load(wf.OUTDIR / f"lda_k{k}.joblib")
    cols = [f"t{j + 1}" for j in range(k)]
    parts = []
    for frame in Frames():
        X = vec.transform(frame.text)
        df = frame.drop(columns=["text", "start_seconds", "end_seconds"]).reset_index(drop=True)
        df["modeled_tokens"] = np.asarray(X.sum(axis=1)).ravel()
        parts.append(pd.concat([df, pd.DataFrame(model.transform(X), columns=cols)], axis=1))
    chunks = pd.concat(parts, ignore_index=True)
    chunks = chunks.loc[chunks.modeled_tokens > 0].reset_index(drop=True)   # empty rows would get a uniform mixture
    chunks.to_parquet(RESULTS / f"theta_chunks_k{k}.parquet", index=False)

    w = chunks[cols].mul(chunks.modeled_tokens, axis=0)
    w[["id", "leaid", "year", "split"]] = chunks[["id", "leaid", "year", "split"]]
    meetings = w.groupby(["id", "leaid", "year", "split"], as_index=False)[cols].sum()
    meetings["modeled_tokens"] = meetings[cols].sum(axis=1)
    meetings[cols] = meetings[cols].div(meetings.modeled_tokens, axis=0)
    meetings.to_parquet(RESULTS / f"theta_meetings_k{k}.parquet", index=False)
    print(f"{len(chunks):,} chunks, {len(meetings):,} meetings, {meetings.leaid.nunique():,} districts", flush=True)
    return meetings


if __name__ == "__main__":
    Main()
