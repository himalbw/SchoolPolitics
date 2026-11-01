"""Rate excerpts with GABRIEL (OpenAI API; costs money).

Run from the repo root, with OPENAI_API_KEY set:
  micromamba run -n stat236 python source/analysis/gabriel/run_gabriel.py models   # confirm model id + reasoning setting
  micromamba run -n stat236 python source/analysis/gabriel/run_gabriel.py pilot    # 200 split-A excerpts, rated twice
  micromamba run -n stat236 python source/analysis/gabriel/run_gabriel.py report   # stability, distributions, cost
  micromamba run -n stat236 python source/analysis/gabriel/run_gabriel.py full     # all excerpts, one pass -> scores_full
  micromamba run -n stat236 python source/analysis/gabriel/run_gabriel.py terra    # 1,000-excerpt check -> scores_terra
  ... run_gabriel.py extend         # score excerpts_extension (2020, 2021 top-up, 2024), append to scores_full
  ... run_gabriel.py public_comment # public-comment share for every excerpt -> public_comment_share
  ... run_gabriel.py extra_random   # score excerpts_extra_random, append to scores_full
Every paid step of the 2026-10-28 session appends its actual cost (API usage fields) to LEDGER and refuses to start once
the ledger total passes SESSION_CAP. Paid calls use the key in ~/.openai_key_5eeu (run with
OPENAI_API_KEY=$(cat ~/.openai_key_5eeu)); `keycheck` makes one tiny call and prints the key suffix and organization.
The first ~$4.40 of the session went to a different key by mistake (session_costs_nwaHuy.csv).
Pilot = 100 random + 100 flagged split-A excerpts (seed 236), random balanced by period, flagged balanced across
lexicon category x period. One rate() call per excerpt with all ATTRIBUTES; GABRIEL's default template returns JSON only.
GABRIEL retries failed calls itself (max_retries=3 by default). Stance = conservative_positions - progressive_positions.
"""
import asyncio
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attributes import ATTRIBUTES, VERSION  # noqa: E402

OUTDIR = Path("Datastore/DistrictView/derived/gabriel/")
SEED = 236
MODEL = "gpt-6-luna"
EFFORTS = ["none", "minimal", "low"]   # lowest first; the first one the model accepts is used
N_PARALLELS = 10
N_RANDOM, N_FLAGGED = 100, 100
RUNS = [1, 2]
PRICE_IN, PRICE_OUT = 0.10, 0.50   # $ per 1M tokens, gpt-6-luna (OpenAI announcement 2026-09-22; GABRIEL's table is stale)
EST_PROMPT_TOKENS, EST_OUT_TOKENS = 608, 300   # per call, as assumed in build_excerpts.py
FULL_EXCERPTS = 10_879
TERRA_MODEL = "gpt-6.1-terra"
TERRA_PRICE_IN, TERRA_PRICE_OUT = 1.00, 5.00   # $ per 1M tokens, gpt-6.1-terra (confirmed by Lillian 2026-10-27)
N_TERRA = 1000
SESSION_CAP = 8.50   # $ of further spend on the 5eeu key (Lillian's $20 total cap across sessions, ~$10.80 already spent)
LEDGER = "session_costs_5eeu.csv"
EXTRA_RANDOM_YEARS = [2021, 2023]
PRIOR_ROWS = {"public_comment": 2000, "extra_random": 1000}   # first rows of these save dirs were billed to nwaHuy   # extra random excerpts scored only for these years (budget)
PUBLIC_COMMENT = {
    "public_comment_share": (
        "Share of this excerpt that is public comment by members of the public speaking to the board, as opposed to "
        "board member or staff discussion. Cues: \"state your name and address\", \"three minutes\", \"next speaker\", "
        "\"public comment\", a speaker introducing themselves as a parent or resident. "
        "0 = no public comment; 100 = the whole excerpt is public comment."
    ),
}


def Main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "keycheck":
        KeyCheck()
    elif mode == "models":
        CheckModel(MODEL, "pilot_model.txt")
        CheckModel(TERRA_MODEL, "terra_model.txt")
    elif mode == "pilot":
        RunPilot()
    elif mode == "report":
        Report()
    elif mode == "repilot":
        Repilot()
    elif mode == "full":
        RunFull()
    elif mode == "terra":
        RunTerra()
    elif mode == "extend":
        RunAppend("excerpts_extension.parquet", "extension")
    elif mode == "extra_random":
        RunAppend("excerpts_extra_random.parquet", "extra_random")
    elif mode == "public_comment":
        RunPublicComment()
    else:
        sys.exit(__doc__)
    return


def Repilot():
    # Rate the 200 pilot excerpts once with the current attributes and compare with the v2 pilot (run 1).
    model, effort = (OUTDIR / "pilot_model.txt").read_text().split()
    sample = PilotSample()
    save_dir = OUTDIR / f"pilot_{VERSION}"
    new = asyncio.run(RateOnce(sample, model, effort, save_dir))
    new.to_parquet(OUTDIR / f"pilot_{VERSION}.parquet", index=False)
    old = pd.read_parquet(OUTDIR / "pilot_run1.parquet")
    old["stance"] = Stance(old)
    kinds = sample.set_index("excerpt_id").excerpt_type
    cols = list(ATTRIBUTES) + ["stance"]
    means = pd.concat({"v2": old.set_index("excerpt_id")[cols], VERSION: new.set_index("excerpt_id")[cols]}, axis=1)
    print(f"Mean scores by excerpt type, v2 pilot run 1 vs {VERSION} (same 200 excerpts):")
    print(means.groupby(kinds).mean().T.unstack(0).round(1).to_string())
    for x in ["h2uNRx7jglg_0", "soM9kXBemSk_0"]:
        print(f"\n{x}:")
        print(means.loc[x].unstack(0).round(0).to_string())
    print("\n" + Usage(save_dir, PRICE_IN, PRICE_OUT))
    return


def CheckModel(model, out_name):
    # Exact id must be in the account's model list; then find the lowest reasoning effort it accepts (tiny calls).
    from openai import OpenAI
    client = OpenAI()
    ids = sorted(m.id for m in client.models.list())
    if model not in ids:
        sys.exit(f"{model} is not in this account's model list. Similar ids: "
                 f"{[i for i in ids if 'luna' in i or 'gpt-6' in i or 'terra' in i]}")
    for effort in EFFORTS:
        try:
            client.responses.create(model=model, input="Reply with OK.", reasoning={"effort": effort})
        except Exception as e:  # noqa: BLE001 - any rejection means try the next setting
            print(f"reasoning effort {effort!r} rejected: {str(e)[:120]}")
            continue
        print(f"Model {model} found; lowest accepted reasoning effort: {effort!r}")
        (OUTDIR / out_name).write_text(f"{model}\n{effort}\n")
        return model, effort
    sys.exit("No reasoning effort setting accepted")


async def RateOnce(df, model, effort, save_dir):
    import gabriel
    rated = await gabriel.rate(df[["excerpt_id", "text"]], "text", attributes=ATTRIBUTES, save_dir=str(save_dir),
                               model=model, n_parallels=N_PARALLELS, n_runs=1, reasoning_effort=effort,
                               reset_files=False)
    rated = rated.drop(columns="text")
    rated["stance"] = Stance(rated)
    return rated.assign(model=model, attributes_version=VERSION)


def Usage(save_dir, price_in, price_out):
    raw = pd.read_csv(Path(save_dir) / "ratings_raw_responses.csv")
    tok_in, tok_out, tok_reason = (int(raw[c].fillna(0).sum()) for c in ["Input Tokens", "Output Tokens", "Reasoning Tokens"])
    ok = int(raw.Successful.fillna(False).astype(bool).sum())
    line = f"API calls: {len(raw):,} ({ok:,} successful); {tok_in:,} input, {tok_out:,} output ({tok_reason:,} reasoning)"
    if price_in is None:
        return line + " -> cost: price not set"
    return line + f" -> ${tok_in / 1e6 * price_in + tok_out / 1e6 * price_out:.2f} at ${price_in}/M in, ${price_out}/M out"


def RunFull():
    model, effort = (OUTDIR / "pilot_model.txt").read_text().split()
    excerpts = pd.read_parquet(OUTDIR / "excerpts.parquet")
    save_dir = OUTDIR / f"full_{VERSION}"
    print(f"Full run: {len(excerpts):,} excerpts, {model}, reasoning {effort}, attributes {VERSION}")
    scores = asyncio.run(RateOnce(excerpts, model, effort, save_dir))
    scores.to_parquet(OUTDIR / "scores_full.parquet", index=False)
    print(f"Fully rated: {scores[list(ATTRIBUTES)].notna().all(axis=1).sum():,} of {len(scores):,}")
    print(Usage(save_dir, PRICE_IN, PRICE_OUT))
    return


def SessionCost(step=None, save_dir=None, price_in=PRICE_IN, price_out=PRICE_OUT):
    # Ledger of actual costs this session (API usage fields); with no step, check the cap before a paid call.
    path = OUTDIR / LEDGER
    ledger = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["step", "calls", "input_tokens",
                                                                           "output_tokens", "cost"])
    if step is None:
        if ledger.cost.sum() > SESSION_CAP:
            sys.exit(f"Session spend ${ledger.cost.sum():.2f} is past the ${SESSION_CAP:.0f} cap")
        return ledger.cost.sum()
    raw = pd.read_csv(Path(save_dir) / "ratings_raw_responses.csv").iloc[PRIOR_ROWS.get(step, 0):]
    tok_in, tok_out = int(raw["Input Tokens"].fillna(0).sum()), int(raw["Output Tokens"].fillna(0).sum())
    row = {"step": step, "calls": len(raw), "input_tokens": tok_in, "output_tokens": tok_out,
           "cost": tok_in / 1e6 * price_in + tok_out / 1e6 * price_out}
    ledger = pd.concat([ledger[ledger.step != step], pd.DataFrame([row])], ignore_index=True)
    ledger.to_csv(path, index=False)
    print(f"Step {step}: ${row['cost']:.2f}; session total ${ledger.cost.sum():.2f} (cap ${SESSION_CAP:.0f})")
    return ledger.cost.sum()


def KeyCheck():
    import os
    from openai import OpenAI
    raw = OpenAI().responses.with_raw_response.create(model=MODEL, input="Reply with OK.", reasoning={"effort": "none"})
    resp = raw.parse()
    print(f"Key ...{os.environ['OPENAI_API_KEY'][-4:]}; openai-organization: {raw.headers.get('openai-organization')}; "
          f"reply {resp.output_text!r}; usage {resp.usage.input_tokens} in / {resp.usage.output_tokens} out")
    return


def HalfYear(dates):
    return dates.dt.year.astype(str) + np.where(dates.dt.month <= 6, "H1", "H2")


def RunAppend(excerpt_file, batch):
    # Score new excerpts with the frozen attributes and append to scores_full with batch and half_year columns.
    SessionCost()
    model, effort = (OUTDIR / "pilot_model.txt").read_text().split()
    excerpts = pd.read_parquet(OUTDIR / excerpt_file)
    if batch == "extra_random":
        excerpts = excerpts[excerpts.year.isin(EXTRA_RANDOM_YEARS)]
    full = pd.read_parquet(OUTDIR / "scores_full.parquet")
    backup = OUTDIR / "scores_full_original.parquet"
    if not backup.exists():
        full.to_parquet(backup, index=False)
    if "batch" not in full:   # label the original rows
        dates = pd.read_parquet(OUTDIR / "excerpts.parquet", columns=["excerpt_id", "meeting_date"])
        full = full.merge(dates, on="excerpt_id", validate="one_to_one")
        full = full.assign(batch="original", half_year=HalfYear(full.meeting_date)).drop(columns="meeting_date")
    excerpts = excerpts[~excerpts.excerpt_id.isin(full.excerpt_id)]
    save_dir = OUTDIR / f"{batch}_{VERSION}"
    print(f"Scoring {len(excerpts):,} {batch} excerpts, {model}, reasoning {effort}, attributes {VERSION}")
    scores = asyncio.run(RateOnce(excerpts, model, effort, save_dir))
    scores = scores.merge(excerpts[["excerpt_id", "meeting_date"]], on="excerpt_id", validate="one_to_one")
    scores = scores.assign(batch=batch, half_year=HalfYear(scores.meeting_date)).drop(columns="meeting_date")
    print(f"Fully rated: {scores[list(ATTRIBUTES)].notna().all(axis=1).sum():,} of {len(scores):,}")
    full = pd.concat([full, scores[full.columns]], ignore_index=True)
    full.to_parquet(OUTDIR / "scores_full.parquet", index=False)
    print(f"scores_full.parquet: {full.batch.value_counts().to_dict()}")
    print(Usage(save_dir, PRICE_IN, PRICE_OUT))
    SessionCost(batch, save_dir)
    return


def RunPublicComment():
    # One-attribute pass over the original and extension excerpts (not the extra random ones: budget).
    SessionCost()
    model, effort = (OUTDIR / "pilot_model.txt").read_text().split()
    files = ["excerpts.parquet", "excerpts_extension.parquet"]
    excerpts = pd.concat([pd.read_parquet(OUTDIR / f, columns=["excerpt_id", "text"]) for f in files], ignore_index=True)
    save_dir = OUTDIR / "public_comment"
    print(f"Public-comment share: {len(excerpts):,} excerpts, {model}, reasoning {effort}")
    import gabriel
    rated = asyncio.run(gabriel.rate(excerpts, "text", attributes=PUBLIC_COMMENT, save_dir=str(save_dir),
                                     model=model, n_parallels=N_PARALLELS, n_runs=1, reasoning_effort=effort,
                                     reset_files=False))
    out = rated.drop(columns="text").assign(model=model)
    out.to_parquet(OUTDIR / "public_comment_share.parquet", index=False)
    print(f"Rated: {out.public_comment_share.notna().sum():,} of {len(out):,}; distribution:\n"
          + out.public_comment_share.describe().round(1).to_string())
    print(Usage(save_dir, PRICE_IN, PRICE_OUT))
    SessionCost("public_comment", save_dir)
    return


def TerraSample():
    # 1,000 excerpts: half random (balanced by period), half flagged (balanced across category x period).
    excerpts = pd.read_parquet(OUTDIR / "excerpts.parquet")
    parts = []
    for kind, cell in [("random", "period"), ("flagged", ["lexicon_category", "period"])]:
        pool = excerpts[excerpts.excerpt_type == kind]
        quota = Allocate(pool.groupby(cell).size(), N_TERRA // 2)
        parts += [g.sample(quota[k], random_state=SEED) for k, g in pool.groupby(cell)]
    return pd.concat(parts).reset_index(drop=True)


def RunTerra():
    model, effort = CheckModel(TERRA_MODEL, "terra_model.txt")
    sample = TerraSample()
    save_dir = OUTDIR / f"terra_{VERSION}"
    print(f"Terra check: {len(sample):,} excerpts, {model}, reasoning {effort}, attributes {VERSION}")
    scores = asyncio.run(RateOnce(sample, model, effort, save_dir))
    scores.to_parquet(OUTDIR / "scores_terra.parquet", index=False)
    luna = pd.read_parquet(OUTDIR / "scores_full.parquet")
    both = scores.merge(luna, on="excerpt_id", suffixes=("_terra", "_luna"))
    rows = [(a, both[f"{a}_terra"].corr(both[f"{a}_luna"]), (both[f"{a}_terra"] - both[f"{a}_luna"]).mean())
            for a in list(ATTRIBUTES) + ["stance"]]
    print(f"Terra vs Luna on the same {len(both):,} excerpts:")
    print(pd.DataFrame(rows, columns=["attribute", "corr", "mean_diff_terra_minus_luna"]).round(3).to_string(index=False))
    print(Usage(save_dir, TERRA_PRICE_IN, TERRA_PRICE_OUT))
    return


def Allocate(sizes, n):
    # Spread n as evenly as possible over cells, small cells taken whole (same water-filling as the district sample).
    quota, left = {}, n
    for i, (cell, size) in enumerate(sizes.sort_values().items()):
        quota[cell] = min(size, left // (len(sizes) - i))
        left -= quota[cell]
    return quota


def PilotSample():
    path = OUTDIR / "pilot_sample.parquet"
    if path.exists():
        return pd.read_parquet(path)
    e = pd.read_parquet(OUTDIR / "excerpts.parquet")
    e = e[e.split == "A"]
    parts = []
    for kind, n, cell in [("random", N_RANDOM, "period"), ("flagged", N_FLAGGED, ["lexicon_category", "period"])]:
        pool = e[e.excerpt_type == kind]
        quota = Allocate(pool.groupby(cell).size(), n)
        parts += [g.sample(quota[k], random_state=SEED) for k, g in pool.groupby(cell)]
    sample = pd.concat(parts).reset_index(drop=True)
    sample.to_parquet(path, index=False)
    return sample


def RunPilot():
    import gabriel
    model_file = OUTDIR / "pilot_model.txt"
    if not model_file.exists():
        sys.exit("Run the 'models' step first to confirm the model id and reasoning setting.")
    model, effort = model_file.read_text().split()
    sample = PilotSample()
    print(f"Pilot: {len(sample)} excerpts ({sample.excerpt_type.value_counts().to_dict()}), model {model}, "
          f"reasoning {effort}, attributes {VERSION}")
    asyncio.run(RateRuns(gabriel, sample, model, effort))   # one event loop for all runs (a second asyncio.run
    return                                                  # leaves GABRIEL's client on a closed loop)


async def RateRuns(gabriel, sample, model, effort):
    for run in RUNS:
        rated = await gabriel.rate(sample[["excerpt_id", "text"]], "text", attributes=ATTRIBUTES,
                                   save_dir=str(OUTDIR / f"pilot_run{run}"), model=model, n_parallels=N_PARALLELS,
                                   n_runs=1, reasoning_effort=effort, reset_files=False)
        rated.drop(columns="text").assign(run=run, model=model, attributes_version=VERSION).to_parquet(
            OUTDIR / f"pilot_run{run}.parquet", index=False)
        print(f"Run {run}: {rated[list(ATTRIBUTES)].notna().all(axis=1).sum()} of {len(rated)} excerpts fully rated")
    return


def Stance(df):
    return df.conservative_positions - df.progressive_positions


def Report():
    sample = PilotSample()
    runs = {r: pd.read_parquet(OUTDIR / f"pilot_run{r}.parquet") for r in RUNS}
    attrs = list(ATTRIBUTES)
    both = runs[1].merge(runs[2], on="excerpt_id", suffixes=("_1", "_2"))
    for d in (runs[1], runs[2]):
        d["stance"] = Stance(d)
    both["stance_1"], both["stance_2"] = Stance(runs[1]).values, Stance(runs[2]).values

    print("Run-to-run stability (n = %d excerpts):" % len(both))
    rows = [(a, both[f"{a}_1"].corr(both[f"{a}_2"]), (both[f"{a}_1"] - both[f"{a}_2"]).abs().mean())
            for a in attrs + ["stance"]]
    print(pd.DataFrame(rows, columns=["attribute", "corr", "mean_abs_diff"]).round(3).to_string(index=False))

    scores = runs[1].merge(sample.drop(columns="text"), on="excerpt_id")
    print("\nScore distributions by excerpt type (run 1):")
    print(scores.groupby("excerpt_type")[attrs + ["stance"]].describe(percentiles=[.25, .5, .75])
          .T.round(1).to_string())

    # Leaked school/place names: proper-looking words left next to place nouns after stripping.
    leak_re = r"\b(?!the\b|our\b|this\b|a\b|new\b)[A-Za-z]+\s+(?:High|Middle|Elementary|School|County|City|Academy|District)\b"
    scores = scores.merge(sample[["excerpt_id", "text"]], on="excerpt_id")
    scores["leak"] = scores.text.str.count(leak_re) > 0
    print(f"\nExcerpts with a possible leaked school/place name: {scores.leak.sum()} of {len(scores)}")
    print("Mean scores, leak vs no leak, within excerpt type (run 1):")
    print(scores.groupby(["excerpt_type", "leak"])[attrs + ["stance"]].mean().round(1).to_string())

    # 30 excerpts to read: 10 most conservative, 10 most progressive, 10 at random from the middle.
    scores = scores.sort_values("stance")
    middle = scores.iloc[10:-10].sample(10, random_state=SEED)
    show = pd.concat([scores.tail(10).assign(group="conservative"), scores.head(10).assign(group="progressive"),
                      middle.assign(group="middle")])
    with open(OUTDIR / "pilot_read_30.txt", "w") as f:
        for r in show.itertuples():
            f.write(f"===== {r.group} | {r.excerpt_id} | {r.excerpt_type} {r.lexicon_category or ''} | {r.period} | "
                    f"GOP {r.gop_2p_2020:.2f}\n"
                    + " ".join(f"{a}={getattr(r, a):.0f}" for a in attrs) + f" | stance={r.stance:.0f}\n\n{r.text}\n\n")
    print(f"\n30 excerpts with scores written to {OUTDIR / 'pilot_read_30.txt'}")

    # Actual usage from the API responses (both runs) vs the pre-run estimate, both at PRICE_IN / PRICE_OUT.
    raw = pd.concat([pd.read_csv(OUTDIR / f"pilot_run{r}" / "ratings_raw_responses.csv") for r in RUNS])
    tok_in, tok_out, tok_reason = (int(raw[c].fillna(0).sum()) for c in ["Input Tokens", "Output Tokens", "Reasoning Tokens"])
    actual = tok_in / 1e6 * PRICE_IN + tok_out / 1e6 * PRICE_OUT
    est_in = len(RUNS) * (int(sample.n_tokens.sum()) + len(sample) * EST_PROMPT_TOKENS)
    est_out = len(RUNS) * len(sample) * EST_OUT_TOKENS
    estimate = est_in / 1e6 * PRICE_IN + est_out / 1e6 * PRICE_OUT
    print(f"\nAPI calls: {len(raw)} ({int(raw.Successful.fillna(False).astype(bool).sum())} successful)")
    print(f"Actual (API usage fields): {tok_in:,} input, {tok_out:,} output (of which {tok_reason:,} reasoning) "
          f"-> ${actual:.3f}")
    print(f"Pre-run estimate:          {est_in:,} input, {est_out:,} output -> ${estimate:.3f}")
    print(f"Prices: ${PRICE_IN}/M input, ${PRICE_OUT}/M output (gpt-6-luna, OpenAI announcement 2026-09-22). "
          f"Scaled to the full {FULL_EXCERPTS:,} excerpts at the actual per-call usage: "
          f"${actual / len(raw) * FULL_EXCERPTS:.2f} per run.")
    return


if __name__ == "__main__":
    Main()
