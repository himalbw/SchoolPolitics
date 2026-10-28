"""Run one GABRIEL task (defined in prompts.py) on the GABRIEL transcript sample.

Reads  Datastore/DistrictView/derived/gabriel_sample/sample.parquet   (built by source/derived/sb_transcripts/build_gabriel_sample.py)
Writes Datastore/DistrictView/derived/gabriel/{task}_{model}[_{tag}]/  (GABRIEL checkpoints + raw responses)
       .../results.parquet                                             (id, leaid, year, state + GABRIEL output columns, no text)

Needs `pip install openai-gabriel` and OPENAI_API_KEY in the environment. GABRIEL resumes from its checkpoints in the
save dir, so a crashed run can be restarted with the same command; change --tag after editing a definition in prompts.py.
--pilot N runs on a stratified subset (about N transcripts spread over state x year, same draw every time) to compare
models and check the definitions before the full run.
Run from the repo root, e.g.:
  python3 source/analysis/gabriel/run_gabriel.py meeting_type --pilot 200
  python3 source/analysis/gabriel/run_gabriel.py conflict --pilot 200 --model gpt-5.6-terra
  python3 source/analysis/gabriel/run_gabriel.py conflict
"""
import argparse
import asyncio
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import TASKS

INDIR_SAMPLE = Path("Datastore/DistrictView/derived/gabriel_sample/sample.parquet")
OUTDIR_GABRIEL = Path("Datastore/DistrictView/derived/gabriel/")
ID_COLS = ["id", "leaid", "year", "state"]
PILOT_SEED = 236
TOKENS_PER_WORD = 1.3  # rough, for the size printout only

def Main():
    args = ParseArgs()
    task = TASKS[args.task]
    df = pd.read_parquet(INDIR_SAMPLE, columns=ID_COLS + ["n_words", "text"])
    if args.pilot:
        df = PilotSubset(df, args.pilot)
    if task.get("max_words"):
        df["text"] = [" ".join(t.split()[:task["max_words"]]) for t in df.text]
    save_dir = OUTDIR_GABRIEL / "_".join(filter(None, [args.task, args.model, f"pilot{args.pilot}" if args.pilot else "", args.tag]))
    n_tokens = sum(len(t.split()) for t in df.text) * TOKENS_PER_WORD
    print(f"{args.task} ({task['fn']}) with {args.model}: {len(df):,} transcripts, ~{n_tokens / 1e6:,.1f}M input tokens -> {save_dir}")
    df_out = asyncio.run(RunTask(df[ID_COLS + ["text"]], task, args, save_dir))
    df_out.drop(columns="text", errors="ignore").to_parquet(save_dir / "results.parquet", index=False)
    print(f"Saved {len(df_out):,} rows to {save_dir / 'results.parquet'}")
    return

def ParseArgs():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--pilot", type=int, default=0, help="run on about this many transcripts (0 = full sample)")
    parser.add_argument("--n-runs", type=int, default=1, help="repeat ratings to measure model noise (rate/classify)")
    parser.add_argument("--reasoning-effort", default=None, choices=[None, "none", "low", "medium", "high"])
    parser.add_argument("--tag", default="", help="suffix for the save dir, e.g. v2 after editing prompts.py")
    return parser.parse_args()

def PilotSubset(df, n):
    df_groups = df.groupby(["state", "year"])
    per_group = math.ceil(n / df_groups.ngroups)
    return df.sample(frac=1, random_state=PILOT_SEED).groupby(["state", "year"]).head(per_group).reset_index(drop=True)

async def RunTask(df, task, args, save_dir):
    import gabriel
    save_dir.mkdir(parents=True, exist_ok=True)
    common = dict(save_dir=str(save_dir), model=args.model, additional_instructions=task["instructions"],
                  reasoning_effort=args.reasoning_effort)
    if task["fn"] == "rate":
        return await gabriel.rate(df, "text", attributes=task["attributes"], n_runs=args.n_runs, **common)
    if task["fn"] == "classify":
        return await gabriel.classify(df, "text", labels=task["labels"], n_runs=args.n_runs, **common)
    if task["fn"] == "codify":
        return await gabriel.codify(df, "text", categories=task["categories"], **common)
    raise ValueError(f"unknown GABRIEL function {task['fn']}")

if __name__ == "__main__":
    Main()
