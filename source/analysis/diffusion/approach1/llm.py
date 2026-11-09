"""OpenAI helper with a hard spend cap: every call's actual usage is appended to a ledger and the cap is checked
before each call. Prices/models follow source/analysis/gabriel/run_gabriel_stance.py."""
import asyncio
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_OUT

MODELS = {"gpt-6-luna": (0.10, 0.50), "gpt-6.1-terra": (1.00, 5.00)}   # $ per 1M tokens in / out
EFFORT = "none"            # lowest reasoning effort both models accept (checked 2026-11-09)
CAP = 20.00                # hard cap ($) for this task, all calls
STOP_AT = 18.00            # stop issuing calls once the ledger reaches this
LEDGER = DATA_OUT / "llm_ledger.csv"
FIELDS = ["time", "task", "kw", "item", "model", "input_tokens", "output_tokens", "cost"]


def spent():
    if not LEDGER.exists():
        return 0.0
    with open(LEDGER) as f:
        return sum(float(r["cost"]) for r in csv.DictReader(f))


class Client:
    def __init__(self, concurrency=16):
        from openai import AsyncOpenAI
        self.c = AsyncOpenAI(timeout=120, max_retries=3)
        self.sem = asyncio.Semaphore(concurrency)
        self.total = spent()
        new = not LEDGER.exists()
        self.f = open(LEDGER, "a", newline="")
        self.w = csv.DictWriter(self.f, FIELDS)
        if new:
            self.w.writeheader()

    async def json_call(self, model, system, user, schema, task, kw, item):
        async with self.sem:
            if self.total >= STOP_AT:
                raise RuntimeError(f"spend ${self.total:.2f} reached the stop level ${STOP_AT} (cap ${CAP})")
            r = await self.c.responses.create(
                model=model, reasoning={"effort": EFFORT}, instructions=system, input=user,
                text={"format": {"type": "json_schema", "name": "label", "schema": schema, "strict": True}})
            pin, pout = MODELS[model]
            cost = r.usage.input_tokens / 1e6 * pin + r.usage.output_tokens / 1e6 * pout
            self.total += cost
            self.w.writerow(dict(time=time.strftime("%H:%M:%S"), task=task, kw=kw, item=item, model=model,
                                 input_tokens=r.usage.input_tokens, output_tokens=r.usage.output_tokens, cost=cost))
            self.f.flush()
            return json.loads(r.output_text)

    def close(self):
        self.f.close()
