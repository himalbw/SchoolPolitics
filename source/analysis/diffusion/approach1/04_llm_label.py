"""Step 4: LLM labelling of keyword hits (OpenAI; costs money, hard cap in llm.py).

  python3 .../04_llm_label.py check            # 40 hand-coded indoctrination hits, gpt-6-luna and gpt-6.1-terra
  python3 .../04_llm_label.py label [model]    # (a) every hit in a talking-point cluster + (b) 150 random hits / kw
  python3 .../04_llm_label.py pairs [model]    # borderline pairs just below tau: same talking point?
Each hit is sent with its +-250-word window (re-read from the transcript), keyword marked with << >>.
Outputs: Datastore/.../keywords/{kw}/llm_hits_{model}.parquet, llm_pairs_{model}.parquet; ledger llm_ledger.csv.
"""
import asyncio
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_KW, KEYWORDS, SEED, TAB_OUT, TRANSCRIPTS, WIDE_HALF
from llm import Client, spent

N_RANDOM = 150
N_PAIRS = 200          # borderline pairs per keyword
PAIR_HALF = 40

TOPIC = {   # statement the stance is measured against (pro = agrees)
    "indoctrination": "Schools or teachers are indoctrinating students.",
    "mask": "Students and staff should be required to wear masks at school.",
    "crt": "Critical race theory (or what is called CRT) is acceptable to teach or discuss in schools.",
    "parental_rights": "Parents' rights over curriculum and school decisions should be asserted or expanded "
                       "(the parental-rights position).",
    "groom": "Schools, teachers or school materials are grooming children (sexual-grooming accusation).",
    "pornography": "Some school library books or materials are pornographic and should be removed.",
    "equity": "The district should pursue equity initiatives (racial or educational equity).",
    "vaccine": "Vaccination (e.g. against COVID-19) should be required or encouraged for students or staff.",
    "transgender": "Transgender students' rights and inclusion (names, pronouns, facilities, sports) should be "
                   "supported.",
}
SYSTEM = (
    "You label excerpts of automatic captions from US school board meetings (lowercase, no punctuation, no "
    "speaker labels, caption errors). One keyword occurrence is marked with << >>. Judge the speaker of the "
    "marked words only, using the whole excerpt for context.\n"
    "speaker_role: public_commenter (a member of the public, parent, resident, teacher or student speaking during "
    "public comment / audience participation), board_member, staff_administrator (superintendent, staff, "
    "presenter, attorney), student (student board representative or student report, not public comment), "
    "reading_official_text (reading aloud policy, statute, resolution, agenda or procedural rules), unclear.\n"
    "in_public_comment_period: yes if the marked words fall inside a public comment / public participation "
    "segment, no if not, unclear.\n"
    "is_quoting: true if the marked words are part of text quoted or read from another source; quote_source then "
    "names it: statute_policy, dictionary, news_media, advocacy_group, other (none if not quoting).\n"
    "stance: agree / disagree / neutral with the topic statement given, as expressed by the speaker (neutral if "
    "procedural, informational or unclear).")
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["speaker_role", "in_public_comment_period", "is_quoting", "quote_source", "stance"],
          "properties": {
              "speaker_role": {"type": "string", "enum": ["public_commenter", "board_member", "staff_administrator",
                                                          "student", "reading_official_text", "unclear"]},
              "in_public_comment_period": {"type": "string", "enum": ["yes", "no", "unclear"]},
              "is_quoting": {"type": "boolean"},
              "quote_source": {"type": "string", "enum": ["none", "statute_policy", "dictionary", "news_media",
                                                          "advocacy_group", "other"]},
              "stance": {"type": "string", "enum": ["agree", "disagree", "neutral"]}}}
PAIR_SYSTEM = (
    "You compare two excerpts of automatic captions (lowercase, caption errors) from school board meetings in "
    "DIFFERENT districts. Each contains a keyword marked with << >>. Decide whether the two marked passages make "
    "the SAME specific point in recognisably shared wording or phrasing (a shared talking point, slogan, quote or "
    "recited text), allowing paraphrase and caption errors. Merely discussing the same topic, or sharing stock "
    "phrases (\"thank you\", \"our students\"), is not enough.")
PAIR_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["same_talking_point", "kind"],
               "properties": {"same_talking_point": {"type": "boolean"},
                              "kind": {"type": "string", "enum": ["verbatim_or_near", "paraphrase", "same_topic_only",
                                                                  "unrelated"]}}}


def windows(df, half=WIDE_HALF):
    out = {}
    for f, g in df.groupby("file"):
        w = gzip.open(TRANSCRIPTS / f, "rt").read().split()
        for r in g.itertuples():
            a = w[max(0, r.pos - half): r.pos]
            k = w[r.pos: r.pos + r.kw_len]
            b = w[r.pos + r.kw_len: r.pos + r.kw_len + half]
            out[r.hit_id] = " ".join(a + ["<<" + " ".join(k) + ">>"] + b)
    return df.hit_id.map(out)


def user_msg(kw, text):
    return f"Topic statement for stance: {TOPIC[kw]}\n\nExcerpt:\n{text}"


async def label_hits(client, kw, df, model, task):
    df = df.assign(text=windows(df))

    async def one(r):
        try:
            return r.hit_id, await client.json_call(model, SYSTEM, user_msg(kw, r.text), SCHEMA, task, kw, r.hit_id)
        except RuntimeError:
            raise
        except Exception as e:  # noqa: BLE001
            return r.hit_id, {"error": str(e)[:200]}
    res = await asyncio.gather(*(one(r) for r in df.itertuples()))
    return pd.DataFrame([dict(hit_id=h, model=model, **v) for h, v in res])


def selection(kw):
    h = pd.read_parquet(DATA_KW(kw) / "hits.parquet")
    cl = pd.read_parquet(DATA_KW(kw) / "clusters.parquet")
    rnd = h.sample(min(N_RANDOM, len(h)), random_state=SEED).hit_id
    h["in_cluster"] = h.hit_id.isin(cl.hit_id)
    h["in_random"] = h.hit_id.isin(rnd)
    return h[h.in_cluster | h.in_random]


async def run_check():
    hand = pd.read_csv(TAB_OUT / "hand_check_public_comment.csv")
    h = pd.read_parquet(DATA_KW("indoctrination") / "hits.parquet")
    df = h[h.hit_id.isin(hand.hit_id)]
    client = Client()
    try:
        outs = [await label_hits(client, "indoctrination", df, m, "check") for m in ("gpt-6-luna", "gpt-6.1-terra")]
    finally:
        client.close()
    res = pd.concat(outs).merge(hand[["hit_id", "hand_label", "flag_public_comment", "flag_public_comment_loose"]],
                                on="hit_id")
    res.to_csv(TAB_OUT / "llm_check_vs_hand.csv", index=False)
    clear = res[res.hand_label != "ambiguous"]
    for m, g in clear.groupby("model"):
        llm_pc = g.speaker_role.eq("public_commenter")
        hand_pc = g.hand_label.eq("public_comment")
        print(f"{m}: agreement with hand (public commenter vs not) {np.mean(llm_pc == hand_pc):.0%} "
              f"(n={len(g)}); LLM pc share {llm_pc.mean():.0%}, hand {hand_pc.mean():.0%}")
        print(pd.crosstab(g.hand_label, g.speaker_role))
    print(f"spent so far ${spent():.3f}")


async def run_label(model):
    client = Client()
    try:
        for kw in KEYWORDS:
            out = DATA_KW(kw) / f"llm_hits_{model}.parquet"
            sel = selection(kw)
            done = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["hit_id"])
            if "error" in done:
                done = done[done.error.isna()]
            todo = sel[~sel.hit_id.isin(done.hit_id)]
            if len(todo):
                res = await label_hits(client, kw, todo, model, "label")
                done = pd.concat([done, res], ignore_index=True)
                done.to_parquet(out, index=False)
            print(f"{kw}: {len(sel)} selected ({sel.in_cluster.sum()} in clusters, {sel.in_random.sum()} random); "
                  f"labelled {len(done)}; spent ${client.total:.2f}", flush=True)
    finally:
        client.close()


async def run_pairs(model):
    rng = np.random.default_rng(SEED)
    client = Client()
    try:
        for kw in KEYWORDS:
            h = pd.read_parquet(DATA_KW(kw) / "hits.parquet")
            E = pd.read_parquet(DATA_KW(kw) / "edges.parquet")
            linked = (E.lex >= 80) | (E.sh5f >= 4)
            bord = E[~linked & (E.sh5f.between(2, 3) | E.lex.between(65, 79))]
            if len(bord) > N_PAIRS:
                bord = bord.iloc[rng.choice(len(bord), N_PAIRS, replace=False)]
            ids = pd.unique(np.r_[bord.hit_i, bord.hit_j])
            txt = windows(h[h.hit_id.isin(ids)], PAIR_HALF)
            txt = dict(zip(h[h.hit_id.isin(ids)].hit_id, txt))

            async def one(r):
                u = f"Excerpt A:\n{txt[r.hit_i]}\n\nExcerpt B:\n{txt[r.hit_j]}"
                try:
                    v = await client.json_call(model, PAIR_SYSTEM, u, PAIR_SCHEMA, "pairs", kw, f"{r.hit_i}-{r.hit_j}")
                except RuntimeError:
                    raise
                except Exception as e:  # noqa: BLE001
                    v = {"error": str(e)[:200]}
                return dict(hit_i=r.hit_i, hit_j=r.hit_j, lex=r.lex, sh5f=r.sh5f, **v)
            res = pd.DataFrame(await asyncio.gather(*(one(r) for r in bord.itertuples())))
            res.to_parquet(DATA_KW(kw) / f"llm_pairs_{model}.parquet", index=False)
            print(f"{kw}: {len(E[~linked & (E.sh5f.between(2, 3) | E.lex.between(65, 79))]):,} borderline pairs, "
                  f"{len(res)} judged, same talking point {res.same_talking_point.mean():.0%}; spent ${client.total:.2f}",
                  flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    mode = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-6-luna"
    if mode == "check":
        asyncio.run(run_check())
    elif mode == "label":
        asyncio.run(run_label(model))
    elif mode == "pairs":
        asyncio.run(run_pairs(model))
