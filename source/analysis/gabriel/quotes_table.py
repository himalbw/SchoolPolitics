"""Appendix table of the verified Indian River County (FL) quotes from quotes.py.

Reads  output/tables/quotes_1200930.csv
Writes output/tables/quotes_1200930.tex (one-page table of SELECT quotes, sorted by meeting date). Run from the repo root:
  python3 source/analysis/gabriel/quotes_table.py
"""
from pathlib import Path

import pandas as pd

INFILE = Path("output/tables/quotes_1200930.csv")
OUTFILE = Path("output/tables/quotes_1200930.tex")
ATTR = {"conservative_positions": "Cons.", "hostility": "Host."}
# One-page selection for the appendix: the three quotes cited in the text plus a spread across dates and disputes
SELECT = ["i'm here to ask you to please make masks", "but for you to force behaviors", "if you and the board go through",
          "we do however oppose crt", "i cannot and i will not support", "sacrificing children's mental",
          "i also know that there's a parent's bill", "it's not too late to rescind", "this board broke the law",
          "I thank you all in advance", "I respectfully request that you not approve", "boys whatever their inclination"]

def Main():
    d_all = pd.read_csv(INFILE)
    n_pc, n_scored = (d_all.public_comment_share >= 50).sum(), d_all.public_comment_share.notna().sum()
    keep = d_all.quote.apply(lambda q: any(q.startswith(x) for x in SELECT))
    assert keep.sum() == len(SELECT), "a selected quote was not found"
    d = d_all[keep].sort_values(["meeting_date", "attribute", "score"], ascending=[True, True, False])
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             rf"\caption{{Selected verified quotes from Indian River County, Florida (county Republican share 0.61), "
             rf"{len(d)} of {len(d_all)}. For the 25 excerpts scoring highest on conservative positions (Cons.) and the 10 "
             r"highest on hostility (Host.), \texttt{gpt-6.1-terra} returned the single sentence that most strongly argues "
             r"the position or is most hostile; a quote is kept only if it is an exact substring of the raw caption window. "
             r"Score is the excerpt's 0--100 score on that attribute; PC is its public-comment share (blank = not scored). "
             rf"Of all {n_scored} scored quotes, {n_pc} are mostly public comment (PC $\geq$ 50). Names are replaced by "
             r"[NAME]; caption errors are preserved. The full list is in \texttt{output/tables/quotes\_1200930.csv}.}",
             r"\label{tab:quotes}", r"\begin{tabularx}{\linewidth}{@{}l l r r X@{}}", r"\toprule",
             r"Date & Attr. & Score & PC & Quote \\", r"\midrule"]
    for r in d.itertuples():
        pc = "" if pd.isna(r.public_comment_share) else f"{r.public_comment_share:.0f}"
        lines.append(rf"{r.meeting_date} & {ATTR[r.attribute]} & {r.score:.0f} & {pc} & ``{Escape(r.quote)}'' \\[2pt]")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    OUTFILE.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUTFILE} ({len(d)} of {len(d_all)} quotes)")
    return

def Escape(s):
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"), ('"', "''")]:
        s = s.replace(a, b)
    return s

if __name__ == "__main__":
    Main()
