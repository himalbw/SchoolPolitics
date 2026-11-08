"""Appendix table: for each dictionary theme, its most frequent words and phrases across the whole screened corpus.

Single terms are counted from the district-quarter matrix, phrases from phrase_counts.parquet (count_phrases.py).
Rates are mentions per 10,000 words of the full corpus. Lexicon terms cut from the matrix (too rare) are reported as absent.
Writes output/tables/{BOW_OUTPUT_TAG}/lexicon_top_terms.{csv,tex}. Run from the repo root:
  python3 source/analysis/bow/lexicon_table.py
"""
import numpy as np
import pandas as pd

from bow_data import INDIR_PHRASES, LoadDQ, OUTDIR_TABLES
from lexicon import LEXICON

N_TOP = 10
LABELS = {
    "parental_rights": "Parental rights", "book_content": "Book content", "crt_race": "CRT / race",
    "gender_lgbtq": "Gender / LGBTQ", "school_choice": "School choice", "covid_mandates": "COVID mandates",
    "equity_sel": "Equity / SEL", "guns_safety": "Guns / safety", "unions_pay": "Unions / pay",
    "immigration": "Immigration", "religion": "Religion", "conflict_legal": "Conflict (legal)",
    "meeting_conflict": "Meeting conflict", "ctrl_budget": "Budget (control)", "ctrl_facilities": "Facilities (control)",
    "ctrl_athletics": "Athletics (control)", "ctrl_academics": "Academics (control)",
}

def Main():
    dtm, vocab, units = LoadDQ()
    n_words = units.n_words.sum()
    term_counts = pd.Series(np.asarray(dtm.sum(0)).ravel(), index=vocab)
    phrase_counts = pd.read_parquet(INDIR_PHRASES).drop(columns="file").sum()
    rows = []
    for cat, spec in LEXICON.items():
        for term in spec["terms"]:
            rows.append((cat, term, "term", term_counts.get(term, np.nan)))
        for phrase in spec["phrases"]:
            rows.append((cat, phrase, "phrase", phrase_counts.get(phrase, np.nan)))
    df = pd.DataFrame(rows, columns=["category", "entry", "kind", "count"])
    df["per_10k"] = df["count"] / n_words * 1e4
    df = df.sort_values(["category", "count"], ascending=[True, False])
    df.to_csv(OUTDIR_TABLES / "lexicon_top_terms.csv", index=False)
    print(f"Corpus: {n_words:,.0f} words. Absent (cut as rare): {df['count'].isna().sum()} entries; "
          f"zero count: {(df['count'] == 0).sum()}")
    WriteTex(df, n_words)
    return

def WriteTex(df, n_words):
    lines = [r"\begin{landscape}", r"\small", r"\setlength{\tabcolsep}{5pt}", r"\setlength{\LTcapwidth}{\linewidth}",
             r"\begin{longtable}{@{}p{3.3cm}p{17.2cm}r@{}}",
             r"\caption{Most frequent words and phrases in each dictionary theme, 2018--2023. Entries are ranked from most "
             r"to least frequent; numbers in parentheses are mentions per 10,000 words across the full screened corpus "
             rf"({n_words / 1e9:.2f} billion words). Phrases are counted in the cleaned text and single words in the "
             r"document-term matrix, so a phrase's words are also counted as single words. The last column gives the "
             r"number of entries in the theme's list.}\label{tab:lexicon-terms}\\",
             r"\toprule", r"Theme & Top entries (mentions per 10k words) & Entries \\", r"\midrule", r"\endfirsthead",
             r"\multicolumn{3}{@{}l}{\emph{\Cref{tab:lexicon-terms}, continued}}\\", r"\toprule",
             r"Theme & Top entries (mentions per 10k words) & Entries \\", r"\midrule", r"\endhead",
             r"\midrule \multicolumn{3}{r@{}}{\emph{continued on next page}}\\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot"]
    for cat in LABELS:
        d = df[(df.category == cat) & (df["count"] > 0)].head(N_TOP)
        items = ", ".join(f"{e} ({FormatRate(r)})" for e, r in zip(d.entry, d.per_10k))
        lines.append(rf"{LABELS[cat]} & {items} & {(df.category == cat).sum()} \\[3pt]")
    lines += [r"\end{longtable}", r"\end{landscape}"]
    (OUTDIR_TABLES / "lexicon_top_terms.tex").write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUTDIR_TABLES / 'lexicon_top_terms.tex'}")
    return

def FormatRate(r):
    return f"{r:.2f}" if r >= 0.01 else f"{r:.3f}" if r >= 0.001 else "<0.001"

if __name__ == "__main__":
    Main()
