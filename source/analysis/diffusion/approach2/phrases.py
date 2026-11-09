"""Step 1: the tracked talking-point phrases, with caption-spelling variants.

Text is the cleaned transcript: lowercase, punctuation -> space (hyphens too: "co-parent" -> "co parent"), in-word
apostrophes kept ("parent's"), edge apostrophes dropped ("parents'" -> "parents"). Each phrase is ONE regex (variants as
alternations) matched with word boundaries. `exclude` (optional) is a regex searched in a +-EXCL_WINDOW-character
window around each match; matches with such context are counted separately as procedural/false-positive hits and are
not used to define adoption.

source: S = hand seed (national talking point named in the project brief / advocacy & news coverage),
        N = data-driven (mine_candidates.py: 3-5-gram near-absent in 2018-19, >= 20 districts and >= 5 states 2020-23),
        Q = recurring n-gram in the GABRIEL illustrative quotes (output/tables/quotes_*.csv).
side: our prior about which side uses it (R / D / covid); descriptive label only, never used to define anything.
"""
EXCL_WINDOW = 150

PHRASES = {
    # --- parents' rights ---
    "parents bill of rights": dict(regex=r"parent(?:s|'s|al)? bill of rights", source="S,N,Q", side="R"),
    "parental rights": dict(regex=r"parental rights|parent(?:s|'s)? rights", source="S,N", side="R",
                            exclude=r"ferpa|procedural safeguards?|special education|\biep\b|\b504\b|due process"
                                    r"|handbook|notice of (?:parent|procedural)|annual notice"),
    "moms for liberty": dict(regex=r"moms? for liberty", source="N", side="R"),
    "let parents decide": dict(regex=r"let (?:the )?parents (?:decide|choose)|parents should (?:be able to )?decide",
                               source="N", side="R"),
    "parent's choice": dict(regex=r"parent'?s choice|parental choice", source="N,Q", side="R",
                            exclude=r"distance|virtual|remote|hybrid|in person|learning model|online"),
    "co-parent with government": dict(regex=r"co ?parent(?:ing|s)? with", source="S,N", side="R"),
    "not what to think": dict(regex=r"not what to think|how to think not what", source="S", side="R"),
    # --- masks / vaccines ---
    "mask mandate": dict(regex=r"(?:mask|masks|mass|math) mandates?", source="S,N,Q", side="covid"),
    "masks optional": dict(regex=r"(?:mask|masks|mass) optional|optional mask(?:s|ing)?", source="N,Q", side="covid"),
    "unmask our children": dict(regex=r"unmask (?:our|the|my|these) (?:children|kids|students)", source="S", side="R"),
    "let them breathe": dict(regex=r"let (?:them|our kids|our children|the kids|the children|my kids) breathe",
                             source="S", side="R"),
    "right to breathe": dict(regex=r"right to breathe|breathe freely|breathe fresh air", source="N", side="R"),
    "medical freedom": dict(regex=r"medical freedom|health freedom", source="S", side="R"),
    "medical decisions for my child": dict(regex=r"(?:medical|health care|healthcare) decisions for (?:my|our|their)"
                                                 r" (?:child|children|kids|family|families)", source="N", side="R"),
    "my child my choice": dict(regex=r"my (?:child|children|kids?) my choice", source="S", side="R"),
    "my body my choice": dict(regex=r"my body my choice", source="N", side="R"),
    "vaccine mandate": dict(regex=r"(?:vaccine|vaccination|vax) mandates?", source="S", side="covid"),
    "experimental vaccine": dict(regex=r"experimental (?:vaccines?|shots?|injections?|gene therapy|jab)",
                                 source="S", side="R"),
    "god-given rights": dict(regex=r"god given rights?", source="N", side="R"),
    # --- curriculum, race, gender ---
    "critical race theory": dict(regex=r"critical race theor(?:y|ies)", source="S,N", side="R"),
    "crt (acronym)": dict(regex=r"crt", source="S,N", side="R",
                          exclude=r"crisis|response team|crt team|crt training|crt member|cathode"),
    "1619 project": dict(regex=r"(?:1619|sixteen nineteen) project", source="S", side="R"),
    "indoctrination": dict(regex=r"indoctrination (?:centers?|camps?)|indoctrinat\w* (?:our|the|my|these) "
                                 r"(?:children|kids|students)", source="S,N", side="R"),
    "gender ideology": dict(regex=r"(?:gender|transgender|trans) ideology", source="S", side="R"),
    "groomer / grooming kids": dict(regex=r"groomers?|grooming (?:our |the |these )?(?:children|kids|students)",
                                    source="S", side="R", exclude=r"hygiene|dress code|dog|pet\b"),
    "sexually explicit": dict(regex=r"sexually explicit", source="S,N", side="R"),
    "pornographic": dict(regex=r"pornographic", source="S", side="R", exclude=r"child pornography"),
    "book bans": dict(regex=r"ban(?:ning|ned|s)? books|book bans?|banning of books", source="N,Q", side="D",
                      exclude=r"banned books week"),
    "my pronouns are": dict(regex=r"my pronouns are", source="N", side="D"),
    "lgbtq plus": dict(regex=r"lgbtq(?:ia)? plus|lgbtqia", source="N", side="D"),
    "diversity equity inclusion": dict(regex=r"diversity equity (?:and )?inclusion|equity diversity (?:and )?inclusion"
                                             r"|diversity inclusion (?:and )?equity", source="N,Q", side="D"),
    "anti-racist": dict(regex=r"anti ?racis[mt]s?", source="N", side="D"),
    "systemic racism": dict(regex=r"systemic(?:ally)? racis[mt]|structural racism", source="N", side="D"),
    "grading for equity": dict(regex=r"grading for equity|equitable grading", source="N", side="D"),
}


def Compiled():
    """{phrase: (pattern, exclude_pattern or None)}"""
    import re
    out = {}
    for k, v in PHRASES.items():
        pat = re.compile(r"\b(?:" + v["regex"] + r")\b")
        exc = re.compile(v["exclude"]) if v.get("exclude") else None
        out[k] = (pat, exc)
    return out
