"""Attribute definitions for the single GABRIEL rate() call (each scored 0-100 per excerpt).

Every change to a name or wording must bump VERSION and be logged in attribute_log.md (p-hacking guard: we report
every attribute tried). Develop wording on split A only; split B is reserved for reported results.
Stance is not rated directly: it is computed later as conservative_positions - progressive_positions.
"""

VERSION = "v3"

ATTRIBUTES = {
    "hostility": (
        "How hostile the exchange is: personal attacks or insults, shouting or interrupting, accusations of lying, "
        "corruption or bad faith, threats, and calls for officials to resign or be removed or for speakers to be ejected. "
        "0 = civil and procedural; 100 = sustained, intense hostility. Polite disagreement is low."
    ),
    "culture_war_share": (
        "Share of the excerpt spent on culture-war issues: race in the curriculum (critical race theory, DEI), "
        "library or classroom books and content, gender identity or LGBTQ issues (pronouns, bathrooms, sports teams), "
        "parental rights, and masks or vaccines. 0 = none of it; 100 = all of it. "
        "Routine business (budgets, buildings, athletics, test scores) counts as none."
    ),
    "parental_rights_framing": (
        "How strongly speakers frame issues as parents' authority over their children's schooling (the right to know, "
        "consent, opt out or decide) rather than the district's or educators' authority. "
        "0 = no such framing; 100 = parental-rights framing dominates."
    ),
    "covid_contention": (
        "How much disagreement there is over COVID-19 policies: masks, school closures or remote learning, vaccines, "
        "quarantines, testing. 0 = no COVID discussion, or only neutral updates; 100 = intense, sustained disagreement."
    ),
    "progressive_positions": (
        "How strongly speakers argue for the progressive side of a contested political or social dispute, e.g. "
        "defending books against removal, defending equity, DEI or LGBTQ-inclusive policies against criticism, "
        "defending inclusive race or history curriculum against those criticisms, "
        "supporting masks or vaccine requirements against opposition. Rate positions argued, NOT topics discussed. "
        "Routine descriptions of district programs, plans or services (e.g. staff presenting an equity plan or SEL "
        "curriculum without arguing against an opposing view) count as 0. "
        "0 = no progressive side argued; 100 = the progressive side argued strongly throughout."
    ),
    "conservative_positions": (
        "How strongly speakers argue for the conservative side of a contested political or social dispute, e.g. "
        "demanding books be removed, opposing equity, DEI or LGBTQ-inclusive policies, "
        "asserting parents' authority over curriculum, opposing CRT or 'woke' content, opposing masks or vaccine "
        "requirements. Rate positions argued, NOT topics discussed. "
        "Routine descriptions of district programs, plans or services (e.g. staff presenting a safety plan or "
        "curriculum review without arguing against an opposing view) count as 0. "
        "0 = no conservative side argued; 100 = the conservative side argued strongly throughout."
    ),
}
