"""Hand-built lexicon of contested education-politics vocabulary, used to ask which agendas are partisan.

Each category has single `terms` (counted from the unigram document-term matrix) and multi-word `phrases` (counted
directly from the cleaned text by count_phrases.py, since the matrix has no bigrams). `expected` is our PRIOR about which
side talks more about it ("R", "D", or "none" for technical controls); the analysis checks it against the data and never
uses it to define partisanship. Terms rarer than MIN_DISTRICTS districts are cut from the matrix, so some will be absent.
Text is lowercase with punctuation removed: "parents' rights" appears as "parents rights".
"""

LEXICON = {
    # --- contested agendas, hypothesised right-coded ---
    "parental_rights": {
        "expected": "R",
        "terms": ["parental", "moms"],
        "phrases": ["parental rights", "parents rights", "parent rights", "parents bill of rights", "right to know",
                    "moms for liberty", "parent choice", "parents choice", "opt out"],
    },
    "book_content": {
        "expected": "R",
        "terms": ["ban", "banned", "censorship", "pornography", "pornographic", "obscene", "explicit", "inappropriate"],
        "phrases": ["book ban", "book bans", "banned books", "sexually explicit", "age appropriate",
                    "inappropriate material", "inappropriate materials", "pornographic material"],
    },
    "crt_race": {
        "expected": "R",
        "terms": ["crt", "woke", "indoctrination", "indoctrinate", "indoctrinating", "marxism", "marxist"],
        "phrases": ["critical race theory", "anti racist", "antiracist", "white privilege", "systemic racism",
                    "divisive concepts", "1619 project"],
    },
    "gender_lgbtq": {
        "expected": "R",
        "terms": ["lgbtq", "lgbt", "transgender", "pronouns", "groomer", "groomers", "grooming", "gender", "bathroom", "bathrooms"],
        "phrases": ["gender identity", "biological sex", "gender ideology", "title ix", "gender affirming",
                    "sexual orientation", "locker room", "girls sports"],
    },
    "school_choice": {
        "expected": "R",
        "terms": ["vouchers", "voucher", "charter", "charters", "homeschool", "homeschooling"],
        "phrases": ["school choice", "education savings account", "education savings accounts", "open enrollment",
                    "tax credit scholarship"],
    },
    "covid_mandates": {
        "expected": "R",
        "terms": ["mask", "masks", "masking", "mandate", "mandates", "mandated", "vaccine", "vaccines", "vaccination",
                  "vaccinated", "quarantine", "lockdown"],
        "phrases": ["mask mandate", "mask mandates", "mask optional", "vaccine mandate", "social distancing",
                    "contact tracing", "remote learning", "virtual learning", "hybrid learning"],
    },
    # --- contested agendas, hypothesised left-coded ---
    "equity_sel": {
        "expected": "D",
        "terms": ["equity", "inclusion", "inclusive", "diversity", "dei", "sel", "restorative", "marginalized",
                  "underserved", "bipoc"],
        "phrases": ["social emotional learning", "social emotional", "culturally responsive", "racial equity", "equity lens",
                    "implicit bias", "restorative justice", "restorative practices", "safe space"],
    },
    "guns_safety": {
        "expected": "D",
        "terms": ["shooting", "shooter", "shootings", "firearm", "firearms", "gun", "guns", "sro", "armed"],
        "phrases": ["school resource officer", "school resource officers", "active shooter", "gun violence",
                    "arming teachers", "school safety"],
    },
    "unions_pay": {
        "expected": "D",
        "terms": ["union", "unions", "strike", "bargaining"],
        "phrases": ["collective bargaining", "teacher pay", "teacher salaries", "living wage", "teachers union"],
    },
    "immigration": {
        "expected": "D",
        "terms": ["immigrants", "immigration", "undocumented", "asylum", "refugees", "migrant", "migrants"],
        "phrases": ["english learners", "english language learners", "dual language"],
    },
    # --- no clear prior ---
    "religion": {
        "expected": "R",
        "terms": ["prayer", "religion", "religious", "christian", "bible", "atheist", "satanic"],
        "phrases": ["moment of silence", "separation of church", "ten commandments", "in god we trust"],
    },
    "conflict_legal": {
        "expected": "none",
        "terms": ["lawsuit", "lawsuits", "recall", "transparency", "censure", "resign", "resignation", "misconduct",
                  "unlawful", "illegal"],
        "phrases": ["public comment", "open meetings act", "first amendment", "recall election", "no confidence"],
    },
    # --- in-meeting conflict (decorum, disruption, hostility); used for the conflict index/map in conflict_map.py ---
    # Deliberately excludes routine/ambiguous words: police, security, threat (school-safety items), harassment (a policy agenda
    # item), disruption (instructional), shame (casual speech).
    "meeting_conflict": {
        "expected": "none",
        "terms": ["decorum", "outburst", "trespass", "trespassing", "arrested", "arrest", "escorted", "heckling", "booing",
                  "shouting", "yelling", "screaming", "intimidation", "intimidate", "threatened", "liar", "lying",
                  "disgrace", "disgusting", "hypocrite", "corrupt", "corruption"],
        "phrases": ["out of order", "point of order", "call the police", "please be seated", "please sit down",
                    "your time is up", "time is up", "remove him", "remove her", "stop the meeting", "recess the meeting",
                    "ask you to leave", "you will be removed", "this is a warning"],
    },
    # --- technical controls (should NOT be partisan) ---
    "ctrl_budget": {
        "expected": "none",
        "terms": ["budget", "revenue", "bond", "levy", "millage", "expenditures", "audit", "fiscal", "deficit", "reserves"],
        "phrases": ["fund balance", "general fund"],
    },
    "ctrl_facilities": {
        "expected": "none",
        "terms": ["construction", "roof", "hvac", "building", "facilities", "parking", "renovation", "bus", "buses", "transportation"],
        "phrases": ["capital improvement", "capital improvements"],
    },
    "ctrl_athletics": {
        "expected": "none",
        "terms": ["football", "basketball", "coach", "coaches", "athletics", "athletic", "stadium", "volleyball"],
        "phrases": ["athletic director"],
    },
    "ctrl_academics": {
        "expected": "none",
        "terms": ["assessment", "standards", "literacy", "math", "graduation", "curriculum", "testing", "achievement", "scores"],
        "phrases": ["test scores", "reading level"],
    },
}

def AllPhrases():
    return sorted({p for c in LEXICON.values() for p in c["phrases"]})
