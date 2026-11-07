"""Word lists for closure talk and for pandemic-era educational outcomes (same format as bow/lexicon.py).

These are kept apart from bow/lexicon.py so adding them does not change the phrase counts behind the BoW tables. `terms` come from
the unigram matrix and `phrases` from count_closure_phrases.py. Terms cut from the vocabulary are skipped when rates are built.
"""

CLOSURE_LEXICON = {
    # Reopening and modes of instruction. "virtual" and "zoom" are left out because they also describe the meeting itself.
    "closure_talk": {
        "terms": ["reopen", "reopening", "reopened", "closure", "closures", "hybrid", "asynchronous", "synchronous", "chromebook",
                  "chromebooks", "hotspot", "hotspots"],
        "phrases": ["in person learning", "in person instruction", "remote learning", "virtual learning", "hybrid learning",
                    "distance learning", "remote instruction", "school closure", "school closures", "return to school",
                    "return to learn", "reopening plan"],
    },
    "learning_loss": {
        "terms": ["tutoring", "tutors", "absenteeism", "esser", "remediation", "proficiency", "proficient"],
        "phrases": ["learning loss", "unfinished learning", "lost learning", "learning recovery", "learning gap", "learning gaps",
                    "achievement gap", "test scores", "chronic absenteeism", "summer learning", "high dosage"],
    },
    "enrollment_loss": {
        "terms": ["homeschooling", "homeschool"],
        "phrases": ["declining enrollment", "enrollment decline", "enrollment drop", "enrollment loss", "lost students",
                    "loss of students", "left the district"],
    },
}

def AllClosurePhrases():
    return sorted({p for c in CLOSURE_LEXICON.values() for p in c["phrases"]})
