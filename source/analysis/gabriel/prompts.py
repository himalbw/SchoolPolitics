"""GABRIEL task definitions for school-board transcripts: conflict, culture-war salience, framing, meeting type.

Each TASK names a GABRIEL function ("rate" -> 0-100 per attribute, "classify" -> boolean per label, "codify" -> quoted
passages per category) and the attribute / label / category definitions. Edit definitions here, and use a new save_dir
(run_gabriel.py does this via --tag) whenever a definition changes, since GABRIEL resumes from checkpoints.

Design notes:
- Salience (how much a topic is discussed) and stance (which side is advocated) are separate tasks. The bag-of-words
  results showed that mentions measure who is discussing/implementing a topic, not who opposes it.
- Every definition has inclusion AND exclusion criteria, because routine administrative language ("equity goals",
  "point of order", "closed session with legal counsel") otherwise inflates the scores.
- Captions carry no tone or volume, so conflict must be evidenced in words (see CONTEXT).
- The topics line up with the categories in source/analysis/bow/lexicon.py, so LLM scores can be validated against them.
"""

CONTEXT = (
    "Each text is an automatic (YouTube) caption transcript of one video posted by a U.S. public school district, "
    "usually a school board meeting. It is lowercased, has no punctuation and no speaker labels, and contains "
    "transcription errors, filler words and run-on text. Infer who is speaking (board members, the superintendent or "
    "staff, public commenters, students) from context. Judge only what is said in this transcript: do not use outside "
    "knowledge about the district or its politics. Captions do not record tone or volume, so do not assume anger, "
    "shouting or disruption unless the words show it (for example the chair asking people to stop, calling a recess, "
    "speakers describing being yelled at, or explicit insults)."
)

RATE_SCALE = (
    "Score each attribute from 0 to 100 using these anchors: 0 = absent; 10-25 = a brief or passing instance; "
    "40-60 = a substantial, recurring part of the meeting; 75-100 = dominates the meeting or is extreme in intensity. "
    "Score each attribute independently of the others."
)

TASKS = {
    # --- filter: is this a board meeting at all? cheap, run first ---------------------------------------------------
    "meeting_type": {
        "fn": "classify",
        "max_words": 3000,  # the opening of the video is enough to tell what it is
        "instructions": CONTEXT + " Choose exactly one label: the one that best describes the video as a whole.",
        "labels": {
            "regular_board_meeting": "A regular business meeting of the school board (governing board / board of "
                "education / trustees): roll call, agenda, reports, votes, usually public comment.",
            "special_board_meeting": "A special, emergency or single-topic meeting of the school board, including "
                "public hearings, budget hearings and superintendent-search meetings.",
            "workshop_or_committee": "A board study session, work session, retreat or committee meeting with "
                "presentations and discussion but few or no formal votes.",
            "other_district_video": "Not a board or committee meeting: graduation, awards, sports, community forum "
                "run by staff, town hall, training, news segment, or a meeting of another body (city council, county "
                "commission, parent-teacher organization).",
        },
    },
    # --- in-meeting conflict ------------------------------------------------------------------------------------------
    "conflict": {
        "fn": "rate",
        "instructions": CONTEXT + " " + RATE_SCALE,
        "attributes": {
            "disorder": "Breakdown of meeting order: the chair warning about decorum, calling speakers out of order, "
                "gaveling, cutting off microphones, audience interruptions or outbursts that the words make evident, "
                "recesses or adjournments because of disruption, removal of attendees, police or security involvement. "
                "Exclude routine procedure (motions, seconds, reading the public-comment rules without incident, a "
                "speaker simply running out of time).",
            "hostility_toward_board": "Speakers (usually public commenters) expressing anger, contempt or distrust "
                "toward the board, superintendent or staff: accusations of lying, corruption, indoctrination, "
                "grooming, negligence or abuse; demands to resign; threats to vote members out or recall them. "
                "Exclude polite disagreement, ordinary complaints and requests.",
            "board_infighting": "Open acrimony among board members, or between board members and the "
                "superintendent: sharp personal exchanges, accusations, complaints about colleagues' conduct, censure "
                "motions, walkouts, members refusing to participate. Exclude ordinary deliberation and split votes "
                "without acrimony.",
            "escalation": "Disputes escalated beyond the meeting: lawsuits or threats to sue, complaints to state "
                "agencies or the attorney general, recall petitions, forced resignations, investigations, police "
                "reports, threats against officials. Exclude routine legal business (contract approvals, closed "
                "session on property or personnel, standard counsel updates) not tied to a dispute.",
            "us_vs_them": "Speakers framing an issue as a fight between opposing camps rather than a shared problem: "
                "'parents versus the board', 'radicals', 'extremists', 'the left', 'the right', 'woke', 'bigots', "
                "'they want to...', describing opponents as enemies or as harming children on purpose.",
            "civil_deliberation": "Respectful, reasoned engagement on contested or substantive issues: speakers give "
                "reasons and evidence, acknowledge other views, respond to each other's points, disagree civilly. "
                "Score 0 if nothing substantive is discussed; routine unanimous consent items do not count.",
        },
    },
    # --- culture-war salience (attention, regardless of side) --------------------------------------------------------
    "culture_war": {
        "fn": "rate",
        "instructions": CONTEXT + " " + RATE_SCALE + " Rate how much attention each topic receives in the meeting "
            "(by board, staff or public), regardless of which position is taken. Count substantive discussion, "
            "debate, presentations and public comments on the topic; do not count a word used in passing, a "
            "routine item title read aloud, or generic administrative language.",
        "attributes": {
            "race_equity": "Race, equity and diversity as education issues: critical race theory, DEI programs or "
                "staff, equity plans, ethnic studies, anti-racism resolutions, how race and racism are taught, "
                "achievement or discipline gaps by race. Exclude 'equity' used as boilerplate in plan or budget "
                "read-outs without discussion.",
            "gender_sexuality": "Gender identity and sexuality: policies on transgender students (names, pronouns, "
                "bathrooms, locker rooms, sports teams), parental notification about gender identity, LGBTQ content "
                "in curriculum or libraries, pride flags or events, sex education content, Title IX rules on gender.",
            "book_content": "Challenges to books, library or curriculum materials as obscene, explicit or "
                "inappropriate; book removal or reconsideration policies and committees; accusations of censorship "
                "or book banning.",
            "parental_rights": "Parents' rights and control over schooling: rights to review curriculum and "
                "materials, opt-outs, consent and notification policies, parents' bills of rights, claims that the "
                "school usurps or excludes parents.",
            "covid_policy": "COVID-19 policy: masks, vaccines, testing, quarantine, school closures, remote or hybrid "
                "learning, reopening plans. Logistical updates count, at lower intensity than debate.",
            "religion_patriotism": "Religion and patriotism in schools: prayer, chaplains, religious displays or "
                "clubs, the Pledge of Allegiance as an issue (not its routine recitation), flags, 'Christian' or "
                "'traditional values', teaching American history as patriotic or not.",
            "safety_policing": "School safety and policing: school resource officers or police in schools, guns, "
                "armed staff, shootings or threats, metal detectors, student discipline and suspension policy.",
            "routine_governance": "Routine, non-contested governance: budgets, facilities, construction, "
                "transportation, food service, staffing and contracts, academic program reports, consent agendas, "
                "recognitions. Used as a baseline for how much of the meeting is ordinary business.",
        },
    },
    # --- stance and nationalization ---------------------------------------------------------------------------------
    "framing": {
        "fn": "rate",
        "instructions": CONTEXT + " " + RATE_SCALE + " Rate advocacy by anyone in the meeting (board, staff or "
            "public). Advocacy means arguing for a position, not merely mentioning or neutrally reporting a topic. A "
            "meeting can score high on both of the first two attributes if both sides speak.",
        "attributes": {
            "conservative_advocacy": "Arguing positions associated with the national conservative education movement: "
                "opposing CRT, DEI or 'woke' teaching; opposing transgender-inclusive policies; opposing mask or "
                "vaccine mandates or closures; supporting removal of sexually explicit books, parents' rights "
                "policies, school choice; warnings about indoctrination or grooming.",
            "progressive_advocacy": "Arguing positions associated with the national progressive education movement: "
                "supporting DEI, anti-racism or ethnic studies; supporting LGBTQ-inclusive policies and students; "
                "opposing book removals as censorship; supporting masks, vaccination or caution on reopening; "
                "warnings about hate, discrimination or extremism.",
            "nationalization": "Connecting local school issues to national or state politics: naming national or "
                "state politicians, parties or elections; national organizations (e.g. Moms for Liberty, teachers' "
                "unions as national actors, ACLU, Turning Point); federal or state culture-war legislation; cable "
                "news or viral controversies; 'this is happening all over the country'. Exclude routine state "
                "funding, testing or compliance updates.",
            "organized_mobilization": "Signs of organized participation: many speakers making the same coordinated "
                "points, speakers identifying as members of advocacy groups, petitions, scripted talking points, "
                "mentions of rallies, protests or campaigns to fill the room or flood public comment.",
        },
    },
    # --- audit: pull the passages behind high scores (Terra, small subset only) ---------------------------------------
    "codify_audit": {
        "fn": "codify",
        "instructions": CONTEXT + " Quote passages verbatim from the transcript.",
        "categories": {
            "hostile_exchange": "A passage where a speaker attacks, insults or accuses the board, staff, a board "
                "member or another speaker, or where the chair has to intervene to restore order.",
            "culture_war_claim": "A passage where a speaker argues a position on race and equity, gender and "
                "sexuality, book content, parental rights, COVID mandates or religion in schools.",
            "national_reference": "A passage that ties the local issue to national or state politics, politicians, "
                "parties or national advocacy organizations.",
        },
    },
}
