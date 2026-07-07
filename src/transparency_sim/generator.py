"""Minimal synthetic environment generator (draft v0.4, §5.3, Assumptions 1-4).

Design contract:
  Assumption 1 (pure distraction): distractor bodies are built from vocabulary
    pools disjoint from the core pools, so they contain no Y0 value; a string
    leak check enforces this per instance.
  Assumption 2 (exchangeability): document ids are random hex codes drawn from
    a common space and the display order is shuffled, so neither id nor
    position identifies core status.
  Assumption 3 (references among cores only): resolvable "REF:DOC_XXXX"
    references are inserted only between core documents, each ordered pair
    independently with probability c. Distractors carry no references.
  Assumption 4 (linear recovery-distortion map): Y0 has exactly r components
    and each core document exclusively carries exactly one of them, so the
    conditional Bayes risk is D0 * (1 - |recovered|/r) by construction.

The generator is fully deterministic given (q, r, c, seed). No LLM is used;
LLM surface realization is a separate, optional layer outside this module.
"""
from __future__ import annotations

import random
import re

from .corpus import Corpus, Document

# --------------------------------------------------------------------------
# Vocabulary pools. CORE_* and DISTRACTOR_* are disjoint by construction and
# a runtime leak check re-verifies disjointness on every generated instance.
# None of the entries contains the banned body labels (see _BANNED_BODY).
# --------------------------------------------------------------------------

_COMPONENT_TYPES = [
    "authority", "rationale", "rejected_option", "timeline",
    "consultation", "venue", "budget_line", "vote_split",
]

CORE_POOLS: dict[str, list[str]] = {
    "authority": [
        "the Harbor Renewal Board", "the Meridian Oversight Council",
        "the Northgate Planning Bureau", "the Cascade Infrastructure Authority",
        "the Lakeshore Development Commission", "the Auburn Transit Directorate",
    ],
    "rationale": [
        "projected maintenance overruns", "groundwater contamination findings",
        "seismic retrofit obligations", "unresolved easement litigation",
        "shortfalls in the drainage assessment", "conflicting soil survey results",
    ],
    "rejected_option": [
        "the elevated causeway proposal", "the phased tunnel option",
        "the western bypass alignment", "the modular terminal scheme",
        "the deferred procurement plan", "the split-site variant",
    ],
    "timeline": [
        "March 12, 2031", "June 04, 2029", "October 27, 2032",
        "January 19, 2030", "August 08, 2033", "November 23, 2031",
    ],
    "consultation": [
        "a closed briefing with the Meridian caucus",
        "an off-record session with the harbor lessees",
        "a private consultation with the bond underwriters",
        "an unlisted meeting with the district surveyors",
        "a restricted call with the transit concessionaire",
        "an informal exchange with the levee contractors",
    ],
    "venue": [
        "the Rotunda annex in Fairview", "the Dockside chamber at Pier Nine",
        "the Ridgeline conference hall", "the Old Mint deliberation room",
        "the Aqueduct house in Millbrook", "the Granary hall at Kestrel Yard",
    ],
    "budget_line": [
        "a contingency reserve of 74.6 million", "an allocation ceiling of 41.7 million",
        "a phased outlay of 58.2 million", "an escrow tranche of 63.9 million",
        "a bridging facility of 82.3 million", "a settlement fund of 96.1 million",
    ],
    "vote_split": [
        "a nine-to-four division", "a seven-to-six division",
        "an eleven-to-two division", "an eight-to-five division",
        "a ten-to-three division", "a twelve-to-one division",
    ],
}

_CORE_TEMPLATES: dict[str, str] = {
    "authority": "After deliberation, final signing power for the matter rested with {v}.",
    "rationale": "The determination was grounded in {v}, as entered in the minutes.",
    "rejected_option": "The committee set aside {v} and did not return to it.",
    "timeline": "The operative decision was entered into the register on {v}.",
    "consultation": "Prior to the session, {v} took place and was not minuted.",
    "venue": "Proceedings were held at {v} under the standing arrangements.",
    "budget_line": "The financing clause fixed {v} for the works described above.",
    "vote_split": "The motion carried on {v} of the members present.",
}

_CORE_FILLER = [
    "The session opened at half past nine and quorum was confirmed.",
    "Procedural compliance under standing order 12 was noted for the file.",
    "No further items were tabled before adjournment.",
    "The clerk certified the circulation list in the usual manner.",
]

DISTRACTOR_ORGS = [
    "the Riverside Sanitation Panel", "the Elm Street Licensing Desk",
    "the Foxglove Parks Committee", "the Quarry Road Signage Unit",
    "the Bellamy Archives Working Group", "the Halstead Canteen Board",
    "the Pinecrest Noise Review Cell", "the Warrick Lane Fencing Office",
]
DISTRACTOR_REASONS = [
    "an incomplete stationery inventory", "a lapsed mowing contract",
    "a disputed parking rota", "an outstanding signage audit",
    "a postponed roof inspection", "a duplicate invoice query",
    "a stalled newsletter tender", "an unreturned key register",
]
DISTRACTOR_DATES = [
    "April 03, 2019", "September 15, 2020", "February 21, 2021",
    "July 30, 2022", "December 05, 2023", "May 17, 2024",
    "October 09, 2019", "March 26, 2023",
]
DISTRACTOR_PEOPLE = [
    "Officer Hained", "Clerk Bellrose", "Deputy Marwick", "Registrar Tolvey",
    "Surveyor Quist", "Warden Ashcombe", "Auditor Penhale", "Bailiff Corven",
]
_DISTRACTOR_TEMPLATE = (
    "Status memorandum concerning {org}. The panel deferred action citing {reason}. "
    "A follow-up review was scheduled for {date} and assigned to {person}. "
    "No decision of record was taken at this stage."
)

# Banned body labels (draft instruction): the generator must never emit these.
_BANNED_BODY = [
    re.compile(r"CORE"), re.compile(r"Y0"),
    re.compile(r"\btruth\b", re.IGNORECASE),
    re.compile(r"\bcentral\b", re.IGNORECASE),
    re.compile(r"\bcore\b", re.IGNORECASE),
    re.compile(r"component_", re.IGNORECASE),
]

_ID_FORMAT = re.compile(r"^DOC_[0-9A-F]{4}$")
_REF_TOKEN = re.compile(r"REF:(DOC_[0-9A-F]{4})")


def generate_corpus(q: int, r: int, c: float, seed: int, language: str = "en") -> Corpus:
    """Generate one synthetic environment satisfying Assumptions 1-4."""
    if language != "en":
        raise NotImplementedError("only language='en' is implemented in this round")
    if not (1 <= r <= q):
        raise ValueError(f"need 1 <= r <= q, got r={r}, q={q}")
    if not (0.0 <= c <= 1.0):
        raise ValueError(f"need 0 <= c <= 1, got c={c}")
    if q > 60000:
        raise ValueError("q too large for the 4-hex id space")
    max_r = min(len(pool) for pool in CORE_POOLS.values()) * len(_COMPONENT_TYPES)
    if r > max_r:
        raise ValueError(f"r={r} exceeds built-in vocabulary capacity ({max_r})")

    rng = random.Random(seed)

    # --- ground truth Y0: r components, one exclusive value each -----------
    remaining = {t: list(pool) for t, pool in CORE_POOLS.items()}
    y0: dict[str, str] = {}
    for i in range(1, r + 1):
        ctype = _COMPONENT_TYPES[(i - 1) % len(_COMPONENT_TYPES)]
        value = remaining[ctype].pop(rng.randrange(len(remaining[ctype])))
        y0[f"component_{i}"] = value
    _assert_no_substring_collision(list(y0.values()))

    # --- ids: random hex codes from a common space, then shuffled order ----
    codes = rng.sample(range(0x10000), q)
    ids = [f"DOC_{v:04X}" for v in codes]
    rng.shuffle(ids)
    core_ids = tuple(ids[:r])
    distractor_ids = ids[r:]

    # --- reference structure: core -> core only, iid with probability c ----
    edges: list[tuple[str, str]] = []
    for a in core_ids:
        for b in core_ids:
            if a != b and rng.random() < c:
                edges.append((a, b))
    out: dict[str, list[str]] = {cid: [] for cid in core_ids}
    for a, b in edges:
        out[a].append(b)

    # --- documents ----------------------------------------------------------
    docs: list[Document] = []
    for i, cid in enumerate(core_ids, start=1):
        key = f"component_{i}"
        ctype = _COMPONENT_TYPES[(i - 1) % len(_COMPONENT_TYPES)]
        sentences = [
            rng.choice(_CORE_FILLER),
            _CORE_TEMPLATES[ctype].format(v=y0[key]),
            rng.choice(_CORE_FILLER),
        ]
        for tgt in out[cid]:
            sentences.append(f"Cross-reference: REF:{tgt}.")
        docs.append(Document(doc_id=cid, body=" ".join(sentences),
                             is_core=True, component_key=key,
                             refs=tuple(out[cid])))
    for did in distractor_ids:
        body = _DISTRACTOR_TEMPLATE.format(
            org=rng.choice(DISTRACTOR_ORGS),
            reason=rng.choice(DISTRACTOR_REASONS),
            date=rng.choice(DISTRACTOR_DATES),
            person=rng.choice(DISTRACTOR_PEOPLE),
        )
        docs.append(Document(doc_id=did, body=body, is_core=False,
                             component_key=None, refs=()))

    rng.shuffle(docs)  # display order carries no information (Assumption 2)

    corpus = Corpus(q=q, r=r, c=c, seed=seed, language=language, y0=y0,
                    docs=docs, core_ids=core_ids, edges=tuple(edges))
    validate_corpus(corpus)
    return corpus


def _assert_no_substring_collision(values: list[str]) -> None:
    for i, a in enumerate(values):
        for j, b in enumerate(values):
            if i != j and a in b:
                raise AssertionError(f"Y0 value collision: {a!r} inside {b!r}")


def validate_corpus(corpus: Corpus) -> None:
    """Leak and discipline checks (draft §5.3). Raises on any violation."""
    docs = corpus.docs
    cores = corpus.core_documents
    distractors = corpus.distractor_documents
    core_set = set(corpus.core_ids)
    values = list(corpus.y0.values())

    # counts
    if len(docs) != corpus.q or len(cores) != corpus.r:
        raise AssertionError("document counts do not match (q, r)")
    if len(distractors) != corpus.q - corpus.r:
        raise AssertionError("distractor count does not match q - r")

    # exclusivity: each core carries exactly its own component, no other
    seen_keys = set()
    for d in cores:
        if d.component_key is None or d.component_key not in corpus.y0:
            raise AssertionError(f"core {d.doc_id} lacks a valid component key")
        seen_keys.add(d.component_key)
        own = corpus.y0[d.component_key]
        if own not in d.body:
            raise AssertionError(f"core {d.doc_id} does not contain its component value")
        for v in values:
            if v != own and v in d.body:
                raise AssertionError(f"core {d.doc_id} leaks another component: {v!r}")
    if seen_keys != set(corpus.y0.keys()):
        raise AssertionError("component keys are not covered one-to-one")

    # distractor leak: no Y0 value, no references, no reference tokens
    for d in distractors:
        low = d.body.lower()
        for v in values:
            if v in d.body or v.lower() in low:
                raise AssertionError(f"distractor {d.doc_id} leaks Y0 value {v!r}")
        if d.refs or "REF:" in d.body:
            raise AssertionError(f"distractor {d.doc_id} carries references")
        if d.component_key is not None:
            raise AssertionError(f"distractor {d.doc_id} has a component key")

    # reference discipline: every token resolves to a core, refs match edges
    edge_set = set(corpus.edges)
    for d in docs:
        tokens = set(_REF_TOKEN.findall(d.body))
        if not d.is_core:
            if tokens:
                raise AssertionError(f"distractor {d.doc_id} embeds reference tokens")
            continue
        if tokens != set(d.refs):
            raise AssertionError(f"core {d.doc_id}: body tokens and refs disagree")
        for t in d.refs:
            if t not in core_set:
                raise AssertionError(f"core {d.doc_id} references non-core {t}")
            if (d.doc_id, t) not in edge_set:
                raise AssertionError(f"edge ({d.doc_id}, {t}) missing from edge list")

    # banned body labels
    for d in docs:
        for pat in _BANNED_BODY:
            if pat.search(d.body):
                raise AssertionError(f"banned label {pat.pattern!r} in {d.doc_id}")

    # id format and order shuffle
    for d in docs:
        if not _ID_FORMAT.match(d.doc_id):
            raise AssertionError(f"malformed id {d.doc_id}")
    positions = [i for i, d in enumerate(docs) if d.is_core]
    if corpus.q > corpus.r and positions == list(range(corpus.r)):
        raise AssertionError("cores occupy the leading block; shuffle failed")
COMPONENT_TYPES = _COMPONENT_TYPES
CORE_TEMPLATES = _CORE_TEMPLATES
