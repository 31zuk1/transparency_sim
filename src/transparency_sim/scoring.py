"""Deterministic scoring for structured transparency-sim answers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .corpus import Corpus
from .generator import COMPONENT_TYPES


def normalize(s: str) -> str:
    """Normalize by collapsing whitespace and applying casefold."""
    return " ".join(s.split()).casefold()


def component_schema(r: int) -> tuple[tuple[str, str], ...]:
    """Return component keys and types in the same cycle as generate_corpus."""
    return tuple(
        (f"component_{i}", COMPONENT_TYPES[(i - 1) % len(COMPONENT_TYPES)])
        for i in range(1, r + 1)
    )


@dataclass(frozen=True)
class ScoreResult:
    distortion: float
    per_component: dict[str, bool]


def score_answer(y0: dict[str, str], answer: dict[str, str]) -> ScoreResult:
    """Score a structured answer by normalized exact match."""
    unknown = set(answer) - set(y0)
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise ValueError(f"answer contains unknown component keys: {keys}")

    per_component = {
        key: key in answer and normalize(answer[key]) == normalize(value)
        for key, value in y0.items()
    }
    wrong = sum(not ok for ok in per_component.values())
    return ScoreResult(distortion=wrong / len(y0), per_component=per_component)


def recovery_distortion(corpus: Corpus, obtained_ids: Iterable[str]) -> float:
    """Evaluate recovery using ground-truth metadata; policies must not call this."""
    recovered = set()
    for doc_id in obtained_ids:
        doc = corpus.doc(doc_id)
        if doc.is_core and doc.component_key is not None:
            recovered.add(doc.component_key)
    return 1.0 - len(recovered) / corpus.r
