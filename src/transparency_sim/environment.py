"""Sequential fetch environment for the Blind-ID observer class."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .corpus import Corpus


class TransparencySimEnvError(Exception):
    """Base class for environment API errors."""


class BudgetExhausted(TransparencySimEnvError):
    """Raised when a paid fetch would exceed the direct-acquisition budget."""


class InvalidResolve(TransparencySimEnvError):
    """Raised when a resolve request is not licensed by obtained references."""


class DepthExceeded(TransparencySimEnvError):
    """Raised when a resolve request would exceed the configured depth."""


class UnknownDocument(TransparencySimEnvError):
    """Raised when a document id is not present in the corpus."""


@dataclass(frozen=True)
class FetchView:
    doc_id: str
    body: str
    refs: tuple[str, ...]


@dataclass(frozen=True)
class Event:
    op: str
    doc_id: Optional[str]
    cost: int
    depth: Optional[int]


class BlindIDEnvironment:
    def __init__(
        self,
        corpus: Corpus,
        budget: int,
        depth: int | str = "inf",
        kappa: float = 0.0,
    ) -> None:
        if not isinstance(budget, int) or not (0 <= budget <= corpus.q):
            raise ValueError("budget must be an integer in [0, corpus.q]")
        if depth != "inf" and not (isinstance(depth, int) and depth >= 1):
            raise ValueError("depth must be a positive integer or 'inf'")
        if kappa != 0.0:
            raise NotImplementedError("kappa > 0 is a future extension")

        self._corpus = corpus
        self._budget = budget
        self._budget_spent = 0
        self._depth = depth
        self._kappa = kappa
        self._directly_fetched: set[str] = set()
        self._depth_of: dict[str, int] = {}
        self._obtained_order: list[str] = []
        self._transcript: list[Event] = []

    def list_ids(self) -> tuple[str, ...]:
        self._transcript.append(Event(op="list", doc_id=None, cost=0, depth=None))
        return tuple(d.doc_id for d in self._corpus.docs)

    def fetch(self, doc_id: str) -> FetchView:
        doc = self._doc_or_raise(doc_id)
        if doc_id in self._directly_fetched:
            depth = self._depth_of[doc_id]
            self._transcript.append(Event(op="fetch", doc_id=doc_id, cost=0, depth=depth))
            return FetchView(doc_id=doc.doc_id, body=doc.body, refs=doc.refs)

        if self.budget_remaining == 0:
            raise BudgetExhausted("direct-acquisition budget is exhausted")

        self._budget_spent += 1
        self._directly_fetched.add(doc_id)
        self._depth_of[doc_id] = 0
        if doc_id not in self._obtained_order:
            self._obtained_order.append(doc_id)
        self._transcript.append(Event(op="fetch", doc_id=doc_id, cost=1, depth=0))
        return FetchView(doc_id=doc.doc_id, body=doc.body, refs=doc.refs)

    def resolve(self, src_id: str, target_id: str) -> FetchView:
        if src_id not in self._depth_of:
            raise InvalidResolve("source document has not been obtained")
        src = self._doc_or_raise(src_id)
        if target_id not in src.refs:
            raise InvalidResolve("target is not a seen reference from source")
        if self._depth != "inf" and self._depth_of[src_id] >= self._depth:
            raise DepthExceeded("resolve depth limit reached")

        target = self._doc_or_raise(target_id)
        new_depth = self._depth_of[src_id] + 1
        if target_id in self._depth_of:
            self._depth_of[target_id] = min(self._depth_of[target_id], new_depth)
        else:
            self._depth_of[target_id] = new_depth
            self._obtained_order.append(target_id)
        depth = self._depth_of[target_id]
        self._transcript.append(Event(op="resolve", doc_id=target_id, cost=0, depth=depth))
        return FetchView(doc_id=target.doc_id, body=target.body, refs=target.refs)

    @property
    def budget_remaining(self) -> int:
        return self._budget - self._budget_spent

    @property
    def budget_spent(self) -> int:
        return self._budget_spent

    @property
    def n_components(self) -> int:
        # K = r is part of the questionnaire shape and does not reveal document roles.
        return self._corpus.r

    def obtained_ids(self) -> tuple[str, ...]:
        return tuple(self._obtained_order)

    def transcript(self) -> tuple[Event, ...]:
        return tuple(self._transcript)

    def _doc_or_raise(self, doc_id: str):
        try:
            return self._corpus.doc(doc_id)
        except KeyError as exc:
            raise UnknownDocument(doc_id) from exc
