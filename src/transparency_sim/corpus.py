"""Data structures for synthetic disclosure environments (draft v0.4, §5.3).

A corpus consists of q documents: r core documents (jointly sufficient and
minimal for the ground truth Y0, Assumption 1) and q - r distractors whose
content is independent of Y0. Resolvable references exist only among core
documents (Assumption 3, L subseteq K x K). Each core document carries
exactly one component of Y0, which enforces the linear recovery-distortion
map by construction (Assumption 4, with K = r components).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Document:
    doc_id: str
    body: str
    is_core: bool
    component_key: Optional[str]  # None for distractors
    refs: tuple[str, ...]         # resolvable outgoing references (doc ids)


@dataclass
class Corpus:
    q: int
    r: int
    c: float
    seed: int
    language: str
    y0: dict[str, str]                    # component_key -> ground-truth value
    docs: list[Document]                  # display order (shuffled)
    core_ids: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]    # (src_id, dst_id), core -> core only
    _index: dict[str, Document] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {d.doc_id: d for d in self.docs}
        if len(self._index) != len(self.docs):
            raise ValueError("duplicate document ids")

    def doc(self, doc_id: str) -> Document:
        return self._index[doc_id]

    @property
    def core_documents(self) -> list[Document]:
        return [d for d in self.docs if d.is_core]

    @property
    def distractor_documents(self) -> list[Document]:
        return [d for d in self.docs if not d.is_core]

    def core_adjacency(self) -> dict[str, set[str]]:
        """Out-neighborhoods of the core reference digraph."""
        adj: dict[str, set[str]] = {cid: set() for cid in self.core_ids}
        for src, dst in self.edges:
            adj[src].add(dst)
        return adj
