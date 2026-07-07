"""A0 calibration baseline (draft v0.4, §5.2 and Table 1).

A0 is the scripted seed baseline: it computes, exactly, the expected
recovery and achieved distortion of the batch-then-track restricted policy
class Pi^seed on one fixed corpus. It is a calibration reference for the
theory surface, i.e. it computes D_seed_d (in particular D_seed_inf for
d = "inf"). It is *not* the definitional infimum D* over the full adaptive
class Pi_theta, and it is not an LLM: it never reads document bodies and
uses only the generator's internal metadata (core set and reference graph).

Exact computation (r small, subsets of the core set K enumerated):
  Pr(seed set = T) = C(q - r, B - |T|) / C(q, B)   for 0 <= B-|T| <= q-r,
  E[F_d] = sum_T Pr(seed set = T) * |Reach_d(T)|,
  D/D0   = 1 - E[F_d] / r          (Assumption 4, linear map).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Union

from .corpus import Corpus

Depth = Union[int, str]  # positive int, or "inf"

_MAX_R_ENUM = 20  # 2^r subset enumeration guard


def reach_set(adj: dict[str, set[str]], seeds: frozenset, depth: Depth) -> frozenset:
    """Cores reachable from `seeds` by following references up to `depth` hops."""
    if depth == "inf":
        limit = None
    elif isinstance(depth, int) and depth >= 1:
        limit = depth
    else:
        raise ValueError(f"depth must be a positive int or 'inf', got {depth!r}")
    reached = set(seeds)
    frontier = set(seeds)
    hops = 0
    while frontier and (limit is None or hops < limit):
        nxt = set()
        for u in frontier:
            nxt |= adj.get(u, set())
        frontier = nxt - reached
        reached |= frontier
        hops += 1
    return frozenset(reached)


def expected_recovery(adj: dict[str, set[str]], core_ids: tuple[str, ...],
                      q: int, B: int, depth: Depth) -> float:
    """E[F_d] for the batch-then-track baseline on a fixed reference graph.

    The graph is held fixed; q enters only through the hypergeometric
    weights over seed sets. This separation lets callers vary q for the
    same graph (used to check the Proposition 1 direction on one instance).
    """
    r = len(core_ids)
    if r > _MAX_R_ENUM:
        raise ValueError(f"r={r} too large for exact subset enumeration")
    if not (0 <= B <= q) or r > q:
        raise ValueError(f"invalid (q={q}, r={r}, B={B})")
    denom = comb(q, B)
    ef = 0.0
    core_sorted = tuple(sorted(core_ids))
    for k in range(0, min(r, B) + 1):
        rest = B - k
        if rest < 0 or rest > q - r:
            continue
        p_each = comb(q - r, rest) / denom  # probability of each specific T, |T| = k
        if p_each == 0.0:
            continue
        for T in combinations(core_sorted, k):
            ef += p_each * len(reach_set(adj, frozenset(T), depth))
    return ef


@dataclass(frozen=True)
class A0Result:
    q: int
    r: int
    B: int
    depth: Depth
    expected_recovery: float
    distortion: float  # D / D0 = 1 - E[F_d]/r  (Assumption 4)


def a0_exact(corpus: Corpus, B: int, depth: Depth = "inf") -> A0Result:
    """Exact D_seed_d on one fixed corpus (calibration baseline)."""
    ef = expected_recovery(corpus.core_adjacency(), corpus.core_ids,
                           corpus.q, B, depth)
    return A0Result(q=corpus.q, r=corpus.r, B=B, depth=depth,
                    expected_recovery=ef, distortion=1.0 - ef / corpus.r)
