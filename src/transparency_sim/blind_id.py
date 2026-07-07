"""Scripted Blind-ID policies and deterministic run harness."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Iterable, Protocol

from .corpus import Corpus
from .environment import BlindIDEnvironment, DepthExceeded
from .generator import COMPONENT_TYPES, CORE_TEMPLATES
from .scoring import component_schema, recovery_distortion, score_answer


class BlindIDPolicy(Protocol):
    def run(self, env: BlindIDEnvironment) -> dict[str, str]:
        """Operate the environment and return component_key -> value."""


class NullPolicy:
    def run(self, env: BlindIDEnvironment) -> dict[str, str]:
        return {}


def extract_answers(bodies: Iterable[str], r: int) -> dict[str, str]:
    if r > len(COMPONENT_TYPES):
        raise ValueError("type-to-key mapping is ambiguous for r > 8 in this round")

    ctype_to_key = {ctype: key for key, ctype in component_schema(r)}
    answer: dict[str, str] = {}
    for body in bodies:
        for ctype, key in ctype_to_key.items():
            template = CORE_TEMPLATES[ctype]
            prefix, suffix = template.split("{v}")
            # Non-greedy matching is stable here: period-ending templates use
            # value pools without periods, while budget_line has a longer suffix.
            pattern = re.escape(prefix) + r"(.+?)" + re.escape(suffix)
            match = re.search(pattern, body)
            if not match:
                continue
            value = match.group(1)
            if key in answer and answer[key] != value:
                raise ValueError(f"conflicting extracted values for {key}")
            answer[key] = value
    return answer


class ScriptedSequentialPolicy:
    def __init__(self, policy_seed: int) -> None:
        self.policy_seed = policy_seed

    def run(self, env: BlindIDEnvironment) -> dict[str, str]:
        rng = random.Random(self.policy_seed)
        pool = list(env.list_ids())
        rng.shuffle(pool)
        bodies: dict[str, str] = {}

        for doc_id in pool:
            if env.budget_remaining == 0:
                break
            if doc_id in set(env.obtained_ids()):
                continue
            view = env.fetch(doc_id)
            bodies[view.doc_id] = view.body
            frontier = [view]
            while frontier:
                view = frontier.pop()
                for target_id in view.refs:
                    if target_id in set(env.obtained_ids()):
                        continue
                    try:
                        resolved = env.resolve(view.doc_id, target_id)
                    except DepthExceeded:
                        continue
                    bodies[resolved.doc_id] = resolved.body
                    frontier.append(resolved)

        return extract_answers(bodies.values(), env.n_components)


@dataclass(frozen=True)
class BlindIDRunResult:
    answer: dict[str, str]
    distortion_answer: float
    distortion_recovery: float
    n_fetch_paid: int
    n_resolve: int
    obtained: tuple[str, ...]


def run_blind_id(
    corpus: Corpus,
    policy: BlindIDPolicy,
    budget: int,
    depth: int | str = "inf",
) -> BlindIDRunResult:
    """Run one policy and compute both answer and recovery diagnostics.

    Sequential exclusion can weakly improve on the batch seeded baseline, so
    a mean D_hat slightly below D_seed_inf is expected for some fixed corpora.
    """
    env = BlindIDEnvironment(corpus=corpus, budget=budget, depth=depth, kappa=0.0)
    answer = policy.run(env)
    answer_score = score_answer(corpus.y0, answer)
    recovery = recovery_distortion(corpus, env.obtained_ids())
    transcript = env.transcript()
    return BlindIDRunResult(
        answer=answer,
        distortion_answer=answer_score.distortion,
        distortion_recovery=recovery,
        n_fetch_paid=sum(e.op == "fetch" and e.cost == 1 for e in transcript),
        n_resolve=sum(e.op == "resolve" for e in transcript),
        obtained=env.obtained_ids(),
    )
