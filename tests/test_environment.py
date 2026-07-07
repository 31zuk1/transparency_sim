import pytest

from transparency_sim.environment import (
    BlindIDEnvironment,
    BudgetExhausted,
    DepthExceeded,
    InvalidResolve,
)
from transparency_sim.generator import generate_corpus


def test_list_ids_is_free_and_matches_display_order():
    corpus = generate_corpus(q=20, r=4, c=0.5, seed=1)
    env = BlindIDEnvironment(corpus, budget=3)

    assert env.list_ids() == tuple(d.doc_id for d in corpus.docs)
    assert env.budget_remaining == 3
    assert env.transcript()[-1].op == "list"
    assert env.transcript()[-1].cost == 0


def test_fetch_costs_one_and_returns_body():
    corpus = generate_corpus(q=20, r=4, c=0.5, seed=2)
    env = BlindIDEnvironment(corpus, budget=3)
    doc_id = corpus.docs[0].doc_id

    view = env.fetch(doc_id)

    assert env.budget_remaining == 2
    assert view.body == corpus.doc(doc_id).body


def test_fetch_view_has_no_core_status_fields():
    corpus = generate_corpus(q=20, r=4, c=0.5, seed=3)
    env = BlindIDEnvironment(corpus, budget=3)

    view = env.fetch(corpus.docs[0].doc_id)

    assert hasattr(view, "is_core") is False
    assert hasattr(view, "component_key") is False


def test_refetch_of_directly_fetched_doc_is_free():
    corpus = generate_corpus(q=20, r=4, c=0.5, seed=4)
    env = BlindIDEnvironment(corpus, budget=3)
    doc_id = corpus.docs[0].doc_id

    env.fetch(doc_id)
    view = env.fetch(doc_id)

    assert view.body == corpus.doc(doc_id).body
    assert env.budget_remaining == 2
    assert env.transcript()[-1].cost == 0
    assert env.transcript()[-1].depth == 0


def test_fetch_when_budget_exhausted_raises():
    corpus = generate_corpus(q=20, r=4, c=0.5, seed=5)
    env = BlindIDEnvironment(corpus, budget=1)

    env.fetch(corpus.docs[0].doc_id)
    with pytest.raises(BudgetExhausted):
        env.fetch(corpus.docs[1].doc_id)


def test_resolve_requires_obtained_source_and_seen_reference():
    corpus = generate_corpus(q=20, r=4, c=1.0, seed=6)
    env = BlindIDEnvironment(corpus, budget=2)
    src, target = corpus.core_ids[:2]

    with pytest.raises(InvalidResolve):
        env.resolve(src, target)

    env.fetch(src)
    outsider = next(d.doc_id for d in corpus.docs if d.doc_id not in corpus.doc(src).refs)
    with pytest.raises(InvalidResolve):
        env.resolve(src, outsider)


def test_resolve_is_free_at_kappa_zero():
    corpus = generate_corpus(q=20, r=4, c=1.0, seed=7)
    env = BlindIDEnvironment(corpus, budget=2)
    src, target = corpus.core_ids[:2]

    env.fetch(src)
    before = env.budget_remaining
    view = env.resolve(src, target)

    assert view.body == corpus.doc(target).body
    assert env.budget_remaining == before
    assert env.transcript()[-1].cost == 0


def test_depth_one_blocks_second_hop():
    corpus = generate_corpus(q=20, r=4, c=1.0, seed=8)
    env = BlindIDEnvironment(corpus, budget=2, depth=1)
    src, mid, target = corpus.core_ids[:3]

    env.fetch(src)
    env.resolve(src, mid)
    with pytest.raises(DepthExceeded):
        env.resolve(mid, target)


def test_paid_refetch_reanchors_depth():
    corpus = generate_corpus(q=20, r=4, c=1.0, seed=9)
    env = BlindIDEnvironment(corpus, budget=2, depth=1)
    src, mid, target = corpus.core_ids[:3]

    env.fetch(src)
    env.resolve(src, mid)
    env.fetch(mid)
    view = env.resolve(mid, target)

    assert view.doc_id == target
    assert env.budget_remaining == 0
    assert env.transcript()[-2].op == "fetch"
    assert env.transcript()[-2].cost == 1
    assert env.transcript()[-2].depth == 0


def test_kappa_positive_not_implemented():
    corpus = generate_corpus(q=20, r=4, c=0.5, seed=10)

    with pytest.raises(NotImplementedError):
        BlindIDEnvironment(corpus, budget=2, kappa=0.5)


def test_depth_and_kappa_are_read_only_public_resource_profile():
    corpus = generate_corpus(q=20, r=4, c=0.5, seed=12)
    env = BlindIDEnvironment(corpus, budget=2, depth=1, kappa=0.0)

    assert env.depth == 1
    assert env.kappa == 0.0


def test_transcript_records_ops_costs_depths():
    corpus = generate_corpus(q=20, r=4, c=1.0, seed=11)
    env = BlindIDEnvironment(corpus, budget=2, depth=1)
    src, target = corpus.core_ids[:2]

    env.list_ids()
    env.fetch(src)
    env.fetch(src)
    env.resolve(src, target)

    assert [(e.op, e.cost, e.depth) for e in env.transcript()] == [
        ("list", 0, None),
        ("fetch", 1, 0),
        ("fetch", 0, 0),
        ("resolve", 0, 1),
    ]
