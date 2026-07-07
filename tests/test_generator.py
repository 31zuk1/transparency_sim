import re

import pytest

from transparency_sim.generator import generate_corpus, validate_corpus


@pytest.fixture(scope="module")
def corp():
    return generate_corpus(q=60, r=5, c=0.5, seed=7)


def test_counts(corp):
    assert len(corp.docs) == 60
    assert len(corp.core_documents) == 5
    assert len(corp.distractor_documents) == 55


def test_each_core_carries_exactly_one_component(corp):
    values = list(corp.y0.values())
    seen = set()
    for d in corp.core_documents:
        present = [v for v in values if v in d.body]
        assert present == [corp.y0[d.component_key]]
        seen.add(d.component_key)
    assert seen == set(corp.y0.keys())


def test_distractors_do_not_leak_y0(corp):
    for d in corp.distractor_documents:
        for v in corp.y0.values():
            assert v not in d.body
            assert v.lower() not in d.body.lower()


def test_references_only_among_cores(corp):
    core_ids = set(corp.core_ids)
    for src, dst in corp.edges:
        assert src in core_ids and dst in core_ids
    for d in corp.core_documents:
        for t in d.refs:
            assert t in core_ids


def test_distractors_have_no_resolvable_references(corp):
    for d in corp.distractor_documents:
        assert d.refs == ()
        assert "REF:" not in d.body


def test_core_status_not_inferable_from_id_or_order(corp):
    # same id format for everyone
    fmt = re.compile(r"^DOC_[0-9A-F]{4}$")
    assert all(fmt.match(d.doc_id) for d in corp.docs)
    # cores are neither the leading nor trailing display block
    positions = [i for i, d in enumerate(corp.docs) if d.is_core]
    assert positions != list(range(corp.r))
    assert positions != list(range(corp.q - corp.r, corp.q))
    # cores are not the lexicographically smallest ids
    sorted_ids = sorted(d.doc_id for d in corp.docs)
    assert set(corp.core_ids) != set(sorted_ids[: corp.r])


def test_reference_probability_extremes():
    c0 = generate_corpus(q=30, r=4, c=0.0, seed=11)
    assert c0.edges == ()
    c1 = generate_corpus(q=30, r=4, c=1.0, seed=12)
    assert len(c1.edges) == 4 * 3  # complete digraph on cores


def test_validate_runs_clean(corp):
    validate_corpus(corp)  # must not raise


def test_determinism():
    a = generate_corpus(q=40, r=5, c=0.3, seed=99)
    b = generate_corpus(q=40, r=5, c=0.3, seed=99)
    assert a.y0 == b.y0
    assert [d.doc_id for d in a.docs] == [d.doc_id for d in b.docs]
    assert a.edges == b.edges
