import pytest

from transparency_sim.generator import COMPONENT_TYPES, generate_corpus
from transparency_sim.scoring import component_schema, recovery_distortion, score_answer


def test_perfect_answer_scores_zero():
    corpus = generate_corpus(q=20, r=4, c=0.5, seed=20)

    result = score_answer(corpus.y0, dict(corpus.y0))

    assert result.distortion == 0.0
    assert all(result.per_component.values())


def test_empty_answer_scores_one():
    corpus = generate_corpus(q=20, r=4, c=0.5, seed=21)

    result = score_answer(corpus.y0, {})

    assert result.distortion == 1.0
    assert not any(result.per_component.values())


def test_partial_answer_is_linear():
    corpus = generate_corpus(q=20, r=5, c=0.5, seed=22)
    answer = dict(list(corpus.y0.items())[:2])

    result = score_answer(corpus.y0, answer)

    assert result.distortion == pytest.approx(0.6)


def test_normalization_absorbs_whitespace_and_case():
    y0 = {"component_1": "the Harbor renewal board"}
    answer = {"component_1": "  THE Harbor   renewal board "}

    assert score_answer(y0, answer).distortion == 0.0


def test_unknown_key_raises():
    y0 = {"component_1": "value"}

    with pytest.raises(ValueError):
        score_answer(y0, {"component_2": "value"})


def test_schema_matches_generator_cycle():
    for r in (5, 8):
        corpus = generate_corpus(q=20, r=r, c=0.5, seed=30 + r)
        schema = component_schema(r)

        assert tuple(key for key, _ in schema) == tuple(corpus.y0)
        assert tuple(ctype for _, ctype in schema) == tuple(
            COMPONENT_TYPES[(i - 1) % 8] for i in range(1, r + 1)
        )


def test_recovery_distortion_counts_core_components():
    corpus = generate_corpus(q=20, r=5, c=0.5, seed=40)
    distractor = next(d.doc_id for d in corpus.docs if not d.is_core)
    obtained = [corpus.core_ids[0], corpus.core_ids[1], distractor]

    assert recovery_distortion(corpus, obtained) == pytest.approx(1.0 - 2 / 5)
