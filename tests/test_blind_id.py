import pytest

from transparency_sim.a0 import a0_exact
from transparency_sim.blind_id import (
    NullPolicy,
    ScriptedSequentialPolicy,
    extract_answers,
    run_blind_id,
)
from transparency_sim.generator import generate_corpus


def test_extractor_roundtrip_recovers_y0():
    corpus = generate_corpus(q=30, r=8, c=0.5, seed=50)
    bodies = [d.body for d in corpus.core_documents]

    assert extract_answers(bodies, corpus.r) == corpus.y0


def test_extractor_rejects_r_above_eight():
    with pytest.raises(ValueError):
        extract_answers([], 9)


def test_null_policy_distortion_one():
    corpus = generate_corpus(q=30, r=5, c=0.5, seed=51)
    run = run_blind_id(corpus, NullPolicy(), budget=10, depth="inf")

    assert run.distortion_answer == 1.0


def test_full_budget_recovers_everything():
    corpus = generate_corpus(q=30, r=5, c=0.0, seed=52)
    run = run_blind_id(corpus, ScriptedSequentialPolicy(policy_seed=0), budget=30, depth="inf")

    assert run.distortion_answer == 0.0


def test_pigeonhole_budget_at_c1_gives_zero_distortion():
    q, r = 30, 4
    corpus = generate_corpus(q=q, r=r, c=1.0, seed=53)
    run = run_blind_id(
        corpus,
        ScriptedSequentialPolicy(policy_seed=0),
        budget=q - r + 1,
        depth="inf",
    )

    assert run.distortion_answer == 0.0


def test_answer_distortion_equals_recovery_distortion():
    corpus = generate_corpus(q=50, r=5, c=0.5, seed=2)

    for seed in range(10):
        run = run_blind_id(
            corpus,
            ScriptedSequentialPolicy(policy_seed=seed),
            budget=10,
            depth="inf",
        )
        assert run.distortion_answer == pytest.approx(run.distortion_recovery)


def test_mean_distortion_near_a0_baseline():
    corpus = generate_corpus(q=50, r=5, c=0.5, seed=2)
    runs = [
        run_blind_id(corpus, ScriptedSequentialPolicy(policy_seed=seed), budget=10, depth="inf")
        for seed in range(200)
    ]
    mean = sum(run.distortion_answer for run in runs) / len(runs)
    baseline = a0_exact(corpus, 10, "inf").distortion

    assert mean == pytest.approx(baseline, abs=0.15)


def test_policy_spends_exactly_budget():
    corpus = generate_corpus(q=50, r=5, c=0.5, seed=2)

    for seed in range(10):
        run = run_blind_id(
            corpus,
            ScriptedSequentialPolicy(policy_seed=seed),
            budget=10,
            depth="inf",
        )
        assert run.n_fetch_paid == 10
