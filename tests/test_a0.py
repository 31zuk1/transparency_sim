from pathlib import Path

import pytest

from transparency_sim import theory
from transparency_sim.a0 import a0_exact, expected_recovery
from transparency_sim.generator import generate_corpus

ROOT = Path(__file__).resolve().parents[1]


def test_c0_matches_linear_identity():
    corp = generate_corpus(q=50, r=5, c=0.0, seed=1)
    for depth in (1, "inf"):
        res = a0_exact(corp, B=10, depth=depth)
        assert res.distortion == pytest.approx(1 - 10 / 50)


def test_c1_matches_seed_failure_probability():
    corp = generate_corpus(q=50, r=5, c=1.0, seed=3)
    for depth in (1, "inf"):
        res = a0_exact(corp, B=10, depth=depth)
        assert res.distortion == pytest.approx(theory.hypergeom_p_zero(50, 5, 10))


def test_zero_budget_gives_full_distortion():
    corp = generate_corpus(q=40, r=5, c=0.5, seed=2)
    assert a0_exact(corp, B=0, depth="inf").distortion == pytest.approx(1.0)


def test_full_budget_gives_zero_distortion():
    corp = generate_corpus(q=40, r=5, c=0.5, seed=2)
    assert a0_exact(corp, B=40, depth="inf").distortion == pytest.approx(0.0)


def test_distortion_increases_in_q_on_fixed_graph():
    # same reference graph, larger q -> strictly larger distortion (Prop. 1 direction)
    corp = generate_corpus(q=20, r=5, c=0.5, seed=4)
    adj = corp.core_adjacency()
    for depth in (1, "inf"):
        d_small = 1 - expected_recovery(adj, corp.core_ids, q=20, B=8, depth=depth) / 5
        d_large = 1 - expected_recovery(adj, corp.core_ids, q=30, B=8, depth=depth) / 5
        assert d_large > d_small


def test_deeper_tracking_weakly_helps():
    corp = generate_corpus(q=50, r=6, c=0.4, seed=8)
    d1 = a0_exact(corp, B=12, depth=1).distortion
    di = a0_exact(corp, B=12, depth="inf").distortion
    assert di <= d1 + 1e-12


def test_no_banned_phrases_in_code_or_docs():
    banned = [
        "D* " + "oracle", "optimal " + "observer", "class-" + "optimal",
        "true " + "D*", "full " + "recovery", "stopping " + "time",
        "r/(r+1)", "r(q+1)/(r+1)",
    ]
    targets = list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py"))
    targets.append(ROOT / "README.md")
    for path in targets:
        text = path.read_text(encoding="utf-8").lower()
        for phrase in banned:
            assert phrase.lower() not in text, f"{phrase!r} found in {path}"


def test_graph_averaged_d1_matches_closed_form_by_full_enumeration():
    # E over ALL reference graphs (weights c^e (1-c)^(r(r-1)-e)) of the
    # fixed-graph A0 value at d = 1 equals the closed form of eq. (3).
    # Single fixed graphs need not match; only the graph average does.
    from itertools import product
    r, q, B = 3, 12, 4
    cores = ("DOC_0001", "DOC_0002", "DOC_0003")
    pairs = [(a, b) for a in cores for b in cores if a != b]  # 6 ordered pairs
    for c in (0.3, 0.7):
        avg = 0.0
        for mask in product((0, 1), repeat=len(pairs)):
            w = 1.0
            adj = {x: set() for x in cores}
            for bit, (a, b) in zip(mask, pairs):
                if bit:
                    adj[a].add(b)
                    w *= c
                else:
                    w *= 1.0 - c
            avg += w * expected_recovery(adj, cores, q=q, B=B, depth=1)
        assert avg == pytest.approx(theory.ef_one_hop(q, r, B, c), abs=1e-12)


def test_single_fixed_graph_need_not_match_closed_form():
    # The empty graph is a positive-probability realization at c = 0.5, but
    # its conditional value (= E[m] = Br/q) differs from the graph-averaged
    # closed form. This documents the distinction A0 must respect.
    r, q, B, c = 3, 12, 4, 0.5
    cores = ("DOC_0001", "DOC_0002", "DOC_0003")
    empty = {x: set() for x in cores}
    fixed = 1 * expected_recovery(empty, cores, q=q, B=B, depth=1)
    assert fixed == pytest.approx(B * r / q)
    assert abs(fixed - theory.ef_one_hop(q, r, B, c)) > 0.1
