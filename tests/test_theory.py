import pytest

from transparency_sim import theory
from transparency_sim.plots import FIG1B_LABELS, fig1b_lines


def test_p_zero_no_budget_is_one():
    assert theory.hypergeom_p_zero(50, 5, 0) == 1.0


def test_p_zero_at_full_distractor_budget_positive():
    assert theory.hypergeom_p_zero(50, 5, 45) > 0.0


def test_p_zero_beyond_pigeonhole_is_zero():
    # B = q - r + 1: every draw set must hit a core; handled explicitly as 0.
    assert theory.hypergeom_p_zero(50, 5, 46) == 0.0


def test_bounds_sandwich_exact_value():
    q, r = 150, 5
    for B in [0, 10, 30, 90, 145]:
        p = theory.hypergeom_p_zero(q, r, B)
        assert theory.p_zero_lower(q, r, B) <= p + 1e-12
        assert p <= theory.p_zero_upper(q, r, B) + 1e-12


def test_budget_c0_linear_identity():
    q, alpha = 200, 0.05
    assert theory.budget_c0_linear(q, alpha) == pytest.approx(q * (1 - alpha))


def test_fig1b_c0_line_is_q_times_one_minus_alpha():
    r, alpha = 5, 0.05
    lines = fig1b_lines([10.0, 100.0, 500.0], r, alpha)
    for rho, val in zip(lines["rho"], lines["c0"]):
        q = r * rho
        assert val == pytest.approx(q * (1 - alpha))
    # the legend advertises the same identity and no banned comparison quantity
    assert r"q(1-\alpha)" in FIG1B_LABELS["c0"]
    banned = ["full " + "recovery", "stopping " + "time", "r/(r+1)", "r(q+1)/(r+1)"]
    for label in FIG1B_LABELS.values():
        for phrase in banned:
            assert phrase.lower() not in label.lower()


def test_budget_bracket_ordering_and_width():
    q, r, alpha = 2500, 5, 0.05
    lo = theory.budget_lower(q, r, alpha)
    hi = theory.budget_c1_upper(q, r, alpha)
    c0 = theory.budget_c0_linear(q, alpha)
    assert lo < hi < c0
    assert hi - lo < r  # bracket width (r-1)(1-alpha^(1/r)) < r
