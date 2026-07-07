"""Closed-form quantities from draft v0.4, Proposition 3 and Appendix A.4.

All quantities refer to the batch-then-track restricted policy class
(Pi^seed) unless stated otherwise. Distortion is normalized: D0 = 1.
"""
from __future__ import annotations

from math import comb


def _check_qrB(q: int, r: int, B: int) -> None:
    if not (isinstance(q, int) and isinstance(r, int) and isinstance(B, int)):
        raise TypeError("q, r, B must be integers")
    if not (1 <= r <= q):
        raise ValueError(f"need 1 <= r <= q, got r={r}, q={q}")
    if not (0 <= B <= q):
        raise ValueError(f"need 0 <= B <= q, got B={B}")


def hypergeom_p_zero(q: int, r: int, B: int) -> float:
    """Exact seed-failure probability Pr(m = 0) = C(q-r, B) / C(q, B).

    For B > q - r the pigeonhole principle forces at least one core hit,
    so the probability is exactly 0 (math.comb returns 0 for k > n,
    which handles this domain edge explicitly).
    """
    _check_qrB(q, r, B)
    return comb(q - r, B) / comb(q, B)


def p_zero_upper(q: int, r: int, B: int) -> float:
    """Upper bound (1 - B/q)^r on Pr(m = 0). (Appendix A.4.)"""
    _check_qrB(q, r, B)
    return (1.0 - B / q) ** r


def p_zero_lower(q: int, r: int, B: int) -> float:
    """Lower bound (1 - B/(q-r+1))^r on Pr(m = 0). (Appendix A.4.)

    Clamped at 0 for B > q - r + 1, where the bound is vacuous.
    """
    _check_qrB(q, r, B)
    return max(0.0, 1.0 - B / (q - r + 1)) ** r


def budget_lower(q: int, r: int, alpha: float) -> float:
    """Universal lower bound on the required direct-acquisition budget,
    B*(delta; q, c) >= (q - r + 1)(1 - alpha^(1/r)) for every c and every
    reference structure L. (Proposition 3(i), eq. (10); alpha = delta/D0.)
    """
    _check_alpha(alpha)
    return (q - r + 1) * (1.0 - alpha ** (1.0 / r))


def budget_c1_upper(q: int, r: int, alpha: float) -> float:
    """Achievable upper bound at complete connectivity,
    B*(delta; q, 1) <= q(1 - alpha^(1/r)). (Proposition 3(ii).)
    """
    _check_alpha(alpha)
    return q * (1.0 - alpha ** (1.0 / r))


def budget_c0_linear(q: int, alpha: float) -> float:
    """Required budget with no reference structure, under the linear
    recovery-distortion map (Assumption 4): B*(delta; q, 0) = q(1 - alpha).

    At c = 0, E[D]/D0 = 1 - B/q holds exactly, hence the identity.
    """
    _check_alpha(alpha)
    return q * (1.0 - alpha)


def _check_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"need 0 < alpha < 1, got alpha={alpha}")
