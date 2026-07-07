"""Figure 1 (draft v0.4, §5.6): two-sided bounds of Proposition 3.

Panel (a): exact Pr(m = 0) with its two-sided bounds (q = 150, r = 5).
Panel (b): the three budget lines against the dilution ratio rho = q/r:
  - universal lower bound      (q - r + 1)(1 - alpha^(1/r))   [all c, all L]
  - complete structure, c = 1  q(1 - alpha^(1/r))             [achievable upper]
  - no structure, c = 0        q(1 - alpha)                   [linear loss, exact]

The c = 0 comparison quantity is the required budget under the linear
recovery-distortion map (Assumption 4), B*(delta; q, 0) = q(1 - alpha).
The lower bound nearly coincides with the c = 1 line by construction:
the bracket width is (r - 1)(1 - alpha^(1/r)) < r documents (Prop. 3(ii)).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import theory

FIG1A_PARAMS = {"q": 150, "r": 5}
FIG1B_PARAMS = {"r": 5, "alpha": 0.05, "rho_min": 10, "rho_max": 500}

FIG1B_LABELS = {
    "lower": r"universal lower bound (all $c$, all $L$): $(q{-}r{+}1)(1-\alpha^{1/r})$",
    "c1": r"$c = 1$:  $B^*(\delta;q,1) \leq q(1-\alpha^{1/r})$",
    "c0": r"$c = 0$:  $B^*(\delta;q,0) = q(1-\alpha)$  (linear loss)",
}


def fig1a_curves(q: int, r: int) -> dict[str, np.ndarray]:
    Bs = np.arange(0, q - r + 1)
    return {
        "B": Bs,
        "exact": np.array([theory.hypergeom_p_zero(q, r, int(b)) for b in Bs]),
        "upper": np.array([theory.p_zero_upper(q, r, int(b)) for b in Bs]),
        "lower": np.array([theory.p_zero_lower(q, r, int(b)) for b in Bs]),
    }


def fig1b_lines(rhos, r: int, alpha: float) -> dict[str, np.ndarray]:
    qs = r * np.asarray(rhos, dtype=float)
    return {
        "rho": np.asarray(rhos, dtype=float),
        "lower": (qs - r + 1) * (1.0 - alpha ** (1.0 / r)),
        "c1": qs * (1.0 - alpha ** (1.0 / r)),
        "c0": qs * (1.0 - alpha),
    }


def make_fig1(out_png: Path, out_pdf: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    # ---- panel (a) ----
    q, r = FIG1A_PARAMS["q"], FIG1A_PARAMS["r"]
    a = fig1a_curves(q, r)
    ax = axes[0]
    ax.plot(a["B"], a["upper"], "--", lw=1.2, color="#888888",
            label=r"upper: $(1 - B/q)^{r}$")
    ax.plot(a["B"], a["exact"], "-", lw=1.8, color="#1a5276",
            label=r"exact: $\Pr(m{=}0) = C(q{-}r,\,B)\,/\,C(q,\,B)$")
    ax.plot(a["B"], a["lower"], ":", lw=1.4, color="#b03a2e",
            label=r"lower: $(1 - B/(q{-}r{+}1))^{r}$")
    ax.set_xlabel(r"direct-acquisition budget $B$")
    ax.set_ylabel(r"$\Pr(m = 0)$")
    ax.set_title("(a) Seed-failure probability and two-sided bounds\n"
                 + rf"($q = {q}$, $r = {r}$; Prop. 3, App. A.4)", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(0, q - r)
    ax.set_ylim(0, 1)

    # ---- panel (b) ----
    r, alpha = FIG1B_PARAMS["r"], FIG1B_PARAMS["alpha"]
    rhos = np.linspace(FIG1B_PARAMS["rho_min"], FIG1B_PARAMS["rho_max"], 200)
    b = fig1b_lines(rhos, r, alpha)
    ax = axes[1]
    ax.plot(b["rho"], b["c0"], "-", lw=1.8, color="#7d6608", label=FIG1B_LABELS["c0"])
    ax.fill_between(b["rho"], b["lower"], b["c1"], color="#1a5276", alpha=0.25, lw=0)
    ax.plot(b["rho"], b["c1"], "-", lw=1.6, color="#1a5276", label=FIG1B_LABELS["c1"])
    ax.plot(b["rho"], b["lower"], ":", lw=1.6, color="#b03a2e", label=FIG1B_LABELS["lower"])
    ax.annotate("lower bound nearly coincides with the $c = 1$ line:\n"
                r"bracket width $(r{-}1)(1-\alpha^{1/r}) < r$ documents",
                xy=(330, float(fig1b_lines([330], r, alpha)["c1"][0])),
                xytext=(150, 1350), fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#444444"))
    ax.set_xlabel(r"dilution ratio $\rho = q/r$")
    ax.set_ylabel(r"required budget $B^*(\delta;\,q,\,c)$")
    ax.set_title("(b) Structure buys the coefficient, not the scaling\n"
                 + rf"($r = {r}$, $\alpha = \delta/D_0 = {alpha}$; Assumption 4)",
                 fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_xlim(FIG1B_PARAMS["rho_min"], FIG1B_PARAMS["rho_max"])
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
