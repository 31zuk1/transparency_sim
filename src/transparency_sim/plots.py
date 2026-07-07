"""Figure 1 (draft v0.4, §5.6): two-sided bounds of Proposition 3.

Panel (a): exact Pr(m = 0) with its two-sided bounds (q = 150, r = 5).
Panel (b): the three budget lines against the dilution ratio rho = q/r:
  - universal lower bound      (q - r + 1)(1 - alpha^(1/r))   [all c, all L]
  - complete structure, c = 1  q(1 - alpha^(1/r))             [achievable upper]
  - no structure, c = 0        q(1 - alpha)                   [linear loss, exact]

The c = 0 comparison quantity is the required budget under the linear
recovery-distortion map (Assumption 4), B*(delta; q, 0) = q(1 - alpha).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import theory

FIG1A_PARAMS = {"q": 150, "r": 5}
FIG1B_PARAMS = {"r": 5, "alpha": 0.05, "rho_min": 10, "rho_max": 500}

FIG1B_LABELS = {
    "lower": "all L:  B*(δ; q, c) ≥ (q − r + 1)(1 − α^(1/r))",
    "c1": "c = 1:  B*(δ; q, 1) ≤ q(1 − α^(1/r))",
    "c0": "c = 0:  B*(δ; q, 0) = q(1 − α)  (linear loss)",
}


def fig1a_curves(q: int, r: int) -> dict[str, np.ndarray]:
    Bs = np.arange(0, q - r + 1)
    return {
        "B": Bs,
        "exact": np.array([theory.hypergeom_p_zero(q, r, int(b)) for b in Bs]),
        "upper": np.array([theory.p_zero_upper(q, r, int(b)) for b in Bs]),
        "lower": np.array([theory.p_zero_lower(q, r, int(b)) for b in Bs]),
    }


def fig1b_lines(rhos: np.ndarray, r: int, alpha: float) -> dict[str, np.ndarray]:
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
            label="upper:  (1 − B/q)^r")
    ax.plot(a["B"], a["exact"], "-", lw=1.8, color="#1a5276",
            label="exact:  Pr(m = 0) = C(q−r, B)/C(q, B)")
    ax.plot(a["B"], a["lower"], ":", lw=1.4, color="#b03a2e",
            label="lower:  (1 − B/(q−r+1))^r")
    ax.set_xlabel("direct-acquisition budget B")
    ax.set_ylabel("Pr(m = 0)")
    ax.set_title(f"(a) Seed-failure probability and two-sided bounds\n"
                 f"(q = {q}, r = {r}; Prop. 3, App. A.4)", fontsize=10)
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
    ax.set_xlabel("dilution ratio ρ = q/r")
    ax.set_ylabel("required budget B*(δ; q, c)")
    ax.set_title(f"(b) Structure buys the coefficient, not the scaling\n"
                 f"(r = {r}, α = δ/D₀ = {alpha}; Assumption 4)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_xlim(FIG1B_PARAMS["rho_min"], FIG1B_PARAMS["rho_max"])

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
