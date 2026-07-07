"""Regenerate Figure 1 (draft v0.4, §5.6) into outputs/figures/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transparency_sim.plots import make_fig1, fig1b_lines, FIG1B_PARAMS  # noqa: E402

out_png = ROOT / "outputs" / "figures" / "fig1_budget_bounds.png"
out_pdf = ROOT / "outputs" / "figures" / "fig1_budget_bounds.pdf"
make_fig1(out_png, out_pdf)

r, alpha = FIG1B_PARAMS["r"], FIG1B_PARAMS["alpha"]
lines = fig1b_lines([FIG1B_PARAMS["rho_max"]], r, alpha)
width = float(lines["c1"][0] - lines["lower"][0])
print(f"saved: {out_png}")
print(f"saved: {out_pdf}")
print(f"bracket width at rho={FIG1B_PARAMS['rho_max']}: {width:.3f} (< r - 1 = {r - 1}: {width < r - 1})")
