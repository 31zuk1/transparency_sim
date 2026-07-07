"""Light smoke run of the A0 calibration baseline (no experiments, no LLM)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transparency_sim import theory  # noqa: E402
from transparency_sim.a0 import a0_exact  # noqa: E402
from transparency_sim.generator import generate_corpus  # noqa: E402

settings = [
    {"q": 50, "r": 5, "c": 0.0, "B": 10, "seed": 1},
    {"q": 50, "r": 5, "c": 0.5, "B": 10, "seed": 2},
    {"q": 50, "r": 5, "c": 1.0, "B": 10, "seed": 3},
]

lines = ["A0 smoke run: D_seed_d on one fixed corpus per setting (D0 = 1)", ""]
header = f"{'q':>4} {'r':>3} {'c':>4} {'B':>4} {'seed':>4} | {'E[F] d=1':>9} {'D d=1':>7} | {'E[F] d=inf':>10} {'D d=inf':>8} | theory reference"
lines.append(header)
lines.append("-" * len(header))
for s in settings:
    corp = generate_corpus(q=s["q"], r=s["r"], c=s["c"], seed=s["seed"])
    r1 = a0_exact(corp, s["B"], depth=1)
    ri = a0_exact(corp, s["B"], depth="inf")
    if s["c"] == 0.0:
        ref = f"c=0 exact: D = 1 - B/q = {1 - s['B']/s['q']:.4f}"
    elif s["c"] == 1.0:
        ref = f"c=1 exact: D = Pr(m=0) = {theory.hypergeom_p_zero(s['q'], s['r'], s['B']):.4f}"
    else:
        ref = "0 < c < 1: graph-specific value; d=inf distortion <= d=1"
    lines.append(f"{s['q']:>4} {s['r']:>3} {s['c']:>4} {s['B']:>4} {s['seed']:>4} | "
                 f"{r1.expected_recovery:>9.4f} {r1.distortion:>7.4f} | "
                 f"{ri.expected_recovery:>10.4f} {ri.distortion:>8.4f} | {ref}")
    assert ri.distortion <= r1.distortion + 1e-12, "deeper tracking must not hurt"

text = "\n".join(lines)
print(text)
log = ROOT / "outputs" / "logs" / "a0_smoke.txt"
log.parent.mkdir(parents=True, exist_ok=True)
log.write_text(text + "\n", encoding="utf-8")
print(f"\nlog written: {log}")
